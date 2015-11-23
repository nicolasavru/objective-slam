#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>                // Stops underlining of __global__
#include <thrust/sort.h>
#include <thrust/inner_product.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/scan.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/device_vector.h>
#include <thrust/binary_search.h>

#include "model.h"
#include "impl/ppf_utils.hpp"
#include "impl/util.hpp"
#include "impl/parallel_hash_array.hpp"
#include "kernel.h"
#include "book.h"

struct high_32_bits : public thrust::unary_function<unsigned long,unsigned int>{
    __host__ __device__
    unsigned int operator()(unsigned long i) const {
        return (unsigned int) (i >> 32);
    }
};


struct low_32_bits : public thrust::unary_function<unsigned long,unsigned int>{
    __host__ __device__
    unsigned int operator()(unsigned long i) const {
        return (unsigned int) (i & (-1ul >> 32));
    }
};

struct float16 {
    float f[16];
};

Model::Model(thrust::host_vector<float3> *points, thrust::host_vector<float3> *normals, int n){
    this->initPPFs(points, normals, n);
    this->search_array = ParallelHashArray<float4>(*(this->modelPPFs));
}

Model::~Model(){
    // TODO
}

void Model::ppf_lookup(Scene *scene){

    #ifdef DEBUG
        cudaEvent_t start, stop;
        HANDLE_ERROR(cudaEventCreate(&start));
        HANDLE_ERROR(cudaEventCreate(&stop));
        HANDLE_ERROR(cudaEventRecord(start, 0));
    #endif

    thrust::device_vector<std::size_t> *sceneIndices =
        this->search_array.GetIndices(*(scene->getHashKeys()));
    // Steps 1-3
    // launch voting kernel instance for each scene reference point
    unsigned int lastIndex, lastCount;
    thrust::device_vector<unsigned long> *votes_old = new thrust::device_vector<unsigned long>(scene->getModelPPFs()->size()*this->modelPoints->size(),0);

    // populates parallel arrays votes and vecs_old
    int blocks = std::min(((int)(scene->getHashKeys()->size()) + BLOCK_SIZE - 1) / BLOCK_SIZE, MAX_NBLOCKS);
    ppf_vote_kernel<<<blocks,BLOCK_SIZE>>>
        (RAW_PTR(scene->getHashKeys()), RAW_PTR(sceneIndices),
         RAW_PTR(this->search_array.GetHashkeys()), RAW_PTR(this->search_array.GetCounts()),
         RAW_PTR(this->search_array.GetFirstHashkeyIndices()),
         RAW_PTR(this->search_array.GetHashkeyToDataMap()),
         RAW_PTR(this->modelPoints), RAW_PTR(this->modelNormals),
         this->n, RAW_PTR(scene->getModelPoints()),
         RAW_PTR(scene->getModelNormals()), scene->numPoints(),
         RAW_PTR(votes_old),
         scene->getHashKeys()->size());

    thrust::sort(votes_old->begin(), votes_old->end());
    this->votes = new thrust::device_vector<unsigned long>();
    this->voteCounts = new thrust::device_vector<unsigned int>();
    histogram_destructive(*votes_old, *(this->votes), *(this->voteCounts));

    this->transformations = new thrust::device_vector<float>(this->votes->size()*16);
    /* DEBUG */
    fprintf(stderr, "votes_size: %d\n", this->votes->size());
    /* DEBUG */

    blocks = std::min(((int)(this->votes->size()) + BLOCK_SIZE - 1) / BLOCK_SIZE, MAX_NBLOCKS);


    trans_calc_kernel2<<<blocks,BLOCK_SIZE>>>
        (RAW_PTR(votes),
         RAW_PTR(this->modelPoints), RAW_PTR(this->modelNormals),
         RAW_PTR(scene->getModelPoints()), RAW_PTR(scene->getModelNormals()),
         RAW_PTR(this->transformations),
         this->votes->size());

    this->transformation_trans = new thrust::device_vector<float3>(this->votes->size());
    this->transformation_rots = new thrust::device_vector<float4>(this->votes->size());
    mat2transquat_kernel<<<blocks,BLOCK_SIZE>>>
        (RAW_PTR(this->transformations),
         RAW_PTR(this->transformation_trans),
         RAW_PTR(this->transformation_rots),
         this->votes->size());

    thrust::device_vector<unsigned int> *nonunique_trans_hash =
        new thrust::device_vector<unsigned int>(this->votes->size());
    thrust::device_vector<unsigned int> *adjacent_trans_hash =
        new thrust::device_vector<unsigned int>(27*this->votes->size());
    trans2idx_kernel<<<blocks,BLOCK_SIZE>>>
        (RAW_PTR(this->transformation_trans),
         RAW_PTR(nonunique_trans_hash),
         RAW_PTR(adjacent_trans_hash),
         this->votes->size());

    this->key2transMap = new thrust::device_vector<unsigned int>(this->votes->size());
    thrust::sequence(key2transMap->begin(), key2transMap->end());
    thrust::sort_by_key(nonunique_trans_hash->begin(),
                        nonunique_trans_hash->end(),
                        key2transMap->begin());

    this->trans_hash = new thrust::device_vector<unsigned int>(this->votes->size());
    this->transCount = new thrust::device_vector<unsigned int>(this->votes->size());
    histogram_destructive(*nonunique_trans_hash, *(this->trans_hash), *(this->transCount));
    delete nonunique_trans_hash;
    this->firstTransIndex = new thrust::device_vector<unsigned int>(this->votes->size());
    thrust::exclusive_scan(this->transCount->begin(),
                           this->transCount->end(),
                           this->firstTransIndex->begin());

    thrust::device_vector<unsigned int> *transIndices =
        new thrust::device_vector<unsigned int>(adjacent_trans_hash->size());
    thrust::lower_bound(this->trans_hash->begin(),
                        this->trans_hash->end(),
                        adjacent_trans_hash->begin(),
                        adjacent_trans_hash->end(),
                        transIndices->begin());

    // write_device_vector("adjacent_trans_hash", adjacent_trans_hash);
    // write_device_vector("transCount", transCount);

    this->vote_counts_out = new thrust::device_vector<unsigned int>(*(this->voteCounts));
    blocks = std::min(((int)(this->votes->size()) + BLOCK_SIZE - 1) / BLOCK_SIZE, MAX_NBLOCKS);
    rot_clustering_kernel<<<blocks,BLOCK_SIZE>>>
        (RAW_PTR(this->transformation_trans), RAW_PTR(this->transformation_rots),
         RAW_PTR(this->voteCounts), RAW_PTR(adjacent_trans_hash),
         RAW_PTR(transIndices), RAW_PTR(this->trans_hash),
         RAW_PTR(this->transCount), RAW_PTR(this->firstTransIndex),
         RAW_PTR(this->key2transMap), RAW_PTR(this->vote_counts_out),
         this->votes->size());

    thrust::sort_by_key(this->vote_counts_out->begin(),
                        this->vote_counts_out->end(),
                        thrust::device_ptr<struct float16>((struct float16 *) RAW_PTR(this->transformations)),
                        thrust::greater<unsigned int>());


    // thrust::device_vector<unsigned int> *uniqueSceneRefPts =
    //     new thrust::device_vector<unsigned int>(this->votes->size());
    // this->maxval = new thrust::device_vector<unsigned int>(this->votes->size());
    // thrust::device_vector<unsigned int> *maxModelAngleCode =
    //     new thrust::device_vector<unsigned int>(this->votes->size());

    // thrust::reduce_by_key
    //     (// key input: step function that increments for every row
    //      thrust::make_transform_iterator(votes->begin()+1, high_32_bits()),
    //      thrust::make_transform_iterator(votes->end(), high_32_bits()),
    //      // value input: (value, index) tuple
    //      thrust::make_zip_iterator(thrust::make_tuple(voteCounts->begin()+1,
    //                                                   thrust::make_transform_iterator(votes->begin()+1,
    //                                                                                   low_32_bits()))),
    //      uniqueSceneRefPts->begin(),
    //      thrust::make_zip_iterator(thrust::make_tuple(this->maxval->begin(),
    //                                                   maxModelAngleCode->begin())),
    //      thrust::equal_to<unsigned int>(),
    //      // compare by first element of tuple
    //      thrust::maximum<thrust::tuple<unsigned int, unsigned int> >());



    // Step 8, 9
    // call trans_calc_kernel
    // this->transformations = new thrust::device_vector<float>(this->votes->size()*16);

    // blocks = std::min(((int)(this->votes->size()) + BLOCK_SIZE - 1) / BLOCK_SIZE, MAX_NBLOCKS);
    // trans_calc_kernel<<<blocks,BLOCK_SIZE>>>
    //     (RAW_PTR(uniqueSceneRefPts), RAW_PTR(maxModelAngleCode),
    //      RAW_PTR(this->modelPoints), RAW_PTR(this->modelNormals),
    //      RAW_PTR(scene->getModelPoints()), RAW_PTR(scene->getModelNormals()),
    //      RAW_PTR(this->transformations),
    //      this->votes->size());

    #ifdef DEBUG
        // end cuda timer
        HANDLE_ERROR(cudaEventRecord(stop, 0));
        HANDLE_ERROR(cudaEventSynchronize(stop));
        float elapsedTime;
        HANDLE_ERROR(cudaEventElapsedTime(&elapsedTime, start, stop));
        fprintf(stderr, "Time to lookup model:  %3.1f ms\n", elapsedTime);
    #endif
}

void Model::accumulateVotes(){
    this->voteCodes = new thrust::device_vector<unsigned long>();
    this->voteCounts = new thrust::device_vector<unsigned int>();
    histogram(*(this->votes), *(this->voteCodes), *(this->voteCounts));
}

thrust::device_vector<float>* Model::getTransformations(){
    return this->transformations;
}

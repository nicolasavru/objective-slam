#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>                // Stops underlining of __global__
#include <thrust/sort.h>
#include <thrust/inner_product.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/scan.h>
#include <thrust/sequence.h>
#include <thrust/device_vector.h>
#include <thrust/binary_search.h>

#include "model.h"
#include "impl/ppf_utils.hpp"
#include "impl/util.hpp"
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


Model::Model(thrust::host_vector<float3> *points, thrust::host_vector<float3> *normals, int n){
    this->initPPFs(points, normals, n);

    // SLAM++ Algorithm 1 lines 1-5
    // key2ppfMap: indices into modelPPFs
    this->key2ppfMap = new thrust::device_vector<unsigned int>(this->modelPPFs->size());
    thrust::sequence(key2ppfMap->begin(), key2ppfMap->end());

    // nonunique_hashkeys: array of hashKeys
    thrust::device_vector<unsigned int> *nonunique_hashkeys =
        new thrust::device_vector<unsigned int>(this->modelPPFs->size());

    // for each ppf, compute a 32-bit hash
    int blocks = std::min(((int)(this->modelPPFs->size()) + BLOCK_SIZE - 1) / BLOCK_SIZE, MAX_NBLOCKS);
    ppf_hash_kernel<<<blocks,BLOCK_SIZE>>>(RAW_PTR(this->modelPPFs),
                                           RAW_PTR(nonunique_hashkeys),
                                           this->modelPPFs->size());

    // SLAM++ Algorithm 1 line 6
    // Instead of concatenating the hash and the index and sorting the
    // 64-bit values, we're doing the equivalent operation of sorting
    // the indices using the hashkeys as the sort key.
    thrust::sort_by_key(nonunique_hashkeys->begin(), nonunique_hashkeys->end(), key2ppfMap->begin());

    // key2ppfMap is now sorted such that the indices of identical
    // PPFs are adjacent. We can now count the number of occurances of
    // each unique PPF by taking a histogram of the hash keys. Sorting
    // the hash keys allows the histogram to be computed efficiently
    // using a reduce_by_key.

    this->hashKeys = new thrust::device_vector<unsigned int>();
    this->ppfCount = new thrust::device_vector<unsigned int>();
    histogram_destructive(*nonunique_hashkeys, *(this->hashKeys), *(this->ppfCount));
    delete nonunique_hashkeys;

    // SLAM++ Algorithm 1 line 16
    // Find the indices in key2ppfMap of the beginning of each block of identical PPFs.
    this->firstPPFIndex = new thrust::device_vector<unsigned int>(this->hashKeys->size());
    thrust::exclusive_scan(this->ppfCount->begin(),
                           this->ppfCount->end(),
                           this->firstPPFIndex->begin());
    /* DEBUG */
    fprintf(stderr, "numkeys: %u\n", this->hashKeys->size());
    /* DEBUG */

    this->votes = NULL;
    this->voteCodes = NULL;
    this->voteCounts = NULL;
    this->firstVecIndex = NULL;
}

// TODO: Deallocate memory for things not here yet
Model::~Model(){
    delete this->ppfCount;
    delete this->firstPPFIndex;
    delete this->key2ppfMap;
    if (this->votes != NULL) delete this->votes;
    if (this->voteCodes != NULL) delete this->voteCodes;
    if (this->voteCounts != NULL) delete this->voteCounts;
}

// TODO: finish
void Model::ppf_lookup(Scene *scene){

    #ifdef DEBUG
        cudaEvent_t start, stop;
        HANDLE_ERROR(cudaEventCreate(&start));
        HANDLE_ERROR(cudaEventCreate(&stop));
        HANDLE_ERROR(cudaEventRecord(start, 0));
    #endif

    // find possible starting indices of blocks matching Model hashKeys
    thrust::device_vector<unsigned int> *sceneIndices =
        new thrust::device_vector<unsigned int>(scene->getModelPPFs()->size());
    thrust::lower_bound(this->hashKeys->begin(),
                        this->hashKeys->end(),
                        scene->getHashKeys()->begin(),
                        scene->getHashKeys()->end(),
                        sceneIndices->begin());

    // Steps 1-3
    // launch voting kernel instance for each scene reference point
    unsigned int lastIndex, lastCount;
    thrust::device_vector<unsigned long> *votes_old = new thrust::device_vector<unsigned long>(scene->getModelPPFs()->size()*this->modelPoints->size(),0);

    // populates parallel arrays votes and vecs_old
    int blocks = std::min(((int)(scene->getHashKeys()->size()) + BLOCK_SIZE - 1) / BLOCK_SIZE, MAX_NBLOCKS);
    ppf_vote_kernel<<<blocks,BLOCK_SIZE>>>
        (RAW_PTR(scene->getHashKeys()), RAW_PTR(sceneIndices),
         RAW_PTR(this->hashKeys), RAW_PTR(this->ppfCount),
         RAW_PTR(this->firstPPFIndex), RAW_PTR(this->key2ppfMap),
         RAW_PTR(this->modelPoints), RAW_PTR(this->modelNormals),
         this->n, RAW_PTR(scene->getModelPoints()),
         RAW_PTR(scene->getModelNormals()), scene->numPoints(),
         RAW_PTR(votes_old),
         scene->getHashKeys()->size());

#ifdef DEBUG
    {
        using namespace std;
//        cout << "votes[0] = " << (*this->votes)[0] << endl;
//        thrust::host_vector<float3> *hah = new thrust::host_vector<float3>(*vecs_old);
//        cout << "vecs_old" << endl;
//        for(int i = 0; i < vecs_old->size(); i++){
//            if((*hah)[i].y > 0){
//                cout << i << ", " << (*hah)[i] << endl;
//            }
//        }
    }
#endif
    thrust::sort(votes_old->begin(), votes_old->end());
    this->votes = new thrust::device_vector<unsigned long>();
    this->voteCounts = new thrust::device_vector<unsigned int>();
    histogram_destructive(*votes_old, *(this->votes), *(this->voteCounts));

    thrust::device_vector<unsigned int> *uniqueSceneRefPts =
        new thrust::device_vector<unsigned int>(this->votes->size());
    this->maxval = new thrust::device_vector<unsigned int>(this->votes->size());
    thrust::device_vector<unsigned int> *maxModelAngleCode =
        new thrust::device_vector<unsigned int>(this->votes->size());

    thrust::reduce_by_key
        (// key input: step function that increments for every row
         thrust::make_transform_iterator(votes->begin()+1, high_32_bits()),
         thrust::make_transform_iterator(votes->end(), high_32_bits()),
         // value input: (value, index) tuple
         thrust::make_zip_iterator(thrust::make_tuple(voteCounts->begin()+1,
                                                      thrust::make_transform_iterator(votes->begin()+1,
                                                                                      low_32_bits()))),
         uniqueSceneRefPts->begin(),
         thrust::make_zip_iterator(thrust::make_tuple(this->maxval->begin(),
                                                      maxModelAngleCode->begin())),
         thrust::equal_to<unsigned int>(),
         // compare by first element of tuple
         thrust::maximum<thrust::tuple<unsigned int, unsigned int> >());

    // // populates voteCodes and voteCounts, sorts votes
    // this->accumulateVotes();

    // this->vec2VoteMap = new thrust::device_vector<unsigned int>(vecs_old->size());
    // thrust::sequence(vec2VoteMap->begin(), vec2VoteMap->end());

    // thrust::sort_by_key(vecs_old->begin(), vecs_old->end(), vec2VoteMap->begin());

    // // Step 4

    // // accumulator is an n*n_angle matrix where the ith row
    // // corresponds to the translation vector vecs[i] and the jth
    // // column corresponds to the jth angle bin. accumulator[i*n_angle + j]
    // // is the number of votes that correspond to that translation vector
    // // and angle.
    // //
    // // We need to do linear indexing since using 1d array to model 2d
    // // array. A device_vector is a host-side wrapper for device
    // // memory, so we can't create a device_vector<device_vector>. We
    // // could create a vector of device_vectors on the host, but then
    // // backing memory would be non-contiguous (only vector-wise
    // // continuous).

    // unsigned int num_bins = thrust::inner_product(vecs_old->begin(), vecs_old->end() - 1,
    //                                               vecs_old->begin() + 1,
    //                                               (unsigned int) 1,
    //                                               thrust::plus<unsigned int>(),
    //                                               thrust::not_equal_to<float3>());

    // // allocated by histogram_desctructive?
    // this->vecs = new thrust::device_vector<float3>(num_bins);
    // this->vecCounts = new thrust::device_vector<unsigned int>(num_bins);

    // histogram_destructive(*vecs_old, *(this->vecs), *(this->vecCounts));
    // delete vecs_old;

    // // create list of beginning indices of blocks of ppfs having equal hashes
    // this->firstVecIndex = new thrust::device_vector<unsigned int>(this->vecs->size());

    // thrust::exclusive_scan(this->vecCounts->begin(),
    //                        this->vecCounts->end(),
    //                        this->firstVecIndex->begin());

    // Step 5
    // Can almost represent this (and Step 4) as a reduction or transformation, but not quite.
    // this->accumulator = new thrust::device_vector<unsigned int>(truncVotes->size()*N_ANGLE, 0);
    // std::cout << "votes_size:" << this->votes->size() << std::endl;

    // blocks = std::min(((int)(this->votes->size()) + BLOCK_SIZE - 1) / BLOCK_SIZE, MAX_NBLOCKS);
    // ppf_reduce_rows_kernel<<<blocks,BLOCK_SIZE>>>(RAW_PTR(this->votes),
    //                                               RAW_PTR(this->voteCounts),
    //                                               RAW_PTR(this->firstVoteIndex),
    //                                               N_ANGLE,
    //                                               RAW_PTR(accumulator),
    //                                               this->votes->size());

    // // Steps 6, 7
    // this->maxidx = new thrust::device_vector<unsigned int>(this->votes->size());
    // rowwise_max(*accumulator, this->votes->size(), N_ANGLE, *maxidx);

    // thrust::device_vector<unsigned int> *scores =
    //     new thrust::device_vector<unsigned int>(this->votes->size());

    // blocks = std::min(((int)(this->votes->size()) + BLOCK_SIZE - 1) / BLOCK_SIZE, MAX_NBLOCKS);
    // ppf_score_kernel<<<blocks,BLOCK_SIZE>>>(RAW_PTR(accumulator),
    //                                         RAW_PTR(maxidx),
    //                                         N_ANGLE, SCORE_THRESHOLD,
    //                                         RAW_PTR(scores),
    //                                         this->votes->size());

    // Step 8, 9
    // call trans_calc_kernel
    this->transformations = new thrust::device_vector<float>(this->votes->size()*16);

    blocks = std::min(((int)(this->votes->size()) + BLOCK_SIZE - 1) / BLOCK_SIZE, MAX_NBLOCKS);
    trans_calc_kernel<<<blocks,BLOCK_SIZE>>>
        (RAW_PTR(uniqueSceneRefPts), RAW_PTR(maxModelAngleCode),
         RAW_PTR(this->modelPoints), RAW_PTR(this->modelNormals),
         RAW_PTR(scene->getModelPoints()), RAW_PTR(scene->getModelNormals()),
         RAW_PTR(this->transformations),
         this->votes->size());

    #ifdef DEBUG
        {
            // using namespace std;
            // for (int i=0; i<maxidx->size(); i++){
            //     std::cerr << (*maxidx)[i] << std::endl;
            // }
            // std::cerr << std::endl;
            // for (int i=0; i<accumulator->size(); i++){
            //     std::cerr << (*accumulator)[i] << std::endl;
            // }
            // std::cerr << std::endl;
            // for (int i=0; i<vecs->size(); i++){
            //     std::cerr << (*vecs)[i] << std::endl;
            // }
//            std::cerr << std::endl;
//            for (int i=0; i<votes->size(); i++){
//                std::cerr << (*votes)[i] << std::endl;
//            }
            // std::cerr << std::endl;
            // for (int i=0; i<hashKeys->size(); i++){
            //     std::cerr << (*hashKeys)[i] << std::endl;
            // }
//            std::cerr << std::endl;
//            for (int i=0; i<modelPPFs->size(); i++){
//                std::cerr << (*modelPPFs)[i] << std::endl;
//            }
        }

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

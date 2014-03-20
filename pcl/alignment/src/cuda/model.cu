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
#include "kernel.h"
#include "book.h"

Model::Model(thrust::host_vector<float3> *points, thrust::host_vector<float3> *normals, int n){
    this->initPPFs(points, normals, n);

    // key2ppfMap: associated indices ppf indices
    this->key2ppfMap = new thrust::device_vector<unsigned int>(this->modelPPFs->size());
    thrust::sequence(key2ppfMap->begin(), key2ppfMap->end());

    // hashKeys_old: array of hashKeys
    thrust::device_vector<unsigned int> *hashKeys_old =
        new thrust::device_vector<unsigned int>(this->modelPPFs->size());

    // for each ppf, compute a 32-bit hash
    ppf_hash_kernel<<<32768/BLOCK_SIZE,BLOCK_SIZE>>>(RAW_PTR(this->modelPPFs),
                                                     RAW_PTR(hashKeys_old),
                                                     this->modelPPFs->size());
// #ifdef DEBUG
//     {
//         using namespace std;
//         /* DEBUG */
//         fprintf(stderr, "%d, %d\n", this->modelPPFs->size(), this->modelPPFs->size()/BLOCK_SIZE);
//         /* DEBUG */

//         thrust::host_vector<float4> *ppfs = new thrust::host_vector<float4>(*(this->modelPPFs));
//         for(int i = 0; i < 20; i++){
//             cout << "PPF Number: " << i << endl;
//             cout << (*ppfs)[i].x << endl;
//             cout << (*ppfs)[i].y << endl;
//             cout << (*ppfs)[i].z << endl;
//             cout << (*ppfs)[i].w << endl;
//         }

//         thrust::host_vector<unsigned int> *hah = new thrust::host_vector<unsigned int>(*hashKeys_old);
//         cout << "hashKeys_old" << endl;
//         for(int i = 0; i < 20; i++){
//             cout << (*hah)[i] << endl;
//         }
//     }
// #endif

    thrust::sort_by_key(hashKeys_old->begin(), hashKeys_old->end(), key2ppfMap->begin());

    this->hashKeys = new thrust::device_vector<unsigned int>();
    this->ppfCount = new thrust::device_vector<unsigned int>();
    histogram_destructive(*hashKeys_old, *(this->hashKeys), *(this->ppfCount));
    delete hashKeys_old;

    // create list of beginning indices of blocks of ppfs having equal hashes
    this->firstPPFIndex = new thrust::device_vector<unsigned int>(this->hashKeys->size());

    thrust::exclusive_scan(this->ppfCount->begin(),
                           this->ppfCount->end(),
                           this->firstPPFIndex->begin());

    this->votes = NULL;
    this->voteCodes = NULL;
    this->voteCounts = NULL;

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
    this->votes = new thrust::device_vector<unsigned long>(scene->getModelPPFs()->size());

    // vecs_old is an array of (soon to be) sorted translation vectors
    thrust::device_vector<float3> *vecs_old =
        new thrust::device_vector<float3>(scene->getModelPPFs()->size());

    // populates parallel arrays votes and vecs_old
    ppf_vote_kernel<<<32768,BLOCK_SIZE>>>
        (RAW_PTR(scene->getHashKeys()), RAW_PTR(sceneIndices),
         RAW_PTR(this->hashKeys), RAW_PTR(this->ppfCount),
         RAW_PTR(this->firstPPFIndex), RAW_PTR(this->key2ppfMap),
         RAW_PTR(this->modelPoints), RAW_PTR(this->modelNormals),
         this->n, RAW_PTR(scene->getModelPoints()),
         RAW_PTR(scene->getModelNormals()), scene->numPoints(),
         RAW_PTR(this->votes), RAW_PTR(vecs_old),
         scene->getHashKeys()->size());

#ifdef DEBUG
    {
        using namespace std;

        thrust::host_vector<float3> *hah = new thrust::host_vector<float3>(*vecs_old);
        cout << "vecs_old" << endl;
        for(int i = 0; i < 20; i++){
            if((*hah)[i].y > 0){
                cout << i << ", " << (*hah)[i] << endl;
            }
        }
    }
#endif

    thrust::sort_by_key(vecs_old->begin(), vecs_old->end(), votes->begin());
    this->vecs = new thrust::device_vector<float3>();
    this->vecCounts = new thrust::device_vector<unsigned int>();
    histogram_destructive(*vecs_old, *(this->vecs), *(this->vecCounts));
    this->firstVecIndex = new thrust::device_vector<unsigned int>(this->vecs->size());
    thrust::exclusive_scan(this->vecCounts->begin(),
                           this->vecCounts->end(),
                           this->firstVecIndex->begin());
    // votes is sorted by vecs!!


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
    thrust::device_vector<unsigned int> *accumulator =
        new thrust::device_vector<unsigned int>(this->vecs->size()*N_ANGLE);

     ppf_reduce_rows_kernel<<<this->vecs->size()/BLOCK_SIZE,BLOCK_SIZE>>>(RAW_PTR(this->vecs),
                                                                          RAW_PTR(this->vecCounts),
                                                                          RAW_PTR(this->firstVecIndex),
                                                                          RAW_PTR(this->votes),
                                                                          N_ANGLE,
                                                                          RAW_PTR(accumulator),
                                                                          this->vecs->size());

    // Steps 6, 7
    thrust::device_vector<unsigned int> *maxidx =
        new thrust::device_vector<unsigned int>(this->vecs->size());
    rowwise_max(*accumulator, this->vecs->size(), N_ANGLE, *maxidx);

    thrust::device_vector<unsigned int> *scores =
        new thrust::device_vector<unsigned int>(this->vecs->size());
    ppf_score_kernel<<<this->vecs->size()/BLOCK_SIZE,BLOCK_SIZE>>>(RAW_PTR(accumulator),
                                                                   RAW_PTR(maxidx),
                                                                   N_ANGLE, SCORE_THRESHOLD,
                                                                   RAW_PTR(scores),
                                                                   this->vecs->size());

    // Step 8, 9
    // call trans_calc_kernel
    this->transformations = new thrust::device_vector<float>(this->vecs->size()*16);

     trans_calc_kernel<<<this->vecs->size()/BLOCK_SIZE,BLOCK_SIZE>>>
             (RAW_PTR(this->vecs), RAW_PTR(this->vecCounts),
              RAW_PTR(this->firstVecIndex), RAW_PTR(this->votes),
              RAW_PTR(maxidx), RAW_PTR(scores),
              N_ANGLE,
              RAW_PTR(this->modelPoints), RAW_PTR(this->modelNormals),
              this->modelPoints->size(),
              RAW_PTR(scene->getModelPoints()), RAW_PTR(scene->getModelNormals()),
              scene->getModelPoints()->size(),
              RAW_PTR(this->transformations),
              this->vecs->size());

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

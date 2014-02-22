#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>                // Stops underlining of __global__
#include <thrust/sort.h>
#include <thrust/device_ptr.h>
#include <thrust/inner_product.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/scan.h>
#include <thrust/device_vector.h>
#include <thrust/binary_search.h>

#include "SearchStructure.h"
#include "kernel.h"
#include "book.h"

SearchStructure::SearchStructure(thrust::host_vector<float3> *points, thrust::host_vector<float3> *normals, int n){
    this->n = n;
    this->modelPoints = new thrust::device_vector<float3>(*points);
    this->modelNormals = new thrust::device_vector<float3>(*normals);
    this->modelPPFs = new thrust::device_vector<float4>(n*n);
    /* DEBUG */
    fprintf(stderr, "n: %d\n", n);
    /* DEBUG */

    // start cuda timer
    cudaEvent_t start, stop;
    HANDLE_ERROR(cudaEventCreate(&start));
    HANDLE_ERROR(cudaEventCreate(&stop));
    HANDLE_ERROR(cudaEventRecord(start, 0));

    ppf_kernel<<<n/BLOCK_SIZE,BLOCK_SIZE>>>(RAW_PTR(this->modelPoints),
                                            RAW_PTR(this->modelNormals),
                                            RAW_PTR(this->modelPPFs),
                                            n);

    // end cuda timer
    HANDLE_ERROR(cudaEventRecord(stop, 0));
    HANDLE_ERROR(cudaEventSynchronize(stop));
    float elapsedTime;
    HANDLE_ERROR(cudaEventElapsedTime(&elapsedTime, start, stop));
    printf("Time to generate PPFs:  %3.1f ms\n", elapsedTime);


    // for each ppf, compute a 32-bit hash and concatenate it with
    // a 32-bit int representing the index of the ppf in d_ppfs
    thrust::device_vector<unsigned long> *d_codes =
        new thrust::device_vector<unsigned long>(this->modelPPFs->size());

    ppf_encode_kernel<<<this->modelPPFs->size()/BLOCK_SIZE,BLOCK_SIZE>>>(RAW_PTR(this->modelPPFs),
                                                                         RAW_PTR(d_codes),
                                                                         this->modelPPFs->size());
    thrust::sort(d_codes->begin(), d_codes->end());

    // split codes into array of hashKeys (high 32 bits) and
    // key2ppfMap, the associated indices (low 32 bits)
    this->key2ppfMap = new thrust::device_vector<unsigned int>(this->modelPPFs->size());
    thrust::device_vector<unsigned int> *hashKeys_old =
        new thrust::device_vector<unsigned int>(this->modelPPFs->size());

    ppf_decode_kernel<<<this->modelPPFs->size()/BLOCK_SIZE,BLOCK_SIZE>>>(RAW_PTR(d_codes),
                                                                          RAW_PTR(this->key2ppfMap),
                                                                          RAW_PTR(hashKeys_old),
                                                                          this->modelPPFs->size());
    delete d_codes;

    // create histogram of hash keys
    // https://code.google.com/p/thrust/source/browse/examples/histogram.cu
    unsigned int num_bins = thrust::inner_product(hashKeys_old->begin(), hashKeys_old->end() - 1,
                                                  hashKeys_old->begin() + 1,
                                                  (unsigned int) 1,
                                                  thrust::plus<unsigned int>(),
                                                  thrust::not_equal_to<unsigned int>());

    /* DEBUG */
    fprintf(stderr, "num_bins: %d\n", num_bins);
    /* DEBUG */

    this->hashKeys = new thrust::device_vector<unsigned int>(num_bins);
    this->ppfCount = new thrust::device_vector<unsigned int>(num_bins);

    thrust::reduce_by_key(hashKeys_old->begin(), hashKeys_old->end(),
                          thrust::constant_iterator<unsigned int>(1),
                          this->hashKeys->begin(),
                          this->ppfCount->begin());
    delete hashKeys_old;

    // create list of beginning indices of blocks of ppfs having equal hashes
    this->firstPPFIndex = new thrust::device_vector<unsigned int>(num_bins);

    thrust::exclusive_scan(this->ppfCount->begin(),
                           this->ppfCount->end(),
                           this->firstPPFIndex->begin());
}

SearchStructure::~SearchStructure(){
    delete this->modelPoints;
    delete this->modelNormals;
    delete this->modelPPFs;
    delete this->hashKeys;
    delete this->ppfCount;
    delete this->firstPPFIndex;
    delete this->key2ppfMap;
}

// TODO: finish
thrust::device_vector<unsigned int> *SearchStructure::ppf_lookup(thrust::device_vector<float4> *scene_ppfs){

    thrust::device_vector<unsigned int> *sceneKeys =
        new thrust::device_vector<unsigned int>(scene_ppfs->size());

    ppf_hash_kernel<<<n/BLOCK_SIZE,BLOCK_SIZE>>>(RAW_PTR(scene_ppfs),
                                                 RAW_PTR(sceneKeys),
                                                 scene_ppfs->size());

    thrust::device_vector<unsigned int> *sceneIndeces =
        new thrust::device_vector<unsigned int>(scene_ppfs->size());
    thrust::lower_bound(this->hashKeys->begin(),
                        this->hashKeys->end(),
                        sceneKeys->begin(),
                        sceneKeys->end(),
                        sceneIndeces->begin());

    thrust::device_vector<unsigned int> *found_ppf_starts =
        new thrust::device_vector<unsigned int>(scene_ppfs->size());
    thrust::device_vector<unsigned int> *found_ppf_count =
        new thrust::device_vector<unsigned int>(scene_ppfs->size());
    // WHAT THIS SHOULD DO:
    // FOR EACH INDEX RETURNED BY LOWER BOUND
    //      CHECK IF THIS->HASH_KEYS[i] == SCENE_KEYS[i]
    //      IF IS EQUAL LOOK UP PPFCOUNTS[i] AND FIRSTPPFINDEX[i}
    //      
    ppf_lookup_kernel<<<n/BLOCK_SIZE,BLOCK_SIZE>>>(RAW_PTR(sceneKeys), RAW_PTR(sceneIndeces),
                                                   RAW_PTR(this->hashKeys), RAW_PTR(this->ppfCount),
                                                   RAW_PTR(this->firstPPFIndex), RAW_PTR(this->key2ppfMap),
                                                   RAW_PTR(found_ppf_starts), RAW_PTR(found_ppf_count),
                                                   scene_ppfs->size());

    return sceneIndeces;
}


thrust::device_vector<float3> *SearchStructure::getModelPoints(){
    return this->modelPoints;
}

thrust::device_vector<float3> *SearchStructure::getModelNormals(){
    return this->modelNormals;
}
thrust::device_vector<float4> *SearchStructure::getModelPPFs(){
    return this->modelPPFs;
}

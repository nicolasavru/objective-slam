#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>                // Stops underlining of __global__
#include <thrust/sort.h>
#include <thrust/inner_product.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/scan.h>
#include <thrust/device_vector.h>
#include <thrust/binary_search.h>

#include "model.h"
#include "imp/ppf_utils.hpp"
#include "kernel.h"
#include "book.h"

Model::Model(thrust::host_vector<float3> *points, thrust::host_vector<float3> *normals, int n):
    Scene(points, normals, n){

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

    this->hashKeys = new thrust::device_vector<unsigned int>();
    this->ppfCount = new thrust::device_vector<unsigned int>();
    histogram_destructive(*hashKeys_old, *(this->hashKeys), *(this->ppfCount));
    delete hashKeys_old;

    // create list of beginning indices of blocks of ppfs having equal hashes
    this->firstPPFIndex = new thrust::device_vector<unsigned int>(this->hashKeys->size());

    thrust::exclusive_scan(this->ppfCount->begin(),
                           this->ppfCount->end(),
                           this->firstPPFIndex->begin());
}

Model::~Model(){
    delete this->hashKeys;
    delete this->ppfCount;
    delete this->firstPPFIndex;
    delete this->key2ppfMap;
    // TODO: Check if votes exist and delete them
}

// TODO: finish
thrust::device_vector<unsigned int> *Model::ppf_lookup(Model *Model){

    // compute hashKeys for Model PPFs
    thrust::device_vector<unsigned int> *ModelKeys =
        new thrust::device_vector<unsigned int>(Model->getModelPPFs()->size());

    ppf_hash_kernel<<<n/BLOCK_SIZE,BLOCK_SIZE>>>(RAW_PTR(Model->getModelPPFs()),
                                                 RAW_PTR(ModelKeys),
                                                 Model->getModelPPFs()->size());

    // find possible starting indices of blocks matching Model hashKeys
    thrust::device_vector<unsigned int> *ModelIndices =
        new thrust::device_vector<unsigned int>(Model->getModelPPFs()->size());
    thrust::lower_bound(this->hashKeys->begin(),
                        this->hashKeys->end(),
                        ModelKeys->begin(),
                        ModelKeys->end(),
                        ModelIndices->begin());

    thrust::device_vector<unsigned int> *found_ppf_starts =
        new thrust::device_vector<unsigned int>(Model->getModelPPFs()->size());
    thrust::device_vector<unsigned int> *found_ppf_count =
        new thrust::device_vector<unsigned int>(Model->getModelPPFs()->size());
    // WHAT THIS SHOULD DO:
    // FOR EACH INDEX RETURNED BY LOWER BOUND
    //      CHECK IF THIS->HASH_KEYS[i] == Model_KEYS[i]
    //      IF IS EQUAL LOOK UP PPFCOUNTS[i] AND FIRSTPPFINDEX[i}
    //
    // launch kernel for Model point
    thrust::device_vector<unsigned int> *votes =
        new thrust::device_vector<unsigned int>(Model->getModelPPFs()->size());
    // ppf_lookup_kernel<<<n/BLOCK_SIZE,BLOCK_SIZE>>>(RAW_PTR(ModelKeys), RAW_PTR(ModelIndices),
    //                                                RAW_PTR(this->hashKeys), RAW_PTR(this->ppfCount),
    //                                                RAW_PTR(this->firstPPFIndex), RAW_PTR(this->key2ppfMap),
    //                                                RAW_PTR(this->modelPoints),
    //                                                RAW_PTR(this->modelNormals),
    //                                                this->n,
    //                                                RAW_PTR(Model->getModelPoints()),
    //                                                RAW_PTR(Model->getModelNormals()),
    //                                                Model->numPoints(),
    //                                                RAW_PTR(votes),
    //                                                Model->numPoints());

    return ModelIndices;
}

void Model::accumulateVotes(){
    thrust::sort(this->votes->begin(), this->votes->end());

//    // create histogram of hash keys
//    // https://code.google.com/p/thrust/source/browse/examples/histogram.cu
//    unsigned int num_bins = thrust::inner_product(hashKeys_old->begin(), hashKeys_old->end() - 1,
//                                                  hashKeys_old->begin() + 1,
//                                                  (unsigned int) 1,
//                                                  thrust::plus<unsigned int>(),
//                                                  thrust::not_equal_to<unsigned int>());
//
//    /* DEBUG */
//    fprintf(stderr, "num_bins: %d\n", num_bins);
//    /* DEBUG */
//
//    this->hashKeys = new thrust::device_vector<unsigned int>(num_bins);
//    this->ppfCount = new thrust::device_vector<unsigned int>(num_bins);
//
//    thrust::reduce_by_key(hashKeys_old->begin(), hashKeys_old->end(),
//                          thrust::constant_iterator<unsigned int>(1),
//                          this->hashKeys->begin(),
//                          this->ppfCount->begin());
}

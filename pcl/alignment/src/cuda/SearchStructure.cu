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

#define RAW_PTR(V) thrust::raw_pointer_cast(V->data())

SearchStructure::SearchStructure(float4 *d_ppfs, int n){
    this->n = n;

    // for each ppf, compute a 32-bit hash and concatenate it with
    // a 32-bit int representing the index of the ppf in d_ppfs
    thrust::device_vector<unsigned long> *d_codes = new thrust::device_vector<unsigned long>(n);

    ppf_encode_kernel<<<n/BLOCK_SIZE,BLOCK_SIZE>>>(d_ppfs, RAW_PTR(d_codes), n);
    thrust::sort(d_codes->begin(), d_codes->end());

    // split codes into array of hashKeys (high 32 bits) and
    // key2ppfMap, the associated indices (low 32 bits)
    this->key2ppfMap = new thrust::device_vector<unsigned int>(n);
    thrust::device_vector<unsigned int> *hashKeys_old = new thrust::device_vector<unsigned int>(n);

    ppf_decode_kernel<<<n/BLOCK_SIZE,BLOCK_SIZE>>>(RAW_PTR(d_codes),
                                                   RAW_PTR(this->key2ppfMap),
                                                   RAW_PTR(hashKeys_old),
                                                   n);
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

    //HANDLE_ERROR(cudaMalloc(&(this->hashKeys), num_bins*sizeof(unsigned int)));
    //HANDLE_ERROR(cudaMalloc(&(this->ppfCount), num_bins*sizeof(unsigned int)));
    this->ppfCount = new thrust::device_vector<unsigned int>(num_bins);
    this->hashKeys = new thrust::device_vector<unsigned int>(num_bins);

    thrust::reduce_by_key(hashKeys_old->begin(), hashKeys_old->end(),
                          thrust::constant_iterator<unsigned int>(1),
                          this->hashKeys->begin(),
                          this->ppfCount->begin());
    delete hashKeys_old;

    // create list of beginning indices of blocks of ppfs having equal hashes
    this->firstPPFIndex = new thrust::device_vector<unsigned int>(num_bins);

    thrust::exclusive_scan(this->ppfCount->begin(), this->ppfCount->end(), this->firstPPFIndex->begin());
}

SearchStructure::~SearchStructure(){
    delete this->hashKeys;
    delete this->ppfCount;
    delete this->firstPPFIndex;
    delete this->key2ppfMap;
}

// TODO: finish
thrust::device_vector<unsigned int> *SearchStructure::ppf_lookup(thrust::device_vector<float4> *d_ppfs){

    thrust::device_vector<unsigned int> *sceneKeys = new thrust::device_vector<unsigned int>(d_ppfs->size());
    thrust::device_vector<unsigned int> *sceneIndeces = new thrust::device_vector<unsigned int>(d_ppfs->size());

    ppf_hash_kernel<<<n/BLOCK_SIZE,BLOCK_SIZE>>>(RAW_PTR(d_ppfs), RAW_PTR(sceneKeys), d_ppfs->size());

    thrust::lower_bound(this->hashKeys->begin(),
                        this->hashKeys->end(),
                        sceneKeys->begin(),
                        sceneKeys->end(),
                        sceneIndeces->begin());

    return sceneIndeces;
}

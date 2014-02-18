#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>                // Stops underlining of __global__
#include <thrust/sort.h>
#include <thrust/device_ptr.h>
#include <thrust/inner_product.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/scan.h>

#include "book.h"
#include "SearchStructure.h"


// FNV-1a hash function
// http://programmers.stackexchange.com/questions/49550/which-hashing-algorithm-is-best-for-uniqueness-and-speed
__device__ unsigned int hash(void *f, int n){
    char *s = (char *) f;
    unsigned int hash = 2166136261;
    while(n--){
        hash ^= *s++;
        hash *= 16777619;
    }
    return hash;
}

// TODO: increase thread work
__global__ void ppf_encode_kernel(float4 *ppfs, unsigned long *codes, int count){
    if(count <= 1) return;

    int ind = threadIdx.x;
    int idx = ind + blockIdx.x * blockDim.x;

    if(idx < count){
        unsigned int hk = hash(ppfs+idx, sizeof(float4));
        codes[idx] = (((unsigned long) hk) << 32) + idx;
    }
}

// TODO: increase thread work
__global__ void ppf_decode_kernel(unsigned long *codes, unsigned int *key2ppfMap,
                                  unsigned int *hashKeys, int count){
    if(count <= 1) return;

    int ind = threadIdx.x;
    int idx = ind + blockIdx.x * blockDim.x;

    unsigned long low32 = ((unsigned long) -1) >> 32;

    if(idx < count){
        // line 11 in algorithm 1, typo on their part?!?
        key2ppfMap[idx] = (unsigned int) (codes[idx] & low32);
        hashKeys[idx] = (unsigned int) (codes[idx] >> 32);
    }
}

SearchStructure::SearchStructure(float4 *d_ppfs, int n, int block_size){
    this->n = n;

    // for each ppf, compute a 32-bit hash and concatenate it with
    // a 32-bit int representing the index of the ppf in d_ppfs
    unsigned long *d_codes;
    HANDLE_ERROR(cudaMalloc(&d_codes, n*sizeof(unsigned long)));
    thrust::device_ptr<unsigned long> dev_ptr(d_codes);

    ppf_encode_kernel<<<n/block_size,block_size>>>(d_ppfs, d_codes, n);
    thrust::sort(dev_ptr, dev_ptr+n);

    // split codes into array of hashKeys (high 32 bits) and
    // key2ppfMap, the associated indices (low 32 bits)
    unsigned int *hashKeys_old;
    HANDLE_ERROR(cudaMalloc(&(this->key2ppfMap), n*sizeof(unsigned int)));
    HANDLE_ERROR(cudaMalloc(&hashKeys_old, n*sizeof(unsigned int)));
    thrust::device_ptr<unsigned int> hashKeys_old_ptr(hashKeys_old);
    thrust::device_ptr<unsigned int> key2ppfMap_ptr(this->key2ppfMap);

    ppf_decode_kernel<<<n/block_size,block_size>>>(d_codes, this->key2ppfMap, hashKeys_old, n);
    cudaFree(d_codes);


    // create histogram of hash keys
    // https://code.google.com/p/thrust/source/browse/examples/histogram.cu
    unsigned int num_bins = thrust::inner_product(hashKeys_old_ptr, hashKeys_old_ptr + n - 1,
                                                  hashKeys_old_ptr + 1,
                                                  (unsigned int) 1,
                                                  thrust::plus<unsigned int>(),
                                                  thrust::not_equal_to<unsigned int>());

    /* DEBUG */
    fprintf(stderr, "num_bins: %d\n", num_bins);
    /* DEBUG */

    HANDLE_ERROR(cudaMalloc(&(this->hashKeys), num_bins*sizeof(unsigned int)));
    HANDLE_ERROR(cudaMalloc(&(this->ppfCount), num_bins*sizeof(unsigned int)));
    thrust::device_ptr<unsigned int> hashKeys_ptr(this->hashKeys);
    thrust::device_ptr<unsigned int> ppfCount_ptr(this->ppfCount);

    thrust::reduce_by_key(hashKeys_old_ptr, hashKeys_old_ptr + n,
                          thrust::constant_iterator<unsigned int>(1),
                          hashKeys_ptr,
                          ppfCount_ptr);
    cudaFree(hashKeys_old);


    // create list of beginning indices of blocks of ppfs having equal hashes
    HANDLE_ERROR(cudaMalloc(&(this->firstPPFIndex), num_bins*sizeof(unsigned int)));
    thrust::device_ptr<unsigned int> firstPPFIndex_ptr(this->firstPPFIndex);

    thrust::exclusive_scan(ppfCount_ptr, ppfCount_ptr+num_bins, firstPPFIndex_ptr);
}

SearchStructure::~SearchStructure(){
    cudaFree(this->hashKeys);
    cudaFree(this->ppfCount);
    cudaFree(this->firstPPFIndex);
    cudaFree(this->key2ppfMap);
}

// TODO: finish
//
// TODO: redo using thrust vectorized search
// __device__ int SearchStructure::ppf_lookup(float4 *ppf, float4 *results){
//     int hashKey = hash(ppf, sizeof(float4));
//     int hash_idx;
//     // Iterator iter = thrust::lower_bound(thrust::device, s.hashKeys, s.hashKeys + n, hashKey);

//     return 0;
// }

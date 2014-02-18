#ifndef SEARCH_STRUCTURE_H
#define SEARCH_STRUCTURE_H

#include <cuda.h>
#include <cuda_runtime.h>                // Stops underlining of __global__
#include <device_launch_parameters.h>    // Stops underlining of threadIdx etc.

// TODO: Have ppf_lookup be a method. Replace
// (unsigned int *)s with thrust::device_vectors and modify everything
// else accordingly.
class SearchStructure {

    public:

        SearchStructure(float4 *d_ppfs, int n, int block_size);

        ~SearchStructure();

        __device__ int ppf_lookup(float4 *ppf, float4 *results);

    private:

        // Number of PPF in the mode. I.e., number of elements in each of
        // the following arrays;
        int n;

        // List of all hash keys. Use a parallel binary search to find
        // index of desired hash key.
        unsigned int *hashKeys;

        // ppfCount[i] is the number of PPFs whose hash is hashKeys[i];
        unsigned int *ppfCount;

        // firstPPFIndex[i] is the index of the first entry in key2ppfMap
        // corresponding to hashKey[i]. The following ppfCount[i]-1
        // entries also correspond to hashKey[i].
        unsigned int *firstPPFIndex;

        // key2ppfMap[i] is the index in d_ppfs that contains (one of) the
        // PPF(s) whose hash is hashKeys[i]. From there, the indices of
        // the points that were used to generate the PPF can be
        // calculated.
        unsigned int *key2ppfMap;
};


#endif /* SEARCH_STRUCTURE_H */

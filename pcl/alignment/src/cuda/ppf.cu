#include <iostream>
#include <ctime>
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>                // Stops underlining of __global__
#include <device_launch_parameters.h>    // Stops underlining of threadIdx etc.
#include <sys/types.h>
#include <sys/stat.h>
#include <thrust/sort.h>
#include <thrust/device_ptr.h>
#include <thrust/inner_product.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/scan.h>
#include <math_constants.h>
#include <functional>

#include "book.h"

#define BLOCK_SIZE 256

__const__ int n_angle = 32;
__const__ float d_angle = 2*CUDART_PI_F/n_angle;
__const__ float d_dist = 0.05;

using namespace std;

__device__ __forceinline__ float dot(float3 v1, float3 v2){
    return (v1.x*v2.x + v1.y*v2.y + v1.z*v2.z);
}

__device__ __forceinline__ float norm(float3 v){
    return sqrtf(dot(v, v));
}

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

__device__ float4 disc_feature(float4 f, float d_dist, float d_angle){
    f.x = f.x - fmodf(f.x, d_dist);
    f.y = f.y - fmodf(f.y, d_angle);
    f.z = f.z - fmodf(f.z, d_angle);
    f.w = f.w - fmodf(f.w, d_angle);
    return f;
}

__device__ float4 compute_ppf(float3 p1, float3 n1, float3 p2, float3 n2){
    float3 d;
    d.x = p2.x - p1.x;
    d.y = p2.y - p1.y;
    d.z = p2.z - p1.z;

    float4 f;
    f.x = norm(d);
    f.y = acosf(dot(n1,d) / (norm(n1)*norm(d)));
    f.z = acosf(dot(n2,d) / (norm(n2)*norm(d)));
    f.w = acosf(dot(n1,n2) / (norm(n1)*norm(n2)));

    return f;
}

__global__ void ppf_kernel(float3 *points, float3 *norms, float4 *out, int count){
    if(count <= 1) return;

    int ind = threadIdx.x;
    int idx = ind + blockIdx.x * blockDim.x;

    if(idx < count) {

        __shared__ float3 Spoints[BLOCK_SIZE];
        __shared__ float3 Snorms[BLOCK_SIZE];

        float3 thisPoint = points[idx];
        float3 thisNorm  = norms[idx];

        for(int i = 0; i < count/BLOCK_SIZE; i++){

            Spoints[ind] = points[i*BLOCK_SIZE+ind];
            Snorms[ind]  = norms[i*BLOCK_SIZE+ind];
            __syncthreads();

            for(int j = 0; j < BLOCK_SIZE; j++) {
                if((idx*count + j + i*BLOCK_SIZE) % (count+1) == 0) continue;
                out[idx*count + j + i*BLOCK_SIZE] = compute_ppf(thisPoint, thisNorm, Spoints[j], Snorms[j]);
                out[idx*count + j + i*BLOCK_SIZE] = disc_feature(out[idx*count + j + i*BLOCK_SIZE],
                                                                 d_dist, d_angle);
            }
        }
    }
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

// TODO: Re-write as a class. Have the constructor take n as an
// argument and cudaMalloc() everything. Have the destructor
// cudaFree() everything. Have ppf_lookup be a method. Replace
// (unsigned int *)s with thrust::device_vectors and modify everything
// else accordingly.
//
// structure for searching a model description
struct search_structure{
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

// TODO: finish
//
// TODO: redo using thrust vectorized search
// __device__ int ppf_lookup(struct search_structure s, float4 *ppf, float4 *results){
//     int hashKey = hash(ppf, sizeof(float4));
//     int hash_idx;
//     // Iterator iter = thrust::lower_bound(thrust::device, s.hashKeys, s.hashKeys + n, hashKey);

//     return 0;
// }

struct search_structure build_model_description(float4 *d_ppfs, int n){
    // for each ppf, compute a 32-bit hash and concatenate it with
    // a 32-bit int representing the index of the ppf in d_ppfs
    unsigned long *d_codes;
    HANDLE_ERROR(cudaMalloc(&d_codes, n*sizeof(unsigned long)));
    thrust::device_ptr<unsigned long> dev_ptr(d_codes); 

    ppf_encode_kernel<<<n/BLOCK_SIZE,BLOCK_SIZE>>>(d_ppfs, d_codes, n);
    thrust::sort(dev_ptr, dev_ptr+n);


    // split codes into array of hashKeys (high 32 bits) and
    // key2ppfMap, the associated indices (low 32 bits)
    unsigned int *key2ppfMap, *hashKeys;
    HANDLE_ERROR(cudaMalloc(&key2ppfMap, n*sizeof(unsigned int)));
    HANDLE_ERROR(cudaMalloc(&hashKeys, n*sizeof(unsigned int)));
    thrust::device_ptr<unsigned int> hashKeys_ptr(hashKeys);
    thrust::device_ptr<unsigned int> key2ppfMap_ptr(key2ppfMap);

    ppf_decode_kernel<<<n/BLOCK_SIZE,BLOCK_SIZE>>>(d_codes, key2ppfMap, hashKeys, n);
    cudaFree(d_codes);


    // create histogram of hash keys
    // https://code.google.com/p/thrust/source/browse/examples/histogram.cu
    unsigned int num_bins = thrust::inner_product(hashKeys_ptr, hashKeys_ptr + n - 1,
                                                  hashKeys_ptr + 1,
                                                  (unsigned int) 1,
                                                  thrust::plus<unsigned int>(),
                                                  thrust::not_equal_to<unsigned int>());

    /* DEBUG */
    fprintf(stderr, "num_bins: %d\n", num_bins);
    /* DEBUG */

    unsigned int *hashKeys_new, *ppfCount;
    HANDLE_ERROR(cudaMalloc(&hashKeys_new, num_bins*sizeof(unsigned int)));
    HANDLE_ERROR(cudaMalloc(&ppfCount, num_bins*sizeof(unsigned int)));
    thrust::device_ptr<unsigned int> hashKeys_new_ptr(hashKeys_new);
    thrust::device_ptr<unsigned int> ppfCount_ptr(ppfCount);

    thrust::reduce_by_key(hashKeys_ptr, hashKeys_ptr + n,
                          thrust::constant_iterator<unsigned int>(1),
                          hashKeys_new_ptr,
                          ppfCount_ptr);
    cudaFree(hashKeys);


    // create list of beginning indices of blocks of ppfs having equal hashes
    unsigned int *firstPPFIndex;
    HANDLE_ERROR(cudaMalloc(&firstPPFIndex, num_bins*sizeof(unsigned int)));
    thrust::device_ptr<unsigned int> firstPPFIndex_ptr(firstPPFIndex);

    thrust::exclusive_scan(ppfCount_ptr, ppfCount_ptr+num_bins, firstPPFIndex_ptr);

    struct search_structure result;
    result.n = n;
    result.hashKeys = hashKeys_new;
    result.ppfCount = ppfCount;
    result.firstPPFIndex = firstPPFIndex;
    result.key2ppfMap = key2ppfMap;
    return result;
}

int ply_load_main(char *point_path, char *norm_path, int N){
    // file input
    FILE *points_fin, *norms_fin;
    size_t result1, result2;

    points_fin = fopen(point_path, "rb");
    norms_fin  = fopen(norm_path, "rb");
    if(points_fin==NULL){fputs ("File error: point_fin",stderr); exit (1);}
    if(norms_fin==NULL){fputs ("File error: norms_fin",stderr); exit (1);}

    float3 *points = new float3[N];
    float3 *norms = new float3[N];
    if (points == NULL) {fputs ("Memory error: points",stderr); exit (2);}
    if (norms  == NULL) {fputs ("Memory error: norms",stderr); exit (2);}

    long startTime0 = clock();
    result1 = fread(points,sizeof(float3),N,points_fin);
    result2 = fread(norms,sizeof(float3),N,norms_fin);
    long finishTime0 = clock();

    if(result1 != N){fputs ("Reading error: points",stderr); exit (3);}
    if(result2 != N){fputs ("Reading error: norms",stderr); exit (3);}


    // cuda setup
    cudaDeviceProp  prop;
    HANDLE_ERROR(cudaGetDeviceProperties(&prop, 0));
    int blocks = prop.multiProcessorCount;
    /* DEBUG */
    fprintf(stderr, "blocks: %d\n", blocks);
    /* DEBUG */


    // start cuda timer
    cudaEvent_t start, stop;
    HANDLE_ERROR(cudaEventCreate(&start));
    HANDLE_ERROR(cudaEventCreate(&stop));
    HANDLE_ERROR(cudaEventRecord(start, 0));


    // compute ppfs
    float3 *d_points, *d_norms;
    float4 *d_ppfs;
    HANDLE_ERROR(cudaMalloc(&d_points, N*sizeof(float3)));
    HANDLE_ERROR(cudaMalloc(&d_norms, N*sizeof(float3)));
    HANDLE_ERROR(cudaMalloc(&d_ppfs, N*N*sizeof(float4)));

    HANDLE_ERROR(cudaMemcpy(d_points, points, N*sizeof(float3), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(d_norms, norms, N*sizeof(float3), cudaMemcpyHostToDevice));

    ppf_kernel<<<N/BLOCK_SIZE,BLOCK_SIZE>>>(d_points, d_norms, d_ppfs, N);


    // build model description
    struct search_structure model;
    model = build_model_description(d_ppfs, N*N);


    // end cuda timer
    HANDLE_ERROR(cudaEventRecord(stop, 0));
    HANDLE_ERROR(cudaEventSynchronize(stop));
    float elapsedTime;
    HANDLE_ERROR(cudaEventElapsedTime(&elapsedTime, start, stop));
    printf("Time to generate:  %3.1f ms\n", elapsedTime);


    // copy ppfs back to host
    float4 *ppfs = new float4[N*N];
    HANDLE_ERROR(cudaMemcpy(ppfs, d_ppfs, N*N*sizeof(float4), cudaMemcpyDeviceToHost));

    // write out ppfs
    for(int i = 0; i < 100; i++){
        cout << "PPF Number: " << i << endl;
        cout << ppfs[i].x << endl;
        cout << ppfs[i].y << endl;
        cout << ppfs[i].z << endl;
        cout << ppfs[i].w << endl;
    }

    cout<<"Data Load Time"<<" "<<(finishTime0 - startTime0)<<" ms"<<endl;

    // Deallocate ram
    delete[] points;
    delete[] norms;
    delete[] ppfs;

    cudaFree(d_points);
    cudaFree(d_norms);
    cudaFree(d_ppfs);

    cudaFree(model.hashKeys);
    cudaFree(model.key2ppfMap);
    cudaFree(model.ppfCount);
    cudaFree(model.firstPPFIndex);

    cudaDeviceReset();

    // close input file
    fclose(points_fin);
    fclose(norms_fin);
    return 0;
}

// int ppf_run(Eigen::MatrixXf &points, Eigen::MatrixXf &normals){
//     float *point_data = points.data();
//     float *normal_data = normals.data();
//     int size = points.rows()*points.cols()*sizeof(float);

//     float *dev_point_data, *dev_normal_data;
//     HANDLE_ERROR(cudaMalloc((void **) &dev_point_data, size));
//     HANDLE_ERROR(cudaMemcpy(dev_point_data, point_data. size));

//     HANDLE_ERROR(cudaMalloc((void **) &dev_normal_data, size));
//     HANDLE_ERROR(cudaMemcpy(dev_normal_data, point_data. size));

//     cudaDeviceProp prop;
//     HANDLE_ERROR(cudaGetDeviceProperties(&prop, 0));
//     int blocks = prop.multiProcessorCount;

//     ppf_kernel<<<blocks*2,256>>>(dev_buffer, SIZE, dev_histo);

// }

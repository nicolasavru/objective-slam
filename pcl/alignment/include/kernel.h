#ifndef __KERNEL_H
#define __KERNEL_H

#include <cuda.h>
#include <cuda_runtime.h>                // Stops underlining of __global__
#include <device_launch_parameters.h>    // Stops underlining of threadIdx etc.
#include <math_constants.h>

#include "debug.h"

#define BLOCK_SIZE 256

__const__ int n_angle = 32;
__const__ float d_angle = 2*CUDART_PI_F/n_angle;
__const__ float d_dist = 0.05;
__const__ float score_threshold = 0.8;

__device__ unsigned int hash(void *f, int n);
__device__ __forceinline__ float dot(float3 v1, float3 v2);
__device__ __forceinline__ float dot(float4 v1, float4 v2);
__device__ __forceinline__ float norm(float3 v);
__device__ float3 cross(float3 u, float3 v);
__device__ __forceinline__ float quant_downf(float x, float y);
__device__ float3 discretize(float3 f, float d_dist);
__device__ float4 disc_feature(float4 f, float d_dist, float d_angle);
__device__ float4 compute_ppf(float3 p1, float3 n1, float3 p2, float3 n2);
__device__ void trans(float3 v, float T[4][4]);
__device__ void rotx(float theta, float T[4][4]);
__device__ void roty(float theta, float T[4][4]);
__device__ void rotz(float theta, float T[4][4]);
__device__ void mat4f_mul(const float A[4][4], const float B[4][4], float C[4][4]);
__device__ float4 mat4f_vmul(const float A[4][4], const float4 b);
__device__ float4 homogenize(float3 v);
__device__ float3 dehomogenize(float4 v);
__device__ float3 times(float a, float3 v);
__device__ float4 times(float a, float4 v);
__device__ float3 plus(float3 u, float3 v);
__device__ float4 plus(float4 u, float4 v);
__device__ float3 minus(float3 u, float3 v);
__device__ float4 minus(float4 u, float4 v);

__device__ void trans_model_scene(float3 m_r, float3 n_r_m, float3 m_i,
                                  float3 s_r, float3 n_r_s, float3 s_i,
                                  float d_dist, float3 &trans_vec,
                                  unsigned int &alpha);

__device__ float4 compute_ppf(float3 p1, float3 n1, float3 p2, float3 n2);


__global__ void ppf_kernel(float3 *points, float3 *norms, float4 *out, int count);

__global__ void ppf_encode_kernel(float4 *ppfs, unsigned long *codes, int count);

__global__ void ppf_decode_kernel(unsigned long *codes, unsigned int *key2ppfMap,
                                  unsigned int *hashKeys, int count);

__global__ void vec_decode_kernel(float4 *vecs, unsigned int *key2VecMap,
                                  float3 *vecCodes, int count);

__global__ void ppf_hash_kernel(float4 *ppfs, unsigned int *codes, int count);

__global__ void ppf_vote_kernel(unsigned int *sceneKeys, unsigned int *sceneIndices,
                                unsigned int *hashKeys, unsigned int *ppfCount,
                                unsigned int *firstPPFIndex, unsigned int *key2ppfMap,
                                float3 *modelPoints, float3 *modelNormals, int modelSize,
                                float3 *scenePoints, float3 *sceneNormals, int sceneSize,
                                unsigned long *votes, float3 *vecs_old, int count);

__global__ void ppf_reduce_rows_kernel(float3 *vecs, unsigned int *vecCounts,
                                       unsigned int *firstVecIndex,
                                       unsigned int *key2VecMap,
                                       unsigned long *voteCodes,
                                       unsigned int *voteCounts,
                                       int n_angle,
                                       unsigned int *accumulator,
                                       int count);

__global__ void ppf_score_kernel(unsigned int *accumulator,
                                 unsigned int *maxidx,
                                 int n_angle, float threshold,
                                 unsigned int *scores,
                                 int count);

#endif /* __KERNEL_H */

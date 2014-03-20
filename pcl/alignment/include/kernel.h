#ifndef __KERNEL_H
#define __KERNEL_H

#include <cuda.h>
#include <cuda_runtime.h>                // Stops underlining of __global__
#include <device_launch_parameters.h>    // Stops underlining of threadIdx etc.
#include <math_constants.h>

#include "debug.h"

//Launch configuration macros
#define BLOCK_SIZE 512
#define MAX_NBLOCKS 1024
//Algorithm macros
#define N_ANGLE 32
#define D_ANGLE0 (2.0f*float(CUDART_PI_F))/float(N_ANGLE);  //this one is for discretizing the feature in ppf_kernel
#define D_ANGLE1 (2.0f*float(CUDART_PI_F))/float(N_ANGLE - 1);  //this one is for assigining alpha indices [0 31]
#define D_DIST 0.1f;
#define SCORE_THRESHOLD 0

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
                                       unsigned long *votes,
                                       int n_angle,
                                       unsigned int *accumulator,
                                       int count);

__global__ void ppf_score_kernel(unsigned int *accumulator,
                                 unsigned int *maxidx,
                                 int n_angle, int threshold,
                                 unsigned int *scores,
                                 int count);

__global__ void trans_calc_kernel(float3 *vecs, unsigned int *vecCounts,
                                  unsigned int *firstVecIndex, unsigned long *votes,
                                  unsigned int *maxidx, unsigned int *scores, int n_angle,
                                  float3 *model_points, float3 *model_normals, int model_size,
                                  float3 *scene_points, float3 *scene_normals, int scene_size,
                                  float *transforms, int count);

#endif /* __KERNEL_H */

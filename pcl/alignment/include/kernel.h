#ifndef __KERNEL_H
#define __KERNEL_H

#include <cstdlib>
#include <cuda.h>
#include <cuda_runtime.h>                // Stops underlining of __global__
#include <device_launch_parameters.h>    // Stops underlining of threadIdx etc.
#include <math_constants.h>

#include "debug.h"

//Launch configuration macros
#define BLOCK_SIZE 512
#define MAX_NBLOCKS 1024
//Algorithm macros
#define MIN(X,Y) ((X) < (Y) ? (X) : (Y))
#define N_ANGLE 30
#define D_ANGLE0 ((2.0f*float(CUDART_PI_F))/float(N_ANGLE))  //this one is for discretizing the feature in ppf_kernel
#define D_DIST 0.035317969013662f  // generated based on MATLAB model_description.m:12, specifically for chair model
#define TRANS_THRESH (0.5*D_DIST)
#define ROT_THRESH (2*D_ANGLE0)
#define SCORE_THRESHOLD 0

__global__ void ppf_kernel(float3 *points, float3 *norms, float4 *out, int count);

__global__ void ppf_encode_kernel(float4 *ppfs, unsigned long *codes, int count);

__global__ void ppf_decode_kernel(unsigned long *codes, unsigned int *key2ppfMap,
                                  unsigned int *hashKeys, int count);

__global__ void vec_decode_kernel(float4 *vecs, unsigned int *key2VecMap,
                                  float3 *vecCodes, int count);

__global__ void ppf_hash_kernel(float4 *ppfs, unsigned int *codes, int count);

__global__ void ppf_vote_kernel(unsigned int *sceneKeys, std::size_t *sceneIndices,
                                unsigned int *hashKeys, std::size_t *ppfCount,
                                std::size_t *firstPPFIndex, std::size_t *key2ppfMap,
                                float3 *modelPoints, float3 *modelNormals, int modelSize,
                                float3 *scenePoints, float3 *sceneNormals, int sceneSize,
                                unsigned long *votes, int count);

__global__ void ppf_reduce_rows_kernel(unsigned long *votes, unsigned int *voteCounts,
                                       unsigned int *firstVoteIndex,
                                       int n_angle,
                                       unsigned int *accumulator,
                                       int count);

__global__ void ppf_score_kernel(unsigned int *accumulator,
                                 unsigned int *maxidx,
                                 int n_angle, int threshold,
                                 unsigned int *scores,
                                 int count);

__global__ void trans_calc_kernel(unsigned int *uniqueSceneRefPts,
                                  unsigned int *maxModelAngleCodes,
                                  float3 *model_points, float3 *model_normals,
                                  float3 *scene_points, float3 *scene_normals,
                                  float *transforms, int count);

__global__ void trans_calc_kernel2(unsigned long *votes,
                                   float3 *model_points, float3 *model_normals,
                                   float3 *scene_points, float3 *scene_normals,
                                   float *transforms, int count);

__global__ void mat2transquat_kernel(float *transformations,
                                     float3 *transformation_trans,
                                     float4 *transformation_rots,
                                     int count);

__global__ void rot_clustering_kernel(float3 *translations,
                                      float4 *quaternions,
                                      float *vote_counts,
                                      unsigned int *adjacent_trans_hash,
                                      std::size_t *transIndices,
                                      unsigned int *transKeys,  std::size_t *transCount,
                                      std::size_t *firstTransIndex, std::size_t *key2transMap,
                                      // float3 *translations_out,
                                      // float4 *quaternions_out,
                                      float *vote_counts_out,
                                      int count);

__global__ void trans2idx_kernel(float3 *translations,
                                 unsigned int *trans_hash,
                                 unsigned int *adjacent_trans_hash,
                                 int count);

__global__ void vote_weight_kernel(unsigned long *votes, unsigned int *vote_counts,
                                   float *modelPointWeights, float *weightedVoteCounts,
                                   int count);

#endif /* __KERNEL_H */

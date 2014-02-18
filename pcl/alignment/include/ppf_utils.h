#ifndef __PPF_UTILS
#define __PPF_UTILS

#include <cuda.h>
#include <cuda_runtime.h>                // Stops underlining of __global__
#include <device_launch_parameters.h>    // Stops underlining of threadIdx etc.

__device__ unsigned int hash(void *f, int n);

__device__ __forceinline__ float dot(float3 v1, float3 v2);

__device__ __forceinline__ float norm(float3 v);

__device__ float4 disc_feature(float4 f, float d_dist, float d_angle);

__device__ float4 compute_ppf(float3 p1, float3 n1, float3 p2, float3 n2);

#endif /* __PPF_UTILS */

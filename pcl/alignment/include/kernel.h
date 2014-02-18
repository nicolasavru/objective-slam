#ifndef __KERNEL_H
#define __KERNEL_H

#include <cuda.h>
#include <cuda_runtime.h>                // Stops underlining of __global__
#include <device_launch_parameters.h>    // Stops underlining of threadIdx etc.
#include <math_constants.h>

#define BLOCK_SIZE 256

__const__ int n_angle = 32;
__const__ float d_angle = 2*CUDART_PI_F/n_angle;
__const__ float d_dist = 0.05;

__global__ void ppf_kernel(float3 *points, float3 *norms, float4 *out, int count);

__global__ void ppf_encode_kernel(float4 *ppfs, unsigned long *codes, int count);

__global__ void ppf_decode_kernel(unsigned long *codes, unsigned int *key2ppfMap,
                                  unsigned int *hashKeys, int count);

__global__ void ppf_hash_kernel(float4 *ppfs, unsigned int *codes, int count);

__global__ void ppf_vote_kernel(float4 *ppfs, unsigned long *codes, int count);


#endif /* __KERNEL_H */

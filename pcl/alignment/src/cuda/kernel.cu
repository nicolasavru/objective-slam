#include "kernel.h"

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

__device__ __forceinline__ float dot(float3 v1, float3 v2){
    return (v1.x*v2.x + v1.y*v2.y + v1.z*v2.z);
}

__device__ __forceinline__ float norm(float3 v){
    return sqrtf(dot(v, v));
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
        // line 11 in algorithm 1, typo on their part
        key2ppfMap[idx] = (unsigned int) (codes[idx] & low32);
        hashKeys[idx] = (unsigned int) (codes[idx] >> 32);
    }
}

// TODO: increase thread work
__global__ void ppf_hash_kernel(float4 *ppfs, unsigned int *codes, int count){
    if(count <= 1) return;

    int ind = threadIdx.x;
    int idx = ind + blockIdx.x * blockDim.x;

    if(idx < count){
        codes[idx] = hash(ppfs+idx, sizeof(float4));
    }
}

// TODO: increase thread work
__global__ void ppf_lookup_kernel(unsigned int *sceneKeys, unsigned int *sceneIndices,
                                  unsigned int *hashKeys, unsigned int *ppfCount,
                                  unsigned int *firstPPFIndex, unsigned int *key2ppfMap,
                                  unsigned int *found_ppf_start, unsigned int *found_ppf_count,
                                  int count){
    if(count <= 1) return;

    int ind = threadIdx.x;
    int idx = ind + blockIdx.x * blockDim.x;

    if(idx < count){
        unsigned int thisSceneKey = sceneKeys[idx];
        unsigned int thisSceneIndex = sceneIndices[idx];
        if (thisSceneKey != hashKeys[thisSceneIndex]){
            return;
        }
        unsigned int thisPPFCount = ppfCount[thisSceneIndex];
        unsigned int thisFirstPPFIndex = firstPPFIndex[thisSceneIndex];

        /*for (int i=0; i<thisPPFCount; i++) {*/
            /*unsigned int modelPPFIndex = key2ppfMap[thisFirstPPFIndex+i];*/
            /*modelPPF = */
        /*}*/


    }
}

// TODO: increase thread work
__global__ void ppf_vote_kernel(float4 *ppfs, unsigned long *codes, int count){
    if(count <= 1) return;

    int ind = threadIdx.x;
    int idx = ind + blockIdx.x * blockDim.x;

    if(idx < count){

        codes[idx] = hash(ppfs+idx, sizeof(float4));
    }

}

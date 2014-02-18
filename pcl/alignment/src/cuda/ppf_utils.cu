#include "ppf_utils.h"


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

#include <string.h>
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
    return v1.x*v2.x + v1.y*v2.y + v1.z*v2.z;
}

__device__ __forceinline__ float dot(float4 v1, float4 v2){
    return v1.x*v2.x + v1.y*v2.y + v1.z*v2.z + v1.w*v2.w;
}


__device__ __forceinline__ float norm(float3 v){
    return sqrtf(dot(v, v));
}

__device__  float3 cross(float3 u, float3 v){
    float3 w = {u.y*v.z - u.z*v.y,
                u.z*v.x - u.x*v.z,
                u.x*v.y - u.y*v.z};
    return w;
}

__device__ __forceinline__ float quant_downf(float x, float y){
    return x - fmodf(x, y);
}

__device__ float4 disc_feature(float4 f, float d_dist, float d_angle){
    f.x = quant_downf(f.x, d_dist);
    f.y = quant_downf(f.y, d_angle);
    f.z = quant_downf(f.z, d_angle);
    f.w = quant_downf(f.w, d_angle);
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

__device__ void trans(float3 v, float T[4][4]){
    memset(T, 0, sizeof(T));
    T[0][0] = 1;
    T[1][1] = 1;
    T[2][2] = 1;
    T[3][3] = 1;
    T[0][3] = v.x;
    T[1][3] = v.y;
    T[2][3] = v.z;
}

__device__ void rotx(float theta, float T[4][4]){
    memset(T, 0, sizeof(T));
    T[0][0] = 1;
    T[1][1] = cosf(theta);
    T[2][1] = sinf(theta);
    T[1][2] = -1*T[2][1];
    T[2][2] = T[1][1];
    T[3][3] = 1;
}

__device__ void roty(float theta, float T[4][4]){
    memset(T, 0, sizeof(T));
    T[0][0] = cosf(theta);
    T[0][2] = sinf(theta);
    T[1][1] = 1;
    T[2][0] = -1*T[0][2];
    T[2][2] = T[0][0];
    T[3][3] = 1;
}

__device__ void rotz(float theta, float T[4][4]){
    memset(T, 0, sizeof(T));
    T[0][0] = cosf(theta);
    T[1][0] = sinf(theta);
    T[0][1] = -1*T[1][0];
    T[1][1] = T[0][0];
    T[2][2] = 1;
    T[3][3] = 1;
}

__device__ void mat4f_mul(const float A[4][4],
                          const float B[4][4],
                          float C[4][4]){
    memset(C, 0, sizeof(C));
    for(int i = 0; i < 4; i++) {
        for(int j = 0; j < 4; j++) {
            for(int k = 0; k < 4; k++)
                C[i][j] += A[i][k]*B[k][j];
        }
    }
}

__device__ float4 mat4f_vmul(const float A[4][4], const float4 b){
    float4 *Af4 = (float4 *) A;
    float4 c;
    c.x = dot(Af4[0], b);
    c.y = dot(Af4[1], b);
    c.z = dot(Af4[2], b);
    c.w = dot(Af4[3], b);
    return c;
}

__device__ float4 homogenize(float3 v){
    float4 w = {v.x, v.y, v.z, 1};
    return w;
}

__device__ float3 dehomogenize(float4 v){
    float3 w = {v.x, v.y, v.z};
    return w;
}

__device__ float3 times(float a, float3 v){
    float3 w = {a*v.x, a*v.y, a*v.z};
    return w;
}

__device__ float4 times(float a, float4 v){
    float4 w = {a*v.x, a*v.y, a*v.z, a*v.z};
    return w;
}

__device__ float3 plus(float3 u, float3 v){
    float3 w = {u.x+v.x, u.y+v.y, u.z+v.z};
    return w;
}

__device__ float4 plus(float4 u, float4 v){
    float4 w = {u.x+v.x, u.y+v.y, u.z+v.z, u.w+v.w};
    return w;
}

__device__ float3 minus(float3 u, float3 v){
    float3 w = {u.x-v.x, u.y-v.y, u.z-v.z};
    return w;
}

__device__ float4 minus(float4 u, float4 v){
    float4 w = {u.x-v.x, u.y-v.y, u.z-v.z, u.w-v.w};
    return w;
}


__device__ void trans_model_scene(float3 m_r, float3 n_r_m, float3 m_i,
                                  float3 s_r, float3 n_r_s, float3 s_i,
                                  float T_m_g[4][4], float T_s_g[4][4], float &alpha){
    float transm[4][4], rot_y[4][4], rot_z[4][4], T_tmp[4][4];
    float4 n_tmp;

    m_r = times(-1, m_r);
    trans(m_r, transm);

    roty(atan2f(n_r_m.z, n_r_m.x), rot_y);

    n_tmp = homogenize(n_r_m);

    mat4f_vmul(rot_y, n_tmp);

    rotz(-1*atan2f(n_tmp.y, n_tmp.x), rot_z);

    mat4f_mul(rot_z, rot_y, T_tmp);
    mat4f_mul(T_tmp, transm, T_m_g);


    s_r = times(-1, s_r);
    trans(s_r, transm);

    roty(atan2f(n_r_s.z, n_r_s.x), rot_y);

    n_tmp = homogenize(n_r_s);

    mat4f_vmul(rot_y, n_tmp);

    rotz(-1*atan2f(n_tmp.y, n_tmp.x), rot_z);

    mat4f_mul(rot_z, rot_y, T_tmp);
    mat4f_mul(T_tmp, transm, T_s_g);


    n_tmp = homogenize(m_i);
    n_tmp = mat4f_vmul(T_m_g, n_tmp);
    float3 u = dehomogenize(n_tmp);

    n_tmp = homogenize(s_i);
    n_tmp = mat4f_vmul(T_s_g, n_tmp);
    float3 v = dehomogenize(n_tmp);

    u.x = 0;
    v.x = 0;

    alpha = atan2f(cross(u, v).x, dot(u, v));
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
                                  float3 *modelPoints, float3 *modelNormals, int modelSize,
                                  float3 *scenePoints, float3 *sceneNormals, int sceneSize,
                                  unsigned int *votes,
                                  int count){
    if(count <= 1) return;

    int ind = threadIdx.x;
    int idx = ind + blockIdx.x * blockDim.x;

    if(idx < count){
        unsigned int thisSceneKey = sceneKeys[idx];
        unsigned int thisSceneIndex = sceneIndices[idx];
        float3 thisScenePoint = scenePoints[idx];
        float3 thisSceneNormal = sceneNormals[idx];
        if (thisSceneKey != hashKeys[thisSceneIndex]){
            return;
        }
        unsigned int thisPPFCount = ppfCount[thisSceneIndex];
        unsigned int thisFirstPPFIndex = firstPPFIndex[thisSceneIndex];

        unsigned int modelPPFIndex, model_r_index, model_i_index;
        float3 model_r_point, model_r_norm, model_i_point, scene_i_point;
        for(int i = 0; i < thisPPFCount; i++){
            modelPPFIndex = key2ppfMap[thisFirstPPFIndex+i];
            model_r_index = modelPPFIndex / modelSize;
            model_i_index = modelPPFIndex % modelSize;

            model_r_point = modelPoints[model_r_index];
            model_r_norm = modelNormals[model_r_index];
            model_i_point = modelPoints[model_i_index];

            // for(int j = 0; j < sceneSize; j++){
            //     scene_i_point = scenePoints[j];
            //     trans_model_scene(model_r_point, model_r_normal, model_i_point,
            //                       thisScenePoint, thisSceneNormal, scene_i_point);

            // }

        }


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

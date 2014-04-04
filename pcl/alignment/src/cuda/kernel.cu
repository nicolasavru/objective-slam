#include <string.h>
#include <cuda.h>
#include <cuda_runtime.h>                // Stops underlining of __global__
#include <device_launch_parameters.h>    // Stops underlining of threadIdx etc.
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

__device__ __forceinline__ void zeroMat4(float T[4][4]){
    T[0][0] = 0;
    T[0][1] = 0;
    T[0][2] = 0;
    T[0][3] = 0;
    T[1][0] = 0;
    T[1][1] = 0;
    T[1][2] = 0;
    T[1][3] = 0;
    T[2][0] = 0;
    T[2][1] = 0;
    T[2][2] = 0;
    T[2][3] = 0;
    T[3][0] = 0;
    T[3][1] = 0;
    T[3][2] = 0;
    T[3][3] = 0;
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

__device__ __forceinline__  float3 cross(float3 u, float3 v){
    float3 w = {u.y*v.z - u.z*v.y,
                u.z*v.x - u.x*v.z,
                u.x*v.y - u.y*v.z};
    return w;
}

__device__ __forceinline__ float quant_downf(float x, float y){
    return x - fmodf(x, y);
}

__device__ __forceinline__ float4 disc_feature(float4 f, float d_dist, float d_angle){
    f.x = quant_downf(f.x, d_dist);
    f.y = quant_downf(f.y, d_angle);
    f.z = quant_downf(f.z, d_angle);
    f.w = quant_downf(f.w, d_angle);
    return f;
}

__device__ __forceinline__ float3 discretize(float3 f, float d_dist){
    f.x = quant_downf(f.x, d_dist);
    f.y = quant_downf(f.y, d_dist);
    f.z = quant_downf(f.z, d_dist);
    return f;
}

__device__ __forceinline__ float4 compute_ppf(float3 p1, float3 n1, float3 p2, float3 n2){
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

__device__ __forceinline__ void trans(float3 v, float T[4][4]){
    zeroMat4(T);
    T[0][0] = 1;
    T[1][1] = 1;
    T[2][2] = 1;
    T[3][3] = 1;
    T[0][3] = v.x;
    T[1][3] = v.y;
    T[2][3] = v.z;
}

__device__ __forceinline__ void rotx(float theta, float T[4][4]){
    zeroMat4(T);
    T[0][0] = 1;
    T[1][1] = cosf(theta);
    T[2][1] = sinf(theta);
    T[1][2] = -1*T[2][1];
    T[2][2] = T[1][1];
    T[3][3] = 1;
}

__device__ __forceinline__ void roty(float theta, float T[4][4]){
    zeroMat4(T);
    T[0][0] = cosf(theta);
    T[0][2] = sinf(theta);
    T[1][1] = 1;
    T[2][0] = -1*T[0][2];
    T[2][2] = T[0][0];
    T[3][3] = 1;
}

__device__ __forceinline__ void rotz(float theta, float T[4][4]){
    zeroMat4(T);
    T[0][0] = cosf(theta);
    T[1][0] = sinf(theta);
    T[0][1] = -1*T[1][0];
    T[1][1] = T[0][0];
    T[2][2] = 1;
    T[3][3] = 1;
}

__device__ __forceinline__ void mat4f_mul(const float A[4][4],
                          const float B[4][4],
                          float C[4][4]){
    zeroMat4(C);

    for(int i = 0; i < 4; i++){
        for(int j = 0; j < 4; j++){
            for(int k = 0; k < 4; k++){
                C[i][j] += A[i][k]*B[k][j];
            }
        }
    }
}

__device__ __forceinline__ float3 mat3f_vmul(const float A[3][3], const float3 b){
    float3 *Af3 = (float3 *) A;
    float3 c;
    c.x = dot(Af3[0], b);
    c.y = dot(Af3[1], b);
    c.z = dot(Af3[2], b);
    return c;
}

__device__ __forceinline__ float4 mat4f_vmul(const float A[4][4], const float4 b){
    float4 *Af4 = (float4 *) A;
    float4 c;
    c.x = dot(Af4[0], b);
    c.y = dot(Af4[1], b);
    c.z = dot(Af4[2], b);
    c.w = dot(Af4[3], b);
    return c;
}

__device__ __forceinline__ float4 homogenize(float3 v){
    float4 w = {v.x, v.y, v.z, 1};
    return w;
}

__device__ __forceinline__ float3 dehomogenize(float4 v){
    float3 w = {v.x, v.y, v.z};
    return w;
}

__device__ __forceinline__ float3 times(float a, float3 v){
    float3 w = {a*v.x, a*v.y, a*v.z};
    return w;
}

__device__ __forceinline__ float4 times(float a, float4 v){
    float4 w = {a*v.x, a*v.y, a*v.z, a*v.z};
    return w;
}

__device__ __forceinline__ float3 plus(float3 u, float3 v){
    float3 w = {u.x+v.x, u.y+v.y, u.z+v.z};
    return w;
}

__device__ __forceinline__ float4 plus(float4 u, float4 v){
    float4 w = {u.x+v.x, u.y+v.y, u.z+v.z, u.w+v.w};
    return w;
}

__device__ __forceinline__ float3 minus(float3 u, float3 v){
    float3 w = {u.x-v.x, u.y-v.y, u.z-v.z};
    return w;
}

__device__ __forceinline__ float4 minus(float4 u, float4 v){
    float4 w = {u.x-v.x, u.y-v.y, u.z-v.z, u.w-v.w};
    return w;
}

__device__ __forceinline__ void invht(float T[4][4], float T_inv[4][4]){
    // T = [R t;0 1]; inv(T) = [R' -R'*t;0 1]; R*R' = I
    // R'
    T_inv[0][0] = T[0][0];
    T_inv[0][1] = T[1][0];
    T_inv[0][2] = T[2][0];

    T_inv[1][0] = T[0][1];
    T_inv[1][1] = T[1][1];
    T_inv[1][2] = T[2][1];

    T_inv[2][0] = T[0][2];
    T_inv[2][1] = T[1][2];
    T_inv[2][2] = T[2][2];

    // -R'
    float neg_Rtranspose[3][3];
    neg_Rtranspose[0][0] = -T_inv[0][0];
    neg_Rtranspose[0][1] = -T_inv[0][1];
    neg_Rtranspose[0][2] = -T_inv[0][2];

    neg_Rtranspose[1][0] = -T_inv[1][0];
    neg_Rtranspose[1][1] = -T_inv[1][1];
    neg_Rtranspose[1][2] = -T_inv[1][2];

    neg_Rtranspose[2][0] = -T_inv[2][0];
    neg_Rtranspose[2][1] = -T_inv[2][1];
    neg_Rtranspose[2][2] = -T_inv[2][2];

    // t
    float3 T_tmp;
    T_tmp.x = T[0][3];
    T_tmp.y = T[1][3];
    T_tmp.z = T[2][3];

    //-R*t
    float3 tmp = mat3f_vmul(neg_Rtranspose, T_tmp);
    T_inv[0][3] = tmp.x;
    T_inv[1][3] = tmp.y;
    T_inv[2][3] = tmp.z;

    T_inv[3][0] = 0;
    T_inv[3][1] = 0;
    T_inv[3][2] = 0;
    T_inv[3][3] = 1;
}


__device__ void trans_model_scene(float3 m_r, float3 n_r_m, float3 m_i,
                                  float3 s_r, float3 n_r_s, float3 s_i,
                                  float d_dist, unsigned int &alpha_idx){
    float transm[4][4], rot_x[4][4], rot_y[4][4], rot_z[4][4], T_tmp[4][4], T_m_g[4][4], T_s_g[4][4],
    T_tmp2[4][4], T[4][4];
    float4 n_tmp;
    float3 s_r_transformed;

    // probably not necessary due to PCL resampling onto voxel grid
    // m_r = discretize(m_r, d_dist);
    m_r = times(-1, m_r);

    trans(m_r, transm);
    roty(atan2f(n_r_m.z, n_r_m.x), rot_y);
    n_tmp = homogenize(n_r_m);
    mat4f_vmul(rot_y, n_tmp);
    rotz(-1*atan2f(n_tmp.y, n_tmp.x), rot_z);
    mat4f_mul(rot_z, rot_y, T_tmp);     //POTENTIALLY SLOW
    mat4f_mul(T_tmp, transm, T_m_g);    //POTENTIALLY SLOW

    s_r = times(-1, s_r);
    trans(s_r, transm);
    roty(atan2f(n_r_s.z, n_r_s.x), rot_y);
    n_tmp = homogenize(n_r_s);
    mat4f_vmul(rot_y, n_tmp);
    rotz(-1*atan2f(n_tmp.y, n_tmp.x), rot_z);
    mat4f_mul(rot_z, rot_y, T_tmp);     //POTENTIALLY SLOW
    mat4f_mul(T_tmp, transm, T_s_g);    //POTENTIALLY SLOW


    n_tmp = homogenize(m_i);
    n_tmp = mat4f_vmul(T_m_g, n_tmp);
    float3 u = dehomogenize(n_tmp);

    n_tmp = homogenize(s_i);
    n_tmp = mat4f_vmul(T_s_g, n_tmp);
    float3 v = dehomogenize(n_tmp);

    u.x = 0;
    v.x = 0;
    float alpha = atan2f(cross(u, v).x, dot(u, v));
    alpha_idx = (int)((alpha + CUDART_PI_F)*N_ANGLE/(2*CUDART_PI_F));
    rotx(alpha, rot_x);

    invht(T_s_g, T_tmp);
    mat4f_mul(T_tmp, rot_x, T_tmp2);    //POTENTIALLY SLOW
    mat4f_mul(T_tmp2, T_m_g, T);        //POTENTIALLY SLOW
    // T is T_ms
}

__device__ void compute_rot_angles(float3 n_r_m, float3 n_r_s,
                                   float *m_roty, float *m_rotz,
                                   float *s_roty, float *s_rotz){
    float rot_y[4][4];
    float4 n_tmp;

    *m_roty = atan2f(n_r_m.z, n_r_m.x);
    roty(*m_roty, rot_y);
    n_tmp = homogenize(n_r_m);
    mat4f_vmul(rot_y, n_tmp);
    *m_rotz = -1*atan2f(n_tmp.y, n_tmp.x);

    *s_roty = atan2f(n_r_s.z, n_r_s.x);
    roty(*s_roty, rot_y);
    n_tmp = homogenize(n_r_s);
    mat4f_vmul(rot_y, n_tmp);
    *s_rotz = -1*atan2f(n_tmp.y, n_tmp.x);
}

__device__ void compute_transforms(unsigned int angle_idx, float3 m_r,
                                   float m_roty, float m_rotz,
                                   float3 s_r, float s_roty,
                                   float s_rotz, float *T){
    float transm[4][4], rot_x[4][4], rot_y[4][4], rot_z[4][4], T_tmp[4][4],
          T_tmp2[4][4], T_m_g[4][4], T_s_g[4][4];

    float (*T_arr)[4] = (float (*)[4])T;

    // m_r = discretize(m_r, D_DIST);
    m_r = times(-1, m_r);

    trans(m_r, transm);
    roty(m_roty, rot_y);
    rotz(m_rotz, rot_z);
    mat4f_mul(rot_z, rot_y, T_tmp);
    mat4f_mul(T_tmp, transm, T_m_g);

    s_r = times(-1, s_r);
    trans(s_r, transm);
    roty(s_roty, rot_y);
    rotz(s_rotz, rot_z);
    mat4f_mul(rot_z, rot_y, T_tmp);
    mat4f_mul(T_tmp, transm, T_s_g);

    rotx(angle_idx*2*CUDART_PI_F/(N_ANGLE-1), rot_x);
    invht(T_s_g, T_tmp);
    mat4f_mul(T_tmp, rot_x, T_tmp2);
    mat4f_mul(T_tmp2, T_m_g, T_arr);
}

__global__ void ppf_kernel(float3 *points, float3 *norms, float4 *out, int count){
    if(count <= 1) return;

    int ind = threadIdx.x;
    int idx = ind + blockIdx.x * blockDim.x;
    int bound;

    while(idx < count) {

        __shared__ float3 Spoints[BLOCK_SIZE];
        __shared__ float3 Snorms[BLOCK_SIZE];

        float3 thisPoint = points[idx];
        float3 thisNorm  = norms[idx];

        for(int i = 0; i < count; i+=BLOCK_SIZE){

            bound = MIN(count - i, BLOCK_SIZE);

            if (ind < bound){
                Spoints[ind] = points[i+ind];
                Snorms[ind]  = norms[i+ind];
            }
            __syncthreads();

            for(int j = 0; j < bound; j++) {
                if((j + i - idx) == 0){
                    out[idx*count + j + i].x = CUDART_NAN_F;
                    continue;
                };
                out[idx*count + j + i] = compute_ppf(thisPoint, thisNorm, Spoints[j], Snorms[j]);
                out[idx*count + j + i] = disc_feature(out[idx*count + j + i], D_DIST, D_ANGLE0);
            }
        }
        //grid stride
        idx += blockDim.x * gridDim.x;
    }
}

// TODO: increase thread work
__global__ void ppf_hash_kernel(float4 *ppfs, unsigned int *codes, int count){
    if(count <= 1) return;

    int ind = threadIdx.x;
    int idx = ind + blockIdx.x * blockDim.x;

    while(idx < count){
        if(ppfs[idx].x == CUDART_NAN_F){
            codes[idx] = 0;
        }
        else{
            codes[idx] = hash(ppfs+idx, sizeof(float4));
        }

        //grid stride
        idx += blockDim.x * gridDim.x;
    }
}



// during trans_model_scene() (while computing translation vector
// between scene ref point and model pt:
// 1) descritize translation vector (probably small multiple of voxel distance)
// 2) encode disc'd translation vector into long:
//    [trans vec|idx]
//    where idx is an index into the global array of votes (slam++ fig 4)
// 3) sort array of translation vec codes
// 4) reduce_by_key translation vec code array to get mapping from
// unique translation vecs to list of code indices (identical to lines 13-16
// slam++ algorithm 1)
// 5) for each unique translation vector:
//       create histogram by accumulating all associated votes
//       (probably identical to slam++ algorithm 2)
// 5a) (maybe) smooth adjacent translation vector histograms
// 6) for each unique translation vector histogram:
//      find max angle
//      score according to max angle + neighbors
// 7) compare score to threshold
// 8) get list of votes associated with unique trans vec and max angle + neighbors
// 9) At this point, we have a list of clusters of (scene point, model point, angle)
//    tuples.
//    for each tuple:
//      call trans_model_scene:
//        a) compute rotation angles for each scene point model point pair
//        b) average computed angles and alphas from each tuple
//        c) compute T_m_g, T_s_g, and rotx(alpha) from averaged angles and alphas
//           and the unique translation vector corresponding to this cluster
// return drost eqn. 2 as final solution(s)
//

// TODO: increase thread work
__global__ void ppf_vote_kernel(unsigned int *sceneKeys, unsigned int *sceneIndices,
                                unsigned int *hashKeys, unsigned int *ppfCount,
                                unsigned int *firstPPFIndex, unsigned int *key2ppfMap,
                                float3 *modelPoints, float3 *modelNormals, int modelSize,
                                float3 *scenePoints, float3 *sceneNormals, int sceneSize,
                                unsigned long *votes_old, unsigned long *truncVotes, int count){
    if(count <= 1) return;

    int ind = threadIdx.x;
    int idx = ind + blockIdx.x * blockDim.x;
    unsigned int alpha_idx;

    while(idx < count){
        unsigned int thisSceneKey = sceneKeys[idx];
        unsigned int thisSceneIndex = sceneIndices[idx];
        if (thisSceneKey == 0 ||
            thisSceneKey != hashKeys[thisSceneIndex]){
            return;
        }
        unsigned int thisPPFCount = ppfCount[thisSceneIndex];
        unsigned int thisFirstPPFIndex = firstPPFIndex[thisSceneIndex];

        unsigned int scene_r_index = idx / sceneSize;
        unsigned int scene_i_index = idx - scene_r_index*sceneSize;
        float3 scene_r_point = scenePoints[scene_r_index];
        float3 scene_r_norm  = sceneNormals[scene_r_index];
        float3 scene_i_point = scenePoints[scene_i_index];

        unsigned int modelPPFIndex, model_r_index, model_i_index;
        float3 model_r_point, model_r_norm, model_i_point;
        for(int i = 0; i < thisPPFCount; i++){
            modelPPFIndex = key2ppfMap[thisFirstPPFIndex+i];
            model_r_index = modelPPFIndex / modelSize;
            model_i_index = modelPPFIndex - model_r_index*modelSize;

            model_r_point = modelPoints[model_r_index];
            model_r_norm  = modelNormals[model_r_index];
            model_i_point = modelPoints[model_i_index];

            trans_model_scene(model_r_point, model_r_norm, model_i_point,
                              scene_r_point, scene_r_norm, scene_i_point,
                              D_DIST, alpha_idx);
            votes_old[thisFirstPPFIndex + i] =
                (((unsigned long) scene_r_index) << 32) | (model_r_index << 6) | (alpha_idx);
            truncVotes[thisFirstPPFIndex + i] = 
            (((unsigned long) scene_r_index) << 26) | model_r_index;
        }

        idx += blockDim.x * gridDim.x;
    }
}

/*
  for(int i = 0; i < unique_vecs.size(); i++){
    int angle_histogram[32];
    // do this in parallel, so atomics should be used
    for(int j = 0; j < vecCounts[i]; j++){
      voteCode = votes[firstVoteIndex[i] + j];
      angle_idx = voteCode & low6;
      angle_histogram[angle_idx]++;
      
    }
 */

// TODO: increase thread work
__global__ void ppf_reduce_rows_kernel(unsigned long *votes, unsigned int *voteCounts,
                                       unsigned int *firstVoteIndex,
                                       int n_angle,
                                       unsigned int *accumulator,
                                       int count){
    if(count <= 1) return;

    int ind = threadIdx.x;
    int idx = ind + blockIdx.x * blockDim.x;
    unsigned int thisVoteCount, thisVoteIndex;
    int angle_idx;
    unsigned long vote;

    unsigned long low6 = ((unsigned long) -1) >> 58;

    while(idx < count){
        thisVoteCount = voteCounts[idx];
        thisVoteIndex = firstVoteIndex[idx];
        for(int i = 0; i < thisVoteCount; i++){
            vote      = votes[thisVoteIndex+i];
            if(vote == 0) continue;
            angle_idx = vote & low6;
            accumulator[idx*n_angle+angle_idx]++;
        }
        //grid stride
        idx += blockDim.x * gridDim.x;
    }
}

// TODO: increase thread work
__global__ void ppf_score_kernel(unsigned int *accumulator,
                                 unsigned int *maxidx,
                                 int n_angle, int threshold,
                                 unsigned int *scores,
                                 int count){
    if(count <= 1) return;

    int ind = threadIdx.x;
    int idx = ind + blockIdx.x * blockDim.x;

    int thisMaxIdx, score, score_left, score_right;

    while(idx < count){
        thisMaxIdx = idx*n_angle + maxidx[idx];
        score_left = thisMaxIdx > 0 ? accumulator[thisMaxIdx-1] : 0;
        score_right = thisMaxIdx < (n_angle-1) ? accumulator[thisMaxIdx+1] : 0;
        score = accumulator[thisMaxIdx] + score_left + score_right;
        scores[idx] = score > threshold ? score : 0;

        //grid stride
        idx += blockDim.x * gridDim.x;
    }
}


// for each transvec in maxidx:
//   binary seach for transvec in vecs
//   nevermind, we already know that - the ith row of the
//   accumulator (and maxidx) corresponds to vec[i]

//   goto found index in firstVecIndex and vecCounts
//   Use that to find voteIndices in vec2VoteMap
//   find possible starting indices of blocks matching Model hashKeys

// TODO: increase thread work
__global__ void trans_calc_kernel(unsigned long *votes, unsigned int *voteCounts,
                                  unsigned int *firstVoteIndex,
                                  unsigned int *maxidx, unsigned int *scores, int n_angle,
                                  float3 *model_points, float3 *model_normals, int model_size,
                                  float3 *scene_points, float3 *scene_normals, int scene_size,
                                  float *transforms, int count){
    if(count <= 1) return;

    int ind = threadIdx.x;
    int idx = ind + blockIdx.x * blockDim.x;

    unsigned int thisVoteCount, thisFirstVoteIndex, angle_idx, model_point_idx, scene_point_idx;
    unsigned long vote;

    unsigned long low6 = ((unsigned long) -1) >> 58;
    unsigned long hi32 = ((unsigned long) -1) << 32;
    unsigned long model_point_mask = ((unsigned long) -1) ^ hi32 ^ low6;
    float m_roty_t, m_rotz_t, s_roty_t, s_rotz_t;

    // angle average accumulators
    float m_roty = 0;
    float m_rotz = 0;
    float s_roty = 0;
    float s_rotz = 0;

    int c;

    while(idx < count){
        thisVoteCount = voteCounts[idx];
        thisFirstVoteIndex = firstVoteIndex[idx];
        c = 0;

        for(int i = 0; i < thisVoteCount; i++){
            vote = votes[thisFirstVoteIndex+i];
            angle_idx = (unsigned int) (vote & low6);
            if(((int) fabsf(angle_idx - maxidx[idx])) > 1) continue;

            scene_point_idx = (unsigned int) ((vote & hi32) >> 32);
            model_point_idx = (unsigned int) ((vote & model_point_mask) >> 6);
            compute_rot_angles(model_normals[model_point_idx],
                               scene_normals[scene_point_idx],
                               &m_roty_t, &m_rotz_t, &s_roty_t, &s_rotz_t);

            // running average
            m_roty = (m_roty_t + c*m_roty)/(c+1);
            m_rotz = (m_rotz_t + c*m_rotz)/(c+1);
            s_roty = (s_roty_t + c*s_roty)/(c+1);
            s_rotz = (s_rotz_t + c*s_rotz)/(c+1);
            c++;
        }

        angle_idx = (unsigned int) (votes[thisFirstVoteIndex] & low6);
        scene_point_idx = (unsigned int) ((votes[thisFirstVoteIndex] & hi32) >> 32);
        model_point_idx = (unsigned int) ((votes[thisFirstVoteIndex] & model_point_mask) >> 6);

        compute_transforms(angle_idx, model_points[model_point_idx],
                           m_roty, m_rotz,
                           scene_points[scene_point_idx],
                           s_roty, s_rotz, ((float *) transforms + idx*16));

        //grid stride
        idx += blockDim.x * gridDim.x;
    }
}

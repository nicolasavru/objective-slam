#ifndef __MODEL_H
#define __MODEL_H

#include <cuda.h>
#include <cuda_runtime.h>                // Stops underlining of __global__
#include <Eigen/Core>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include "impl/parallel_hash_array.hpp"
#include "scene.h"
#include "debug.h"

class Model : public Scene {

  public:
    Model(pcl::PointCloud<pcl::PointNormal> *cloud,
          thrust::host_vector<float3>*points, thrust::host_vector<float3> *normals, int n);
    ~Model();

    void SetModelPointVoteWeights(thrust::device_vector<float> modelPointVoteWeights);
    void ComputeUniqueVotes(Scene *scene);
    thrust::device_vector<float>
    ComputeWeightedVoteCounts(thrust::device_vector<unsigned long> votes,
                              thrust::device_vector<unsigned int> vote_counts,
                              thrust::device_vector<float> modelpoint_vote_weights);
    void ComputeTransformations(Scene *scene);
    thrust::device_vector<float> *ClusterTransformations();

    float ScorePose(const float *weights, Eigen::Matrix4f truth,
                    pcl::PointCloud<pcl::PointNormal> scene);
    thrust::device_vector<float> OptimizeWeights(Scene *scene);

    void ppf_lookup(Scene *scene);

    thrust::device_vector<float> getTransformations();
    pcl::PointCloud<pcl::PointNormal> *cloud_ptr;
    // private:
  public:

        // ppfCount[i] is the number of PPFs whose hash is hashKeys[i];
        thrust::device_vector<unsigned int> *ppfCount;

        // firstPPFIndex[i] is the index of the first entry in key2ppfMap
        // corresponding to hashKey[i]. The following ppfCount[i]-1
        // entries also correspond to hashKey[i].
        thrust::device_vector<unsigned int> *firstPPFIndex;

        // key2ppfMap[i] is the index in d_ppfs that contains (one of) the
        // PPF(s) whose hash is hashKeys[i]. From there, the indices of
        // the points that were used to generate the PPF can be
        // calculated.
        thrust::device_vector<unsigned int> *key2ppfMap;

        // Vector of all vote codes [Scene Ref Point|Model Point|Angle]
        //                           63           32 31        6 5   0
        thrust::device_vector<unsigned long> *votes;

        // Vector of unique vote codes
        thrust::device_vector<unsigned long> *voteCodes;

        // voteCounts[i] is the number of votes associated with the
        // the vote code stored in voteCodes[i].
        thrust::device_vector<unsigned int> *voteCounts;

        // firstVoteIndex[i] is the index of the first entry in voteCodes
        // corresponding to votes[i]. The following voteCounts[i]-1
        // entries also correspond to voteCodes[i].
        thrust::device_vector<unsigned int> *firstVoteIndex;

        // Vector of unique vecs
        thrust::device_vector<float3> *vecs;

        // vecCounts[i] is the number of vecs associated with the
        // the vec code stored in vecCodes[i].
        thrust::device_vector<unsigned int> *vecCounts;

        // firstVecIndex[i] is the index of the first entry in vecCodes
        // corresponding to votes[i]. The following vecCounts[i]-1
        // entries also correspond to vecCodes[i].
        thrust::device_vector<unsigned int> *firstVecIndex;

        // transformations stores 4 by 4 arrays of transformation matrices
        // however it uses linear indexing
        thrust::device_vector<float> transformations;

        thrust::device_vector<float4> *transformation_rots;
        thrust::device_vector<float3> *transformation_trans;
        thrust::device_vector<float> *vote_counts_out;

        thrust::device_vector<unsigned int> *accumulator;
        thrust::device_vector<unsigned int> *maxidx;
        thrust::device_vector<unsigned int> *maxval;

        thrust::device_vector<unsigned int> *trans_hash;
        thrust::device_vector<unsigned int> *key2transMap;
        thrust::device_vector<unsigned int> *transCount;
        thrust::device_vector<unsigned int> *firstTransIndex;

        thrust::device_vector<unsigned int> *adjacent_trans_hash;

        ParallelHashArray<unsigned int> search_array;
        thrust::device_vector<float> modelPointVoteWeights;
        thrust::device_vector<float> weightedVoteCounts;

        void accumulateVotes();
};


#endif /* __MODEL_H */

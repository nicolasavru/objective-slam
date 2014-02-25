#ifndef __MODEL_H
#define __MODEL_H

#include <cuda.h>
#include <cuda_runtime.h>                // Stops underlining of __global__
#include <device_launch_parameters.h>    // Stops underlining of threadIdx etc.
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include "scene.h"
#include "debug.h"

class Model : public Scene {

    public:

        Model(thrust::host_vector<float3>*points, thrust::host_vector<float3> *normals, int n);

        ~Model();

        void ppf_lookup(Scene *scene);

    private:



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

        // Vector of all vote codes
        thrust::device_vector<unsigned long> *votes;

        // Vector of unique vote codes
        thrust::device_vector<unsigned long> *voteCodes;

        // voteCount[i] is the number of votes associated with the
        // the vote code stored in voteCodes[i].
        thrust::device_vector<unsigned int> *voteCounts;

        void accumulateVotes();
};


#endif /* __MODEL_H */

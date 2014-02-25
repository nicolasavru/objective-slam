#ifndef __SCENE_H
#define __SCENE_H

#include <cuda.h>
#include <cuda_runtime.h>                // Stops underlining of __global__
#include <device_launch_parameters.h>    // Stops underlining of threadIdx etc.
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include "debug.h"

#define RAW_PTR(V) thrust::raw_pointer_cast(V->data())

class Scene {

    public:

        Scene(thrust::host_vector<float3>*points, thrust::host_vector<float3> *normals, int n);

        ~Scene();

        int numPoints();
        thrust::device_vector<float3> *getModelPoints();
        thrust::device_vector<float3> *getModelNormals();
        thrust::device_vector<float4> *getModelPPFs();

    protected:

        // Number of PPF in the mode. I.e., number of elements in each of
        // the following arrays;
        int n;

        // Vector of model points
        thrust::device_vector<float3> *modelPoints;

        // Vector of model normals
        thrust::device_vector<float3> *modelNormals;

        // Vector of model point pair features
        thrust::device_vector<float4> *modelPPFs;
};


#endif /* __SCENE_H */

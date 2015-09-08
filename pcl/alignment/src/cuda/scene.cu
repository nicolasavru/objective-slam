#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>                // Stops underlining of __global__
#include <thrust/sort.h>
#include <thrust/inner_product.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/scan.h>
#include <thrust/binary_search.h>

#include "scene.h"
#include "kernel.h"
#include "book.h"
#include "impl/util.hpp"

Scene::Scene(){}

Scene::Scene(thrust::host_vector<float3> *points, thrust::host_vector<float3> *normals, int n){
    this->initPPFs(points, normals, n);
    this->hashKeys = new thrust::device_vector<unsigned int>(this->modelPPFs->size());

    int blocks = std::min(((int)(this->modelPPFs->size()) + BLOCK_SIZE - 1) / BLOCK_SIZE, MAX_NBLOCKS);
    ppf_hash_kernel<<<blocks,BLOCK_SIZE>>>(RAW_PTR(this->modelPPFs),
                                           RAW_PTR(this->hashKeys),
                                           this->modelPPFs->size());
}

Scene::~Scene(){
    delete this->modelPoints;
    delete this->modelNormals;
    delete this->modelPPFs;
    delete this->hashKeys;
}

void Scene::initPPFs(thrust::host_vector<float3> *points, thrust::host_vector<float3> *normals, int n){
    this->n = n;
    // check if these are used later or can be discarded after this function
    this->modelPoints = new thrust::device_vector<float3>(*points);
    this->modelNormals = new thrust::device_vector<float3>(*normals);
    this->modelPPFs = new thrust::device_vector<float4>(n*n);

    #ifdef DEBUG
        fprintf(stderr, "n: %d\n", n);

        // start cuda timer
        cudaEvent_t start, stop;
        HANDLE_ERROR(cudaEventCreate(&start));
        HANDLE_ERROR(cudaEventCreate(&stop));
        HANDLE_ERROR(cudaEventRecord(start, 0));
    #endif

    int blocks = std::min(((int)(this->n + BLOCK_SIZE) - 1) / BLOCK_SIZE, MAX_NBLOCKS);
    // MATLAB drost.m:59, all of model_description.m
    // ppf_kernel computes ppfs and descritizes them, but does *not* hash them
    // hashing is done by ppf_hash_kernel, called only for model, not scene (model.cu:46)
    ppf_kernel<<<blocks,BLOCK_SIZE>>>(RAW_PTR(this->modelPoints),
                                      RAW_PTR(this->modelNormals),
                                      RAW_PTR(this->modelPPFs),
                                      n);

    #ifdef DEBUG
        // end cuda timer
        HANDLE_ERROR(cudaEventRecord(stop, 0));
        HANDLE_ERROR(cudaEventSynchronize(stop));
        float elapsedTime;
        HANDLE_ERROR(cudaEventElapsedTime(&elapsedTime, start, stop));
        fprintf(stderr, "Time to generate PPFs: %3.1f ms\n", elapsedTime);
    #endif
}

int Scene::numPoints(){
    return this->n;
}

thrust::device_vector<float3> *Scene::getModelPoints(){
    return this->modelPoints;
}

thrust::device_vector<float3> *Scene::getModelNormals(){
    return this->modelNormals;
}
thrust::device_vector<float4> *Scene::getModelPPFs(){
    return this->modelPPFs;
}

thrust::device_vector<unsigned int>* Scene::getHashKeys(){
    return this->hashKeys;
}

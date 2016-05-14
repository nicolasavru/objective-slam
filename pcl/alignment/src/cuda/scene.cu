#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>                // Stops underlining of __global__
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
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

Scene::Scene(pcl::PointCloud<pcl::PointNormal> *cloud_ptr){
    /* DEBUG */
    fprintf(stderr, "foo0: %d\n", cloud_ptr->size());
    /* DEBUG */
    this->cloud_ptr = cloud_ptr;
    thrust::host_vector<float3> *scene_points =
        new thrust::host_vector<float3>(cloud_ptr->size());
    thrust::host_vector<float3> *scene_normals =
        new thrust::host_vector<float3>(cloud_ptr->size());

    for (int i = 0; i < cloud_ptr->size(); i++){
        (*scene_points)[i].x = (*cloud_ptr)[i].x;
        (*scene_points)[i].y = (*cloud_ptr)[i].y;
        (*scene_points)[i].z = (*cloud_ptr)[i].z;
        (*scene_normals)[i].x = (*cloud_ptr)[i].normal_x;
        (*scene_normals)[i].y = (*cloud_ptr)[i].normal_y;
        (*scene_normals)[i].z = (*cloud_ptr)[i].normal_z;
    }

    // this->initPPFs(points, normals, n);
    this->initPPFs(scene_points, scene_normals, cloud_ptr->size());
    HANDLE_ERROR(cudaGetLastError());
    HANDLE_ERROR(cudaDeviceSynchronize());
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
    // delete this->hashKeys;
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

    // This will crash if n = 0;
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

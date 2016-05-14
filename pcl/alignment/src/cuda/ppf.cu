#include <iostream>
#include <ctime>
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>                // Stops underlining of __global__
#include <sys/types.h>
#include <sys/stat.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include "kernel.h"
#include "book.h"
#include "model.h"

using namespace std;

#include <thrust/sort.h>
#include <thrust/device_ptr.h>
#include <thrust/inner_product.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/scan.h>

#include <Eigen/Core>

void test_histogram(char *point_path, int N){
    FILE *points_fin;
    size_t result1;

    points_fin = fopen(point_path, "rb");
    if(points_fin==NULL){fputs ("File error: point_fin",stderr); exit (1);}

    thrust::host_vector<unsigned int> *points = new thrust::host_vector<unsigned int>(N);

    if (points == NULL) {fputs ("Memory error: points",stderr); exit (2);}

    result1 = fread(RAW_PTR(points),sizeof(unsigned int),N,points_fin);
    thrust::device_vector<unsigned int> *d_points = new thrust::device_vector<unsigned int>(*points);

    thrust::sort(d_points->begin(), d_points->end());

    // create histogram of hash keys
    // https://code.google.com/p/thrust/source/browse/examples/histogram.cu
    unsigned int num_bins = thrust::inner_product(d_points->begin(), d_points->end() - 1,
                                                  d_points->begin() + 1,
                                                  (unsigned int) 1,
                                                  thrust::plus<unsigned int>(),
                                                  thrust::not_equal_to<unsigned int>());

    /* DEBUG */
    fprintf(stderr, "num_bins: %d\n", num_bins);
    /* DEBUG */

    thrust::device_vector<unsigned int> *hashKeys = new thrust::device_vector<unsigned int>(num_bins);
    thrust::device_vector<unsigned int> *ppfCount = new thrust::device_vector<unsigned int>(num_bins);

    thrust::reduce_by_key(d_points->begin(), d_points->end(),
                          thrust::constant_iterator<unsigned int>(1),
                          hashKeys->begin(),
                          ppfCount->begin());


    thrust::host_vector<unsigned int> *A = new thrust::host_vector<unsigned int>(*ppfCount);

    for (int i=0; i<num_bins; i++){
        fprintf(stderr, "%u: %u %u\n", i, (*A)[i]);
    }
}

void ptr_test_cu(pcl::PointCloud<pcl::PointNormal> *scene_cloud_ptr){
    /* DEBUG */
    fprintf(stderr, "foo-1: %p, %d, %d\n", scene_cloud_ptr, scene_cloud_ptr->points.size(), scene_cloud_ptr->size());
}

void ptr_test_cu2(pcl::PointCloud<pcl::PointNormal> scene_cloud){
    /* DEBUG */
    fprintf(stderr, "foo-2: %d, %d\n", scene_cloud.points.size(), scene_cloud.size());
}

void ptr_test_cu3(pcl::PointCloud<pcl::PointNormal> &scene_cloud){
    /* DEBUG */
    fprintf(stderr, "foo-3: %d, %d\n", scene_cloud.points.size(), scene_cloud.size());
}

void ptr_test_cu4(const pcl::PointCloud<pcl::PointNormal> &scene_cloud){
    /* DEBUG */
    fprintf(stderr, "foo-4: %d, %d\n", scene_cloud.points.size(), scene_cloud.size());
}


Eigen::Matrix4f ply_load_main(pcl::PointCloud<pcl::PointNormal> *scene_cloud_ptr,
                              float3 *scenePoints, float3 *sceneNormals, int sceneN,
                              pcl::PointCloud<pcl::PointNormal> *object_cloud_ptr,
                              float3 *objectPoints, float3 *objectNormals, int objectN,
                              int devUse){
    /* DEBUG */
    fprintf(stderr, "foo1: %p, %d, %d\n", scene_cloud_ptr, scene_cloud_ptr->points.size(), scene_cloud_ptr->size());
    /* DEBUG */
    int *device_array = 0;
    HANDLE_ERROR(cudaMalloc((void**)&device_array, 1024*sizeof(int)));

    int numDevices;
    HANDLE_ERROR(cudaGetDeviceCount(&numDevices));
    fprintf(stderr, "numDevices: %d\n", numDevices);
    cudaDeviceProp prop;
    for(int i = 0; i < numDevices; i++){
        cudaGetDeviceProperties(&prop, i);
        fprintf(stderr, "%d) name: %s\n", i, prop.name);
    }
    // HANDLE_ERROR(cudaSetDevice(devUse));
    HANDLE_ERROR(cudaSetDevice(numDevices > 1 ? devUse : 0));
    int devNum;
    HANDLE_ERROR(cudaGetDevice(&devNum));
    HANDLE_ERROR(cudaGetDeviceProperties(&prop, devNum));
    fprintf(stderr, "Using device %d, %s: \n", devNum, prop.name);
    // thrust::device_vector<float3> foo(1024);

    // convert float3 * to thrust::host_vector<float3>
    thrust::host_vector<float3> *scenePointsVec =
            new thrust::host_vector<float3>(scenePoints, scenePoints+sceneN);
    thrust::host_vector<float3> *sceneNormsVec =
            new thrust::host_vector<float3>(sceneNormals, sceneNormals+sceneN);
    thrust::host_vector<float3> *objectPointsVec =
            new thrust::host_vector<float3>(objectPoints, objectPoints+objectN);
    thrust::host_vector<float3> *objectNormsVec =
            new thrust::host_vector<float3>(objectNormals, objectNormals+objectN);

    free(scenePoints);
    free(sceneNormals);
    free(objectPoints);
    free(objectNormals);

    // cuda setup
    int blocks = prop.multiProcessorCount;
    /* DEBUG */
    fprintf(stderr, "blocks_multiproccount: %d\n", blocks);
    /* DEBUG */

    // build model description
    Model *model = new Model(object_cloud_ptr, objectPointsVec, objectNormsVec, objectN);

    /* DEBUG */
    fprintf(stderr, "foo0: %d\n", scene_cloud_ptr->points.size());
    /* DEBUG */
    Scene *scene = new Scene(scene_cloud_ptr);

    model->ppf_lookup(scene);

    // copy ppfs back to host
    thrust::host_vector<float> transformations = thrust::host_vector<float>(model->getTransformations());
    // thrust::host_vector<unsigned int> *maxval = new thrust::host_vector<unsigned int>(*model->maxval);
    thrust::host_vector<float> *maxval =
        new thrust::host_vector<float>(*model->vote_counts_out);

    // write out transformations
    // (*maxval)[0] is all the unallocated votes
    float threshold = 0.8 * (*maxval)[1];
    for (int i=1; (*maxval)[i] > threshold; i++){
       cout << "num_votes: " << (*maxval)[i] << endl;
       cout << "transforms(:,:," << i << ") = [";
       for (int j=0; j<4; j++){
           for (int k=0; k<4; k++){
               cout << transformations[i*16+j*4+k] << " ";
           }
           cout << ";" << endl;
       }
       cout << "];" << endl;
       cout << endl << endl;
    }

    Eigen::Matrix4f T;
    for (int j=0; j<4; j++){
        for (int k=0; k<4; k++){
            T(j,k) = transformations[16+j*4+k];
        }
    }


    // // Deallocate ram
    // delete points;
    // delete norms;
    // delete ppfs;

    delete model;
    delete scene;

    cudaDeviceReset();

    return T;
}

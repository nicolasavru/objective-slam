#include <iostream>
#include <ctime>
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>                // Stops underlining of __global__
#include <device_launch_parameters.h>    // Stops underlining of threadIdx etc.
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

int ply_load_main(char *point_path, char *norm_path, int N, int devUse){
    // test_histogram("/tmp/hist_test.bin", 10000);
    // return 0;

    // file input
    FILE *points_fin, *norms_fin;
    size_t result1, result2;

    int numDevices;
    cudaGetDeviceCount(&numDevices);
    fprintf(stderr, "numDevices: %d\n", numDevices);
    cudaDeviceProp prop;
    for(int i = 0; i < numDevices; i++){
        cudaGetDeviceProperties(&prop, i);
        fprintf(stderr, "%d) name: %s\n", i, prop.name);
    }
    cudaSetDevice(devUse);
    int devNum;
    cudaGetDevice(&devNum);
    HANDLE_ERROR(cudaGetDeviceProperties(&prop, devNum));
    fprintf(stderr, "Using device %d, %s: \n", devNum, prop.name);

    points_fin = fopen(point_path, "rb");
    norms_fin  = fopen(norm_path, "rb");
    if(points_fin==NULL){fputs ("File error: point_fin",stderr); exit (1);}
    if(norms_fin==NULL){fputs ("File error: norms_fin",stderr); exit (1);}

    thrust::host_vector<float3> *points = new thrust::host_vector<float3>(N);
    thrust::host_vector<float3> *norms = new thrust::host_vector<float3>(N);

    if (points == NULL) {fputs ("Memory error: points",stderr); exit (2);}
    if (norms  == NULL) {fputs ("Memory error: norms",stderr); exit (2);}

    long startTime0 = clock();
    result1 = fread(RAW_PTR(points),sizeof(float3),N,points_fin);
    result2 = fread(RAW_PTR(norms),sizeof(float3),N,norms_fin);
    long finishTime0 = clock();

    cerr<<"Data Load Time"<<" "<<(finishTime0 - startTime0)<<" ms"<<endl;

    if(result1 != N){fputs ("Reading error: points",stderr); exit(3);}
    if(result2 != N){fputs ("Reading error: norms",stderr); exit(3);}

    // cuda setup
    int blocks = prop.multiProcessorCount;
    /* DEBUG */
    fprintf(stderr, "blocks: %d\n", blocks);
    /* DEBUG */

    // build model description
    Model *model = new Model(points, norms, N);

    // model->ppf_lookup();

    // copy ppfs back to host
    thrust::host_vector<float4> *ppfs = new thrust::host_vector<float4>(*model->getModelPPFs());

    // write out ppfs
    for(int i = 0; i < 100; i++){
        cout << "PPF Number: " << i << endl;
        cout << (*ppfs)[i].x << endl;
        cout << (*ppfs)[i].y << endl;
        cout << (*ppfs)[i].z << endl;
        cout << (*ppfs)[i].w << endl;
    }

    // Deallocate ram
    delete points;
    delete norms;
    delete ppfs;

    delete model;

    cudaDeviceReset();

    // close input file
    fclose(points_fin);
    fclose(norms_fin);
    return 0;
}

// int ppf_run(Eigen::MatrixXf &points, Eigen::MatrixXf &normals){
//     float *point_data = points.data();
//     float *normal_data = normals.data();
//     int size = points.rows()*points.cols()*sizeof(float);

//     float *dev_point_data, *dev_normal_data;
//     HANDLE_ERROR(cudaMalloc((void **) &dev_point_data, size));
//     HANDLE_ERROR(cudaMemcpy(dev_point_data, point_data. size));

//     HANDLE_ERROR(cudaMalloc((void **) &dev_normal_data, size));
//     HANDLE_ERROR(cudaMemcpy(dev_normal_data, point_data. size));

//     cudaDeviceProp prop;
//     HANDLE_ERROR(cudaGetDeviceProperties(&prop, 0));
//     int blocks = prop.multiProcessorCount;

//     ppf_kernel<<<blocks*2,256>>>(dev_buffer, SIZE, dev_histo);

// }

#include <iostream>
#include <ctime>
#include <fstream>
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>                // Stops underlining of __global__
#include <device_launch_parameters.h>    // Stops underlining of threadIdx etc.
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>

#include "book.h"

using namespace std;

__device__ __forceinline__ float dot(float3 v1, float3 v2){
    return v1.x*v2.x + v1.y*v2.y + v1.z*v2.z;
}

__device__ __forceinline__ float norm(float3 v){
    return sqrtf(dot(v, v));
}

__device__ float4 compute_ppf(float3 p1, float3 p2, float3 n1, float3 n2){
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

__global__ void ppf_wrapper(float3 *points, float3 *norms, float4 *out, int count){
    if(count <= 1) return;

    int ind = threadIdx.x;
    int idx = ind + blockIdx.x * blockDim.x;

    if(idx < count) {
        float3 thisPoint = points[idx];
        float3 thisNorm = norms[idx];

        for(int j = 0; j < count; j++) {
            // if(j == idx) continue;
            out[ind*count + j] = compute_ppf(thisPoint, points[j], thisNorm, norms[j]);
        }

        // for(int i = 0; i < count; i+= BLOCK_SIZE){
        //     Spoints[ind] = points[i+ind];
        //     __syncthreads();

        //     for(int j = 0; j < BLOCK_SIZE; j++) {
        //         if(i+j == idx) continue;

        //         out[ind*BLOCK_SIZE + j] = compute_ppf(thisPoint, Spoints[j], thisNorm, Snorms[j]);
        //     }
        // }
    }
}


float input(int fd)    // basic input structure
{
    float x;
    read(fd, &x, sizeof(float));
    return x;
}

int ply_load_main(char *point_path, char *norm_path, int N){
    int points_fin = open(point_path, O_RDONLY);    // read in points
    int norms_fin = open(norm_path, O_RDONLY);    // read in norms
    // Array of points
    float3 *points = new float3[N];
    float3 *norms = new float3[N];

    // read in data and time
    long startTime = clock();
    for(int i = 0; i < N; i++)    // loop over rows
    {    
        points[i].x = input(points_fin);    // read in an input entry
        points[i].y = input(points_fin);
        points[i].z = input(points_fin);

        norms[i].x = input(norms_fin);    // read in an input entry
        norms[i].y = input(norms_fin);
        norms[i].z = input(norms_fin);

    }
    long finishTime = clock();

    cudaDeviceProp  prop;
    HANDLE_ERROR(cudaGetDeviceProperties(&prop, 0));
    int blocks = prop.multiProcessorCount;

    cudaEvent_t start, stop;
    HANDLE_ERROR(cudaEventCreate(&start));
    HANDLE_ERROR(cudaEventCreate(&stop));
    HANDLE_ERROR(cudaEventRecord(start, 0));

    float3 *d_points; // GPU version
    float3 *d_norms; // GPU version
    float4 *ppfs = new float4[N*N];
    float4 *d_ppfs; // GPU version

    HANDLE_ERROR(cudaMalloc(&d_points, N*sizeof(float3)));
    HANDLE_ERROR(cudaMemcpy(d_points, points, N*sizeof(float3), cudaMemcpyHostToDevice));

    HANDLE_ERROR(cudaMalloc(&d_norms, N*sizeof(float3)));
    HANDLE_ERROR(cudaMemcpy(d_norms, norms, N*sizeof(float3), cudaMemcpyHostToDevice));

    HANDLE_ERROR(cudaMalloc(&d_ppfs, N*N*sizeof(float4)));

    // call ppf kernel
    ppf_wrapper<<<1,1024>>>(d_points, d_norms, d_ppfs, N);
    HANDLE_ERROR(cudaMemcpy(ppfs, d_ppfs, N*N*sizeof(float4), cudaMemcpyDeviceToHost));

    // end timer
    HANDLE_ERROR(cudaEventRecord(stop, 0));
    HANDLE_ERROR(cudaEventSynchronize(stop));
    float elapsedTime;
    HANDLE_ERROR(cudaEventElapsedTime(&elapsedTime, start, stop));
    printf("Time to generate:  %3.1f ms\n", elapsedTime);

    // write out ppfs
    for(int i = 0; i < 10; i++)    // loop over rows
    {
        cout << "PPF Number: " << i << endl;
        cout << ppfs[i].x << endl;
        cout << ppfs[i].y << endl;
        cout << ppfs[i].z << endl;
        cout << ppfs[i].w << endl;
    }

    cout<<"Data Load Time"<<" "<<(finishTime - startTime)<<" ms"<<endl;

    // Deallocate ram
    delete[] points;
    delete[] norms;
    delete[] ppfs;

    cudaFree(d_points);
    cudaFree(d_norms);
    cudaFree(d_ppfs);

    cudaDeviceReset();

    // close input file
    close(points_fin);
    close(norms_fin);
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

#include <iostream>
#include <ctime>
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>                // Stops underlining of __global__
#include <device_launch_parameters.h>    // Stops underlining of threadIdx etc.
#include <sys/types.h>
#include <sys/stat.h>

#include "book.h"

#define BLOCK_SIZE 256

using namespace std;

__device__ __forceinline__ float dot(float3 v1, float3 v2){
    return (v1.x*v2.x + v1.y*v2.y + v1.z*v2.z);
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

        __shared__ float3 Spoints[BLOCK_SIZE];
        __shared__ float3 Snorms[BLOCK_SIZE];

        float3 thisPoint = points[idx];
        float3 thisNorm  = norms[idx];

        for(int i = 0; i < count; i+= BLOCK_SIZE){

            Spoints[ind] = points[i+ind];
            Snorms[ind]  = norms[i+ind];
            __syncthreads();
            
            for(int j = 0; j < BLOCK_SIZE; j++) {
                if(i + j == ind*count) continue;
                out[ind*count + j] = compute_ppf(thisPoint, Spoints[j], thisNorm, Snorms[j]);
            }
        }
    }
}

int main(int argc, char* argv[]){

    // Parse inputs
    char *point_path = argv[1];
    char *norm_path  = argv[2];
    int N;
    sscanf_s(argv[3],"%d",&N);

    FILE *points_fin, *norms_fin;
    size_t result1, result2;

    points_fin = fopen(point_path, "rb");    // open points binary file
    norms_fin  = fopen(norm_path, "rb");    // open norms binary file
    if (points_fin==NULL) {fputs ("File error: point_fin",stderr); exit (1);}
    if (norms_fin==NULL) {fputs ("File error: norms_fin",stderr); exit (1);}

    // Array of points
    float3 *points = new float3[N];
    float3 *norms = new float3[N];
    if (points == NULL) {fputs ("Memory error: points",stderr); exit (2);}
    if (norms  == NULL) {fputs ("Memory error: norms",stderr); exit (2);}

    // read in data and time
    long startTime0 = clock();
    
    result1 = fread(points,sizeof(float3),N,points_fin);
    result2 = fread(norms,sizeof(float3),N,norms_fin);

    long finishTime0 = clock();

    if (result1 != N) {fputs ("Reading error: points",stderr); exit (3);}
    if (result2 != N) {fputs ("Reading error: norms",stderr); exit (3);}

    cudaDeviceProp  prop;
    HANDLE_ERROR(cudaGetDeviceProperties(&prop, 0));
    int blocks = prop.multiProcessorCount;
    /* DEBUG */
    fprintf(stderr, "blocks: %d\n", blocks);
    /* DEBUG */

    float4 *ppfs = new float4[N*N];
    if (ppfs == NULL) {fputs ("Memory error: ppfs",stderr); exit (2);}
    
    float3 *d_points; // GPU version
    float3 *d_norms; // GPU version
    float4 *d_ppfs; // GPU version

    //START: time cuda memory allocation and transfer to device
    cudaEvent_t start0, stop0;
    HANDLE_ERROR(cudaEventCreate(&start0));
    HANDLE_ERROR(cudaEventCreate(&stop0));
    HANDLE_ERROR(cudaEventRecord(start0, 0));

    HANDLE_ERROR(cudaMalloc(&d_points, N*sizeof(float3)));
    HANDLE_ERROR(cudaMalloc(&d_norms, N*sizeof(float3)));
    HANDLE_ERROR(cudaMalloc(&d_ppfs, N*N*sizeof(float4)));

    HANDLE_ERROR(cudaMemcpy(d_points, points, N*sizeof(float3), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(d_norms, norms, N*sizeof(float3), cudaMemcpyHostToDevice));
    
    // END: cuda processing timer
    HANDLE_ERROR(cudaEventRecord(stop0, 0));
    HANDLE_ERROR(cudaEventSynchronize(stop0));
    float elapsedTime0;
    HANDLE_ERROR(cudaEventElapsedTime(&elapsedTime0, start0, stop0));

    //START: time cuda processing
    cudaEvent_t start1, stop1;
    HANDLE_ERROR(cudaEventCreate(&start1));
    HANDLE_ERROR(cudaEventCreate(&stop1));
    HANDLE_ERROR(cudaEventRecord(start1, 0));

    // call ppf kernel
    ppf_wrapper<<<N/BLOCK_SIZE,BLOCK_SIZE>>>(d_points, d_norms, d_ppfs, N);
    
    // END: cuda processing timer
    HANDLE_ERROR(cudaEventRecord(stop1, 0));
    HANDLE_ERROR(cudaEventSynchronize(stop1));
    float elapsedTime1;
    HANDLE_ERROR(cudaEventElapsedTime(&elapsedTime1, start1, stop1));

    //START: time cuda transfer to host
    cudaEvent_t start2, stop2;
    HANDLE_ERROR(cudaEventCreate(&start2));
    HANDLE_ERROR(cudaEventCreate(&stop2));
    HANDLE_ERROR(cudaEventRecord(start2, 0));

    HANDLE_ERROR(cudaMemcpy(ppfs, d_ppfs, N*N * sizeof(float4), cudaMemcpyDeviceToHost));

    // END: time cuda transfer to host
    HANDLE_ERROR(cudaEventRecord(stop2, 0));
    HANDLE_ERROR(cudaEventSynchronize(stop2));
    float elapsedTime2;
    HANDLE_ERROR(cudaEventElapsedTime(&elapsedTime2, start2, stop2));

    // write out ppfs
    for(int i = 0; i < 10; i++)    // loop over rows
    {
        cout << "PPF Number: " << i << endl;
        cout << ppfs[i].x << endl;
        cout << ppfs[i].y << endl;
        cout << ppfs[i].z << endl;
        cout << ppfs[i].w << endl;
    }

    cout<<"Data Read In Time"<<" "<<(finishTime0 - startTime0)<<" ms"<<endl;
    printf("Time to Transfer to Device:  %3.1f ms\n", elapsedTime0);
    printf("Time to Process on Device:  %3.1f ms\n", elapsedTime1);
    printf("Time to Transfer to Host:  %3.1f ms\n", elapsedTime2);

    // clean up
    HANDLE_ERROR( cudaEventDestroy( start0 ));
    HANDLE_ERROR( cudaEventDestroy( stop0 ));

    HANDLE_ERROR( cudaEventDestroy( start1 ));
    HANDLE_ERROR( cudaEventDestroy( stop1 ));

    HANDLE_ERROR( cudaEventDestroy( start2 ));
    HANDLE_ERROR( cudaEventDestroy( stop2 ));

    // Deallocate ram
    delete[] points;
    delete[] norms;
    delete[] ppfs;

    cudaFree(d_points);
    cudaFree(d_norms);
    cudaFree(d_ppfs);

    cudaDeviceReset();

    // close input file
    fclose(points_fin);
    fclose(norms_fin);
    return 0;
}

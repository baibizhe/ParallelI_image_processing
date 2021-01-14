/* ------------
 * This code is provided solely for the personal and private use of
 * students taking the CSC367H5 course at the University of Toronto.
 * Copying for purposes other than this use is expressly prohibited.
 * All forms of distribution of this code, whether as given or with
 * any changes, are expressly prohibited.
 *
 * Authors: Bogdan Simion, Felipe de Azevedo Piovezan
 *
 * All of the files in this directory and all subdirectories are:
 * Copyright (c) 2019 Bogdan Simion
 * -------------
 */

 #include "kernels.h"
 #include <stdio.h>

 #define THREAD_NUM 512
 void run_kernel2(const int8_t *filter, int32_t dimension, const int32_t *input,
                  int32_t *output, int32_t width, int32_t height, float *tin, float *tcompute, float *tout) {

   int32_t N = width * height;
   
   //set timer
   cudaEvent_t start, stop;
   cudaEventCreate(&start);
   cudaEventCreate(&stop);
 
   //allocate memory on device
   size_t SIZE = width * height * sizeof(int32_t);
   int32_t *input_device, *output_device;
   int8_t *filter_d;
   cudaEventRecord(start);
   cudaMalloc((void**)&input_device, SIZE);
   cudaMalloc((void**)&output_device, SIZE);
   cudaMalloc((void**)&filter_d, dimension*dimension*sizeof(int8_t));
   cudaMemcpy(input_device, input, SIZE, cudaMemcpyHostToDevice);
   cudaMemcpy(output_device, output, SIZE, cudaMemcpyHostToDevice);
   cudaMemcpy(filter_d, filter, dimension*dimension*sizeof(int8_t), cudaMemcpyHostToDevice);
   cudaMemcpy(input_device, input, SIZE, cudaMemcpyHostToDevice);
   cudaEventRecord(stop);
   cudaEventSynchronize(stop);
   cudaEventElapsedTime(tin, start, stop);
 //record transfer in
 
   int numBlocks = (N+ THREAD_NUM - 1)/THREAD_NUM;
 
   int32_t *maximum, *minimum;
   cudaMalloc((void**)&maximum, sizeof(int32_t));
   cudaMalloc((void**)&minimum, sizeof(int32_t));
 
   kernel2 <<< numBlocks, THREAD_NUM, 2*THREAD_NUM * sizeof(int32_t)>>>(filter_d, dimension, input_device, output_device, width, height, maximum, minimum);
   normalize2<<< numBlocks, THREAD_NUM>>>(output_device, width, height, minimum, maximum);
   cudaEventRecord(stop);
   cudaEventSynchronize(stop);
   cudaEventElapsedTime(tcompute, start, stop);
   //computation time

   cudaMemcpy(output, output_device, SIZE, cudaMemcpyDeviceToHost);
   cudaEventRecord(stop);
   cudaEventSynchronize(stop);
   cudaEventElapsedTime(tout, start, stop);
    //transfer out
   cudaEventDestroy(start);
   cudaEventDestroy(stop);
   
   //free memory;
   cudaFree(maximum);
   cudaFree(minimum);
   cudaFree(input_device);
   cudaFree(output_device);
   cudaFree(filter_d);
 }
 
 __global__ void kernel2(const int8_t *filter, int32_t dimension,
                         const int32_t *input, int32_t *output, int32_t width,
                         int32_t height, int32_t *maximum, int32_t *minimum) {
                         
    extern __shared__ int32_t smem[];
    int32_t *smax = smem;
    int32_t *smin = smem + THREAD_NUM;   
    unsigned int tid = threadIdx.x;
    unsigned idx = blockIdx.x * blockDim.x+ threadIdx.x;
    int32_t x , y , offset ,curi ,curj ,ans;

    if(idx < width * height){
        y = (int32_t)idx /  width;
        x = (int32_t)idx % width ;
        //y = row, x = column
        offset = (int32_t) dimension/2;
        ans = 0;
        for(int i = 0 ; i<dimension;i++){
            for(int j=0; j<dimension; j++){   // innner loop is j  , j will update before i
                curj = y + i - offset;
                curi = x + j - offset;   // curi is update 4 times when curj update 1 time
                if(0 <= curi && curj < height && 0<=curj &&curi < width){
                    ans += filter[dimension * i + j] * input[width * curj + curi];
                    //width * curj + curi will loop like  32*0+09,32*0+1 ,32*0+2..... So it is row major
                }
            }
            output[idx] = ans;
            smax[tid] = output[idx];
            smin[tid] = output[idx];
        }
    }
    __syncthreads(); 

//reduction

    for (unsigned int s = 1; s < blockDim.x; s *= 2) {
      if (tid % (2*s) == 0 && idx + s < width * height)  { // In a warp, only thread ids divisible by the step participate
          smax[tid] = max(smax[tid], smax[tid+s]);
          smin[tid] = min(smin[tid+s], smin[tid]);
      }
      __syncthreads();

    }

    if(tid == 0){
        atomicMax(maximum, (int)smax[0]);
        atomicMin(minimum, (int)smin[0]);
    }
    __syncthreads();
 }
 
 __global__ void normalize2(int32_t *image, int32_t width, int32_t height,
                            int32_t *smallest, int32_t *biggest) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    int32_t mmax = *biggest;
    int32_t mmin = *smallest;
    if(mmax == mmin) return;
    if(i < width * height){
        image[i] =((image[i] - mmin) * 255) / (mmax - mmin);
    }
 
 }
 

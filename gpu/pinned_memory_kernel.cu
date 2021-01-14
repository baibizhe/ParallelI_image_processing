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
  #include <sys/time.h>
  __constant__ int8_t c_filter[9]; 
// filter  constant
  void run_kernel5(const int8_t *filter, int32_t dimension,  int32_t *input,
                   int32_t *output, int32_t width, int32_t height, float *tin, float *tcompute, float *tout) {
    // Figure out how to split the work into threads and call the kernel below.
    int32_t N = width * height;
  
    //set timer
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
  
    //allocate memory on device
    size_t SIZE = N * sizeof(int32_t);
    int32_t *output_device;  
    int numBlocks = N/4/512 + 1;
    //4 pixels per thread
  
    int32_t *maximum, *minimum;
    cudaEventRecord(start);
    cudaHostRegister((void*)input,SIZE,cudaHostRegisterDefault);
    //set input to pinned memory .Faster to transfer
    cudaMemcpyToSymbol(c_filter, filter, 9 * sizeof(int8_t));
    cudaMalloc((void**)&output_device, SIZE);
    cudaMemcpy(output_device, output, SIZE, cudaMemcpyHostToDevice);
  
  
    cudaMalloc((void**)&maximum, sizeof(int32_t));
    cudaMalloc((void**)&minimum, sizeof(int32_t));
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(tin, start, stop);


    kernel5 <<< numBlocks, 512, 2*512 * sizeof(int32_t)>>>( dimension, input, output_device, width, height, maximum, minimum);
    normalize5<<< numBlocks, 512>>>(output_device, width, height, minimum, maximum);
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
    cudaHostUnregister(input);

    cudaFree(maximum);
    cudaFree(minimum);
    cudaFree(output_device);
  }
  
  __global__ void kernel5( int32_t dimension,
                          int32_t *input, int32_t *output, int32_t width,
                          int32_t height, int32_t *maximum, int32_t *minimum) {
    //  printf("kernel5 input: %d, %d, %d\n", input[1], input[11], input[11]);
 
    extern __shared__ int32_t smem[];
    int32_t N = height * width;
    int32_t *smax = smem;
    int32_t *smin = smem + 512;
    unsigned int tid = threadIdx.x;
    unsigned int i = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    //4 per thread, consecutive row
    int32_t mymax, mymin;
    int h = dimension/2;
    for(int32_t loc = i; loc < i+4 && loc < N; loc++){
      int32_t col = loc%width;
      int32_t row = loc/width;
      int32_t ans = 0;
      int32_t nrow, ncol;
      for(int m = 0; m < dimension; m++){
          for(int j = 0; j < dimension; j++){
              ncol = col + m - h;
              nrow = row + j - h;
              if(0 <= ncol && ncol < width && 0<=nrow &&nrow < height){
                  ans += c_filter[dimension * m + j] * input[width * nrow + ncol];
              }
          }
      }
      if(loc == i){ mymax = mymin = ans; }
      mymax = max(mymax, ans);
      mymin = min(mymin, ans);
      output[loc] = ans;
    }
    if(i < N){
      smax[tid] = mymax;
      smin[tid] = mymin;
      }
    __syncthreads();  
    for (unsigned int s = 1; s < blockDim.x; s *= 2) {
      if (tid % (2*s) == 0 && i + s < width * height)  { // In a warp, only thread ids divisible by the step participate
          smax[tid] = max(smax[tid], smax[tid+s]);
          smin[tid] = min(smin[tid+s], smin[tid]);
      }
      __syncthreads();
    }  
    if(tid == 0){  
 
          atomicMax(maximum, (int)smax[0]);
          atomicMin(minimum, (int)smin[0]);
      } 
  }
  
  __global__ void normalize5(int32_t *image, int32_t width, int32_t height,
                             int32_t *smallest, int32_t *biggest) {
                             
      unsigned int i = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
      int32_t mmax = *biggest;
      int32_t mmin = *smallest;
      int32_t N = width * height;
      if(mmax == mmin) return;
      for(int32_t loc = i; loc < i+4 && loc < N; loc ++){
          image[loc] =((image[loc] - mmin) * 255) / (mmax - mmin);
      }
      
  }
  

/* ------------
Multiple pixels per thread, consecutive rows, row major
 */

#include "kernels.h"
#include <stdio.h>
void row_major_Multiple_pixels(const int8_t *filter, int32_t dimension, const int32_t *input,
                 int32_t *output, int32_t width, int32_t height, float *tin, float *tcompute, float *tout) {
  int32_t N = width * height;
  
  //set timer
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  //allocate memory on device
  size_t SIZE = N * sizeof(int32_t);
  int32_t *input_device, *output_device;
  int8_t *filter_d;
  cudaEventRecord(start);
  cudaMalloc((void**)&input_device, SIZE);
  cudaMalloc((void**)&output_device, SIZE);
  cudaMalloc((void**)&filter_d, dimension*dimension*sizeof(int8_t));
  cudaMemcpy(input_device, input, SIZE, cudaMemcpyHostToDevice);
  cudaMemcpy(output_device, output, SIZE, cudaMemcpyHostToDevice);
  cudaMemcpy(filter_d, filter, dimension*dimension*sizeof(int8_t), cudaMemcpyHostToDevice);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(tin, start, stop);
//record transfer in

  int numBlocks = (N+2047)/2048;
  //4 pixels per thread

  int32_t max_h = -7000; int32_t min_h = 7000;
  int32_t *maximum, *minimum;
  cudaMalloc((void**)&maximum, sizeof(int32_t));
  cudaMalloc((void**)&minimum, sizeof(int32_t));
  cudaMemcpy(maximum, &max_h, sizeof(int32_t), cudaMemcpyHostToDevice);
  cudaMemcpy(minimum, &min_h, sizeof(int32_t), cudaMemcpyHostToDevice);

  kernel3 <<< numBlocks, 512, 2*512 * sizeof(int32_t)>>>(filter_d, dimension, input_device, output_device, width, height, maximum, minimum);

  normalize3<<< numBlocks, 512>>>(output_device, width, height, minimum, maximum);
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

__global__ void kernel3(const int8_t *filter, int32_t dimension,
                        const int32_t *input, int32_t *output, int32_t width,
                        int32_t height, int32_t *maximum, int32_t *minimum) {
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
                ans += filter[dimension * m + j] * input[width * nrow + ncol];
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
  
  //sync results from each block
    if(tid == 0){
        atomicMax(maximum, (int)smax[0]);
        atomicMin(minimum, (int)smin[0]);
    }
}

__global__ void normalize3(int32_t *image, int32_t width, int32_t height,
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

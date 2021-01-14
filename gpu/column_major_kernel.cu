/* ------------

 */

#include "kernels.h"
#include <stdio.h>

void run_kernel1(const int8_t *filter, int32_t dimension, const int32_t *input,
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
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(tin, start, stop);
//record transfer in

  int numBlocks = (N+511)/512;

  int32_t max_h = -7000; int32_t min_h = 7000;
  int32_t *maximum, *minimum;
  cudaMalloc((void**)&maximum, sizeof(int32_t));
  cudaMalloc((void**)&minimum, sizeof(int32_t));
  cudaMemcpy(maximum, &max_h, sizeof(int32_t), cudaMemcpyHostToDevice);
  cudaMemcpy(minimum, &min_h, sizeof(int32_t), cudaMemcpyHostToDevice);
  kernel1 <<< numBlocks, 512, 2*512 * sizeof(int32_t)>>>(filter_d, dimension, input_device, output_device, width, height, maximum, minimum);
  normalize1<<< numBlocks, 512>>>(output_device, width, height, minimum, maximum);
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

__global__ void kernel1(const int8_t *filter, int32_t dimension,
                        const int32_t *input, int32_t *output, int32_t width,
                        int32_t height, int32_t *maximum, int32_t *minimum) {
                        
    extern __shared__ int32_t smem[];
    int32_t *smax = smem;
    int32_t *smin = smem + 512;
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    int32_t col = i/height;
    int32_t row = i%height;
    int h = dimension/2;
    int32_t nrow, ncol;
    int32_t ans = 0;
    if(i < width * height){
        for(int m = 0; m < dimension; m++){
            for(int j = 0; j < dimension; j++){
                ncol = col + m - h;
                nrow = row + j - h;
                if(0 <= ncol && ncol < width && 0<=nrow &&nrow < height){
                    ans += filter[dimension * j + m] * input[width * nrow + ncol];
                }
            }
        }
        output[row * width + col] = ans;
        smax[tid] = output[row * width + col];
        smin[tid] = output[row * width + col];
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
    //update global max/min, once per thread
        atomicMax(maximum, (int)smax[0]);
        atomicMin(minimum, (int)smin[0]);
    }
}

__global__ void normalize1(int32_t *image, int32_t width, int32_t height,
                           int32_t *smallest, int32_t *biggest) {

    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    int32_t mmax = *biggest;
	int32_t mmin = *smallest;
    if(mmax == mmin) return;
    if(i < width * height){
        int32_t col = i/height;
        int32_t row = i%height;
        int32_t id = row * width + col;
        image[id] =((image[id] - mmin) * 255) / (mmax - mmin);
    }

}



#ifndef __KERNELS__H
#define __KERNELS__H


__global__ void column_major(const int8_t *filter, int32_t dimension,
                        const int32_t *input, int32_t *output, int32_t width,
                        int32_t height, int32_t *maximum, int32_t *minimum);

__global__ void row_major(const int8_t *filter, int32_t dimension,
                        const int32_t *input, int32_t *output, int32_t width,
                        int32_t height, int32_t *maximum, int32_t *minimum);


__global__ void row_major_Multiple_pixels(const int8_t *filter, int32_t dimension,
                        const int32_t *input, int32_t *output, int32_t width,
                        int32_t height, int32_t *maximum, int32_t *minimum);

__global__ void stride(const int8_t *filter, int32_t dimension,
                        const int32_t *input, int32_t *output, int32_t width,
                        int32_t height, int32_t *maximum, int32_t *minimum);


__global__ void pinned_memory( int32_t dimension,
                        int32_t *input, int32_t *output, int32_t width,
                        int32_t height, int32_t *maximum, int32_t *minimum);


#endif

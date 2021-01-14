

#include "kernels.h"
#include <pthread.h>
#include <stdlib.h>
#include <sched.h>
#include <unistd.h>
#include <stdio.h>

#define NUM_PTHREADS 10
pthread_mutex_t mutex;

void normalize_pixel(int32_t *target, int32_t pixel_idx, int32_t smallest,
                     int32_t largest) {
  if (smallest == largest) {
    return;
  }

  target[pixel_idx] =
      ((target[pixel_idx] - smallest) * 255) / (largest - smallest);
}

int32_t apply2d(const int8_t *f, int32_t dimension, const int32_t *original, int32_t *target, int32_t width, int32_t height, int row, int column) {
    
    int32_t location = row*width + column;
    int32_t ans = 0;
    int32_t ncol, nrow;
    int h = dimension/2;
    for(int i = 0; i < dimension; i++){
        for(int j = 0; j < dimension; j++){
            ncol = column + i - h;
            nrow = row + j - h;
            if(0 <= ncol && ncol < width && 0<=nrow &&nrow < height){
                ans += f[dimension * i + j] * original[width * nrow + ncol];
            }
        }
    }
    target[location] = ans;
    return ans;
}

typedef struct common_work_t
{
    const int8_t *f;
    int32_t dimension;
    const int32_t *original_image;
    int32_t *output_image;
    int32_t width;
    int32_t height;
    int32_t max_threads;
    pthread_barrier_t barrier;
    int32_t max;
    int32_t min;
} common_work;


typedef struct work_t
{
    common_work *common;
    int32_t id;
} work;

//use shared_row for cpu work
void *sharding_work(void *w) {
    work my_work = *((work *)w);
    common_work* c_w = my_work.common;
    int32_t max, min, cur;
    int32_t id, nthreads, row, col;
    int32_t total = c_w->height * c_w->width;
    nthreads = c_w->max_threads;
    id = my_work.id;
    int32_t num_of_rows, start_row;
        //SHARDED ROWS
        num_of_rows = (c_w->height + nthreads-1)/nthreads;
        start_row = num_of_rows * c_w->width;
        if(num_of_rows * id >= c_w->height){
            pthread_barrier_wait(&c_w->barrier);
            pthread_exit(0);
        }//do no work if already out of bound
        for(int32_t i =num_of_rows * id * c_w->width; i < num_of_rows * id * c_w->width+ start_row && i<total; i++){
            row = i/c_w->width;
            col = i%c_w->width;
            cur = apply2d(c_w->f, c_w->dimension, c_w->original_image, c_w->output_image, c_w->width, c_w->height, row, col);
            if(i == num_of_rows * id) {
                max = min = cur;//initializing max and min to an actual value
            }
            if(cur < min) min = cur;
            if(cur > max) max = cur;
        }
    //Common work
    pthread_mutex_lock(&mutex);
    if(c_w->max < max) c_w->max = max;
    if(c_w->min > min) c_w->min = min;
    pthread_mutex_unlock(&mutex);//get global max and min
    pthread_barrier_wait(&c_w->barrier);
    max = c_w->max;
    min = c_w->min;
    //Normalization
    
        for(int32_t i =num_of_rows * id * c_w->width; i < num_of_rows * id *c_w->width+ start_row && i<total; i++){
            normalize_pixel(c_w->output_image, i, min, max);
        }
        pthread_exit(0);
}

void run_best_cpu(const int8_t *filter, int32_t dimension, const int32_t *input,
                  int32_t *output, int32_t width, int32_t height) {
    common_work *shared = (common_work *)malloc(sizeof(common_work));
    shared->f = filter;
    shared->dimension = dimension;
    shared->original_image = input;
    shared->height=height;
    shared->width=width;
    shared->max_threads = NUM_PTHREADS;
    shared->output_image=output;
    shared->max = -46920;
    shared->min = 46920;
    pthread_barrier_init(&shared->barrier, NULL, NUM_PTHREADS);
    work works[NUM_PTHREADS];
    pthread_t pthreads[NUM_PTHREADS];
    
    //pthreads
    pthread_mutex_init(&mutex, NULL);
    for(int i = 0; i < NUM_PTHREADS; i++){
        works[i].common=shared;
        works[i].id=i;
        pthread_create(&pthreads[i], NULL, sharding_work, &works[i]);
    }
    for(int i = 0; i < NUM_PTHREADS; i++){
        pthread_join(pthreads[i],NULL);
    }

}

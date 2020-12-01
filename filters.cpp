
 #define _GNU_SOURCE

#include <stdio.h>
#include "filters.h"
#include <pthread.h>
#include <stdlib.h>
#include <sched.h>
#include <unistd.h>


/************** FILTER CONSTANTS*****************/
/* laplacian */

filter log_f = {9, log_m};

/* Identity */
int8_t identity_m[] = {1};
filter identity_f = {1, identity_m};

filter *builtin_filters[NUM_FILTERS] = {&lp3_f, &lp5_f, &log_f, &identity_f};

pthread_mutex_t mutex;//to sync max and min values;
struct threadpool* threadpool_init(int thread_num, int queue_max_num);
int threadpool_add_job(struct threadpool *pool, void* (*callback_function)(void *arg), void *arg);
void* threadpool_function(void* arg);
int threadpool_destroy(struct threadpool *pool);
/* Normalizes a pixel given the smallest and largest integer values
 * in the image */
void normalize_pixel(int32_t *target, int32_t pixel_idx, int32_t smallest,
                     int32_t largest) {
  if (smallest == largest) {
    return;
  }

  target[pixel_idx] =
      ((target[pixel_idx] - smallest) * 255) / (largest - smallest);
}

/*************** COMMON WORK ***********************/
/* Process a single pixel and returns the value of processed pixel
 * TODO: you don't have to implement/use this function, but this is a hint
 * on how to reuse your code.
 * */
int32_t apply2d(const filter *f, const int32_t *original, int32_t *target,
                int32_t width, int32_t height, int row, int column) {
	
	int32_t location = row*width + column;
	int32_t num = original[location];
	int32_t ans = 0;
	int32_t ncol, nrow;
	int h = f->dimension/2;
	for(int i = 0; i < f->dimension; i++){
		for(int j = 0; j < f->dimension; j++){
			ncol = column + i - h;
			nrow = row + j - h;
			if(0 <= ncol && ncol < width && 0<=nrow &&nrow < height){
				ans += f->matrix[f->dimension * i + j] * original[width * nrow + ncol];
			}
		}
	}
	target[location] = ans;
	return ans;
}

/*********SEQUENTIAL IMPLEMENTATIONS ***************/
/* TODO: your sequential implementation goes here.
 * IMPORTANT: you must test this thoroughly with lots of corner cases and
 * check against your own manual calculations on paper, to make sure that your
 * code produces the correct image. Correctness is CRUCIAL here, especially if
 * you re-use this code for filtering pieces of the image in your parallel
 * implementations!
 */
void apply_filter2d(const filter *f, const int32_t *original, int32_t *target,
                    int32_t width, int32_t height) {
	int row, col;
	int32_t min, max, cur;
//	printf("start filter!");
	for(int32_t n = 0; n<width*height; n++){
		row = n/width;
		col = n%width;
		cur = apply2d(f, original, target, width, height, row, col);
		if(n == 0) max = min = cur;
		if(cur < min) min = cur;
		if(cur > max) max = cur;
	}
	for(int32_t n = 0; n<width*height; n++){
		normalize_pixel(target, n, min, max);
	}
	return;


}

/****************** ROW/COLUMN SHARDING ************/
/* TODO: you don't have to implement this. It is just a suggestion for the
 * organization of the code.
 */
typedef struct common_work_t
{
	const filter *f;
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
	int type;
} work;

//----------------new struct---------------//
//----------------new struct---------------//

typedef struct chunk_coordinate_t
{
	int xstart;
	int xend;
	int ystart ;
	int yend ; //why dont use topleft[2] , botright[2]? cause I hate use array in struct.Feel safer to use primitive type.
}chunk_coordinate;
// int first last;
typedef struct chunk_job_t{ // this is the argument that we want pass in threadfunction (threadwork)
  chunk_coordinate  coord;
  common_work * common;
//int type; (0 for apply2d, 1 for normalize, 2 for done)
} chunk_job;

struct job
{
    void* (*callback_function)(void *arg);    
	void *arg;    //the args that we need to pass to thread function              
    struct job *next;//the next job in link list 
};

struct threadpool
{
    int thread_num;        //maximum theads in pool            
    int queue_max_num;        //maximum waiting queue    
	int cpuid;       // indicate cpu id 
    struct job *head;                 
    struct job *tail;                 
    pthread_t *pthreads;               // the list to store thread ID 
    pthread_mutex_t mutex;            
    pthread_cond_t queue_empty;       // the condition variable that the queue is empty
    pthread_cond_t queue_not_empty;   
    pthread_cond_t queue_not_full;    
    int queue_cur_num;       // the current jobs in queue         
    int queue_close;                 
    int pool_close;                 
};
//----------------new struct end---------------//
//----------------new struct end---------------//

struct threadpool* threadpool_init(int thread_num, int queue_max_num)// initial the threads and struct thread_pool
{
    struct threadpool *pool = NULL;
    do 
    {
        pool = (threadpool*)malloc(sizeof(struct threadpool));
        if (NULL == pool)
        {
            fprintf(stderr,"failed to malloc threadpool!\n");
            break;
        }
        pool->thread_num = thread_num;
        pool->queue_max_num = queue_max_num;
        pool->queue_cur_num = 0;
        pool->head = NULL;
        pool->tail = NULL;
		pool->cpuid=0;
        if (pthread_mutex_init(&(pool->mutex), NULL))
        {
            fprintf(stderr,"failed to init mutex!\n");
            break;
        }
        if (pthread_cond_init(&(pool->queue_empty), NULL))
        {
            fprintf(stderr,"failed to init queue empty!\n");
            break;
        }
        if (pthread_cond_init(&(pool->queue_not_empty), NULL))
        {
            fprintf(stderr,"failed to init queue_not_empty!\n");
            break;
        }
        if (pthread_cond_init(&(pool->queue_not_full), NULL))
        {
            fprintf(stderr,"failed to init queue_not_full!\n");
            break;
        }
        pool->pthreads =(pthread_t*) malloc(sizeof(pthread_t) * 7);
        if (NULL == pool->pthreads)
        {
            fprintf(stderr,"failed to init pthreads!\n");
            break;
        }
        pool->queue_close = 0;
        pool->pool_close = 0;
        int i;
		int cpu_nums = sysconf(_SC_NPROCESSORS_CONF);
        for (i = 0; i < pool->thread_num; ++i)
        {
            pthread_create(&(pool->pthreads[i]), NULL, threadpool_function, (void *)pool);
			pool->cpuid++;
			if(pool->cpuid >cpu_nums){pool->cpuid=0;}
        }
        
        return pool;    
    } while (0);
    
    return NULL;
}
int threadpool_add_job(struct threadpool* pool, void* (*callback_function)(void *arg), void *arg) // add jobs to queue
{
    pthread_mutex_lock(&(pool->mutex));
    while ((pool->queue_cur_num == pool->queue_max_num) && !(pool->queue_close || pool->pool_close))
    {
        pthread_cond_wait(&(pool->queue_not_full), &(pool->mutex));   
    }
    if (pool->queue_close || pool->pool_close)   
    {
        pthread_mutex_unlock(&(pool->mutex));
        return -1;
    }
    struct job *pjob =(struct job*) malloc(sizeof(struct job));
    if (NULL == pjob)
    {
        pthread_mutex_unlock(&(pool->mutex));
		fprintf(stderr,"failed to init pjob!\n");
        exit(1);
    } 
    pjob->callback_function = callback_function;    
    pjob->arg = arg;
    pjob->next = NULL;
    if (pool->head == NULL)   
    {
        pool->head = pool->tail = pjob;
        pthread_cond_broadcast(&(pool->queue_not_empty)); 
    }
    else
    {
        pool->tail->next = pjob;
        pool->tail = pjob;    
    }
    pool->queue_cur_num++;
    pthread_mutex_unlock(&(pool->mutex));
    return 0;
}

inline int set_cpu(int i)
{
    cpu_set_t mask;
    CPU_ZERO(&mask);
 
    CPU_SET(i,&mask);
 
    printf("thread %lu, i = %d\n", pthread_self(), i);
    if(-1 == pthread_setaffinity_np(pthread_self() ,sizeof(mask),&mask))
    {        
		fprintf(stderr,"pthread_setaffinity_np\n");
		return -1;
    }
    return 0;
}
void* threadpool_function(void* arg) // this is the 
{

    struct threadpool *pool = (struct threadpool*)arg;
    struct job *pjob = NULL;
	if(set_cpu(pool->cpuid))
    {
        fprintf(stderr,"set cpu erro\n");
    }
    while (1)  //死循环
    {
        pthread_mutex_lock(&(pool->mutex));
        while ((pool->queue_cur_num == 0) && !pool->pool_close)   
        {
            pthread_cond_wait(&(pool->queue_not_empty), &(pool->mutex));
        }
        if (pool->pool_close)   
        {
            pthread_mutex_unlock(&(pool->mutex));
            pthread_exit(NULL);
        }
        pool->queue_cur_num--;
        pjob = pool->head;
        if (pool->queue_cur_num == 0)
        {
            pool->head = pool->tail = NULL;
        }
        else 
        {
            pool->head = pjob->next;
        }
        if (pool->queue_cur_num == 0)
        {
            pthread_cond_signal(&(pool->queue_empty));      
                    }
        if (pool->queue_cur_num == pool->queue_max_num - 1)
        {
            pthread_cond_broadcast(&(pool->queue_not_full)); //if queue is not full ,we could  announce threadpool_add_job() to add new job
        }
        pthread_mutex_unlock(&(pool->mutex));
        
        (*(pjob->callback_function))(pjob->arg);   
        pjob = NULL;    
    }
}
int threadpool_destroy(struct threadpool *pool)
{
    pthread_mutex_lock(&(pool->mutex));
    if (pool->queue_close || pool->pool_close)   
    {
        pthread_mutex_unlock(&(pool->mutex));
        return -1;
    }
    
    pool->queue_close = 1;       
    while (pool->queue_cur_num != 0)
    {
        pthread_cond_wait(&(pool->queue_empty), &(pool->mutex)); 
    }    
    
    pool->pool_close = 1;     
    pthread_mutex_unlock(&(pool->mutex));
    pthread_cond_broadcast(&(pool->queue_not_empty));  
    pthread_cond_broadcast(&(pool->queue_not_full));  
    int i;
    for (i = 0; i < pool->thread_num; ++i)
    {
        pthread_join(pool->pthreads[i], NULL);   
    }
    pthread_mutex_destroy(&(pool->mutex));          
    pthread_cond_destroy(&(pool->queue_empty));
    pthread_cond_destroy(&(pool->queue_not_empty));   
    pthread_cond_destroy(&(pool->queue_not_full));    
    
    return 0;
}

/* Recall that, once the filter is applied, all threads need to wait for
 * each other to finish before computing the smallest/largets elements
 * in the resulting matrix. To accomplish that, we declare a barrier variable:
 *      pthread_barrier_t barrier;
 * And then initialize it specifying the number of threads that need to call
 * wait() on it:
 *      pthread_barrier_init(&barrier, NULL, num_threads);
 * Once a thread has finished applying the filter, it waits for the other
 * threads by calling:
 *      pthread_barrier_wait(&barrier);
 * This function only returns after *num_threads* threads have called it.
 */
void *sharding_work(void *w) {
  /* Your algorithm is essentially:
   *  1- Apply the filter on the image
   *  2- Wait for all threads to do the same
   *  3- Calculate global smallest/largest elements on the resulting image
   *  4- Scale back the pixels of the image. For the non work queue
   *      implementations, each thread should scale the same pixels
   *      that it worked on step 1.
   */
	work my_work = *((work *)w);
	common_work* c_w = my_work.common;
	int32_t max, min, cur;
	int32_t id, nthreads, row, col;
	int32_t total = c_w->height * c_w->width;
	nthreads = c_w->max_threads;
	id = my_work.id;
	int32_t num_of_rows, start_row, num_of_cols, start_col;
	if(my_work.type==0){
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
			cur = apply2d(c_w->f, c_w->original_image, c_w->output_image, c_w->width, c_w->height, row, col);
			if(i == num_of_rows * id) {
				max = min = cur;//initializing max and min to an actual value
			}
			if(cur < min) min = cur;
			if(cur > max) max = cur;
		}
	}
	else if(my_work.type == 1){
		//COL MAJOR
		num_of_cols = (c_w->width + nthreads-1)/nthreads;
		start_col = num_of_cols * id;
		if(start_col >= c_w->width) {
			pthread_barrier_wait(&c_w->barrier);
			pthread_exit(0);
		}//do no work if out of bound
		for(int32_t i = 0; i < num_of_cols && start_col + i < c_w->width; i++){
			for(int32_t j = 0; j < c_w->height; j++){
				row = j;
				col = start_col + i;
				cur = apply2d(c_w->f, c_w->original_image, c_w->output_image, c_w->width, c_w->height, row, col);
				if(i == 0 && j == 0) max = min = cur;
				if(cur < min) min = cur;
				if(cur > max) max = cur;
			}
		}
	}
	else{
		num_of_cols = (c_w->width + nthreads-1)/nthreads;
		start_col = num_of_cols * id;
		if(start_col >= c_w->width){
			pthread_barrier_wait(&c_w->barrier);
			pthread_exit(0);
		}
		for(int32_t i = 0; i < c_w->height; i++){
			for(int32_t j = 0; j < num_of_cols && start_col + j < c_w->width; j++){
				row = i;
				col = start_col + j;
				cur = apply2d(c_w->f, c_w->original_image, c_w->output_image, c_w->width, c_w->height, row, col);
				if(i == 0 && j == 0) max = min = cur;
				if(cur < min) min = cur;
				if(cur > max) max = cur;
			}
		}
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
	if(my_work.type == 0){
		for(int32_t i =num_of_rows * id * c_w->width; i < num_of_rows * id *c_w->width+ start_row && i<total; i++){
			normalize_pixel(c_w->output_image, i, min, max);
		}
		pthread_exit(0);
	}
	else if(my_work.type == 1){
		for(int32_t i = 0; i < num_of_cols && start_col + i < c_w->width; i++){
			for(int32_t j = 0; j < c_w->height; j++){
				int32_t loc =c_w->width * j + start_col + i;
				normalize_pixel(c_w->output_image, c_w->width*j + start_col + i, min, max);
			}
		}
		pthread_exit(0);
	}
	else{
		for(int32_t i = 0; i < c_w->height; i++){
			for(int32_t j = 0; j < num_of_cols && start_col + j < c_w->width; j++){
				normalize_pixel(c_w->output_image, c_w->width*i + start_col + j, min, max);
			}
		}
		pthread_exit(0);
	}
}

/***************** WORK QUEUE *******************/
/* TODO: you don't have to implement this. It is just a suggestion for the
 * organization of the code.
 */

// int first last;
int creating_chunks(const int32_t * original,int32_t *target,int32_t width, int32_t height,
                             int32_t num_threads,int32_t work_chunk,chunk_coordinate *waiting_room){
    int chunk_created = 0;  
	int tempweight = 0 ;
	int tempheight =0;	

	if (width<work_chunk){   // this is the case where the width is smaller than chunk_size;
		for(int h=0;h<height-work_chunk;h+=work_chunk){
			chunk_coordinate *coord = (chunk_coordinate *) malloc(sizeof(chunk_coordinate));		
			coord->xstart =0;
			coord->xend = width;
			coord->ystart = h;
			coord->yend = h+work_chunk;
			waiting_room[chunk_created] = *coord;
			chunk_created++;
			tempheight = h;
			// free(coord);
	}
		chunk_coordinate *coord = (chunk_coordinate *) malloc(sizeof(chunk_coordinate));		
		coord->xstart =0;
		coord->xend = width;
		coord->ystart = tempheight+work_chunk;
		coord->yend = height;
		waiting_room[chunk_created] = *coord;
		chunk_created++;
		return chunk_created;
	}
	else if (height<work_chunk){  
		for(int w=0;w<width-work_chunk;w+=work_chunk){
			chunk_coordinate *coord = (chunk_coordinate *) malloc(sizeof(chunk_coordinate));		
			coord->xstart =w;
			coord->xend = w+work_chunk;
			coord->ystart = 0;
			coord->yend = height;
			waiting_room[chunk_created] = *coord;
			chunk_created++;
			tempweight = w;
			// free(coord);
	}
		chunk_coordinate *coord = (chunk_coordinate *) malloc(sizeof(chunk_coordinate));		
		coord->xstart =tempweight+work_chunk;
		coord->xend = width;
		coord->ystart = 0;
		coord->yend = height;
		waiting_room[chunk_created] = *coord;
		chunk_created++;
		return chunk_created;
	}

	else if (height<work_chunk && width<work_chunk){
		chunk_coordinate *coord = (chunk_coordinate *) malloc(sizeof(chunk_coordinate));		
		coord->xstart =0;
		coord->xend = width;
		coord->ystart = 0;
		coord->yend = height;
		waiting_room[chunk_created] = *coord;
		chunk_created++;
		return chunk_created;
	}
	else {
	for (int w =0;w<width-work_chunk;w+=work_chunk){
		// printf("w:%d\n",w);
		for (int h =0;h<height-work_chunk;h+=work_chunk){	
					// printf("h:%d\n",h);
			
			chunk_coordinate *coord = (chunk_coordinate *) malloc(sizeof(chunk_coordinate));		
			coord->xstart =w;
			coord->xend = w+work_chunk;
			coord->ystart = h;
			coord->yend = h+work_chunk;
			waiting_room[chunk_created] = *coord;
			chunk_created++;
			tempheight = h;
			// free(coord);
		}
		chunk_coordinate *coord = (chunk_coordinate *) malloc(sizeof(chunk_coordinate));
		coord->xstart =w;
		coord->xend = w+work_chunk;
		coord->ystart = tempheight+work_chunk;
		coord->yend = height;
		waiting_room[chunk_created] = *coord;	
		chunk_created++;
		// free(coord);
		tempweight =w;
		}
	for(int h=0;h<height-work_chunk;h+=work_chunk){
		chunk_coordinate *coord = (chunk_coordinate *) malloc(sizeof(chunk_coordinate));		
		coord->xstart =tempweight+work_chunk;
		coord->xend = width;
		coord->ystart = h;
		coord->yend = h+work_chunk;
		waiting_room[chunk_created] = *coord;
		chunk_created++;
		tempheight = h;
		// free(coord);
	}
	chunk_coordinate *coord = (chunk_coordinate *) malloc(sizeof(chunk_coordinate));		
	coord->xstart =tempweight+work_chunk;
	coord->xend = width;
	coord->ystart = tempheight+work_chunk;
	coord->yend = height;
	waiting_room[chunk_created] = *coord;
	chunk_created++;
	// free(coord);
	}
	return chunk_created;

}

int finishedwork=0;
int total_chunks =0;
/***************** MULTITHREADED ENTRY POINT ******/
 void* workthread(void* arg)
{ 
  	int min ,max,cur;
    chunk_job * myjob  = (chunk_job *) arg;
      if(finishedwork==total_chunks){
          printf("end thread\n");
      pthread_mutex_unlock(&mutex);
       pthread_exit(0);
      }
    pthread_mutex_lock(&mutex);
    finishedwork ++;
    pthread_mutex_unlock(&mutex);
    int width = myjob->coord.xend-myjob->coord.xstart;
    int height = myjob->coord.yend-myjob->coord.ystart;   
    min = 46920;
    max = -46920;
    for(int x =myjob->coord.xstart;x< myjob->coord.xend;x++){
      for(int y=myjob->coord.ystart;y<myjob->coord.yend;y++){
        cur = apply2d(myjob->common->f,myjob->common->original_image,myjob->common->output_image,myjob->common->width,
        myjob->common->height,y,x);
        if(cur < min) min = cur;
		if(cur > max) max = cur;
      }
    }
    pthread_mutex_lock(&mutex);
    if(myjob->common->max < max) myjob->common->max = max;
    if(myjob->common->min > min) myjob->common->min = min;
    pthread_mutex_unlock(&mutex);//get global max and min  
}

void apply_filter2d_threaded(const filter *f, const int32_t *original,
                             int32_t *target, int32_t width, int32_t height,
                             int32_t num_threads, parallel_method method,
                             int32_t work_chunk) {
  
	common_work *shared = (common_work *)malloc(sizeof(common_work));
	shared->f = f;
	shared->original_image = original;
	shared->height=height;
	shared->width=width;
	shared->max_threads=num_threads;
	shared->output_image=target;
	shared->max = -46920;
	shared->min = 46920;
	pthread_barrier_init(&shared->barrier, NULL, num_threads);
	work works[num_threads];
	pthread_t pthreads[num_threads];
	int cpu_nums = sysconf(_SC_NPROCESSORS_CONF);

	if(method == WORK_QUEUE){
		int w = (int)width%work_chunk ==0 ? width/work_chunk: width/work_chunk+1;
		int h = (int)height%work_chunk ==0 ? height/work_chunk: height/work_chunk+1;
		total_chunks = w*h;	
    	chunk_coordinate chunk_coordinates[total_chunks];
	
		int chunk_created= creating_chunks( original,target, width, height,
                            num_threads, work_chunk, chunk_coordinates);
		struct threadpool *pool = threadpool_init(num_threads, chunk_created);
     
		for(int i = 0;i<chunk_created;i++){	
			chunk_job *queuejob = ( chunk_job *) malloc(sizeof(chunk_job));
			queuejob->coord = chunk_coordinates[i];		
			queuejob->common = shared;
			threadpool_add_job(pool, workthread, queuejob);
			}
		threadpool_destroy(pool);
	
		for(int32_t i = 0; i < width * height; i ++){
			normalize_pixel(target, i, shared->min, shared->max);
		}
	}
	else if(0 <= method && method < 3){

		pthread_mutex_init(&mutex, NULL);
			for(int i = 0; i < num_threads; i++){
				works[i].common=shared;
				works[i].id=i;
				works[i].type=method;
				pthread_create(&pthreads[i], NULL, sharding_work, &works[i]);
			}
			for(int i = 0; i < num_threads; i++){
				pthread_join(pthreads[i],NULL);
			}
	 }
}

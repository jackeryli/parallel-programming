# Mandelbrot Set

## Pthread

My approach is to evenly distribute the height of the images among each thread. For example, if an image is 800*600 and the max_thread_num is 7, then we will assign thread_id=0 to handle pixels with heights (0,7,14,21,28...588), and thread_id=1 to handle pixels with heights (1,8,15,22,29...589). Through this allocation, we can achieve uniform distribution in a simpler way.

---

### Custom Data Structure

The `MandelbrotArg` structure is a set of parameters passed into a pthread.

```c
typedef struct {
    int iters;
    double left;
    double right;
    double lower; 
    double upper; 
    int width;
    int height;
    int j_start;
    int j_end;
}MandelbrotArg;
```

- `j_start`: The starting value of the height.
- `j_end`: The ending value of the height.

---

### Custom Functions

- `calc_repeated`: This function is responsible for calculating the value of "repeated". Using pointers allows direct modification of the original data.
- `calc_mandelbrot`: This is the function pointer passed to `pthread_create`, responsible for the initial setup and calculation of the Mandelbrot set.
- `write_png`: This function is responsible for writing the image to a PNG file.

```c
void calc_repeated(double *x, double *y, double *xx, double *yy, double *length_squared, double *y0, double *x0);
void *calc_mandelbrot(void *argv);
void write_png(const char* filename, int iters, int width, int height, const int* buffer);
```

---

### calc_mandelbrot

```cpp
void *calc_mandelbrot(void *argv) {
    #ifdef TIME
        t1 = std::chrono::steady_clock::now();
    #endif
    MandelbrotArg *arg = (MandelbrotArg*) argv;
    int iters = arg->iters;
    double left = arg->left;
    double right = arg->right;
    double lower = arg->lower;
    double upper = arg->upper;
    int width = arg->width;
    int height = arg->height;
    int j_start = arg->j_start;  // thread_id
    int j_end = arg->j_end;      // Image height
    
    // Extract some commonly used calculations
    double tmp1 = (upper - lower) / height;
    double tmp2 = (right - left) / width;
    
    // Process images in intervals of num_of_threads
    for (int j = j_start; j < j_end; j+=num_of_threads) {
        double y0 = j * tmp1 + lower;
        int tmp3 = j * width;
        
        // Calculate x0 first, as it can be computed independently
        double x0[width];
        #pragma GCC ivdep
        for(int i=0; i<width; ++i){
            x0[i] = i * tmp2 + left;
        }
        
        int i;
        // Calculate repeats; here we compute two pixels at a time
        for(i=1; i < width; i+=2){
            double x[2] = {0};
            double y[2] = {0};
            double x_tmp[2] = {0};
            double y_tmp[2] = {0};
            double xx[2] = {0};
            double yy[2] = {0};
            double length_squared[2] = {0};
            double x0_arr[2] = {x0[i-1], x0[i]};
            int repeats=0;
            int state = 0; // Record state
            // If both pixels meet the condition, continue calculating and the repeats of the two pixels are the same
            // If the condition is not met, exit the while loop
            while(1){
                // Set variable state so that the program knows which condition caused the while loop to exit
                if(length_squared[0] >= 4) { state = 1; break; }
                if(length_squared[1] >= 4) { state = 2; break; }
                if(repeats >= iters)       { state = 3; break; }
                
                // Set two x_tmp, y_tmp to reduce dependency
                #pragma simd vectorlength (2)
                for(int k=0; k < 2; ++k){
                    y_tmp[k] = 2 * x[k] * y[k] + y0;
                    x_tmp[k] = xx[k] - yy[k] + x0_arr[k];
                }

                #pragma simd vectorlength (2)
                for(int k=0; k < 2; ++k){
                    y[k] = y_tmp[k];
                    x[k] = x_tmp[k];
                }
                
                // Set two xx, yy to preprocess the square, for the following use
                #pragma simd vectorlength (2)
                for(int k=0; k < 2; ++k){
                    xx[k] = x[k] * x[k];
                    yy[k] = y[k] * y[k];
                }
                
                #pragma simd vectorlength (2)
                for(int k=0; k < 2; ++k)
                    length_squared[k] = xx[k] + yy[k];
                                            
                ++repeats;
            }

            // Store the index of the two pixels, for later use
            int index1 = tmp3 + i - 1;
            int index2 = tmp3 + i;
            
            switch(state){
                // Pixel[0] is not done yet
                case 2:
                    image[index2] = repeats;
                    while (length_squared[0] < 4 && repeats < iters) {
                        calc_repeated(&x[0], &y[0], &xx[0], &yy[0], &length_squared[0], &y0, &x0_arr[0]);
                        ++repeats;
                    }
                    
                    image[index1] = repeats;
                    break;
                // Pixel[1] is not done yet
                case 1:
                    image[index1] = repeats;
                    while (length_squared[1] < 4 && repeats < iters) {
                        calc_repeated(&x[1], &y[1], &xx[1], &yy[1], &length_squared[1], &y0, &x0_arr[1]);
                        ++repeats;
                    }
                    image[index2] = repeats;
                    break;
                // Both are done
                case 3:
                    image[index1] = repeats;
                    image[index2] = repeats;
                    break;
                default:
                    break;
            }
        }
        // The last pixel may not be processed
        if(( i = i-1) <width){
            int repeats = 0;
            
            double x = 0;
            double y = 0;

            double xx = 0;
            double yy = 0;
            
            double length_squared = 0;
            while (length_squared < 4 && repeats < iters) {
                calc_repeated(&x, &y, &xx, &yy, &length_squared, &y0, &x0[i]);
                ++repeats;
            }
            image[tmp3 + i] = repeats;
        }
        
    }
    #ifdef TIME
        t2 = std::chrono::steady_clock::now();
        std::cout << "[Thread " << j_start << "] " << std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() << "us.\n";
    #endif

    return NULL;
}
```

---

### Program Initialization

- Initialize an array of pthread_t with a length equal to the total number of available cores.
- Initialize an array of MandelbrotArg to provide initial conditions for each thread.

```c
pthread_t threads[num_of_threads];

MandelbrotArg args[num_of_threads];
```

---

### Creating Pthreads, passing parameters, and starting execution.

```c
for(i = 0; i < num_of_threads; i++) {
    args[i].iters = iters;
    args[i].left = left;
    args[i].right = right;
    args[i].lower = lower;
    args[i].upper = upper;
    args[i].width = width;
    args[i].height = height;

    args[i].j_start = i;
    args[i].j_end = height;

    rc = pthread_create(&threads[i], NULL, calc_mandelbrot, (void*)&(args[i]));
    if(rc) {
        printf("ERROR; return code from pthread_create() is %d\n", rc);
        exit(-1);
    }
}
```

---

### Pthread join: Wait for all threads to finish execution and write to the image.

```c    
for(i = 0; i < num_of_threads; i++) {
    pthread_join(threads[i], NULL);
}

/* draw and cleanup */
write_png(filename, iters, width, height, image);
free(image);

```

---

## Hybrid

Task allocation is still based on the height of the image, evenly distributed. The difference compared to pthread lies in the necessity to gather the processed portions of the image from each process and reassemble them into the original image according to the initial allocation. I use `MPI_Gatherv` to perform this operation.

---

### Global Variables

The following global variables are defined for convenience in calculations:

```c
int rank;
int size;
int iters;
double left;
double right;
double lower; 
double upper; 
int width;
int height;
int r;         // Record height % num_of_threads
int d;         // Record height / num_of_threads
int *image;
```

---

### Custom Functions

Essentially similar to the pthread version, except for the difference in the type of `calc_mandelbrot` (void* --> void).

```c
void calc_mandelbrot(int* line_arr, int line_counter);
void write_png(const char* filename, int iters, int width, int height, const int* buffer);
void calc_repeated(double *x, double *y, double *xx, double *yy, double *length_squared, double *y0, double *x0);
```

---

### Main Program Initialization

Use `omp_set_num_threads` to set the number of threads used by OpenMP.

```c
cpu_set_t cpu_set;
sched_getaffinity(0, sizeof(cpu_set), &cpu_set);
//printf("%d cpus available\n", CPU_COUNT(&cpu_set));
int num_of_threads = CPU_COUNT(&cpu_set);
omp_set_num_threads(num_of_threads);
```

---

### Main Program Task Allocation to Processes

Calculate the height each process should be allocated, then store it in `line_arr`, and store the total count in `line_counter`. Execute `calc_mandelbrot` to begin calculations.

```c
r = height % size;
d = height / size; 

int line_counter = (rank < r)?d+1:d;
int line_arr[line_counter];

#pragma omp parallel for
for(int i=0; i < line_counter; ++i)
    line_arr[i] = rank+i*size;

image = (int*)malloc(width * line_counter * sizeof(int));
assert(image);

calc_mandelbrot(line_arr, line_counter);
```

---

### Use `MPI_Gatherv` to collect processed data from different processes.

```c
int* image_buf = (int*)malloc(width * height * sizeof(int));

int* displs = (int*)malloc(size * sizeof(int));
int* recvcounts = (int*)malloc(size * sizeof(int));

displs[0] = 0;

for(int i=0; i < size; i++) {
    if(i < r) recvcounts[i] = (d+1) * width;
    else recvcounts[i] = d * width;
    if(i>=1) displs[i] = displs[i-1] + recvcounts[i-1];
}

MPI_Gatherv(image, width * line_counter, MPI_INT, image_buf, recvcounts, displs, MPI_INT, 0, MPI_COMM_WORLD);
```

---

### Write the pixels distributed across each process into out.png

I have used `pragma omp parallel for` to accelerate the process, but the writing action is not the main bottleneck. The reduced time is limited.

```c

// Write to png
if(rank == 0) {
    #ifdef TIME
        t1 = std::chrono::steady_clock::now();
    #endif
    int* final_image = (int*)malloc(width * height * sizeof(int));

    #pragma omp parallel
    {
        #pragma omp for schedule(dynamic)
        for(int k = 0; k < size; k++){
            for(int i = k, counter = 0; i < height; i+=size, ++counter){
                int l_index = i*width;
                int r_index = displs[k] + counter*width;


                #pragma omp simd
                for(int j=0; j<width; j++){
                    final_image[l_index+j] = image_buf[r_index + j];
                }

                //memcpy(&final_image[l_index], &image_buf[r_index], sizeof(int)*width);
            }
        }
    }




    write_png(filename, iters, width, height, final_image);


}

free(displs);
free(recvcounts);
free(image_buf);
/* draw and cleanup */

free(image);
MPI_Finalize();
return 0;

}
```

---

## void calc_mandelbrot(int* line_arr, int line_counter)

This function is essentially similar to the pthread version, but it includes `#pragma omp parallel` and `#pragma omp for schedule(dynamic)` in the for loop for acceleration.

---

## Experimental Analysis

I conducted experiments using strict35 and increased the number of iterations to 50000 to avoid too small time differences.
I used the following methods for measurement:

1. `t2 = std::chrono::steady_clock::now();`

   Used for measuring time in pthreads and calculating the total time required for the entire process.

2. `omp_timer[rank][omp_get_thread_num()] += omp_get_wtime() - omp_t1;`

   Responsible for calculating how long each thread executes in OMP.

3. `comm_time += MPI_Wtime() - time1;`

   Calculates communication time in MPI.

---

### Experiment 1: Pthread version when n=1

Through the table and the following chart, it can be observed that as the number of cores available increases, the program execution speed accelerates. Basically, doubling the number of cores doubles the program execution speed. However, it can be noticed that there is uneven distribution of tasks when c=8.

|  n  |       c  | Total Time| Average Thread time | Standard deviation | Speed up  |
| --- | -------- | --------- | ------------------- | ------------------ | --------- |
|   1 |        2 |   501.900 |              498.523|              0.587 |     1.000 |
|   1 |        4 |   252.650 |              249.168|              0.125 |     2.001 |   
|   1 |        6 |   169.650 |              166.147|              0.134 |     3.000 |
|   1 |        8 |   136.833 |              127.162|              3.782 |     3.920 |
|   1 |       10 |   102.900 |               99.654|              0.106 |     5.003 | 
|   1 |       12 |    86.083 |               83.120|              0.104 |     5.998 |

![Thread Time, pthread, n=1](https://i.imgur.com/PVjLGjZ.png)
![Speedup, pthread, n=1](https://i.imgur.com/QRtv2Ib.png)

---

### Experiment 2: Hybrid version when c=4


Through the table and the following chart, it can be observed that the main time is basically spent on computation. Observing the speedup, it can also be seen that the speedup is almost proportional to the number of processes. Compared to the pthread version, it can be noticed that the standard deviation of the hybrid version is generally smaller.

|n|c|rearrage(s)|write Image(s)|Total Time(s)|Thread Average Time(s)|STDEV|Speedup|
|-|-|-|-|-|-|-|-|
| 2|4|0.0368|2.5927|	207.630|203.7312|0.0140|1.000|
| 4|4|0.0224|2.5799|	105.630|101.9169|0.0630|1.999|
| 6|4|0.0264|2.5817|	71.7000| 67.9396|0.0745|2.999|
| 8|4|0.0267|2.5765|	67.5800| 50.9315|0.0445|4.000|
|10|4|0.0242|2.5792|	44.4500| 40.7715|0.0191|4.997|
|12|4|0.0219|2.6059|	37.7000| 33.9459|0.0281|6.002|

![Total Time, Hybrid, c=4](https://i.imgur.com/C80Uw4q.png)
![Speedup, Hybrid c=4](https://i.imgur.com/1w5PAEl.png)

---

### Experiment 3: Hybrid version when n=4

Through the table and the chart, it can be observed that the conclusion is basically similar to Experiment 2. Additionally, it can be noticed that the execution times for both (n=4, c=2) and (n=2, c=4) scenarios are similar.

|n|c|rearrage(s)|write Image(s)|Total Time(s)|Process Average Time(s)|STDEV|Speedup|
|-|-|-|-|-|-|-|-|-|-|-|
|4|	1	|0.063	|2.576|	411.033|407.561|0.133| 1.000|
|4|	2	|0.036	|2.577|	207.083|203.671|0.082| 2.001|
|4|	4	|0.022	|2.580|	105.533|101.891|0.033| 4.000|
|4|	6	|0.022	|2.579|	 71.550| 67.942|0.054| 5.999|
|4|	8	|0.022	|2.581|	 62.167| 58.258|0.027| 6.996|
|4|	10	|0.022	|2.578|	 44.317| 40.783|0.026| 9.993|
|4|	12	|0.021	|2.585|	 37.583| 33.975|0.017|11.996|


![Total Time, Hybrid, n=4](https://i.imgur.com/LeFNJdo.png)
![Speedup, Hybrid, n=4](https://i.imgur.com/CDqfgDX.png)

---

## Questions and Discussion

1. Scalability:
   Both versions of the program demonstrate good scalability. Increasing the number of threads leads to improved performance.

2. Load Balance:
   The pthread version does not evenly distribute tasks, sometimes resulting in heavier loads on certain cores. However, the load balance in the hybrid version is better, with execution times for each process being quite close. This could be attributed to OpenMP evenly distributing tasks or not executing enough iterations yet.

---

## Lessons Learned from this Experiment

### 1. Compiling Specific Code using ifdef and Makefile

First, we added the following lines to `sample.c`. The code inside `#ifdef` will only be compiled if `TIME` is defined.
```c
//sample.c
#ifdef TIME
    printf("time=%d", time1);
#endif
```

---

Next, we modified the Makefile. The code below checks whether the `TIME` variable is set to 1. If true, it adds `-DTIME` to `CFLAGS`, allowing the code in `sample.c` to detect it.
```bash
ifeq ($(TIME),1)
    CFLAGS += -DTIME
endif
```

---

Then, we can choose whether to compile the program with time output.

```bash
# With time output
make TIME=1
# Without time output
make
```

### 2. Learning to Vectorize

This experiment taught us how to use vectorized operations to speed up the program. With vectorization, the program runs almost twice as fast. I attempted to utilize AVX for vectorization, but since Apollo only supports up to SSE4.2 and not AVX, I wasn't successful in trying it.

### 3. Learning to Accelerate Code Using OpenMP

This assignment involved using OpenMP, which proved to be very convenient for quickly executing repetitive tasks with multi-threading. The fastest OpenMP schedule appears to be `schedule(dynamic)`, as it dynamically assigns tasks to threads that finish earlier, saving a significant amount of time.

```c
#pragma omp parallel
{
    #pragma omp for schedule(dynamic)
    // ...
}
```


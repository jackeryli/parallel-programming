# All Pairs Shortest Path on CPUs

## Implementation

### Which algorithm do you choose?

Block Floyd Warshall

### Describe your implementation.

Firstly, this time I used clang for compilation and found that clang can perform vectorization for me. Below is the Makefile used this time:

I added some parameters and found that they have an impact on performance before and after adding them.

* msse4 msse3 msse2: Enable specific instruction set compilation.
* -Rpass-missed=loop-vectorize: Tells me which for loops failed to vectorize.
* -Rpass-analysis=loop-vectorize: Analyzes why loop vectorization succeeded or failed.
* -Rpass=loop-vectorize: Forces loop vectorization.

![clang vectorization](https://i.imgur.com/Ny9vpz4.png)

```shell
CC = gcc
CXX = clang++
CXXFLAGS = -O3 -fopenmp -msse4 -msse2 -msse3 -Rpass-missed=loop-vectorize -Rpass-analysis=loop-vectorize -Rpass=loop-vectorize
CFLAGS = -O3 -lm -pthread -fopenmp
TARGETS = hw3

ifeq ($(TIME),1)
	CXXFLAGS += -DTIME
endif

.PHONY: all
all: $(TARGETS)

.PHONY: clean
clean:
	rm -f $(TARGETS)
```


ChatGPT
I also attempted manual vectorization, but found that the performance was not better than auto-vectorization. The difference was approximately 21 seconds compared to 17 seconds when running on the hw3-judge. Below is the manually vectorized code:

```c
int block_internal_start_y = b_j * B;
int block_internal_end_y = (b_j + 1) * B;
block_internal_end_y =  (block_internal_end_y > n)? n : block_internal_end_y;


if (block_internal_end_y > n) {
    //block_internal_end_y = n;
    block_internal_start_y = n-B;
}


for (int k = RB; k < RBB && k < n; ++k) {
    // To calculate original index of elements in the block (b_i, b_j)
    // For instance, original index of (0,0) in block (1,2) is (2,5) for V=6,B=2


    for (int i = block_internal_start_x; i < block_internal_end_x; ++i) {
        /*
        for (int j = block_internal_start_y; j < block_internal_end_y; ++j) {
           int r = Dist[i][k] + Dist[k][j];
           int l = Dist[i][j];
           Dist[i][j] = r * (r < l) + l * (r >= l);
        }
        */

        int j = 0;

        __m128i IK =_mm_set1_epi32(Dist[i][k]);
        for (; j < B; j += 4) {
            int jdx = j+block_internal_start_y;
            __m128i left = _mm_lddqu_si128((__m128i*)&(Dist[i][jdx]));
            __m128i right = _mm_lddqu_si128((__m128i*)&(Dist[k][jdx]));
            right = _mm_add_epi32(IK, right);
            __m128i compare1 = _mm_cmplt_epi32 (right, left);
            __m128i compare2 = _mm_andnot_si128(compare1, left);
            compare1 = _mm_and_si128(right, compare1);
            left = _mm_add_epi32(compare1, compare2);
            //edge = _mm_or_si128(_mm_and_si128(compare, val), _mm_andnot_si128(compare, edge));
            _mm_storeu_si128((__m128i*)&(Dist[i][jdx]), left);
        } 

    }
}
}
```

The code submitted this time utilizes clang for auto-vectorization. Explanations are provided within comments in the code itself.

```c
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <iostream>

#ifdef TIME
    #include <chrono>
    double time1;
    double comm_time = 0;
    std::chrono::steady_clock::time_point t1;
    std::chrono::steady_clock::time_point t2;
    double omp_timer[12]; 
#endif

int num_of_threads;
const int INF = 1073741823;
const int V = 6010;
void input(char* inFileName);
void output(char* outFileName);

void block_FW(int B);
int ceil(int a, int b);
void cal(int B, int Round, int block_start_x, int block_start_y, int block_width, int block_height);

int n, m;
static int Dist[V][V];

int main(int argc, char* argv[]) {
    cpu_set_t cpu_set;
    sched_getaffinity(0, sizeof(cpu_set), &cpu_set);
    //printf("%d cpus available\n", CPU_COUNT(&cpu_set));
    num_of_threads = CPU_COUNT(&cpu_set);
    omp_set_num_threads(num_of_threads);
    input(argv[1]);
    /********************************************
     * I used a blockSize of 64 because subsequent experiments revealed that 64 yielded the fastest results.
     *******************************************/
    int B = 64;
    block_FW(B);
    output(argv[2]);
    return 0;
}

void input(char* infile) {

#ifdef TIME
    t1 = std::chrono::steady_clock::now();
#endif

    FILE* file = fopen(infile, "rb");
    fread(&n, sizeof(int), 1, file);
    fread(&m, sizeof(int), 1, file);
    
    /********************************************
     * Here, I employed a splitting method by using the index of `i` to split the `j` for loop into two blocks, allowing the compiler to vectorize effectively.
     *******************************************/
     
    for (int i = 0; i < n; ++i) {
        #pragma clang loop vectorize(enable) interleave(enable)
        for (int j = 0; j < i; ++j) {
            Dist[i][j] = INF;
        }
        Dist[i][i] = 0;
        #pragma clang loop vectorize(enable) interleave(enable)
        for (int j = i + 1; j < n; ++j) {
            Dist[i][j] = INF;
        }
    }

    int pair[3];
    for (int i = 0; i < m; ++i) {
        fread(pair, sizeof(int), 3, file);
        Dist[pair[0]][pair[1]] = pair[2];
    }
    fclose(file);

#ifdef TIME
    t2 = std::chrono::steady_clock::now();
    std::cout << "[Input] " << std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() << "\n";
#endif

}

void output(char* outFileName) {

#ifdef TIME
    t1 = std::chrono::steady_clock::now();
#endif

    FILE* outfile = fopen(outFileName, "w");
    for (int i = 0; i < n; ++i) {
        fwrite(Dist[i], sizeof(int), n, outfile);
    }
    fclose(outfile);

#ifdef TIME
    t2 = std::chrono::steady_clock::now();
    std::cout << "[Output] " << std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() << "\n";
#endif

}

int ceil(int a, int b) { return (a + b - 1) / b; }

void block_FW(int B) {

#ifdef TIME
    t1 = std::chrono::steady_clock::now();
#endif

    int round = ceil(n, B);
    
    for (int r = 0; r < round; ++r) {
        /********************************************
         * Phase1: Start from the top left corner and gradually move along the diagonal to the bottom right corner, with block lengths of 1.
         *******************************************/
        cal(B, r, r, r, 1, 1);

        /********************************************
         * Phase2: The blocks operated in phase 1 are located on the same row or column as the current block.
         *******************************************/
        cal(B, r,       r,      0,                  r,              1);
        cal(B, r,       r,  r + 1,      round - r - 1,              1);
        cal(B, r,       0,      r,                  1,              r);
        cal(B, r,   r + 1,      r,                  1,  round - r - 1);

        /********************************************
         * Phase3: The remaining region.
         *******************************************/
        cal(B, r,       0,      0,                  r,              r);
        cal(B, r,       0,  r + 1,      round - r - 1,              r);
        cal(B, r,   r + 1,      0,                  r,  round - r - 1);
        cal(B, r,   r + 1,  r + 1,      round - r - 1,  round - r - 1);
    }

#ifdef TIME
    t2 = std::chrono::steady_clock::now();
    std::cout << "[Compute] " << std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() << "\n";

    for(int i=0; i<num_of_threads; ++i)
        printf("[thread_id=%d]%f\n", i, omp_timer[i]);
#endif

}

void cal(
    int B, int Round, int block_start_x, int block_start_y, int block_width, int block_height) {
    
    int block_end_x = block_start_x + block_height;
    int block_end_y = block_start_y + block_width;
    
    /********************************************
     * Move some of the redundant calculations to the outermost level to avoid redundant computations inside.
     *******************************************/
    int RB = Round * B;
    int RBB = (Round+1) * B;
    if(RBB > n) RBB = n;
    
    /********************************************
     * Use OpenMP to accelerate the for loop.
     *******************************************/
    #pragma omp parallel for schedule(dynamic)
    for (int b_i = block_start_x; b_i < block_end_x; ++b_i) {

#ifdef TIME
    double omp_t1 = omp_get_wtime();
#endif
        /********************************************
         * Pre-calculate expressions that can be computed in the outer loop beforehand.
         *******************************************/
        int block_internal_start_x = b_i * B;
        int block_internal_end_x = (b_i + 1) * B;
        block_internal_end_x =  (block_internal_end_x > n)? n : block_internal_end_x;
        
        for (int b_j = block_start_y; b_j < block_end_y; ++b_j) {
            
            int block_internal_start_y = b_j * B;
            int block_internal_end_y = (b_j + 1) * B;
            block_internal_end_y =  (block_internal_end_y > n)? n : block_internal_end_y;

            for (int k = RB; k < RBB; ++k) {
                for (int i = block_internal_start_x; i < block_internal_end_x; ++i) {
                    /********************************************
                     * Pre-fetch Dist[i][k] to avoid constant memory access.
                     *******************************************/
                    int IK = Dist[i][k];
                    #pragma clang loop vectorize(enable) interleave(enable)
                    for (int j = block_internal_start_y; j < block_internal_end_y; ++j) {
                       /********************************************
                         * Rewrite to avoid if cocndition.
                         *******************************************/
                       int l = IK + Dist[k][j];
                       int r = Dist[i][j];
                       Dist[i][j] = l * (l < r) + r * (l >= r);
                    }
                }
            }
        }

#ifdef TIME
    omp_timer[omp_get_thread_num()] += omp_get_wtime() - omp_t1;
#endif

    }
}
```

### What is the time complexity of your algorithm, in terms of number of vertices V, number of edges E, number of CPUs P?

The time complexity of the algorithm is O(V^3/P), where V represents the number of vertices and P represents the number of CPUs. The algorithm's operation is essentially based on Floyd-Warshall (O(V^3)), but distributed across p CPUs for parallel computation, resulting in O(V^3/P).

### How did you design & generate your test case?

The test cases were primarily generated using a random approach. Each pair of vertices would generate an edge with a weight ranging from 501 to 1000.

```c
int main(int argc, char** argv) {

    srand(time(NULL));
    m = 0;
    for (int i = 0; i < n; i++){
        for (int j = 0; j < n; j++){
            if(i != j) {
                int random = rand();
                if( random % 2 == 0 ) {
                    int _distance = ( rand() % 500) + 501;
                    (edges[m]).src = i;
                    (edges[m]).dst = j;
                    (edges[m]).distance = _distance;
                    m++;
                }
                
            }
        }
    }

    std::random_shuffle(edges, edges + m);
    output(argv[1]);
}
```

## Experiment & Analysis

### System Spec

apollo

### Performance Metrics

Time measurement is done using `chrono`, `omp_get_wtime()`, and `srun time`. The usage is as follows:

#### srun time

```shell
#!/bin/bash
#SBATCH -n 1
#SBATCH -c 6
echo "b128"
srun time ./hw3 ./cases/c20.1 c20.1.out 128
```

#### chrono

```c
#ifdef TIME
    t1 = std::chrono::steady_clock::now();
#endif

#ifdef TIME
    t2 = std::chrono::steady_clock::now();
    std::cout << "[Compute] " << std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() << "\n";

    for(int i=0; i<num_of_threads; ++i)
        printf("[thread_id=%d]%f\n", i, omp_timer[i]);
#endif
```

#### omp_get_wtime()
```c
#ifdef TIME
    double omp_t1 = omp_get_wtime();
#endif
 
#ifdef TIME
    omp_timer[omp_get_thread_num()] += omp_get_wtime() - omp_t1;
#endif

#ifdef TIME
    for(int i=0; i<num_of_threads; ++i)
        printf("[thread_id=%d]%f\n", i, omp_timer[i]);
#endif
```

The results of the program execution are as follows, with the following time measurements:

1. Input: Time taken to read the file and initialize (in microseconds)
2. Compute: Total computation time (in microseconds)
3. OMP thread individual execution time (in seconds)
4. Output: Time taken to write the file (in microseconds)
5. Total time spent by the entire program (in seconds)

### Strong scalability & Time profile

I conducted two experiments. The first one tested the performance variation with different thread counts. It can be observed that the performance generally increases with the number of threads. However, it's worth noting that the time taken for reading and writing is somewhat unstable and may require multiple runs to confirm if it's within the margin of error. Additionally, at thread num = 6, it can be seen that the task allocation is not very even.

|c	|INPUT(s)   |Compute(s) 	|output	(s)     |total	(s) |thread avg	 (s)    |thread time stdev|Speedup|
|-|-|-|-|-|-|-|-|
|1	|0.629	|48.148	|0.200|48.98|48.133	|0.000	|1.000|
|2	|0.630	|24.408	|0.207|25.25|24.092	|0.164	|1.940|
|4	|0.625	|12.585	|0.214|13.43|12.105	|0.138	|3.647|
|6	|0.619	|13.460	|0.239|14.33|9.285	|1.610	|3.418|
|8	|0.644	|7.146	|0.207|8.010|6.386	|0.190	|6.115|
|10|0.641	|5.702	|0.212|6.580|5.069	|0.106	|7.444|
|12|0.670	|4.918	|0.241|5.850|4.268	|0.097	|8.373|

![Total Time on Different Thread Number](https://i.imgur.com/zxBKrr8.png)
![Speedup on Different Thread Number](https://i.imgur.com/dAr2vB3.png)

The second experiment examined whether different blockSize values would affect the time. It can be observed that the speed is fastest when blockSize is set to 64. Additionally, as blockSize increases, the distribution of threads becomes less even. This is likely because with larger block sizes, the execution time for each task increases, leading to greater disparities in execution time.


|blockSize	|INPUT	    |Compute	|output	    |total	|thread avg	   |thread time stdev|Speedup|
|-|-|-|-|-|-|-|-|
|4		|0.649|	40.116	|0.233|	41.020|	39.637	|0.045|	1.000|
|8		|0.641|	17.225	|0.209|	18.090|	16.894	|0.049|	2.268|
|16		|0.665|	12.313	|0.219|	13.210|	11.916	|0.053|	3.105|
|32		|0.666|	8.654	|0.227|	9.550	|8.115	|0.096|	4.295|
|64		|0.628|	7.307	|0.246|	8.190	|6.478	|0.184|	5.009|
|128	|0.628|	8.668	|0.261|	9.570	|6.905	|0.227|	4.286|

![Total time on different blockSize](https://i.imgur.com/R9Lq9Nd.png)
![Speedup on different blockSize](https://i.imgur.com/tOcKPP5.png)

## Experience & conclusion

### What have you learned from this homework?

In this assignment, I learned how to write SSE instructions for acceleration and how to use clang to compile auto-vectorized code. I also discovered the importance of performing tasks within for loops as early as possible to minimize redundant computations. Additionally, I realized that the presence of conditional statements (if) in the code can decrease performance. While some conditional statements may not significantly affect performance, I will strive to remove unnecessary conditionals in the future to avoid resource consumption.

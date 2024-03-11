#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <iostream>
#include <omp.h>

const int INF = ((1 << 30) - 1); 
void input(char *inFileName);
void output(char *outFileName);

void block_FW(int B);
int ceil(int a, int b);
__global__ void phase1(int* Dist_gpu, int r, int n, int pitch_int);
__global__ void phase2(int* Dist_gpu, int r, int n, int pitch_int);
__global__ void phase3(int* Dist_gpu, int r, int n, int pitch_int, int thread_id, int Round);

int n, m;   // Number of vertices, edges
int* Dist;

int* Dist_gpu[2];

size_t pitch;

int N;

int main(int argc, char* argv[])
{   
    
    input(argv[1]);
    int B = 64;
    block_FW(B);
    output(argv[2]);
    cudaFreeHost(Dist);
    cudaFree(Dist_gpu);
    return 0;
}

void input(char* infile) { 
    FILE* file = fopen(infile, "rb"); 
    fread(&n, sizeof(int), 1, file); 
    fread(&m, sizeof(int), 1, file);

    N = n;
    n = (!n%64)? n : n + 64 - n%64;

    cudaMallocHost( &Dist, sizeof(int)*(n*n));

    for (int i = 0; i < n; ++i) {
        int IN = i * n;
        #pragma GCC ivdep
        for (int j = 0; j < i; ++j) {
            Dist[IN + j] = INF;
        }
        #pragma GCC ivdep
        for (int j = i + 1; j < n; ++j) {
            Dist[IN + j] = INF;
        }
    }

    int pair[3]; 
    for (int i = 0; i < m; ++i) { 
        fread(pair, sizeof(int), 3, file); 
        Dist[pair[0] * n + pair[1]] = pair[2]; 
    } 
    fclose(file); 
}

void output(char *outFileName) {
    FILE *outfile = fopen(outFileName, "w");
    for (int i = 0; i < N; ++i) {
        fwrite(&Dist[i * n], sizeof(int), N, outfile);
    }
    fclose(outfile);
}

int ceil(int a, int b) {
    return (a + b -1)/b;
}

void block_FW(int B)
{
    int round = ceil(n, B);

    #pragma omp parallel num_threads(2)
    {
        int thread_id = omp_get_thread_num();
        cudaSetDevice(thread_id);
        cudaMalloc((void **)&Dist_gpu[thread_id], n * n * sizeof(int));
        cudaMemcpy(Dist_gpu[thread_id], Dist, n * n * sizeof(int), cudaMemcpyHostToDevice);
        
        dim3 grid3((round/2)+1, round);

        int pitch_int = n;

        for (int r = 0; r < round; ++r) {
            
            #pragma omp barrier
            if(r < (round/2) && thread_id == 1){
                cudaMemcpyPeer((void*) &Dist_gpu[1][r * B * n], 1, (void*) &Dist_gpu[0][r * B * n], 0, B * n * sizeof(int));

            }else if(r >= (round/2) && thread_id == 0){
                if(r == (round-1))
                    cudaMemcpyPeer((void*) &Dist_gpu[0][r * B * n], 0, (void*) &Dist_gpu[1][r * B * n], 1, (n - r * B) * n * sizeof(int));
                else
                    cudaMemcpyPeer((void*) &Dist_gpu[0][r * B * n], 0, (void*) &Dist_gpu[1][r * B * n], 1, B * n * sizeof(int));
            }
            #pragma omp barrier

           
            phase1  <<< 1                     , dim3(32,32),   64*64*sizeof(int) >>>(Dist_gpu[thread_id], r, n, pitch_int);
            phase2  <<< dim3(round, 2)      , dim3(32,32), 2*64*64*sizeof(int) >>>(Dist_gpu[thread_id], r, n, pitch_int);
            phase3  <<< grid3, dim3(32,32), 2*64*64*sizeof(int) >>>(Dist_gpu[thread_id], r, n, pitch_int, thread_id, round);
            
                
            
            
                 
             

        }

        #pragma omp barrier
        if(thread_id == 0)
            cudaMemcpy(Dist, Dist_gpu[0], (round/2) * B * n * sizeof(int), cudaMemcpyDeviceToHost);
        else if(thread_id == 1)
            cudaMemcpy(&Dist[(round/2) * B * n], &Dist_gpu[1][(round/2) * B * n], (n - (round/2) * B) * n * sizeof(int), cudaMemcpyDeviceToHost);

    }
}

__global__ 
void phase1(int* Dist_gpu, int r, int n, int pitch_int) {

    extern __shared__ int shared_mem[]; 

    int sdx = (threadIdx.y * 64) + threadIdx.x;

    shared_mem[sdx]      = Dist_gpu[(r * 64 + threadIdx.y)     *pitch_int + r * 64 + threadIdx.x];
    shared_mem[sdx+32]   = Dist_gpu[(r * 64 + threadIdx.y)     *pitch_int + r * 64 + threadIdx.x + 32];
    shared_mem[sdx+2048] = Dist_gpu[(r * 64 + threadIdx.y + 32)*pitch_int + r * 64 + threadIdx.x];
    shared_mem[sdx+2080] = Dist_gpu[(r * 64 + threadIdx.y + 32)*pitch_int + r * 64 + threadIdx.x + 32];

    //__syncthreads();

    for(int k=0; k < 64; ++k){
        __syncthreads();
        shared_mem[sdx]      = min(shared_mem[sdx]     , shared_mem[threadIdx.y * 64 + k]    + shared_mem[k*64+threadIdx.x]);
        shared_mem[sdx+32]   = min(shared_mem[sdx+32]  , shared_mem[threadIdx.y * 64 + k]    + shared_mem[k*64+threadIdx.x + 32]);
        shared_mem[sdx+2048] = min(shared_mem[sdx+2048], shared_mem[(threadIdx.y+32)*64 + k] + shared_mem[k*64+threadIdx.x]);
        shared_mem[sdx+2080] = min(shared_mem[sdx+2080], shared_mem[(threadIdx.y+32)*64 + k] + shared_mem[k*64+threadIdx.x + 32]);
    }

    Dist_gpu[(r * 64  + threadIdx.y)     *pitch_int + r * 64 + threadIdx.x]       = shared_mem[sdx];
    Dist_gpu[(r * 64  + threadIdx.y)     *pitch_int + r * 64 + threadIdx.x + 32]  = shared_mem[sdx+32];
    Dist_gpu[(r * 64  + threadIdx.y + 32)*pitch_int + r * 64 + threadIdx.x]       = shared_mem[sdx+2048];
    Dist_gpu[(r * 64  + threadIdx.y + 32)*pitch_int + r * 64 + threadIdx.x + 32]  = shared_mem[sdx+2080];
}

__global__ void phase2(int* Dist_gpu, int r, int n, int pitch_int) {

    extern __shared__ int shared_mem[];

    int sdx = threadIdx.y * 64 + threadIdx.x;
    int b_i, b_j, i, j;

    if(blockIdx.y == 0){
        b_i = blockIdx.x;
        b_j = r;

        if(b_i == r) return;

        i = b_i * 64 + threadIdx.y;
        j = b_j * 64 + threadIdx.x;

        shared_mem[sdx]                    = Dist_gpu[i                             * pitch_int + j      ]; // IK
        shared_mem[sdx + 32]               = Dist_gpu[i                             * pitch_int + j + 32 ]; // IK
        shared_mem[sdx + 4096]             = Dist_gpu[(r*64 + threadIdx.y)      * pitch_int + j      ]; // KJ
        shared_mem[sdx + 4128]             = Dist_gpu[(r*64 + threadIdx.y)      * pitch_int + j + 32 ]; // KJ

        shared_mem[sdx + 2048]             = Dist_gpu[(i + 32)                      * pitch_int + j      ];
        shared_mem[sdx + 2080]             = Dist_gpu[(i + 32)                      * pitch_int + j + 32 ];
        shared_mem[sdx + 2048 + 4096]      = Dist_gpu[(r*64 + threadIdx.y + 32) * pitch_int + j      ];
        shared_mem[sdx + 2080 + 4096]      = Dist_gpu[(r*64 + threadIdx.y + 32) * pitch_int + j + 32 ];

        #pragma unroll
        for (int k = 0; k < 64; ++k) {
            __syncthreads();
            
            shared_mem[sdx]      = min(shared_mem[sdx],      shared_mem[threadIdx.y*64+k] + shared_mem[k*64+threadIdx.x + 4096]);
            shared_mem[sdx + 32] = min(shared_mem[sdx + 32], shared_mem[threadIdx.y*64+k] + shared_mem[k*64+threadIdx.x + 4128]);

            shared_mem[sdx + 2048] = min(shared_mem[sdx+2048], shared_mem[(threadIdx.y + 32)*64+k] + shared_mem[k*64+threadIdx.x + 4096]);
            shared_mem[sdx + 2080] = min(shared_mem[sdx+2080], shared_mem[(threadIdx.y + 32)*64+k] + shared_mem[k*64+threadIdx.x + 4128]);
        }

        Dist_gpu[i       *pitch_int + j     ] = shared_mem[sdx            ];  
        Dist_gpu[i       *pitch_int + j + 32] = shared_mem[sdx + 32       ];
        Dist_gpu[(i + 32)*pitch_int + j     ] = shared_mem[sdx + 2048     ];  
        Dist_gpu[(i + 32)*pitch_int + j + 32] = shared_mem[sdx + 2048 + 32];
    }else if(blockIdx.y == 1){
        b_i = r;
        b_j = blockIdx.x;

        if(b_j == r) return;

        i = b_i * 64 + threadIdx.y;
        j = b_j * 64 + threadIdx.x;

        shared_mem[sdx]                    = Dist_gpu[i*pitch_int + j];
        shared_mem[sdx + 32]               = Dist_gpu[i*pitch_int + j + 32];
        shared_mem[sdx + 4096]             = Dist_gpu[i*pitch_int + r * 64 + threadIdx.x];
        shared_mem[sdx + 4096 + 32]        = Dist_gpu[i*pitch_int + r * 64 + threadIdx.x + 32];

        shared_mem[sdx + 2048]             = Dist_gpu[(i + 32)*pitch_int + j];
        shared_mem[sdx + 2080]             = Dist_gpu[(i + 32)*pitch_int + j + 32];
        shared_mem[sdx + 2048 + 4096]      = Dist_gpu[(i + 32)*pitch_int + r * 64 + threadIdx.x];
        shared_mem[sdx + 2080 + 4096]      = Dist_gpu[(i + 32)*pitch_int + r * 64 + threadIdx.x + 32];



        #pragma unroll
        for (int k = 0; k < 64; ++k) {
            __syncthreads();

            
            shared_mem[sdx] = min(shared_mem[sdx], shared_mem[threadIdx.y*64+k+4096] + shared_mem[k*64+threadIdx.x]);
            shared_mem[sdx + 32] = min(shared_mem[sdx + 32], shared_mem[threadIdx.y*64+k+4096] + shared_mem[k*64+threadIdx.x + 32]);

            
            shared_mem[sdx+2048] = min(shared_mem[sdx+2048], shared_mem[(threadIdx.y+32)*64+k+4096] + shared_mem[k*64+threadIdx.x]);
            shared_mem[sdx+2080] = min(shared_mem[sdx+2080], shared_mem[(threadIdx.y+32)*64+k+4096] + shared_mem[k*64+threadIdx.x + 32]);
            
        }

        Dist_gpu[i*pitch_int + j]      = shared_mem[sdx];  
        Dist_gpu[i*pitch_int + j + 32] = shared_mem[sdx + 32];

        Dist_gpu[(i + 32)*pitch_int + j]      = shared_mem[sdx+2048];  
        Dist_gpu[(i + 32)*pitch_int + j + 32] = shared_mem[sdx+2048 + 32];
    }
    
    

    
}

__global__ void phase3(int* Dist_gpu, int r, int n, int pitch_int, int thread_id, int Round) {

    int b_i = blockIdx.x;
    int b_j = blockIdx.y;

    if(thread_id == 1) b_i += (Round/2);
   
    if(b_i == r || b_j == r) return;
    if(b_i == r || b_j == r) return;
    
    int i = b_i * 64 + threadIdx.y;
    int j = b_j * 64 + threadIdx.x;

    if (i >= n || j >= n) return;

    extern __shared__ int shared_mem[];
    
    int d0 = Dist_gpu[i*pitch_int + j];
    int d1 = Dist_gpu[i*pitch_int + j + 32];
    int d2 = Dist_gpu[(i+32)*pitch_int + j];
    int d3 = Dist_gpu[(i+32)*pitch_int + j + 32];
    
    int sdx = threadIdx.y * 64 + threadIdx.x;

    shared_mem[ sdx ]       = Dist_gpu[i*pitch_int + r * 64 + threadIdx.x];
    shared_mem[ sdx + 32]   = Dist_gpu[i*pitch_int + r * 64 + threadIdx.x + 32];
    shared_mem[ sdx + 4096] = Dist_gpu[(r * 64 + threadIdx.y)*pitch_int + j];
    shared_mem[ sdx + 4128] = Dist_gpu[(r * 64 + threadIdx.y)*pitch_int + j + 32];

    sdx += 2048;

    shared_mem[ sdx ]       = Dist_gpu[(i + 32)*pitch_int + r * 64 + threadIdx.x];
    shared_mem[ sdx + 32]   = Dist_gpu[(i + 32)*pitch_int + r * 64 + threadIdx.x + 32];
    shared_mem[ sdx + 4096] = Dist_gpu[(r * 64 + threadIdx.y + 32)*pitch_int + j];
    shared_mem[ sdx + 4128] = Dist_gpu[(r * 64 + threadIdx.y + 32)*pitch_int + j + 32];

    __syncthreads();
    
    #pragma unroll
    for (int k = 0; k < 64; ++k) {
        int idx = threadIdx.y * 64 + k;
        int v0 = shared_mem[idx]        + shared_mem[k*64 + threadIdx.x + 4096];
        int v1 = shared_mem[idx]        + shared_mem[k*64 + threadIdx.x + 4128];
        int v2 = shared_mem[idx + 2048] + shared_mem[k*64 + threadIdx.x + 4096];
        int v3 = shared_mem[idx + 2048] + shared_mem[k*64 + threadIdx.x + 4128];
        d0 = min(d0, v0);
        d1 = min(d1, v1);
        d2 = min(d2, v2);
        d3 = min(d3, v3);
    }

    Dist_gpu[i*pitch_int + j]           = d0;
    Dist_gpu[i*pitch_int + j + 32]      = d1;
    Dist_gpu[(i+32)*pitch_int + j]      = d2;
    Dist_gpu[(i+32)*pitch_int + j + 32] = d3;

}

#include<cuda.h>
#include<cuda_runtime.h>
#include<stdio.h>
#include<stdlib.h>

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) {
            exit(code);
        }
    }
}

__global__ void reduce_baseline(int* sum, int *input, size_t n)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    printf("id = %d\n", id);
    // input: global memory
    // sum: global memory
    int s = 0;
    // s: reg
    for(int i = 0; i < n; ++i) 
    {
        s += input[i];
    }
    // reg write into global
    *sum = s;
}


static bool check(int cpu_res, int gpu_res) {
    return cpu_res == gpu_res;
}

// * host端申请内存并初始化数据
// * device端申请内存
// * host端数据拷贝到device端
// * (计时开始)
// * 启动CUDA kernel
// * (计时结束)
// * device端把结果拷贝回host端
// * 检查device端计算结果和host端计算结果
// * 释放host和device端内存

int main()
{
    const int N = 25600000;
    int *da = nullptr, *ha = nullptr;
    int *ds = nullptr;
    // int hs = 0;
        // tips: cudaMemcpy 当host端内存作为dst时, host内存必须位于堆上; 不能位于栈上
    int *hs = nullptr;
    // float h_res = 0;
    int h_res = 0;
    
    ha = (int*)malloc(N * sizeof(int));         // cpu host mem
    hs = (int*)malloc(sizeof(int));
    gpuErrchk(cudaMalloc(&da, N * sizeof(int)));  // gpu global mem
    gpuErrchk(cudaMalloc(&ds, sizeof(int)));
    for(int i = 0; i < N; ++i) {
        ha[i] = 1;
        h_res += ha[i];
    }
    // printf("ha %p; da %p.\n", ha, da);
    cudaMemcpy(da, ha, N * sizeof(int), cudaMemcpyHostToDevice);  
    
    int grid_shape = 1;
    int block_shape = 1;
    cudaEvent_t start, stop;
    float milliseconds = 0;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    reduce_baseline<<<grid_shape, block_shape>>>(ds, da, N); 
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);

    // check
    gpuErrchk(cudaMemcpy(hs, ds, sizeof(int), cudaMemcpyDeviceToHost));
    if (!check(h_res, *hs)) {
        return 0;
    }
    printf("cpu: %d, gpu: %d\n", h_res, *hs);
    printf("reduce_baseline latency: %.lf ms\n", milliseconds);
    // latency: 562 ms

    cudaFree(da);
    cudaFree(ds);
    free(ha);
    free(hs);
}
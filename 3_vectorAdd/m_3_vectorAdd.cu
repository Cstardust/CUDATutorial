#include <iostream>
#include <cuda.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <stdio.h>

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) {
            exit(code);
        }
    }
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
const int maxn = 10000;
void vec_add_cpu(float *hx, float *hy, float* hz)
{
    for(int i = 0; i < maxn; ++i)
    {
        hz[i] = hx[i] + hy[i];
    }
}

// __global__ 代表要在CUDA上运行该函数
__global__ void vec_add_gpu(float *dx, float *dy, float* dz)
{
    // gridDim.x: grid在x维度上有多少block
    // blockIdx.x: block在grid中处于x方向上的第几个
    // threadIdx.x: thread在block中处于x方向上的第几个
    int id = (gridDim.x * blockIdx.y + blockIdx.x) * blockDim.x + threadIdx.x;
    if (id >= maxn) {
        return ;
    }
    dz[id] = dx[id] + dy[id];
}

int main()
{
    int nbytes = sizeof(float) * maxn;
    cudaError_t err = cudaError::cudaSuccess;

    // host
    // * host端申请内存并初始化数据
    float *hx = static_cast<float*>(malloc(nbytes));
    float *hy = static_cast<float*>(malloc(nbytes));
    float *hz = static_cast<float*>(malloc(nbytes));
    float *hk = static_cast<float*>(malloc(nbytes));
    for(int i = 0; i < maxn; ++i) {
        hx[i] = i;
        hy[i] = i;
        hz[i] = 0;
        hk[i] = 0;
    }
    vec_add_cpu(hx, hy, hz);
    
    // cuda
    // * thread结构
    int bt_x = 256;                  // 1个block在x维度上有多少thread
    int gb_x = 0, gb_y = 0;          // 1个grid在x维度上有多少block
    gb_x = gb_y = ceil(sqrt((maxn + bt_x - 1) / bt_x));
    dim3 grid(gb_x, gb_y);
        
    // * device端申请内存
    float *dx = nullptr, *dy = nullptr, *dz = nullptr;
    err = cudaMalloc(&dx, nbytes);
    gpuErrchk(err);
    cudaMalloc(&dy, nbytes);
    cudaMalloc(&dz, nbytes);
    // * host端数据拷贝到device端
    cudaMemcpy(dx, hx, nbytes, cudaMemcpyHostToDevice);
    cudaMemcpy(dy, hy, nbytes, cudaMemcpyHostToDevice);

    // CUDA Kernel计时API
    float milliseconds;
    cudaEvent_t start, stop;    // * 
    cudaEventCreate(&start);    // **
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    vec_add_gpu<<<grid, bt_x>>>(dx, dy, dz);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop); // 让cpu等待这个stop点打下去
    cudaEventElapsedTime(&milliseconds, start, stop);

    // Ques: 我觉得这里应该*3. 因为add_gpu每次运算都读取了3个内存数据, 所以应该*3.
    // 但是示例没有*3.
    // 不过我不知道这三个数组的内存都位于哪里？应该是一个地方？暂且猜测是private memory.
    printf("Mem BandWitdh %lf GB/s\n", nbytes * 3. * 1000 / 1024 / 1024 / 1024 / milliseconds);
    
    // device端把结果拷贝回host端
    // 这里是GPU自己保证的吗? memcpy的时候, gpu上关于该块内存的运算已经做完?
    cudaMemcpy(hk, dz, nbytes, cudaMemcpyDeviceToHost);

    // check
    for(int i = 0; i < maxn; ++i)
    {
        // check float是否相等. 不是通过等号, 而是通过相减! 因为浮点数有误差!
        if (fabs(hk[i] - hz[i]) > 1e-6) {
            fprintf(stderr, "error!, hk[%d](%lf) != hz[%d](%lf)", i, hk[i], i, hz[i]);
            exit(-1);
        }
    }
    printf("Result right!\n");
    // * 释放host和device端内存
    cudaFree(dx);
    cudaFree(dy);
    cudaFree(dz);
    free(hx);
    free(hy);
    free(hz);
    free(hk);
    return 0;
}
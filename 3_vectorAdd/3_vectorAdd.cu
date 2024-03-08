#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>

using std::cout;
using std::endl;

typedef float FLOAT;

/* CUDA kernel function */
__global__ void vec_add(FLOAT *x, FLOAT *y, FLOAT *z, int N)
{
    /* 2D grid */
    int idx = (blockDim.x * (blockIdx.x + blockIdx.y * gridDim.x) + threadIdx.x);
    /* 1D grid */
    // int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < N) z[idx] = y[idx] + x[idx];
}

void vec_add_cpu(FLOAT *x, FLOAT *y, FLOAT *z, int N)
{
    for (int i = 0; i < N; i++) z[i] = y[i] + x[i];
}

int main()
{
    int N = 10000;
    int nbytes = N * sizeof(FLOAT);

    /* 1D block */
    int bs = 256;

    /* 2D grid */
    // N + bs - 1 / bs: 计算需要多少block. (+bs-1)是为了向上取整. (N/bs是向下取整, 加一个bs-1来向上取整. 防止block数量不够组成那么多线程.)
    // ceil(...): 也是为了向上取整. 防止组成的block数量不够N + bs - 1 / bs
    // 比如bs = 5, N = 37 -> N + bs - 1 / bs = 8. 至少需要8个block才能组成足够的线程. 只是最后1个block会有空余thread没被使用.
    // sqrt(8) = 2.xxx -> ceil(sqrt) = 3. -> 最后是3*3=9个thread
    int s = ceil(sqrt((N + bs - 1.) / bs));
    cout<<s<<endl;
    // 定义套分配的grid形状
    dim3 grid(s, s);
    /* 1D grid */
    // int s = ceil((N + bs - 1.) / bs);
    // dim3 grid(s);

    FLOAT *dx, *hx;     // hx: host_x. CPU上的x向量; dx: dimesion. GPU上的x向量
    FLOAT *dy, *hy;
    FLOAT *dz, *hz;

    /* allocate GPU mem */
    cudaMalloc((void **)&dx, nbytes);
    cudaMalloc((void **)&dy, nbytes);
    cudaMalloc((void **)&dz, nbytes);
    
    /* init time */
    float milliseconds = 0;

    /* alllocate CPU mem */
    hx = (FLOAT *) malloc(nbytes);
    hy = (FLOAT *) malloc(nbytes);
    hz = (FLOAT *) malloc(nbytes);

    /* init */
    for (int i = 0; i < N; i++) {
        hx[i] = 1;
        hy[i] = 1;
        if (hz[i] != 0) {
            cout<<i<<" "<<hz[i]<<endl;
        }
    }

    /* copy data to GPU */
    cudaMemcpy(dx, hx, nbytes, cudaMemcpyHostToDevice);
    cudaMemcpy(dy, hy, nbytes, cudaMemcpyHostToDevice);

    // start, stop 计时
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    /* launch GPU kernel */
    // grid: 描述grid内有多少block
    // bs: 描述block内有多少thread
    vec_add<<<grid, bs>>>(dx, dy, dz, N);   // dz[i] = dx[i] + dy[i]
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);  
    
	/* copy GPU result to CPU */
    // 将结果从dz拷贝到hz
    cudaMemcpy(hz, dz, nbytes, cudaMemcpyDeviceToHost);

    /* CPU compute */
    FLOAT* hz_cpu_res = (FLOAT *) malloc(nbytes);
    vec_add_cpu(hx, hy, hz_cpu_res, N);

    /* check GPU result with CPU*/
    for (int i = 0; i < N; ++i) {
        if (fabs(hz_cpu_res[i] - hz[i]) > 1e-6) {
            printf("Result verification failed at element index %d!\n", i);
        }
    }
    printf("Result right\n");
    printf("Mem BW= %f (GB/sec)\n", (float)N*4/milliseconds/1e6);///1.78gb/s
    cudaFree(dx);
    cudaFree(dy);
    cudaFree(dz);

    free(hx);
    free(hy);
    free(hz);
    free(hz_cpu_res);

    return 0;
}
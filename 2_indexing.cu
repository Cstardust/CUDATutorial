#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>

using std::cout;
using std::endl;

__global__ void sum(float *x)
{
    // 泛指当前block在所有block范围内的id
    int block_id = blockIdx.x;
    // 泛指当前线程在所有block范围内的全局id
    int global_tid = block_id * blockDim.x + threadIdx.x;   // 0 * 32 + idx[0,31]
    // 泛指当前线程在其block内的id
    int local_tid = threadIdx.x;
    printf("current block=%d, thread id in current block =%d, global thread id=%d\n", block_id, local_tid, global_tid);
    x[global_tid] += 1;
}

int main(){
    int N = 32;
    int nbytes = N * sizeof(float);
    float *dx, *hx;
    /* allocate GPU mem */
    cudaMalloc((void **)&dx, nbytes);//思考为什么要用二级指针
    /* allocate CPU mem */
    hx = (float*) malloc(nbytes);
    /* init host data */
    printf("hx original: \n");
    for (int i = 0; i < N; i++) {
        hx[i] = i;
        printf("%g ", hx[i]);
    }
    cout<<endl;
    /* copy data to GPU */
    cudaMemcpy(dx, hx, nbytes, cudaMemcpyHostToDevice);
    /* launch GPU kernel */
    sum<<<1, N>>>(dx);
    // 这里不需要等待同步吗? cudaMemcpy的语义本身就带了同步吗? 等到所有CUDA thread执行完, 才会执行这一句吗?
    /* copy data from GPU */
    cudaMemcpy(hx, dx, nbytes, cudaMemcpyDeviceToHost);
    printf("hx current: \n");
    for (int i = 0; i < N; i++) {
        printf("%g ", hx[i]);
    }
    cout<<endl;
    cudaFree(dx);
    free(hx);
    return 0;
}

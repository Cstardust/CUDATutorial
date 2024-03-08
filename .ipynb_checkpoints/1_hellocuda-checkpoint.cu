#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>

// global: CUDA kernel函数前缀, 被global修饰的函数 被CPU调用启动, 在GPU上运行
// blockIdx.x: 在x维度上, block是grid中的第几个block 
// blockDim.x: 1个block在x维度上有多少线程;       
// threadIdx.x: 该thread在其所属block内的id
__global__ void hello_cuda(){
    // 当前thread在所有block范围内的全局id
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    printf("blockIdx.x [%d], blockDim.x [%d], threadIdx.x [%d], block id = [%d], thread id = [%d] hello cuda\n", blockIdx.x, blockDim.x, threadIdx.x, blockIdx.x, idx);
}

int main() {
    // <<< 1, 1 >>>: 启动CUDA Kernel的标志
    // 第一个1代表分配几个block
    // 第二个1代表1个block内分配几个线程
    // 去做hello_cuda这件事
    hello_cuda<<< 1, 1 >>>();
    
    // 同步: CPU等待GPU上的CUDA kernel执行
    // 不写的话就是CPU可能会比GPU先执行完
    cudaDeviceSynchronize();
    return 0;
}

// 打印多行hello cuda?: 分配的block数量和thread数量>1
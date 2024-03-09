#include <cuda_runtime.h>
#include <cuda.h>
#include <iostream>
#include <string>

int main() {
  int deviceCount = 0;
  // 获取当前机器的GPU数量
  cudaError_t error_id = cudaGetDeviceCount(&deviceCount);
  if (deviceCount == 0) {
    printf("There are no available device(s) that support CUDA\n");
  } else {
    printf("Detected %d CUDA Capable device(s)\n", deviceCount);
  }
  for (int dev = 0; dev < deviceCount; ++dev) {
    cudaSetDevice(dev);
    // 初始化当前device的属性获取对象
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);

    printf("\nDevice %d: \"%s\"\n", dev, deviceProp.name);
    // 显存容量
    // ques: device的global memory应该是显卡上自带的, 与OS本身的DRAM不同.
    printf("  Total amount of global memory:                 %.0f MBytes "
             "(%llu bytes)\n",
             static_cast<float>(deviceProp.totalGlobalMem / 1048576.0f),
             (unsigned long long)deviceProp.totalGlobalMem);
    // 时钟频率
    printf( "  GPU Max Clock rate:                            %.0f MHz (%0.2f "
        "GHz)\n",
        deviceProp.clockRate * 1e-3f, deviceProp.clockRate * 1e-6f);
    // L1, L2 cache 与 private memory, shared memory, global memory之间的关系是?
    // L2 cache大小
    printf("  L2 Cache Size:                                 %d bytes\n",
             deviceProp.l2CacheSize);
    // high-frequent used
    // 注释见每个printf内的字符串
    printf("  Total amount of shared memory per block:       %zu bytes\n",   // 49152
           deviceProp.sharedMemPerBlock);
    printf("  Total shared memory per multiprocessor:        %zu bytes\n",   // 98304. 所以一个SM上最多分配俩block
           deviceProp.sharedMemPerMultiprocessor);
    printf("  Total number of registers available per block: %d\n",          // 65536
           deviceProp.regsPerBlock);
    printf("  Warp size:                                     %d\n",   // 32 thread
           deviceProp.warpSize);
    printf("  Maximum number of threads per multiprocessor:  %d\n",   // 2048 thread
           deviceProp.maxThreadsPerMultiProcessor);
    printf("  Maximum number of threads per block:           %d\n",   // 1024 thread. 也即一个block最多可设定成1024个thread
           deviceProp.maxThreadsPerBlock);
    printf("  Max dimension size of a block size (x,y,z): (%d, %d, %d)\n",   // (1024, 1024, 64) block三个维度的最大值
           deviceProp.maxThreadsDim[0], deviceProp.maxThreadsDim[1],
           deviceProp.maxThreadsDim[2]);
    printf("  Max dimension size of a grid size    (x,y,z): (%d, %d, %d)\n", // grid三个维度的最大值
           deviceProp.maxGridSize[0], deviceProp.maxGridSize[1],
           deviceProp.maxGridSize[2]);
// Detected 1 CUDA Capable device(s)

// Device 0: "Tesla V100-SXM2-16GB"
//   Total amount of global memory:                 16160 MBytes (16945512448 bytes)
//   GPU Max Clock rate:                            1530 MHz (1.53 GHz)
//   L2 Cache Size:                                 6291456 bytes
//   Total amount of shared memory per block:       49152 bytes
//   Total shared memory per multiprocessor:        98304 bytes
//   Total number of registers available per block: 65536
//   Warp size:                                     32
//   Maximum number of threads per multiprocessor:  2048
//   Maximum number of threads per block:           1024
//   Max dimension size of a block size (x,y,z): (1024, 1024, 64)
//   Max dimension size of a grid size    (x,y,z): (2147483647, 65535, 65535)
  }
  return 0;
}
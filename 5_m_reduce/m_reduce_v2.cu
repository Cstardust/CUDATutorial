#include<cuda.h>
#include<cuda_runtime.h>
#include<stdio.h>
#include<cassert>


// v0:
    // 引入并行 + shared mem
    // 并行
        // 每个block之间并行
        // 对于每个block内部的thread, 在不同迭代之间是串行的, 但是在每次迭代内部, thread是并行的.
    // shared mem
        // baseline中, 每次加法都是对global mem进行读取, 故我们现在将其转移至shared mem.
        // 每个block都有一个shared mem, 且大小一般为64KB. 所以256 * 4 / 1024 = 1KB, 完全够用.
        // 由于__shared__变量大小要在编译期间确定, 所以模板偏特化来指明__shared__数组大小.

// v1: 
    // 消除divergence

// v2:
    // 消除bank conflict
template<int block_size>
__global__ void reduce_v2(float *input, float *output, const int N)     // 这个N必须传值, 不能传常量左值引用！不然会有未定义事情发生！! 找了半天才找到！
{
    // block_size = blockDim.x: 但是blockDim.x是在运行时确定, 所以我们需要通过模板传进来.
    // assert(block_size == blockDim.x); // 256                      
    // shared memory
    __shared__ float sm[block_size];
    // grid 一维; block 一维
    int gid = blockIdx.x * blockDim.x + threadIdx.x;        // 总的thread id. 也是当前线程对应的data在input里的索引
    int tid = threadIdx.x;

    // 保存到shared mem中. 避免迭代的时候反复存取global mem
    if (gid >= N) {
        sm[tid] = 0;
    } else  {
        sm[tid] = input[gid];
    }
    // 等待block内的所有thread对shared mem的load和store完成
    __syncthreads();


    // i: 代表第x轮迭代, 所求和的data的间距为2^x.
    // 最终结果存储到0号位.
    // 这些都是在shared memory上进行计算.

    // v0: warp divergence
        // thread tid就对应data[tid]
        // 每轮迭代, 每个thread tid处理两个data: data[tid] 和 data[tid+i]
        // 每轮迭代中, 工作的thread散落在各个warp里面
    // for(int i = 1; i < blockDim.x; i *= 2) {
        // 这样每次都会间隔几个tid, 才会有一个thread进行工作.
    //     if (tid % (i * 2) == 0 && (tid + i) < blockDim.x) {
    //         sm[tid] += sm[tid + i];
    //     }
    //     __syncthreads();        // 应该可以放在分支里面, 因为threadId一次没计算, 剩下的迭代也不会参与. 没进入分支的可以直接return.
    // }

    // v1 after v0: 消除warp divergence
        // thread tid不对应data[tid], 而是直接对应出要处理的两个data的idx, 并对其进行计算
        // 这样, 每轮迭代中, 工作的thread都紧挨在一起
    // 迭代的还是thread处理数据的间隔
    // for (int i = 1; i < blockDim.x; i *= 2) {
    //     int data_idx = 2 * i * tid;             // 计算出当前tid对应处理的data下标.
    //     if (data_idx + i < blockDim.x) {
    //         sm[data_idx] += sm[data_idx + i]; 
    //     }
    //     __syncthreads();
    // }

    // v2 after v1: 消除bank conflict
        // 处理data的间隔反过来. 之前是 1 -> 128; 现在变成 128 -> 1
        // i是间隔, 也是本轮归并data的起点临界, 也是本轮最后一个活跃的thread id). [第一个不活跃的thread id.
        // 如此, 每个warp内的不同thread, 不会读取同一bank.
    for (int i = blockDim.x / 2; i > 0; i >>= 1) {
        if (tid < i) {
            sm[tid] += sm[tid + i];  
        }
        __syncthreads();    // 别忘了. 下一轮迭代会依赖上一轮的结果. 所以本轮要等所有thread都读写shared mem完成. latency 0.045 ms
    }

    if (tid != 0)
        return ;
    
    // 将结果从shared mem写回global memory
    // 每个block的thread计算出来一个结果, 存入output[blockIdx.x].
    // 最后cpu再把所有block的结果合并到一起
    output[blockIdx.x] = sm[0];

    // printf("block[%d], thread[%d], s = %.lf\n", blockIdx.x, tid, output[blockIdx.x]);
}


bool check(float cpu_res, float *gpu_res, const int n) {
    float gpu_merge = 0;
    for(int i = 0; i < n; ++i) {
        gpu_merge += gpu_res[i];    
    }
    printf("cpu_res == %.lf; gpu_res == %.lf\n", cpu_res, gpu_merge);
    return cpu_res == gpu_merge;    
}

int main()
{
    cudaSetDevice(0);
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);

    const int N = 25600000;
    int nbytes = N * sizeof(float);
    const int block_size = 256;     // const修饰才能在编译时已知
    int grid_size = min((N + block_size - 1) / block_size, deviceProp.maxGridSize[0]);      // min(10000, 2147483647)
    
    float *ha = (float*)malloc(nbytes);
    float *hs = (float*)malloc(grid_size * sizeof(float));
    float h_res = N * 1.0f;
    for (int i = 0; i < N; ++i) {
        ha[i] = 1;
        // h_res += ha[i];
    }

    float *da = nullptr;
    float *ds = nullptr;
    cudaMalloc(&da, nbytes);
    cudaMalloc(&ds, grid_size * sizeof(float));
    cudaMemcpy(da, ha, nbytes, cudaMemcpyHostToDevice);

    dim3 grid(grid_size);
    dim3 block(block_size);
    cudaEvent_t start, stop;
    float milliseconds = 0;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    reduce_v2<block_size><<<grid, block>>>(da, ds, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaMemcpy(hs, ds, grid_size * sizeof(float), cudaMemcpyDeviceToHost);
    printf("allcated %d blocks, thread %d per block, data counts are %d\n", grid_size, block_size, N);

    if (!check(h_res, hs, grid_size)) {
        printf("bad ans!\n");
    } else {
        printf("good ans!\n");
        printf("latency %.3lf ms\n", milliseconds);
    }

    // allcated 10000 blocks, thread 256 per block, data counts are 2560000
    // cpu_res == 2560000; gpu_res == 2560000
    // good ans!
    // latency 0.298 ms
    return 0;
}
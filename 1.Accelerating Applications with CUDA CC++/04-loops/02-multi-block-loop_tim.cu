#include <stdio.h>

__global__ void loop()
{
    // 計算全域索引
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    printf("blockIdx.x: %d, blockDim.x: %d, threadIdx.x: %d, Total: %d\n", blockIdx.x, blockDim.x, threadIdx.x, i);
    printf("%d\n", i);
}

int main()
{
    
    // 例如：使用 2 個 blocks，每個 block 5 個 threads
    loop<<<2, 5>>>();
    
    cudaDeviceSynchronize();
    return 0;
}
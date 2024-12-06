#include <stdio.h>
#include <assert.h>

#define N 64
#define BLOCK_SIZE 16

// 錯誤檢查包裝器
inline cudaError_t checkCuda(cudaError_t result)
{
    if (result != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s\n", cudaGetErrorString(result));
        assert(result == cudaSuccess);
    }
    return result;
}

__global__ void matrixMulGPU(const int *a, const int *b, int *c)
{
    // 計算全局索引
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    
    // 累加變數
    int sum = 0;
    
    if (row < N && col < N)  // 邊界檢查
    {
        // 矩陣乘法計算
        for (int k = 0; k < N; ++k)
        {
            sum += a[row * N + k] * b[k * N + col];
        }
        c[row * N + col] = sum;
    }
}

int main()
{
    int *a, *b, *c_cpu, *c_gpu;
    size_t size = N * N * sizeof(int);
    
    // 分配統一記憶體
    checkCuda(cudaMallocManaged(&a, size));
    checkCuda(cudaMallocManaged(&b, size));
    checkCuda(cudaMallocManaged(&c_cpu, size));
    checkCuda(cudaMallocManaged(&c_gpu, size));
    
    // 初始化矩陣
    for(int i = 0; i < N; ++i)
        for(int j = 0; j < N; ++j)
        {
            a[i*N + j] = i;
            b[i*N + j] = j+2;
            c_cpu[i*N + j] = c_gpu[i*N + j] = 0;
        }
    
    // 設定執行配置
    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 numBlocks((N + BLOCK_SIZE - 1) / BLOCK_SIZE, 
                   (N + BLOCK_SIZE - 1) / BLOCK_SIZE);
    
    // 啟動核函數
    matrixMulGPU<<<numBlocks, threadsPerBlock>>>(a, b, c_gpu);
    
    // 檢查啟動錯誤
    checkCuda(cudaGetLastError());
    // 同步等待
    checkCuda(cudaDeviceSynchronize());
    
    // 釋放記憶體
    checkCuda(cudaFree(a));
    checkCuda(cudaFree(b));
    checkCuda(cudaFree(c_cpu));
    checkCuda(cudaFree(c_gpu));
    
    return 0;
}
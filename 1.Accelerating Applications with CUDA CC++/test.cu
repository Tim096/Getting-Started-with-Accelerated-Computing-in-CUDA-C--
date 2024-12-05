#include <stdio.h>
#include <cuda_runtime.h>

// 錯誤處理包裝函數
#define CHECK_CUDA_ERROR(err) \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d - %s\n", \
                __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    }

// Kernel 錯誤檢查包裝函數
#define CHECK_KERNEL_ERROR() \
    { \
        cudaError_t err = cudaGetLastError(); \
        if (err != cudaSuccess) { \
            fprintf(stderr, "Kernel launch error: %s\n", \
                    cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    }

// 示範1：記憶體越界錯誤的 kernel
__global__ void outOfBoundsKernel(int* data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // 故意造成越界訪問
    data[idx] = 42;  // 沒有檢查 idx 是否超過 size
}

// 示範2：正確的 kernel 實現
__global__ void safeKernel(int* data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {  // 添加邊界檢查
        data[idx] = 42;
    }
}

// 示範3：除零錯誤的 kernel
__global__ void divisionByZeroKernel(int* data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] = data[idx] / (idx % 2);  // 當 idx 為偶數時會發生除零錯誤
    }
}

int main() {
    int* d_data;
    const int size = 1000;
    const size_t bytes = size * sizeof(int);

    printf("=== CUDA 錯誤處理示範 ===\n\n");

    // 示範1：記憶體分配錯誤處理
    printf("1. 檢查記憶體分配錯誤...\n");
    cudaError_t err = cudaMalloc(&d_data, bytes);
    CHECK_CUDA_ERROR(err);
    printf("   記憶體分配成功\n\n");

    // 示範2：不安全的 kernel 啟動（可能的記憶體越界）
    printf("2. 執行不安全的 kernel（可能越界）...\n");
    outOfBoundsKernel<<<size, 2>>>(d_data, size);  // 故意使用過多線程
    CHECK_KERNEL_ERROR();
    
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("   檢測到執行時錯誤: %s\n", cudaGetErrorString(err));
    }
    printf("\n");

    // 示範3：安全的 kernel 啟動
    printf("3. 執行安全的 kernel...\n");
    safeKernel<<<(size + 255) / 256, 256>>>(d_data, size);
    CHECK_KERNEL_ERROR();
    err = cudaDeviceSynchronize();
    CHECK_CUDA_ERROR(err);
    printf("   安全的 kernel 執行成功\n\n");

    // 示範4：除零錯誤處理
    printf("4. 執行可能產生除零錯誤的 kernel...\n");
    divisionByZeroKernel<<<(size + 255) / 256, 256>>>(d_data, size);
    CHECK_KERNEL_ERROR();
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("   檢測到執行時錯誤: %s\n", cudaGetErrorString(err));
    }
    printf("\n");

    // 示範5：設備屬性檢查
    printf("5. 檢查設備屬性...\n");
    cudaDeviceProp prop;
    err = cudaGetDeviceProperties(&prop, 0);
    CHECK_CUDA_ERROR(err);
    printf("   使用設備: %s\n", prop.name);
    printf("   最大線程數/區塊: %d\n\n", prop.maxThreadsPerBlock);

    // 清理資源
    printf("6. 清理資源...\n");
    cudaFree(d_data);
    CHECK_CUDA_ERROR(cudaGetLastError());
    printf("   資源清理完成\n\n");

    return 0;
}
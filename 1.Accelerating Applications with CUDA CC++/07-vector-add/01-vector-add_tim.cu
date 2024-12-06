#include <stdio.h>
#include <assert.h>

// 錯誤檢查包裝函數
inline cudaError_t checkCuda(cudaError_t result) {
    if (result != cudaSuccess) {
        fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
        assert(result == cudaSuccess);
    }
    return result;
}

// CPU初始化函數
void initWith(float num, float *a, int N) {
    for(int i = 0; i < N; ++i) {
        a[i] = num;
    }
}

// GPU核函數
__global__ void addVectorsInto(float *result, float *a, float *b, int N) {
    // 計算全局索引
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    // 計算步距
    int stride = blockDim.x * gridDim.x;
    
    // 使用grid-stride loop處理大數據
    for(int i = index; i < N; i += stride) {
        result[i] = a[i] + b[i];
    }
}

// 結果驗證函數
void checkElementsAre(float target, float *array, int N) {
    for(int i = 0; i < N; i++) {
        if(array[i] != target) {
            printf("FAIL: array[%d] - %0.0f does not equal %0.0f\n", 
                   i, array[i], target);
            exit(1);
        }
    }
    printf("SUCCESS! All values added correctly.\n");
}

int main() {
    const int N = 2<<20;  // 約2M元素
    size_t size = N * sizeof(float);

    // 宣告指標
    float *a, *b, *c;

    // 配置統一記憶體
    checkCuda( cudaMallocManaged(&a, size) );
    checkCuda( cudaMallocManaged(&b, size) );
    checkCuda( cudaMallocManaged(&c, size) );

    // 初始化數據
    initWith(3, a, N);
    initWith(4, b, N);
    initWith(0, c, N);

    // 設定執行組態
    size_t threadsPerBlock = 256;
    size_t numberOfBlocks = (N + threadsPerBlock - 1) / threadsPerBlock;

    // 啟動核函數
    addVectorsInto<<<numberOfBlocks, threadsPerBlock>>>(c, a, b, N);

    // 錯誤檢查
    checkCuda( cudaGetLastError() );
    // 同步等待
    checkCuda( cudaDeviceSynchronize() );

    // 驗證結果
    checkElementsAre(7, c, N);

    // 釋放記憶體
    checkCuda( cudaFree(a) );
    checkCuda( cudaFree(b) );
    checkCuda( cudaFree(c) );

    return 0;
}
#include <cuda_runtime.h>

// 定義分塊大小，可根據 GPU 架構調整
#define TILE_SIZE 32
#define BLOCK_SIZE 32

template<typename T>
__global__ void matrixMulKernel(const T* __restrict__ A,
                               const T* __restrict__ B,
                               T* __restrict__ C,
                               const int M, const int N, const int K) {
    // 分配共享記憶體
    __shared__ T As[TILE_SIZE][TILE_SIZE];
    __shared__ T Bs[TILE_SIZE][TILE_SIZE];
    
    // 計算全域索引
    const int row = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    const int col = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    
    // 累加結果
    T sum = 0.0f; // 宣告在暫存器中，每個執行緒都有自己的 sum
    
    // 分塊計算, 同時間會有 TILE_SIZE * TILE_SIZE 個線程
    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; t++) {
        // 協同載入 A 和 B 到共享記憶體
        if (row < M && t * TILE_SIZE + threadIdx.x < K) {
            As[threadIdx.y][threadIdx.x] = A[row * K + t * TILE_SIZE + threadIdx.x];
        } else {
            As[threadIdx.y][threadIdx.x] = 0.0f;
        }
        
        if (col < N && t * TILE_SIZE + threadIdx.y < K) {
            Bs[threadIdx.y][threadIdx.x] = B[(t * TILE_SIZE + threadIdx.y) * N + col];
        } else {
            Bs[threadIdx.y][threadIdx.x] = 0.0f;
        }
        
        // 同步確保資料載入完成
        __syncthreads();
        
        // 計算當前分塊的乘積
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }
        
        // 同步確保計算完成後再載入下一個分塊
        __syncthreads();
    }
    
    // 寫回結果
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

// Host 端包裝函數
template<typename T>
class MatrixMultiplier {
public:
    void multiply(const T* h_A, const T* h_B, T* h_C,
                 int M, int N, int K) {
        // 分配設備記憶體
        T *d_A, *d_B, *d_C;
        cudaMalloc(&d_A, M * K * sizeof(T));
        cudaMalloc(&d_B, K * N * sizeof(T));
        cudaMalloc(&d_C, M * N * sizeof(T));
        
        // 傳輸資料到 GPU
        cudaMemcpy(d_A, h_A, M * K * sizeof(T), cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, h_B, K * N * sizeof(T), cudaMemcpyHostToDevice);
        
        // 設定執行配置
        dim3 gridDim((N + BLOCK_SIZE - 1) / BLOCK_SIZE,
                    (M + BLOCK_SIZE - 1) / BLOCK_SIZE);
        // Threads per block : 同時處理的線程數
        dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
        
        // 建立 CUDA 串流以支援異步操作
        cudaStream_t stream;
        cudaStreamCreate(&stream);
        
        // 啟動核心 grid, block, 0 表示異步操作, stream 表示使用的串流
        // 異步操作 (非同步傳輸) : 不會等待傳輸完成
        matrixMulKernel<T><<<gridDim, blockDim, 0, stream>>>(
            d_A, d_B, d_C, M, N, K);
        
        // 傳輸結果回 CPU
        cudaMemcpyAsync(h_C, d_C, M * N * sizeof(T),
                       cudaMemcpyDeviceToHost, stream);
        
        // 同步和清理
        cudaStreamSynchronize(stream);
        cudaStreamDestroy(stream);
        
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
    }
    
    // 效能優化版本：使用多串流和重疊傳輸
    void multiplyAsync(const T* h_A, const T* h_B, T* h_C,
                      int M, int N, int K,
                      int numStreams = 4) {
        // 分配固定記憶體以支援非同步傳輸
        T *p_A, *p_B, *p_C;
        cudaMallocHost(&p_A, M * K * sizeof(T));
        cudaMallocHost(&p_B, K * N * sizeof(T));
        cudaMallocHost(&p_C, M * N * sizeof(T));
        
        // 複製輸入資料到固定記憶體
        memcpy(p_A, h_A, M * K * sizeof(T));
        memcpy(p_B, h_B, K * N * sizeof(T));
        
        // 分配設備記憶體
        T *d_A, *d_B, *d_C;
        cudaMalloc(&d_A, M * K * sizeof(T));
        cudaMalloc(&d_B, K * N * sizeof(T));
        cudaMalloc(&d_C, M * N * sizeof(T));
        
        // 建立串流
        std::vector<cudaStream_t> streams(numStreams);
        for (int i = 0; i < numStreams; ++i) {
            cudaStreamCreate(&streams[i]);
        }
        
        // 計算每個串流處理的列數
        int rowsPerStream = (M + numStreams - 1) / numStreams;
        
        // 分段處理
        for (int i = 0; i < numStreams; ++i) {
            int startRow = i * rowsPerStream;
            int rowsThis = std::min(rowsPerStream, M - startRow);
            
            if (rowsThis <= 0) continue;
            
            // 非同步傳輸和計算
            cudaMemcpyAsync(d_A + startRow * K,
                          p_A + startRow * K,
                          rowsThis * K * sizeof(T),
                          cudaMemcpyHostToDevice,
                          streams[i]);
            
            dim3 gridDim((N + BLOCK_SIZE - 1) / BLOCK_SIZE,
                        (rowsThis + BLOCK_SIZE - 1) / BLOCK_SIZE);
            dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
            
            matrixMulKernel<T><<<gridDim, blockDim, 0, streams[i]>>>(
                d_A + startRow * K, d_B,
                d_C + startRow * N,
                rowsThis, N, K);
            
            cudaMemcpyAsync(p_C + startRow * N,
                          d_C + startRow * N,
                          rowsThis * N * sizeof(T),
                          cudaMemcpyDeviceToHost,
                          streams[i]);
        }
        
        // 等待所有操作完成
        for (auto& stream : streams) {
            cudaStreamSynchronize(stream);
            cudaStreamDestroy(stream);
        }
        
        // 複製結果到輸出緩衝區
        memcpy(h_C, p_C, M * N * sizeof(T));
        
        // 清理資源
        cudaFreeHost(p_A);
        cudaFreeHost(p_B);
        cudaFreeHost(p_C);
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
    }
};
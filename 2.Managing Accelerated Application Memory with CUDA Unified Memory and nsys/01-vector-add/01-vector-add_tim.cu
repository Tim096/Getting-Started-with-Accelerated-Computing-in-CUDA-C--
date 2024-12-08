#include <stdio.h>
#include <assert.h>

/**
 * 錯誤處理輔助函數
 * 用於檢查 CUDA 操作的結果
 */
inline cudaError_t checkCuda(cudaError_t result)
{
  if (result != cudaSuccess) {
    fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
    assert(result == cudaSuccess);
  }
  return result;
}

/**
 * 向量初始化函數
 * @param num: 初始化的值
 * @param a: 要初始化的向量
 * @param N: 向量長度
 */
void initWith(float num, float *a, int N)
{
  for(int i = 0; i < N; ++i)
  {
    a[i] = num;
  }
}

/**
 * CUDA 核函數：向量加法
 * 使用 grid-stride loop 處理大規模數據
 */
__global__
void addVectorsInto(float *result, float *a, float *b, int N)
{
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  int stride = blockDim.x * gridDim.x;

  for(int i = index; i < N; i += stride)
  {
    result[i] = a[i] + b[i];
  }
}

/**
 * 結果驗證函數
 */
void checkElementsAre(float target, float *vector, int N)
{
  for(int i = 0; i < N; i++)
  {
    if(vector[i] != target)
    {
      printf("FAIL: vector[%d] - %0.0f does not equal %0.0f\n", i, vector[i], target);
      exit(1);
    }
  }
  printf("Success! All values calculated correctly.\n");
}

int main()
{
  const int N = 2<<24;  // 約1600萬個元素
  size_t size = N * sizeof(float);

  float *a, *b, *c;

  // 分配統一記憶體
  checkCuda(cudaMallocManaged(&a, size));
  checkCuda(cudaMallocManaged(&b, size));
  checkCuda(cudaMallocManaged(&c, size));

  // 初始化向量
  initWith(3, a, N);
  initWith(4, b, N);
  initWith(0, c, N);

  // 配置執行參數
  size_t threadsPerBlock = 256;  // 建議值
  size_t numberOfBlocks = (N + threadsPerBlock - 1) / threadsPerBlock;
  
  // 限制 block 數量避免超出硬體限制
  numberOfBlocks = numberOfBlocks > 65535 ? 65535 : numberOfBlocks;

  // 啟動核函數
  addVectorsInto<<<numberOfBlocks, threadsPerBlock>>>(c, a, b, N);

  // 錯誤處理
  checkCuda(cudaGetLastError());
  checkCuda(cudaDeviceSynchronize());

  // 驗證結果
  checkElementsAre(7, c, N);

  // 釋放記憶體
  checkCuda(cudaFree(a));
  checkCuda(cudaFree(b));
  checkCuda(cudaFree(c));

  return 0;
}
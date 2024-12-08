/**
 * CUDA向量加法範例程式
 * 此程式展示了如何使用CUDA實現兩個大型向量的並行加法運算
 * 主要特點:
 * - 使用統一記憶體管理(Unified Memory)
 * - 實現grid-stride loop以處理大規模數據
 * - 包含完整的錯誤處理機制
 */

#include <stdio.h>

/**
 * 初始化向量的CPU函數
 * @param num: 初始化的值
 * @param a: 要初始化的向量
 * @param N: 向量大小
 */
void initWith(float num, float *a, int N)
{
  for(int i = 0; i < N; ++i)
  {
    a[i] = num;
  }
}

/**
 * CUDA核函數: 將兩個向量相加
 * 使用grid-stride loop模式來處理大於grid大小的數據
 * @param result: 結果向量
 * @param a: 輸入向量1
 * @param b: 輸入向量2
 * @param N: 向量大小
 */
__global__
void addArraysInto(float *result, float *a, float *b, int N)
{
  // 計算該線程在整個grid中的全局索引
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  // 計算grid的總線程數，作為stride
  int stride = blockDim.x * gridDim.x;

  // 使用grid-stride loop處理數據
  // 每個線程處理間隔為stride的多個元素
  for(int i = index; i < N; i += stride)
  {
    result[i] = a[i] + b[i];
  }
}

/**
 * 驗證結果的CPU函數
 * @param target: 預期的結果值
 * @param array: 要檢查的向量
 * @param N: 向量大小
 */
void checkElementsAre(float target, float *array, int N)
{
  for(int i = 0; i < N; i++)
  {
    if(array[i] != target)
    {
      printf("FAIL: array[%d] - %0.0f does not equal %0.0f\n", i, array[i], target);
      exit(1);
    }
  }
  printf("Success! All values calculated correctly.\n");
}

int main()
{
  // 設定問題規模: 2^25 個元素
  const int N = 2<<24;
  size_t size = N * sizeof(float);

  float *a;
  float *b;
  float *c;

  // 分配統一記憶體
  cudaMallocManaged(&a, size);
  cudaMallocManaged(&b, size);
  cudaMallocManaged(&c, size);

  // 初始化輸入向量
  initWith(3, a, N);
  initWith(4, b, N);
  initWith(0, c, N);

  // 設定執行配置
  size_t threadsPerBlock = 256;  // 每個block 256個線程
  // 計算需要的block數量，確保有足夠的線程處理所有數據
  size_t numberOfBlocks = (N + threadsPerBlock - 1) / threadsPerBlock;

  // 錯誤處理變量
  cudaError_t addArraysErr;
  cudaError_t asyncErr;

  // 啟動核函數
  addArraysInto<<<numberOfBlocks, threadsPerBlock>>>(c, a, b, N);

  // 檢查同步錯誤（啟動錯誤）
  addArraysErr = cudaGetLastError();
  if(addArraysErr != cudaSuccess) printf("Error: %s\n", cudaGetErrorString(addArraysErr));

  // 檢查異步錯誤（執行錯誤）
  asyncErr = cudaDeviceSynchronize();
  if(asyncErr != cudaSuccess) printf("Error: %s\n", cudaGetErrorString(asyncErr));

  // 驗證結果
  checkElementsAre(7, c, N);

  // 釋放記憶體
  cudaFree(a);
  cudaFree(b);
  cudaFree(c);
}
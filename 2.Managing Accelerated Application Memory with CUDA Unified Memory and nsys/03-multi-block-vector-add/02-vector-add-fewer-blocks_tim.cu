#include <stdio.h>

/**
 * CPU 函數：初始化陣列
 * @param num 要填入的數值
 * @param a 目標陣列
 * @param N 陣列大小
 */
void initWith(float num, float *a, int N)
{
  for(int i = 0; i < N; ++i)
  {
    a[i] = num;
  }
}

/**
 * GPU 核心函數：執行向量加法
 * 使用 grid-stride loop 模式來處理大型數據
 * @param result 結果陣列
 * @param a 輸入陣列1
 * @param b 輸入陣列2
 * @param N 陣列大小
 */
__global__
void addArraysInto(float *result, float *a, float *b, int N)
{
  // 計算當前線程在整個 grid 中的全局索引
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  
  // 計算整個 grid 中的總線程數
  int stride = blockDim.x * gridDim.x;

  // 使用 grid-stride loop 讓每個線程處理多個元素
  for(int i = index; i < N; i += stride)
  {
    result[i] = a[i] + b[i];
  }
}

/**
 * CPU 函數：檢查結果是否正確
 * @param target 預期的目標值
 * @param array 要檢查的陣列
 * @param N 陣列大小
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
  printf("Success! All values correctly calculated.\n");
}

int main()
{
  // 設定問題規模：2^25 個元素
  const int N = 2<<24;
  size_t size = N * sizeof(float);

  float *a, *b, *c;

  // 使用統一記憶體分配三個陣列
  cudaMallocManaged(&a, size);
  cudaMallocManaged(&b, size);
  cudaMallocManaged(&c, size);

  // 初始化輸入陣列
  initWith(3, a, N);
  initWith(4, b, N);
  initWith(0, c, N);

  // 設定 CUDA 執行配置
  size_t threadsPerBlock = 256;  // 每個 block 256 個線程
  size_t numberOfBlocks = 32;    // 使用 32 個 blocks
  
  // 注意！這裡的配置可能不夠處理所有數據
  // 更好的方式是：numberOfBlocks = (N + threadsPerBlock - 1) / threadsPerBlock;

  // 宣告錯誤處理變量
  cudaError_t addArraysErr;
  cudaError_t asyncErr;

  // 啟動 kernel
  addArraysInto<<<numberOfBlocks, threadsPerBlock>>>(c, a, b, N);

  // 檢查 kernel 啟動錯誤
  addArraysErr = cudaGetLastError();
  if(addArraysErr != cudaSuccess) printf("Error: %s\n", cudaGetErrorString(addArraysErr));

  // 檢查 kernel 執行錯誤
  asyncErr = cudaDeviceSynchronize();
  if(asyncErr != cudaSuccess) printf("Error: %s\n", cudaGetErrorString(asyncErr));

  // 驗證結果：3 + 4 = 7
  checkElementsAre(7, c, N);

  // 釋放記憶體
  cudaFree(a);
  cudaFree(b);
  cudaFree(c);
}
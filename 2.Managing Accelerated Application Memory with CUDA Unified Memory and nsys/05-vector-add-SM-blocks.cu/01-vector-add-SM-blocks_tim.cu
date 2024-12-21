// answer.cu: GPU 端加法函數
#include <stdio.h>

/*
 * CPU 端初始化向量函數
 * num: 要填入的值
 * a: 目標向量
 * N: 向量長度
 */ 
void initWith(float num, float *a, int N)
{
  for(int i = 0; i < N; ++i)
  {
    a[i] = num;
  }
}

/*
 * GPU 核心函數 - 向量加法
 * 使用 grid-stride loop 模式來處理任意大小的輸入
 */
__global__
void addArraysInto(float *result, float *a, float *b, int N)
{
  // 計算線程的全局索引
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  // 計算總線程數(跨距)
  int stride = blockDim.x * gridDim.x;

  // Grid-stride loop以處理大於總線程數的數據
  for(int i = index; i < N; i += stride)
  {
    result[i] = a[i] + b[i];
  }
}

/*
 * CPU端驗證函數
 * 檢查結果向量中的所有元素是否等於目標值
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
  // 獲取設備資訊
  int deviceId;
  int numberOfSMs; // SM: Stream Multiprocessor, 即多處理器
  cudaGetDevice(&deviceId);
  cudaDeviceGetAttribute(&numberOfSMs, cudaDevAttrMultiProcessorCount, deviceId);
  printf("Device ID: %d\tNumber of SMs: %d\n", deviceId, numberOfSMs);

  // 設置問題規模
  const int N = 2<<24; // 約33M個元素
  size_t size = N * sizeof(float);

  // 分配統一記憶體
  float *a, *b, *c;
  cudaMallocManaged(&a, size);
  cudaMallocManaged(&b, size);
  cudaMallocManaged(&c, size);

  // 初始化輸入數據
  initWith(3, a, N);
  initWith(4, b, N);
  initWith(0, c, N);

  // 設置執行配置
  size_t threadsPerBlock = 256;
  // 設置區塊數為SM數量的整數倍以優化性能
  size_t numberOfBlocks = 32 * numberOfSMs;

  // 錯誤處理變量
  cudaError_t addArraysErr;
  cudaError_t asyncErr;

  // 啟動核心
  addArraysInto<<<numberOfBlocks, threadsPerBlock>>>(c, a, b, N);

  // 檢查啟動錯誤
  addArraysErr = cudaGetLastError();
  if(addArraysErr != cudaSuccess) printf("Error: %s\n", cudaGetErrorString(addArraysErr));

  // 等待並檢查執行錯誤
  asyncErr = cudaDeviceSynchronize();
  if(asyncErr != cudaSuccess) printf("Error: %s\n", cudaGetErrorString(asyncErr));

  // 驗證結果
  checkElementsAre(7, c, N);

  // 釋放記憶體
  cudaFree(a);
  cudaFree(b);
  cudaFree(c);
}
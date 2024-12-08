#include <stdio.h>
#include <assert.h>

// 錯誤檢查輔助函數
inline cudaError_t checkCuda(cudaError_t result)
{
  if (result != cudaSuccess) {
    fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
    assert(result == cudaSuccess);
  }
  return result;
}

// 初始化陣列
void initWith(float num, float *a, int N)
{
  for(int i = 0; i < N; ++i)
  {
    a[i] = num;
  }
}

// CUDA 核函數：執行向量加法
__global__
void addArraysInto(float *result, float *a, float *b, int N)
{
  // 計算全局索引和步長
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  int stride = blockDim.x * gridDim.x;

  // 使用網格步長循環處理所有元素
  for(int i = index; i < N; i += stride)  {
    result[i] = a[i] + b[i];
  }
}

// 檢查結果
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

int main(){
  
  // 設定問題規模
  const int N = 2<<24;
  size_t size = N * sizeof(float);

  float *a;
  float *b;
  float *c;

  // 分配統一記憶體
  checkCuda(cudaMallocManaged(&a, size));
  checkCuda(cudaMallocManaged(&b, size));
  checkCuda(cudaMallocManaged(&c, size));

  // 初始化陣列
  initWith(3, a, N);
  initWith(4, b, N);
  initWith(0, c, N);

  // 設定執行配置
  size_t threadsPerBlock = 256;
  size_t numberOfBlocks = (N + threadsPerBlock - 1) / threadsPerBlock; // 計算需要的塊數

  // 啟動核函數
  addArraysInto<<<numberOfBlocks, threadsPerBlock>>>(c, a, b, N);

  // 錯誤檢查
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
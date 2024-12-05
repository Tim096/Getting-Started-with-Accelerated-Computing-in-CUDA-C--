#include <stdio.h>

void init(int *a, int N)
{
  int i;
  for (i = 0; i < N; ++i)
  {
    a[i] = i;
  }
}

__global__
void doubleElements(int *a, int N)
{

  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = gridDim.x * blockDim.x;

  /*
   * The previous code (now commented out) attempted
   * to access an element outside the range of `a`.
   */

  // for (int i = idx; i < N + stride; i += stride)
  for (int i = idx; i < N; i += stride)
  {
    a[i] *= 2;
  }
}

bool checkElementsAreDoubled(int *a, int N)
{
  int i;
  for (i = 0; i < N; ++i)
  {
    if (a[i] != i*2) return false;
  }
  return true;
}

int main()
{
  int N = 10000;
  int *a;

  size_t size = N * sizeof(int);
  cudaMallocManaged(&a, size);

  init(a, N);

  /*
   * The previous code (now commented out) attempted to launch
   * the kernel with more than the maximum number of threads per
   * block, which is 1024.
   */

  size_t threads_per_block = 1024;
  /* size_t threads_per_block = 2048; */
  size_t number_of_blocks = 32;

  // cudaError_t: cuda 用來回傳錯誤的資料型態，回傳的錯誤碼是 enum
  // enum: 一種資料型態，用來定義一組命名的整數常數 ex: cudaErrorInvalidValue = 1
  cudaError_t syncErr, asyncErr;

  doubleElements<<<number_of_blocks, threads_per_block>>>(a, N);

  /*
   * Catch errors for both the kernel launch above and any
   * errors that occur during the asynchronous `doubleElements`
   * kernel execution.
   */

  // cudaGetLastError: 回傳最後一個錯誤
  syncErr = cudaGetLastError();
  /* syncErr:
   * 這是同步錯誤檢查
   * 只檢查 CUDA 函式的啟動(launch)是否成功
   * 立即回傳上一個 CUDA 操作的錯誤狀態
   * 可以檢測到像是參數錯誤、記憶體分配失敗等立即可見的錯誤
   * 不會等待 kernel 執行完成
   */

  // cudaDeviceSynchronize: 等待目前的裝置執行完成
  asyncErr = cudaDeviceSynchronize();
  /* asyncErr:
   * 這是非同步錯誤檢查
   * 等待目前的裝置執行完成
   * 並且檢查是否有錯誤發生
   * 這個函式會等待目前裝置執行完成，並且檢查是否有錯誤發生
   * 如果有錯誤發生，會回傳對應的錯誤碼
   */

  /*
   * Print errors should they exist.
   */

  // cudaGetErrorString: 回傳錯誤碼的字串描述
  if (syncErr != cudaSuccess) printf("Error: %s\n", cudaGetErrorString(syncErr));
  if (asyncErr != cudaSuccess) printf("Error: %s\n", cudaGetErrorString(asyncErr));

  bool areDoubled = checkElementsAreDoubled(a, N);
  printf("All elements were doubled? %s\n", areDoubled ? "TRUE" : "FALSE");

  cudaFree(a);
}

#include <stdio.h>

/*
 * Initialize array values on the host.
 */

/*
核心概念和運作邏輯
此程式展示 CUDA 統一記憶體管理（Unified Memory）的使用。主要功能是將陣列中的每個元素值乘以2。

執行流程：
1. 分配記憶體
2. 初始化陣列(0 到 N-1)
3. GPU 平行處理每個元素乘2
4. 驗證結果
5. 釋放記憶體

主要問題：目前使用 malloc() 分配的記憶體只能被 CPU 存取，需改用 CUDA 統一記憶體。
*/

void init(int *a, int N) // 初始化陣列
{
  int i;
  for (i = 0; i < N; ++i)
  {
    a[i] = i;
  }
}

/*
 * Double elements in parallel on the GPU.
 */

__global__
void doubleElements(int *a, int N) // 使用 GPU 平行處理每個元素乘 2
{
  int i;
  i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N) // 確保不超過陣列大小, 避免存取越界
  {
    a[i] *= 2;
  }
  printf("a[%d] = %d\n", i, a[i]); // Output result is random, because the kernel is executed in parallel 
  // above code in one line: a[i] = (i < N) ? a[i] * 2 : a[i];
  // above code in cpu code: for (int i = 0; i < N; i++) { a[i] *= 2; }
}

/*
 * Check all elements have been doubled on the host.
 */

bool checkElementsAreDoubled(int *a, int N) // 驗證結果
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
  int N = 100; // 陣列大小
  int *a; // 陣列指標

  size_t size = N * sizeof(int); // 計算陣列大小, size_t 是一個能儲存任何物件大小的資料型態, 通常用來表示記憶體大小

  /*
   * Refactor this memory allocation to provide a pointer // 重構此記憶體分配以提供一個指標
   * `a` that can be used on both the host and the device. // `a` 可以在主機和裝置上使用
   */

  // a = (int *)malloc(size); // 使用 malloc() 分配記憶體
  // Change the above line to use `cudaMallocManaged` instead of `
  cudaMallocManaged(&a, size); // 使用 CUDA 統一記憶體管理分配記憶體

  init(a, N); // 初始化陣列, init(a, N) 會將陣列 a 的每個元素設為 0 到 N-1, ps : init() 函式在上面(自己寫的)

  size_t threads_per_block = 10; // 每個 block 的執行緒數
  size_t number_of_blocks = 10; // block 數量

  /*
   * This launch will not work until the pointer `a` is also
   * available to the device.
   */

  doubleElements<<<number_of_blocks, threads_per_block>>>(a, N); // 使用 GPU 平行處理每個元素乘 2
  cudaDeviceSynchronize(); // 等待 GPU 完成

  bool areDoubled = checkElementsAreDoubled(a, N); // 驗證結果
  printf("All elements were doubled? %s\n", areDoubled ? "TRUE" : "FALSE"); // 輸出結果

  /*
   * Refactor to free memory that has been allocated to be
   * accessed by both the host and the device.
   */

  // free(a); // 釋放記憶體
  // Change the above line to use `cudaFree` instead of `free`.
  cudaFree(a); // 釋放記憶體
}

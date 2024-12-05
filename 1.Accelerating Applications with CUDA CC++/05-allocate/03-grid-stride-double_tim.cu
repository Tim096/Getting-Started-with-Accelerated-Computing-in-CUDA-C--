/*
 * 1. 核心概念和運作邏輯
 * 這段程式碼展示了 CUDA 中的 grid-stride loop 概念，用於處理大於 grid 大小的資料集。
 * 
 * 主要問題：
 * 1.需要處理 10000 個元素
 * 2.但一個 最多只能處理 8192 個元素，grid 只有 8192 個執行緒(256 * 32)
 * 3.當前的實作無法處理所有元素

*/

#include <stdio.h>

void init(int *a, int N)
{
  int i;
  for (i = 0; i < N; ++i)
  {
    a[i] = i;
  }
}

/*
 * In the current application, `N` is larger than the grid. // N 大於 grid
 * Refactor this kernel to use a grid-stride loop in order that // 重構這個核心，使用 grid-stride loop
 * each parallel thread work on more than one element of the array. // 每個平行執行緒處理多個陣列元素
 */

__global__
void doubleElements(int *a, int N) // a: 輸入陣列, N: 陣列大小
{
  // int i;
  // i = blockIdx.x * blockDim.x + threadIdx.x;
  // if (i < N)
  // {
  //   a[i] *= 2;
  // }
  int idx = blockIdx.x * blockDim.x + threadIdx.x; // 計算執行緒的索引
  int stride = gridDim.x * blockDim.x; // 總執行緒數, 代表 stride 是用來跳過多少個元素, 因為每個執行緒處理多個元素, 下一個執行緒處理的元素是當前元素加上 stride

  
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
  /*
   * `N` is greater than the size of the grid (see below).
   */

  int N = 10000;
  int *a;

  size_t size = N * sizeof(int);
  cudaMallocManaged(&a, size);

  init(a, N);

  /*
   * The size of this grid is 256*32 = 8192.
   */

  size_t threads_per_block = 256;
  size_t number_of_blocks = 32;

  doubleElements<<<number_of_blocks, threads_per_block>>>(a, N);
  // <<<number_of_blocks, threads_per_block>>>: 代表執行多少個 block, 每個 block 有多少執行緒 然後這些執行緒會執行 doubleElements 函式，並且會 parallel 執行
  cudaDeviceSynchronize();

  bool areDoubled = checkElementsAreDoubled(a, N);
  printf("All elements were doubled? %s\n", areDoubled ? "TRUE" : "FALSE");

  cudaFree(a);
}

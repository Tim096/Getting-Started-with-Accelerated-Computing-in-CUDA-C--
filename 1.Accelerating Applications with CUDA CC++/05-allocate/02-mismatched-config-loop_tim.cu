#include <stdio.h>

/*
 * Currently, `initializeElementsTo`, if executed in a thread whose // 目前, `initializeElementsTo`, 如果在一個執行緒中執行
 * `i` is calculated to be greater than `N`, will try to access a value // `i` 被計算為大於 `N`, 將嘗試訪問一個值
 * outside the range of `a`. // 超出 `a` 的範圍
 *
 * Refactor the kernel definition to prevent out of range accesses.
 */


/*
1. 核心概念和運作邏輯
主要功能：
1. 初始化一個大小為 1000 的陣列，將所有元素設為特定值(6)
2. 使用固定的 thread 數(256)進行平行處理
3. 需要處理執行配置和邊界檢查

程式問題：
1. 需要計算正確的 block 數量
2. 需要防止陣列存取越界

*/

__global__ void initializeElementsTo(int initialValue, int *a, int N)
{
  int i = threadIdx.x + blockIdx.x * blockDim.x; // 計算陣列索引
  if (i < N){
    a[i] = initialValue; // 將陣列元素設為特定值
  } // 避免存取越界
}

int main()
{
  /*
   * Do not modify `N`.
   */

  int N = 1000; // 陣列大小

  int *a;
  size_t size = N * sizeof(int);

  cudaMallocManaged(&a, size);

  /*
   * Assume we have reason to want the number of threads // 假設我們有理由想要執行緒數量
   * fixed at `256`: do not modify `threads_per_block`. // 固定在 256: 不要修改 `threads_per_block`.
   */

  size_t threads_per_block = 256; // 每個 block 的執行緒數

  /*
   * Assign a value to `number_of_blocks` that will // 為 `number_of_blocks` 分配一個值
   * allow for a working execution configuration given // 以便在給定的執行配置下工作
   * the fixed values for `N` and `threads_per_block`. // 固定值 `N` 和 `threads_per_block`.
   */

  // 確保有足夠的 block 數量
  size_t number_of_blocks = (N + threads_per_block - 1) / threads_per_block; // 計算 block 數量
  /* 4 = (1000 + 255) / 256 => (1255)/ 256 (利用 -1 避免無條件進位, 達到向上取整效果)
                            =>  ceil(1000/256) 
  */

  // ceil() 函式是向上取整函式；floor() 函式是向下取整函式
  int initialValue = 6; // 設定初始值

  initializeElementsTo<<<number_of_blocks, threads_per_block>>>(initialValue, a, N);
  // 呼叫 kernel 函式 (執行配置, 參數, 陣列大小)
  cudaDeviceSynchronize();

  /*
   * Check to make sure all values in `a`, were initialized.
   * 檢查所有的 `a` 值是否已經初始化
   */

  for (int i = 0; i < N; ++i) // 用 CPU 檢查陣列元素是否正確，是否都為 6
  {
    if(a[i] != initialValue)
    {
      printf("FAILURE: target value: %d\t a[%d]: %d\n", initialValue, i, a[i]);
      cudaFree(a);
      exit(1);
    }
  }
  printf("SUCCESS!\n");

  cudaFree(a); // 釋放記憶體, 如果沒有參數, 則釋放所有 CUDA 記憶體；如果有參數, 則釋放指定的記憶體
}

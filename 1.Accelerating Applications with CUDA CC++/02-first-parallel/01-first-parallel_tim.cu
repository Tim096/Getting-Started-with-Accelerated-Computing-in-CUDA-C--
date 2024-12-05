#include <stdio.h>

/*
1. 核心概念和運作邏輯
主要功能：
這是一個示範如何將串行執行的函數改寫為在 GPU 上平行執行的範例。目前的程式碼是未完成版本，需要修改為支援 GPU 平行執行。
*/

__global__ void firstParallel()
{
  printf("This should be running in parallel.\n");
}

int main()
{
  /*
   * Refactor this call to firstParallel to execute in parallel
   * on the GPU.
   */
  // 啟動 5 個 block, 每個 block 執行 5 個 thread
  firstParallel<<<5, 5>>>();

  /*
   * Some code is needed below so that the CPU will wait
   * for the GPU kernels to complete before proceeding.
   */
  // 等待所有 GPU 工作完成
  cudaDeviceSynchronize();
  return 0;

}
#include <stdio.h>

/*
 * Refactor `loop` to be a CUDA Kernel. The new kernel should
 * only do the work of 1 iteration of the original loop.
 */

/*
1. 核心概念和運作邏輯
主要功能：
這是一個將串行迴圈改造為 GPU 平行執行的練習。原程式碼通過迴圈依序印出 0 到 N-1 的數字，需要改寫為使用 GPU 平行處理。
*/

void loop()
{
  // for (int i = 0; i < N; ++i)
  // {
  //   printf("This is iteration number %d\n", i);
  // }
  printf("This is iteration number %d\n", threadIdx.x);
}

int main()
{
  /*
   * When refactoring `loop` to launch as a kernel, be sure
   * to use the execution configuration to control how many
   * "iterations" to perform.
   *
   * For this exercise, only use 1 block of threads.
   */

  int N = 10;
  loop<<<1, N>>>();
}

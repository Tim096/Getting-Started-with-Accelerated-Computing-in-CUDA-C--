/*
1. 核心概念和運作邏輯
主要功能：
這是一個簡單的 Hello World 程式，目的是展示如何在 CPU 和 GPU 上執行程式並輸出訊息。目前程式碼是未完成的版本，需要進行修改以實現 GPU 功能。

執行流程：
呼叫 helloCPU() 在 CPU 上執行
呼叫 helloGPU() （需要修改為 GPU kernel）
需要添加同步機制等待 GPU 執行完成

關鍵函數：
helloCPU(): 純 CPU 函數
helloGPU(): 需要修改為 GPU kernel
main(): 主程式，負責調用和協調 CPU/GPU 功能

*/

#include <stdio.h>

void helloCPU()
{
  printf("Hello from the CPU.\n");
}

/*
 * Refactor the `helloGPU` definition to be a kernel
 * that can be launched on the GPU. Update its message
 * to read "Hello from the GPU!"
 */

__global__ void helloGPU()  // 將函數宣告為 GPU kernel
{
  printf("Hello also from the CPU.\n");
}

int main()
{

  helloCPU();

  /*
   * Refactor this call to `helloGPU` so that it launches
   * as a kernel on the GPU.
   */
  helloGPU<<<1, 1>>>();  // 使用執行配置啟動 kernel
  // <<<blocks, threads>>>: 定義執行配置

  helloGPU();

  /*
   * Add code below to synchronize on the completion of the
   * `helloGPU` kernel completion before continuing the CPU
   * thread.
   */
  cudaDeviceSynchronize();  // 等待 GPU 執行完成
}

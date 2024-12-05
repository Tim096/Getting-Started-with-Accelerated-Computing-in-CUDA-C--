#include <stdio.h>

/*
1. 核心概念和運作邏輯
主要功能：
這是一個教學練習程式，目的是讓學習者理解如何正確設定 CUDA 的執行配置（Execution Configuration）。程式會在特定的 thread 和 block 索引組合下印出 "Success!"，否則印出失敗訊息。
*/


__global__ void printSuccessForCorrectExecutionConfiguration()
{
    if(threadIdx.x == 1023 && blockIdx.x == 255)
    {
        printf("Success!\n");
    }
}

int main()
{
    // 啟動 256 個 blocks，每個 block 1024 個 threads
    printSuccessForCorrectExecutionConfiguration<<<256, 1024>>>();
    
    // 等待執行完成
    cudaDeviceSynchronize();
    
    return 0;
}

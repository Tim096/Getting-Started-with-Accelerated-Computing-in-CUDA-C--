#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "timer.h"
#include "files.h"
#include <cuda_runtime.h>

#define SOFTENING 1e-9f

/*
 * 每個物體包含 x、y 和 z 坐標位置 (coordinate positions)，
 * 以及在 x、y 和 z 方向上的速度 (velocities)。
 */

typedef struct { float x, y, z, vx, vy, vz; } Body;

/*
 *  `__global__` 指定此函數將在 GPU 上執行，並可從 CPU 上呼叫。這是一個 CUDA 核心函式 (CUDA kernel)。
 *  CUDA 核心函式是並行執行的基本單位。
 */
__global__ void bodyForce_cuda(Body *p, float dt, int n) {
    /*
     * `blockIdx.x` 是目前 Thread所屬的區塊 (block) 在網格 (grid) 中的索引。
     * `blockDim.x` 是每個區塊中的 Thread (thread) 數量。
     * `threadIdx.x` 是目前 Thread在其所屬區塊中的索引。
     * 這個計算確保每個 Thread處理一個唯一的物體。
     */
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float Fx = 0.0f;
        float Fy = 0.0f;
        float Fz = 0.0f;

        for (int j = 0; j < n; j++) {
            float dx = p[j].x - p[i].x;
            float dy = p[j].y - p[i].y;
            float dz = p[j].z - p[i].z;
            float distSqr = dx * dx + dy * dy + dz * dz + SOFTENING;
            float invDist = rsqrtf(distSqr); // 計算平方根的倒數，在 GPU 上通常比除法快
            float invDist3 = invDist * invDist * invDist;

            Fx += dx * invDist3;
            Fy += dy * invDist3;
            Fz += dz * invDist3;
        }

        p[i].vx += dt * Fx;
        p[i].vy += dt * Fy;
        p[i].vz += dt * Fz;
    }
}

/*
 * `__global__` 指定此函數也將在 GPU 上執行。
 */
__global__ void integratePosition_cuda(Body *p, float dt, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        p[i].x += p[i].vx * dt;
        p[i].y += p[i].vy * dt;
        p[i].z += p[i].vz * dt;
    }
}

int main(const int argc, const char** argv) {
    int nBodies = 2 << 11;
    if (argc > 1) nBodies = 2 << atoi(argv[1]);

    const char *initialized_values;
    const char *solution_values;

    if (nBodies == 2 << 11) {
        initialized_values = "09-nbody/files/initialized_4096";
        solution_values = "09-nbody/files/solution_4096";
    } else { // nBodies == 2<<15
        initialized_values = "09-nbody/files/initialized_65536";
        solution_values = "09-nbody/files/solution_65536";
    }

    if (argc > 2) initialized_values = argv[2];
    if (argc > 3) solution_values = argv[3];

    const float dt = 0.01f; // 時間步長 (Time step)
    const int nIters = 10;  // 模擬迭代次數 (Simulation iterations)

    int bytes = nBodies * sizeof(Body);
    float *h_buf; // 主機 (host) 上的緩衝區
    Body *d_p;   // 裝置 (device，即 GPU) 上的指標

    // 主機記憶體配置 (Host memory allocation)
    h_buf = (float *)malloc(bytes);
    Body *h_p = (Body*)h_buf;

    // 從檔案讀取初始值 (Read initial values from file)
    read_values_from_file(initialized_values, h_buf, bytes);

    // 裝置記憶體配置 (Device memory allocation)
    cudaError_t cuda_status = cudaMalloc((void**)&d_p, bytes);
    if (cuda_status != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(cuda_status));
        return 1;
    }

    // 將資料從主機複製到裝置 (Copy data from host to device)
    cuda_status = cudaMemcpy(d_p, h_p, bytes, cudaMemcpyHostToDevice);
    if (cuda_status != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed: %s\n", cudaGetErrorString(cuda_status));
        return 1;
    }

    double totalTime = 0.0;

    // 核心函式啟動配置 (Kernel launch configuration)
    
    int threadsPerBlock = 256;
    //每個區塊的 Thread數量 (Threads per block)。這是一個可以調整的參數，取決於 GPU 的架構。
    
    int blocksPerGrid = (nBodies + threadsPerBlock - 1) / threadsPerBlock; 
    // 網格中的區塊數量 (Blocks per grid)。確保所有物體都被處理。

    for (int iter = 0; iter < nIters; iter++) {
        StartTimer();

        // 啟動 bodyForce 核心函式 (Launch bodyForce kernel)
        bodyForce_cuda<<<blocksPerGrid, threadsPerBlock>>>(d_p, dt, nBodies);
        cuda_status = cudaGetLastError();
        if (cuda_status != cudaSuccess) {
            fprintf(stderr, "bodyForce_cuda launch failed: %s\n", cudaGetErrorString(cuda_status));
            return 1;
        }
        cudaDeviceSynchronize(); // 確保核心函式完成 (Ensure kernel completion)

        // 啟動 integratePosition 核心函式 (Launch integratePosition kernel)
        integratePosition_cuda<<<blocksPerGrid, threadsPerBlock>>>(d_p, dt, nBodies);
        cuda_status = cudaGetLastError();
        if (cuda_status != cudaSuccess) {
            fprintf(stderr, "integratePosition_cuda launch failed: %s\n", cudaGetErrorString(cuda_status));
            return 1;
        }
        cudaDeviceSynchronize(); // 確保核心函式完成 (Ensure kernel completion)

        const double tElapsed = GetTimer() / 1000.0;
        totalTime += tElapsed;
    }

    // 將結果複製回主機 (Copy results back to host)
    cuda_status = cudaMemcpy(h_p, d_p, bytes, cudaMemcpyDeviceToHost);
    if (cuda_status != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed: %s\n", cudaGetErrorString(cuda_status));
        return 1;
    }

    double avgTime = totalTime / (double)(nIters);
    float billionsOfOpsPerSecond = 1e-9 * nBodies * nBodies / avgTime;
    write_values_to_file(solution_values, h_buf, bytes);

    printf("%0.3f Billion Interactions / second\n", billionsOfOpsPerSecond);

    // 釋放裝置記憶體 (Free device memory)
    cudaFree(d_p);
    free(h_buf);

    return 0;
}
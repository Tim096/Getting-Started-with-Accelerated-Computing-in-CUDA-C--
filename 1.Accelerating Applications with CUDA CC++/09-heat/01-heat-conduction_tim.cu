#include <stdio.h>
#include <math.h>

// 將2D索引轉換為1D記憶體訪問的巨集，用於線性化二維數組
#define I2D(num, c, r) ((r)*(num)+(c))

/**
 * GPU核函數：計算熱傳導的一個時間步長
 * 使用有限差分法計算熱擴散
 */
__global__
void step_kernel_mod(int ni, int nj, float fact, float* temp_in, float* temp_out)
{
    int i00, im10, ip10, i0m1, i0p1;
    float d2tdx2, d2tdy2;

    // 計算當前執行緒的2D網格位置
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;

    // 檢查邊界條件，只處理內部點
    if (j > 0 && i > 0 && j < nj-1 && i < ni-1) {
        // 計算中心點和相鄰點的記憶體索引
        i00 = I2D(ni, i, j);    // 當前點
        im10 = I2D(ni, i-1, j); // 左點
        ip10 = I2D(ni, i+1, j); // 右點
        i0m1 = I2D(ni, i, j-1); // 下點
        i0p1 = I2D(ni, i, j+1); // 上點

        // 計算二階導數（中心差分格式）
        d2tdx2 = temp_in[im10]-2*temp_in[i00]+temp_in[ip10];
        d2tdy2 = temp_in[i0m1]-2*temp_in[i00]+temp_in[i0p1];

        // 更新溫度場
        temp_out[i00] = temp_in[i00]+fact*(d2tdx2 + d2tdy2);
    }
}

// CPU參考實現，用於結果驗證
void step_kernel_ref(int ni, int nj, float fact, float* temp_in, float* temp_out)
{
    /* CPU版本的實現與GPU版本邏輯相同，但使用串行迴圈 */
    int i00, im10, ip10, i0m1, i0p1;
    float d2tdx2, d2tdy2;

    for (int j=1; j < nj-1; j++) {
        for (int i=1; i < ni-1; i++) {
            i00 = I2D(ni, i, j);
            im10 = I2D(ni, i-1, j);
            ip10 = I2D(ni, i+1, j);
            i0m1 = I2D(ni, i, j-1);
            i0p1 = I2D(ni, i, j+1);

            d2tdx2 = temp_in[im10]-2*temp_in[i00]+temp_in[ip10];
            d2tdy2 = temp_in[i0m1]-2*temp_in[i00]+temp_in[i0p1];

            temp_out[i00] = temp_in[i00]+fact*(d2tdx2 + d2tdy2);
        }
    }
}

int main()
{
    // 模擬參數設置
    int nstep = 200;           // 時間步數
    const int ni = 200;        // x方向網格點數
    const int nj = 100;        // y方向網格點數
    float tfac = 8.418e-5;     // 銀的熱擴散係數

    // 分配記憶體
    const int size = ni * nj * sizeof(float);
    float *temp1_ref, *temp2_ref, *temp1, *temp2, *temp_tmp;

    // CPU記憶體分配
    temp1_ref = (float*)malloc(size);
    temp2_ref = (float*)malloc(size);
    
    // GPU統一記憶體分配
    cudaMallocManaged(&temp1, size);
    cudaMallocManaged(&temp2, size);

    // 使用隨機數據初始化溫度場
    for(int i = 0; i < ni*nj; ++i) {
        temp1_ref[i] = temp2_ref[i] = temp1[i] = temp2[i] = 
            (float)rand()/(float)(RAND_MAX/100.0f);
    }

    // 執行CPU參考版本
    for (int istep=0; istep < nstep; istep++) {
        step_kernel_ref(ni, nj, tfac, temp1_ref, temp2_ref);
        // 交換指標以準備下一步
        temp_tmp = temp1_ref;
        temp1_ref = temp2_ref;
        temp2_ref = temp_tmp;
    }

    // 設置CUDA執行配置
    dim3 tblocks(32, 16, 1);    // 每個區塊32x16個執行緒
    dim3 grid((nj/tblocks.x)+1, (ni/tblocks.y)+1, 1);
    cudaError_t ierrSync, ierrAsync;

    // 執行GPU版本
    for (int istep=0; istep < nstep; istep++) {
        step_kernel_mod<<<grid, tblocks>>>(ni, nj, tfac, temp1, temp2);

        // 錯誤檢查
        ierrSync = cudaGetLastError();
        ierrAsync = cudaDeviceSynchronize();
        if (ierrSync != cudaSuccess) { 
            printf("Sync error: %s\n", cudaGetErrorString(ierrSync)); 
        }
        if (ierrAsync != cudaSuccess) { 
            printf("Async error: %s\n", cudaGetErrorString(ierrAsync)); 
        }

        // 交換指標
        temp_tmp = temp1;
        temp1 = temp2;
        temp2 = temp_tmp;
    }

    // 計算最大誤差
    float maxError = 0;
    for(int i = 0; i < ni*nj; ++i) {
        if (abs(temp1[i]-temp1_ref[i]) > maxError) { 
            maxError = abs(temp1[i]-temp1_ref[i]); 
        }
    }

    // 驗證結果
    if (maxError > 0.0005f)
        printf("Problem! The Max Error of %.5f is NOT within acceptable bounds.\n", maxError);
    else
        printf("The Max Error of %.5f is within acceptable bounds.\n", maxError);

    // 釋放記憶體
    free(temp1_ref);
    free(temp2_ref);
    cudaFree(temp1);
    cudaFree(temp2);

    return 0;
}
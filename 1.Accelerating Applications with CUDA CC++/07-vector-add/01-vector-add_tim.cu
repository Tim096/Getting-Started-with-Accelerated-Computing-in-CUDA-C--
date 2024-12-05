#include <stdio.h>

void initWith(float num, float *a, int N)
{
  for(int i = 0; i < N; ++i)
  {
    a[i] = num;
  }
}

__global__
void addVectorsInto(float *result, float *a, float *b, int N)
{
  // for(int i = 0; i < N; ++i)
  // {
  //   result[i] = a[i] + b[i];
  // }
  // 將for迴圈改為CUDA核心函式
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N){
    result[idx] = a[idx] + b[idx];
  }
}

void checkElementsAre(float target, float *array, int N)
{
  for(int i = 0; i < N; i++)
  {
    if(array[i] != target)
    {
      printf("FAIL: array[%d] - %0.0f does not equal %0.0f\n", i, array[i], target);
      exit(1);
    }
  }
  printf("SUCCESS! All values added correctly.\n");
}

int main()
{
  const int N = 2<<20;
  size_t size = N * sizeof(float);

  float *a;
  float *b;
  float *c;

  // a = (float *)malloc(size);
  // b = (float *)malloc(size);
  // c = (float *)malloc(size);
  
  // 將malloc改為CUDA統一記憶體管理
  cudaMallocManaged(&a, size);
  cudaMallocManaged(&b, size);
  cudaMallocManaged(&c, size);

  initWith(3, a, N);
  initWith(4, b, N);
  initWith(0, c, N);

  int theadsPerBlock = 256;
  int blocksPerGrid = (N + theadsPerBlock - 1) / theadsPerBlock;
  // addVectorsInto(c, a, b, N);
  // 將函式改為CUDA核心函式
  addVectorsInto<<<blocksPerGrid, threadPerBlock>>>(c, a, b, N);

  checkElementsAre(7, c, N);

  free(a);
  free(b);
  free(c);
}

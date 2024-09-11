#include "cuda_kernels.h"

#include <cuda.h>

__global__ void changeRed() {
  printf("Hello from GPU\n");
}

void changeRed(uint32_t* data, size_t size, newRed) {
  changeRed<<<1, 1>>>();
  cudaDeviceSynchronize();
}

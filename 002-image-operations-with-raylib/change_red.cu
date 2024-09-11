#include "cuda_kernels.h"

#include <stdio.h>

#include <cuda.h>

__global__ void changeRed(uint32_t* data, int size_x, int size_y, int newRed) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < size_x && y < size_y) {
    int index = y * size_x + x;

    uint32_t pixel = data[index];

    // @TODO endianess???? what's the format?
    uint8_t blue = pixel & 0xFF;
    uint8_t green = (pixel >> 8) & 0xFF;
    uint8_t alpha = (pixel >> 24) & 0xFF;

    pixel = (alpha << 24) | (blue << 16) | (green << 8) | newRed;

    data[index] = pixel;
  }
}

void changeRed_host(uint32_t* data, size_t size_x, size_t size_y, int newRed) {
  // Calculate the number of blocks and threads
  dim3 blocks(size_x / 32, size_y / 32, 1);
  dim3 threads(32, 32, 1);

  // Launch the kernel
  changeRed<<<blocks, threads>>>(data, size_x, size_y, newRed);
  cudaDeviceSynchronize();
}

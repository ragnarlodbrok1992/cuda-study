#include <raylib.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <stdio.h>
#include <stdint.h>
#include <string.h>

#include "stb_image.h"
#include "cuda_kernels.h"

#define ASSET_IMAGE_PATH "assets/acorn.png"

/*
__global__ void changeRed(uint32_t *data, size_t size, int newRed) {
  // printf("Hello from block %d, thread %d\n", blockIdx.x, threadIdx.x);
  // printf("Block dimension: x - %d, y - %d\n", blockDim.x, blockDim.y);
  printf("Hello from CUDA!\n");
}
*/

int main() {

  // Raylib initialization
  InitWindow(800, 600, "CUDA examples");
  SetTargetFPS(60);

  // Raylib trace log
  // SetTraceLogCallback(CustomLog);
  SetTraceLogLevel(LOG_ALL);

  // CUDA initialization and information
  int device_count;
  int device;
  unsigned int device_flags;

  // Collect device information
  cudaGetDeviceCount(&device_count);
  cudaGetDevice(&device);
  cudaGetDeviceFlags(&device_flags);

  // Information
  TraceLog(LOG_INFO, "Device count: %d", device_count);
  TraceLog(LOG_INFO, "Device used: %d", device);
  TraceLog(LOG_INFO, "Device flags: 0x%08x", device_flags);

  // Initialization
  cudaSetDevice(device);

  // Main loop
  bool engine_running = true;

  int x, y, n;
  uint32_t *data = (uint32_t *)stbi_load(ASSET_IMAGE_PATH, &x, &y, &n, 4);

  if (data == NULL) {
    TraceLog(LOG_ERROR, "Failed to load image %s", ASSET_IMAGE_PATH);
  }

  size_t image_size = x * y;

  TraceLog(LOG_INFO, "Image loaded:   %dx%d", x, y);
  TraceLog(LOG_INFO, "Image channels: %d", n);
  TraceLog(LOG_INFO, "Image size:     %d", image_size);

  // Test image loading from RAW data
  Image test_image = {
    .data = data,
    .width = x,
    .height = y,
    .mipmaps = 1,
    .format = PIXELFORMAT_UNCOMPRESSED_R8G8B8A8
  };

  // Allocating stuff for CUDA
  uint32_t *d_data;
  cudaMalloc((void **)&d_data, image_size * sizeof(uint32_t));

  // TEST
  changeRed(d_data, image_size, 255);

  while (engine_running) {
    // Check for input
    if (IsKeyPressed(KEY_ESCAPE) || WindowShouldClose()) {
      engine_running = false;
    }

    // Render frame
    BeginDrawing();
    ClearBackground(RAYWHITE);
    DrawText("CUDA examples", 10, 10, 20, BLACK);

    // Render test image
    // Creating texture on the fly
    Texture2D texture = LoadTextureFromImage(test_image);
    DrawTexture(texture, 20, 10, WHITE);

    EndDrawing();

    // After rendering stuff
    UnloadTexture(texture);
  }

  // De-Initialization
  CloseWindow();
  return 0;
}

#include <raylib.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <stdio.h>
#include <string.h>

#define IMAGE_WIDTH 581
#define IMAGE_HEIGHT 394
#define RAYLIB_INTERNAL_PIXEL_FORMAT 7

static unsigned char TEST_IMAGE_DATA[IMAGE_WIDTH * IMAGE_HEIGHT * RAYLIB_INTERNAL_PIXEL_FORMAT];

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

  // Loading image
  Image test_image = LoadImage("assets/acorn.png"); // @TODO: This path should be calculated on runtime
  int test_image_filesize = 0;
  unsigned char* pointer_to_exported_image = ExportImageToMemory(test_image, ".png", &test_image_filesize);

  memcpy(&TEST_IMAGE_DATA[0],
          ExportImageToMemory(test_image, ".png", &test_image_filesize),
          test_image_filesize);

  while (engine_running) {
    // Check for input
    if (IsKeyPressed(KEY_ESCAPE) || WindowShouldClose()) {
      engine_running = false;
    }

    // Render frame
    BeginDrawing();
    ClearBackground(RAYWHITE);
    DrawText("CUDA examples", 10, 10, 20, BLACK);
    EndDrawing();
  }

  // De-Initialization
  CloseWindow();
  return 0;
}

#include <raylib.h>

// Static engine variables
static bool engine_running = true;

// @TODO: Change path to not be hardcoded
static const char* font_file_path = "D:\\engines\\cuda-study\\003-voxel-water-engine\\assets\\fonts\\FiraCode-Medium.ttf";

int main() {
  // Raylib initialization
  InitWindow(800, 600, "CUDA examples");
  SetTargetFPS(60);

  // Raylib trace log
  // SetTraceLogCallback(CustomLog);
  SetTraceLogLevel(LOG_ALL);

  // Raylib - set font
  Font default_font = LoadFont(font_file_path);
  Vector2 default_text_vec = { 10.0f, 10.0f };

  // Engine variables
  while (engine_running) {
    // Update
    if (WindowShouldClose()) {
      engine_running = false;
    }

    // Draw
    BeginDrawing();
      ClearBackground(RAYWHITE);

      // Not so fun function...
      DrawTextEx(default_font,
          "I am using Fira Code Medium font.",
          default_text_vec,
          (float)default_font.baseSize,
          2,
          MAROON);

    EndDrawing();
  }

  // De-initialization of Raylib
  UnloadFont(default_font);
  CloseWindow();
  return 0;
}

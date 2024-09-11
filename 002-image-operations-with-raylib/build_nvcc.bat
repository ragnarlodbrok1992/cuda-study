@echo off
SET compiler_nvcc="nvcc"

SET RAYLIB_VERSION=raylib-5.0_win64_msvc16

SET raylib_include_dir="D:\libraries\%RAYLIB_VERSION%\include"
SET raylib_lib_dir="D:\libraries\%RAYLIB_VERSION%\lib"

echo Building...

IF NOT EXIST build_nvcc mkdir build_nvcc

pushd build_nvcc

nvcc ../main.cpp -o raylib-cuda-test.exe^
  -I %raylib_include_dir%^
  -L %raylib_lib_dir%^
  -Xcompiler "/EHsc /Zi /DEBUG:FULL /INCREMENTAL:NO /std:c++20"^
  -Xlinker "/SUBSYSTEM:console /ENTRY:mainCRTStartup"^
  -Xlinker "/NODEFAULTLIB:libcmt"^
  -l raylib -l gdi32 -l user32 -l shell32 -l winmm -l kernel32


popd

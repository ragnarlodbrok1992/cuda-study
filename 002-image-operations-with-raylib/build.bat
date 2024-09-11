@echo off
SET compiler_dir="C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC\14.40.33807\bin\Hostx64\x64\cl.exe"

SET RAYLIB_VERSION=raylib-5.0_win64_msvc16

SET raylib_include_dir="D:\libraries\%RAYLIB_VERSION%\include"
SET raylib_lib_dir="D:\libraries\%RAYLIB_VERSION%\lib"

echo Building...

IF NOT EXIST build mkdir build

pushd build

%compiler_dir% /EHsc /Zi^
  /DEBUG:FULL^
  /INCREMENTAL:NO^
  /Fe:"raylib-cuda-test.exe"^
  ../main.cpp^
  /I %raylib_include_dir%^
  /link /LIBPATH:%raylib_lib_dir% /NODEFAULTLIB:libcmt^
  raylib.lib gdi32.lib User32.lib Shell32.lib winmm.lib kernel32.lib

popd

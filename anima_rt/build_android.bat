@echo off
REM Build libanima_rt.so for Android aarch64 (NDK cross-compilation)
REM Usage: build_android.bat
REM Output: libanima_rt.so

set NDK=D:\android-ndk-r27d-windows\android-ndk-r27d
set TC=%NDK%\toolchains\llvm\prebuilt\windows-x86_64
set SRC=src/anima_tensor.cpp src/cpu_backend.cpp src/anima_rt.cpp
set OUT=libanima_rt.so
set INC=-Iinclude

echo Building %OUT% for aarch64-android...
"%TC%\bin\clang++.exe" ^
  --target=aarch64-none-linux-android28 ^
  --sysroot="%TC%\sysroot" ^
  -O2 -std=c++17 -fPIC -shared ^
  %INC% ^
  -o %OUT% ^
  %SRC% ^
  -Wl,-z,max-page-size=16384 -Wl,-z,common-page-size=16384 -Wl,--no-rosegment ^
  -lm ^
  -L"%TC%\sysroot\usr\lib\aarch64-linux-android\28" ^
  -static-libstdc++

if %ERRORLEVEL% == 0 (
  echo [OK] %OUT% built successfully
  dir %OUT%
) else (
  echo [FAIL] Build failed
  exit /b 1
)

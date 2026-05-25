@echo off
set NDK=D:\android-ndk-r27d-windows\android-ndk-r27d
set TC=%NDK%\toolchains\llvm\prebuilt\windows-x86_64
set VK=D:\Vulkan_SDK

echo Building libvk_gemm.so for Android ARM64...
"%TC%\bin\clang++.exe" ^
  --target=aarch64-none-linux-android28 ^
  --sysroot="%TC%\sysroot" ^
  -O2 -std=c++17 -fPIC -shared ^
  -I"%VK%\Include" ^
  -o libvk_gemm.so main.cpp ^
  -Wl,-z,max-page-size=16384 ^
  -Wl,-z,common-page-size=16384 ^
  -Wl,--no-rosegment ^
  -llog -landroid -lvulkan ^
  -L"%TC%\sysroot\usr\lib\aarch64-linux-android\28" ^
  -static-libstdc++
if %ERRORLEVEL% EQU 0 (echo Build OK: libvk_gemm.so) else (echo BUILD FAILED)

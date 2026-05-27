@echo off
set NDK=D:\android-ndk-r27d-windows\android-ndk-r27d
set TC=%NDK%\toolchains\llvm\prebuilt\windows-x86_64
set VK=D:\Vulkan_SDK

echo Building libdit_vk.so for Android ARM64...
echo.

:: Step 1: compile GLSL → SPIR-V
echo === Compiling shaders ===
set GLSLC=%VK%\Bin\glslangValidator.exe
for %%s in (gemm_fp16 rms_norm_fp16 layernorm_fp16 silu_fp16 softmax_fp16 add_fp16 scale_shift_fp16 rope_fp16 attention_fp16 broadcast_fp16) do (
    if exist %%s.comp (
        echo   %%s.comp
        "%GLSLC%" -V %%s.comp -o %%s.spv >nul 2>&1
        if %ERRORLEVEL% NEQ 0 echo   ERROR: %%s.comp failed! && exit /b 1
    )
)
echo.

:: Step 2: build libdit_vk.so
echo === Building libdit_vk.so ===
"%TC%\bin\clang++.exe" ^
  --target=aarch64-none-linux-android28 ^
  --sysroot="%TC%\sysroot" ^
  -O2 -std=c++17 -fPIC -shared ^
  -I"%VK%\Include" ^
  -o libdit_vk.so dit_engine.cpp ^
  -Wl,-z,max-page-size=16384 ^
  -Wl,-z,common-page-size=16384 ^
  -Wl,--no-rosegment ^
  -llog -landroid -lvulkan ^
  -L"%TC%\sysroot\usr\lib\aarch64-linux-android\28" ^
  -static-libstdc++
if %ERRORLEVEL% EQU 0 (
    echo.
    echo === Build OK ===
    for %%F in (libdit_vk.so) do echo Size: %%~zF bytes
    echo.
) else (
    echo.
    echo === BUILD FAILED ===
    exit /b 1
)

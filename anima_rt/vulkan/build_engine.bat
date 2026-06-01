@echo off
REM Build libhybrid_engine.so for Android (Adreno 730)
set NDK=D:\android-ndk-r27d-windows\android-ndk-r27d
set TC=%NDK%\toolchains\llvm\prebuilt\windows-x86_64
set VK_SDK=D:\Vulkan_SDK
set OUT=D:\AI\anima_phone\hybridops\vulkan

echo === Compiling shaders ===
REM The .comp sources are in D:\AI\anima_phone\vulkan\ — reuse them
set VKSRC=D:\AI\anima_phone\vulkan

REM GEMM fp16vec4 (fast path)
"%VK_SDK%\Bin\glslangValidator.exe" -V "%VKSRC%\gemm_fp16.comp" -o "%OUT%\gemm_fp16.spv"
if %ERRORLEVEL% neq 0 echo ERROR: gemm_fp16 compile failed && exit /b 1

REM LayerNorm FP32
"%VK_SDK%\Bin\glslangValidator.exe" -V "%VKSRC%\layernorm_fp32.comp" -o "%OUT%\layernorm_fp32.spv"
if %ERRORLEVEL% neq 0 echo ERROR: layernorm_fp32 compile failed && exit /b 1

REM RMSNorm FP16
"%VK_SDK%\Bin\glslangValidator.exe" -V "%VKSRC%\rms_norm_fp16.comp" -o "%OUT%\rms_norm_fp16.spv"
if %ERRORLEVEL% neq 0 echo ERROR: rms_norm_fp16 compile failed && exit /b 1

REM GELU FP16
"%VK_SDK%\Bin\glslangValidator.exe" -V "%VKSRC%\gelu_fp16.comp" -o "%OUT%\gelu_fp16.spv"
if %ERRORLEVEL% neq 0 echo ERROR: gelu_fp16 compile failed && exit /b 1

echo === Compiling libhybrid_engine.so ===
"%TC%\bin\clang++.exe" --target=aarch64-none-linux-android28 ^
  --sysroot="%TC%\sysroot" -O2 -std=c++17 -fPIC -shared ^
  -I"%VK_SDK%\Include" ^
  -o "%OUT%\libhybrid_engine.so" "%OUT%\hybrid_engine.cpp" ^
  -Wl,-z,max-page-size=16384 -Wl,-z,common-page-size=16384 -Wl,--no-rosegment ^
  -llog -landroid -lvulkan ^
  -L"%TC%\sysroot\usr\lib\aarch64-linux-android\28" -static-libstdc++

if %ERRORLEVEL% equ 0 (
    echo === SUCCESS ===
    echo   %OUT%\libhybrid_engine.so
    echo   %OUT%\gemm_fp16.spv
    echo   %OUT%\layernorm_fp32.spv
    echo   %OUT%\rms_norm_fp16.spv
    echo   %OUT%\gelu_fp16.spv
) else (
    echo === FAILED ===
    exit /b 1
)

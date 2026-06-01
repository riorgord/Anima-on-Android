@echo off
set NDK=D:\android-ndk-r27d-windows\android-ndk-r27d
set TC=%NDK%\toolchains\llvm\prebuilt\windows-x86_64
set VK=D:\Vulkan_SDK

echo Building libdit_vk_v2.so for Android ARM64...
echo.

:: Step 1: compile GLSL → SPIR-V (fp32 shaders only)
echo === Compiling fp32 shaders ===
set GLSLC=%VK%\Bin\glslangValidator.exe
for %%s in (gemm_bf16 rms_norm_fp32 layernorm_fp32 silu_fp32 gelu_fp32 scale_shift_fp32 rope_fp32 broadcast_fp32 attn_qkt_fp32 attn_softmax_fp32 attn_out_fp32 gate_fp32) do (
    if exist %%s.comp (
        echo   %%s.comp
        "%GLSLC%" -V %%s.comp -o %%s.spv >nul 2>&1
        if %ERRORLEVEL% NEQ 0 (
            echo   ERROR: %%s.comp failed!
            "%GLSLC%" -V %%s.comp -o %%s.spv
            exit /b 1
        )
    ) else (
        echo   WARNING: %%s.comp not found, skipping
    )
)
echo.

:: Step 2: build libdit_vk_v2.so
echo === Building libdit_vk_v2.so ===
"%TC%\bin\clang++.exe" ^
  --target=aarch64-none-linux-android28 ^
  --sysroot="%TC%\sysroot" ^
  -O2 -std=c++17 -fPIC -shared ^
  -I"%VK%\Include" ^
  -o libdit_vk_v2.so dit_engine_v2.cpp ^
  -Wl,-z,max-page-size=16384 ^
  -Wl,-z,common-page-size=16384 ^
  -Wl,--no-rosegment ^
  -llog -landroid -lvulkan ^
  -L"%TC%\sysroot\usr\lib\aarch64-linux-android\28" ^
  -static-libstdc++
if %ERRORLEVEL% EQU 0 (
    echo.
    echo === Build OK ===
    for %%F in (libdit_vk_v2.so) do echo Size: %%~zF bytes
    echo.
) else (
    echo.
    echo === BUILD FAILED ===
    exit /b 1
)

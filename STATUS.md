# Anima 项目状态摘要 (2026-05-25)

## 我们在做什么
在 Snapdragon 8+ Gen1 手机上运行 Anima 动漫风格图像生成模型 (2B DiT)。

## 工作区
- **主仓库**: `D:\AI\anima_phone\` (整理后的核心代码)
  - `src/` — predict2, llm_adapter, wan_vae 等
  - `scripts/` — phone_pipeline, precompute_context, run_diffsynth_final
  - `vulkan/` — GLSL 着色器, SPIR-V, vk_ops, vk_linear, main.cpp, 编译脚本
  - `models/` — 占位 (权重太大，从模型原路径读取)
- **WSL**: `/home/riorg/anima-work/` (开发、调试)
- **手机**: `/sdcard/anima_on_android/` (运行时脚本、权重、输出)
- **编译**: `D:\android_vk\build_android.bat\` (NDK + Vulkan)
- **NDK**: `D:\android-ndk-r27d-windows\`
- **Vulkan SDK**: `D:\Vulkan_SDK\`
- **模型**: `D:\AI\手坤的anima\models\` (diffusion, text_encoder, vae, lora)

## 已完成
- ✅ PC DiffSynth 推理基线 (3060 12G, base+ turbo)
- ✅ 手机 CPU 管线 (DiT 2B + VAE，256×256，123s/步，正确出图)
- ✅ VAE 格子 bug 根因：wan_vae.py 缺失 latent_mean/std 归一化，已修复
- ✅ Vulkan GEMM benchmark：Adreno 730 7.8× 加速，max_err=0.0001 (独立测试正确)

## 当前状态
- ⚠️ Vulkan GEMM 在管线内/`.so` 内出错，但独立 C++ 二进制完全正确。根因深于预期。

## Vulkan 加速调试进展 (2026-05-25 今天)

### Adreno 730 内存类型 (vk_mem_probe)
只有 2 种内存可用于 STORAGE_BUFFER (`memoryTypeBits=0x41`)：
- **Type 0** (`DEVICE_LOCAL`) — 纯 GPU 内存，CPU 不可访问
- **Type 6** (`DEVICE_LOCAL | HOST_VISIBLE | HOST_COHERENT | HOST_CACHED`) — CPU/GPU 共享

Type 3 (`HOST_VISIBLE | HOST_CACHED`，无 COHERENT) **不在 memoryTypeBits 里**，无法用于 storage buffer。

### 已验证的策略（独立 C++ 二进制 vk_type_test，全部 max_err=0）

| 策略 | 描述 | 结果 |
|------|------|------|
| COHERENT | Type 6 直用，无 flush/invalidate | standalone ✅ .so ❌ |
| NOCOHERENT | Type 6 + vkFlush/Invalidate (多余) | standalone ✅ |
| DEVICE_LOCAL | Type 0 计算 + Type 6 staging + vkCmdCopyBuffer | standalone ✅ .so ❌ |

### 诡异现象
- **独立 C++ 二进制** (`vk_type_test`, `vk_bridge`)：三种策略全部 max_err=0 ✅
- **`.so` 形式** (`libvk_gemm.so`)：通过 ctypes / dlopen 加载，输出全是 inf/nan ❌
- **CPU-only `.so`** (`libvk_gemm_cpu.so`)：通过 ctypes 加载，完全正确 ✅ → 排除 ctypes 传参问题
- 连 **direct staging**（不经过 device-local copy，只把 staging buffer 绑 descriptor dispatch）在 `.so` 里也出垃圾
- 即使 **每次 run 创建全新 buffer**（不重用），`.so` 也出垃圾
- **vk_bridge 独立二进制**（完全相同的 device-local+staging 逻辑）输出也是垃圾！

结论：**Vulkan compute 在 Adreno 730 上，只要代码以 `.so` 或特定方式编译就会出 bug**。独立 `main()` 二进制可能因为链接方式（非 PIC）或初始化时机不同而避开 bug。根本原因尚未确认。

### 已修改文件
- `D:\android_vk\build_android.bat\main.cpp` — 多版迭代（Type 3 尝试、device-local+staging、fresh buffers per call），当前版本是 device-local+staging+每次创建新鲜 buffer
- `D:\android_vk\build_android.bat\bridge.cpp` — 同上逻辑的独立二进制版本
- `D:\shoukunshangde_Anima_zancuun\` 下有 `vk_mem_probe`, `vk_type_test`, `dl_test` 等诊断工具
- `build_so.bat` — 编译 libvk_gemm.so
- `build_bridge.bat` — 编译 vk_bridge

### 后续可能方向
1. 调查为何 `.so` 里的 Vulkan 与独立 binary 行为不同（PIC vs non-PIC? 链接差异?）
2. 如果独立 binary 能稳定工作，改用 subprocess 调用 binary（如 vk_bridge）代替 ctypes .so
3. INT8 CPU 量化作为保底方案

## 关键技术路径
```
prompt → PC预计算context → 手机 DiT (FP16 CPU) denoising → WanVAE decode → PNG
```

## 快速恢复命令
```bash
# WSL 环境
conda activate /home/riorg/anima-work/.conda

# 手机 ADB
adb connect 192.168.0.104:5555  (WiFi)
MSYS_NO_PATHCONV=1 adb shell ...  (防止路径转换)

# 手机跑管线
adb shell "su -c 'taskset f0 /data/data/com.termux/files/usr/bin/python -u /sdcard/anima_on_android/scripts/phone_pipeline.py'"
adb logcat -d -s "VkGEMM:*"  # 看 Vulkan 日志

# Android NDK 编译
set NDK=D:\android-ndk-r27d-windows\android-ndk-r27d
set TC=%NDK%\toolchains\llvm\prebuilt\windows-x86_64
"%TC%\bin\clang++.exe" --target=aarch64-none-linux-android28 --sysroot="%TC%\sysroot" -O2 -std=c++17 -ID:\Vulkan_SDK\Include -o output source.cpp -Wl,-z,max-page-size=16384 -Wl,-z,common-page-size=16384 -Wl,--no-rosegment -llog -landroid -lvulkan -L"%TC%\sysroot\usr\lib\aarch64-linux-android\28" -static-libstdc++

# 手机 ADB 推文件 (必须用 MSYS_NO_PATHCONV)
MSYS_NO_PATHCONV=1 adb push D:/file /sdcard/anima_on_android/
```

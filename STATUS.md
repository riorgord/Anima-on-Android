# Anima 项目状态摘要 (2026-05-26 更新)

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
- ⚠️ Vulkan GEMM 在真实 shape（M=512, N=2048）下出错，输出有"漏算位置 vk=0.0000"。
  之前 STATUS.md 中关于 ".so 出垃圾" 的所有诊断已被推翻：根因是 binary/dl_test 当时用 `tile=4` 跑触发 shader 退化，参数污染。
  当前怀疑：shape 较大 + 真实数据触发的 shader 或 driver bug，跟 K 是否 16 倍数无关、跟 .so vs binary 无关。

## Vulkan 加速调试进展 (2026-05-25 今天)

### Adreno 730 内存类型 (vk_mem_probe)
只有 2 种内存可用于 STORAGE_BUFFER (`memoryTypeBits=0x41`)：
- **Type 0** (`DEVICE_LOCAL`) — 纯 GPU 内存，CPU 不可访问
- **Type 6** (`DEVICE_LOCAL | HOST_VISIBLE | HOST_COHERENT | HOST_CACHED`) — CPU/GPU 共享

Type 3 (`HOST_VISIBLE | HOST_CACHED`，无 COHERENT) **不在 memoryTypeBits 里**，无法用于 storage buffer。

### 已验证的策略（独立 C++ 二进制 vk_type_test，全部 max_err=0）

| 策略 | 描述 | 结果 |
|------|------|------|
| COHERENT | Type 6 直用，无 flush/invalidate | standalone ✅ ~~.so ❌~~ |
| NOCOHERENT | Type 6 + vkFlush/Invalidate (多余) | standalone ✅ |
| DEVICE_LOCAL | Type 0 计算 + Type 6 staging + vkCmdCopyBuffer | standalone ✅ ~~.so ❌~~ |

~~.so ❌~~ 这两条结论 2026-05-25 时是基于参数污染的测试得出的，已推翻。详见下方"现象（2026-05-26 重新核实后修正）"。

### 现象（2026-05-26 重新核实后修正）

**Binary 路径基线（已重新核实，全部正确）：**
- `vk_type_test`：三种 memory 策略（COHERENT/NOCOHERENT/DEVICE_LOCAL），64×64×64 → max_err=0 ✅
- `vk_bridge`：64×64×64 tile=16，identity@ones → max_err=0 ✅
- `dl_test64`（独立 binary + dlopen libvk_gemm.so）：64×64×64 tile=16 → max_err=0 ✅

**~~vk_bridge 出垃圾~~** ← 错。原因是当时跑 `tile=4`（gemm.spv specialization=16，tile=4 触发 workgroup 配置退化）。改 tile=16 后正常。

**~~`.so` 通过 dlopen 加载出 inf/nan~~** ← 错。原因是 dl_test.cpp 当时用 `M=N=K=4`（即使 init(64,64,64,16)，run 的 4×4 同样触发 shader 退化）。改 64×64 后 dl_test64 完美。

**~~direct staging / fresh buffer / .so 路径 各种"诡异 garbage"~~** ← 全部基于同一参数污染，不可信。

**真实翻车点（2026-05-26 真管线 diagnostic 发现）：**
- `phone_pipeline` 1 步 DiT 第一个 GEMM = `x_embedder.proj` (M=512, N=2048, K=68)，max_err=2.6，nan=0 inf=0
- 用 `dl_test_K68` 复现：M=512, N=2048, K∈{64,68,80} randn 数据 → 全部 max_err≈0.55，**worst 位置 vk_out=0.0000**（shader 漏算）
- 跟 K 是否 16 倍数无关；跟 .so vs binary 无关；跟数据是否非平凡有关（identity+ones 对，randn 错）

**当前怀疑**：大 shape 下 shader 或 driver 漏算了某些 dispatch group。下一步 B+A 结合定位（读 shader + 针对性扫边界）。

### CPU-only `.so` 排除项
- `libvk_gemm_cpu.so` 通过 ctypes 加载完全正确 → 排除 ctypes 传参问题，问题在 Vulkan 路径本身。

### 已修改文件
- `D:\android_vk\build_android.bat\main.cpp` — 多版迭代（Type 3 尝试、device-local+staging、fresh buffers per call），当前版本是 device-local+staging+每次创建新鲜 buffer
- `D:\android_vk\build_android.bat\bridge.cpp` — 同上逻辑的独立二进制版本
- `D:\android_vk\build_android.bat\dl_test64.cpp` + `build_dl_test64.bat` — 2026-05-26 新加，binary 内 dlopen .so 跑 64×64×64 验证（已 PASS）
- `D:\android_vk\build_android.bat\dl_test_K68.cpp` — 2026-05-26 新加，复现真实管线 shape (M=512 N=2048 K=68) 翻车
- `D:\shoukunshangde_Anima_zancuun\` 下有 `vk_mem_probe`, `vk_type_test`, `dl_test` 等诊断工具
- `build_so.bat` — 编译 libvk_gemm.so
- `build_bridge.bat` — 编译 vk_bridge
- `vulkan/vk_ops_diag.py` (anima_phone 仓库新加) — 强制启用 Vulkan + 每次 CPU 对照 + 异常 raise 的诊断版 HybridOps；常态 `vk_ops.py` 保持 `_VK_AVAILABLE=False` 干净，需要诊断时 phone_pipeline 把 `import vk_ops` 换成 `import vk_ops_diag as vk_ops`

### 后续可能方向（2026-05-26 更新）
1. **B+A 结合定位**（进行中）：读 `vulkan/gemm.comp` shader 找边界处理漏洞 + 用 `dl_test_K68` 扫描 M/N/data 维度找阈值
2. 如果是 shader 漏边界检查 → 修 shader 重编 .so
3. 如果是 driver bug → 在 vk_ops 加 shape/数据范围阈值绕过；或者 subprocess 调 binary
4. INT8 CPU 量化作为保底方案

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

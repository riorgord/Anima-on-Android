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
- ✅ **Vulkan GEMM dispatch swap bug 定位并修复** (2026-05-26)：gemm.comp 里 row/col 映射跟 dispatch 的 X/Y 轴对调 — M=N 时隐藏，M≠N 时 columns 被截断输出 0。修法只改 shader 2 行（row/col 与 local_row/col 同时 swap），libvk_gemm.so 不变。修复后 281 GEMM 全部 0 翻车，max_err<0.02 ✅

## 当前状态
- ⚠️ Vulkan GEMM dispatch swap bug **已修复**（gemm.comp + gemm.spv 入仓），但 .so 尚未重编（运行时读 gemm.spv 自动生效）。待定：是否尝试完整 3 步管线出图（Vulkan 加速 vs CPU 参考）、是否入仓。
- ⚠️ 手机端 `/data/local/tmp/gemm.spv` = 修复版；`/data/local/tmp/gemm_broken.spv` = 旧备份

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

### Dispatch swap 根因定位全记录 (2026-05-26)

**早上的混乱状态**：STATUS.md 声称 .so 路径出 garbage，但所有证据被参数污染（tile=4 退化触发 inf/nan）。重新核实后 binary baseline 干净。

**D2 阶段（GEMM shape 分析）**：推出 phone_pipeline 走 .so 的 shape 列表 → 全是 16 倍数，K-divisible 假说证伪。

**真管线 diagnostic**：phone_pipeline 1 步 + 每次 CPU 对照 → 第一个 GEMM `x_embedder.proj` (M=512,N=2048,K=68) max_err=2.6, vk_out 有 0 位置。

**dl_test_scan 边界扫描**：M=512,N=64 → 88% zeros；M=64,N=2048 → 97% zeros。假说为"N 维度截断"。

**B 阶段（读 shader）**：`gemm.comp:55-56 row=gl_GlobalInvocationID.y, col=gl_GlobalInvocationID.x` ← 跟 `main.cpp:103 vkCmdDispatch(..., (M+15)/16, (N+15)/16, 1)` 配合，M→X groups、N→Y groups。结果是：
- col (N 维度) 由 X 轴供给 → 只覆盖 `0..M_padded-1` → **M < N 时 N 维度被截断**
- row (M 维度) 由 Y 轴供给 → 只覆盖 `0..N_padded-1` → M < N 时超出部分被边界检查 prune（浪费但不错）

**修复**：`gemm.comp` 交换全局 + 局部两对 ID：
```glsl
// 前    uint row = gl_GlobalInvocationID.y;  uint col = gl_GlobalInvocationID.x;
// 后    uint row = gl_GlobalInvocationID.x;  uint col = gl_GlobalInvocationID.y;
// 前    uint local_row = gl_LocalInvocationID.y;  uint local_col = gl_LocalInvocationID.x;
// 后    uint local_row = gl_LocalInvocationID.x;  uint local_col = gl_LocalInvocationID.y;
```
重编 gemm.spv（glslangValidator -V），推 `/data/local/tmp/gemm.spv`，.so 运行时自动读入无需重编。

**修复验证**：
- dl_test_scan 512×64×64: max_err=0.0077, 0% zero ✅（前：5.11, 88% zero）
- dl_test_scan 512×2048×68: max_err=0.0082, 0% zero ✅（前：0.56, 75% zero）
- phone_pipeline 1 步 diagnostic: 281 Vulkan GEMM 全部 max_err<0.02, OK ✅

**教训**：不是 driver bug，不是 .so vs binary 差异，不是 K 不整除 16。是一个 dispatch 轴映射错误在 M=N 的 benchmark 中完美隐藏了**一个多月**。

### 后续可能方向（2026-05-26 更新）
1. ~~B+A 结合定位~~ **已完成**，dispatch swap bug 修复
2. 重新 benchmark：M=512,N=2048（真实 shape）的 7.8× 加速是否还在
3. 完整 3 步管线 + Vulkan 加速出图 vs CPU 参考图，确认无误
4. 如果加速无误 → 将 `_VK_AVAILABLE` 开关打开，正式启用
5. INT8 CPU 量化作为保底方案

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

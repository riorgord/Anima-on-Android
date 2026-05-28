# anima-phone

在骁龙 Android 手机上运行 [Anima](https://huggingface.co/circlestone-labs/Anima)（2B 动漫风格 DiT 模型），纯 CPU 推理。

## 硬件

- Redmi K50 Ultra / Xiaomi 12T Pro
- Snapdragon 8+ Gen 1, 12GB RAM, Android 12
- Termux + PyTorch 2.11

## 快速开始

```bash
# 手机端 (Termux + root, 配合 Scene 调度器优化)
python -B /sdcard/anima_on_android/scripts/phone_pipeline.py
```

输出：256×256 PNG，约 50s/步（HybridOps Vulkan GEMM），3 步共约 3 分钟。

## 管线

```
prompt → PC 预计算 context → 手机 DiT 去噪 (FP16 CPU) → WanVAE 解码 → PNG
```

## 手机部署步骤

### 1. Termux 环境

安装 Termux 并配置 root 权限。需要安装 Python 和 PyTorch：

```bash
# Termux 内
pkg install python python-numpy
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install pillow
```

### 2. 推送文件到手机

```bash
# PC 端（Git Bash 需要 MSYS_NO_PATHCONV=1 防止路径转换）
MSYS_NO_PATHCONV=1 adb push src/ /sdcard/anima_on_android/src/
MSYS_NO_PATHCONV=1 adb push scripts/ /sdcard/anima_on_android/scripts/
MSYS_NO_PATHCONV=1 adb push models/ /sdcard/anima_on_android/models/
```

### 3. 预计算 context（PC 端）

在 PC 上运行一次，生成 `context_cond.pt` 和 `context_uncond.pt`，推送到手机：

```bash
python scripts/precompute_context.py
MSYS_NO_PATHCONV=1 adb push models/context_cond.pt /sdcard/anima_on_android/models/
MSYS_NO_PATHCONV=1 adb push models/context_uncond.pt /sdcard/anima_on_android/models/
```

### 4. 转换模型权重

在 PC 上将 DiffSynth 权重导出为 `state_dict`，保存为 FP16 `.pt` 文件，推送到手机 `models/` 目录：
- `diffusion_weights_fp16.pt` — DiT 权重
- `vae_weights_fp16.pt` — WanVAE 权重

### 5. 运行

```bash
# 1. 先杀残留进程（必须分两步，不能和 python 拼在同一条命令里）
adb shell "su -c 'am force-stop com.termux'"

# 2. 清空旧日志和 pyc 缓存（否则可能读到上次运行的过期输出）
adb shell "su -c 'rm -rf /sdcard/anima_on_android/scripts/__pycache__'"

# 3. 运行管线（建议 USB 连接，WiFi ADB 可能断连）
adb shell "su -c 'taskset f0 /data/data/com.termux/files/usr/bin/python -u -B /sdcard/anima_on_android/scripts/phone_pipeline.py'"
```

**注意事项：**
- `taskset f0` 绑定大核，避免小核拖慢
- `python -B` 禁止 `.pyc` 缓存；`python -u` 禁用 stdout 缓冲，确保日志实时输出
- **不能**把 `am force-stop` 和 `python` 写在同一行 — `am force-stop` 会杀掉 adb shell 自身
- **MIUI 熄屏降频**：熄屏后 CPU 约 4.5× 慢，务必 `screen_off_timeout=30min` 或保持亮屏
- 每次运行前杀掉残留进程，否则多个 Python 进程叠加导致 OOM
- **USB 优于 WiFi ADB**：WiFi ADB 在长时间运行时容易断连，USB 更稳定

### 6. 查看日志

管线 stdout 直接打印进度，无需额外配置。从 PC 监控：

```bash
# 实时看 stdout（adb shell 运行中时可见）
adb logcat -d -s "python:*"

# 或者把输出重定向到文件再拉回来
adb shell "su -c 'taskset f0 ... > /sdcard/anima_on_android/output/run.log 2>&1'"
adb pull /sdcard/anima_on_android/output/run.log .
```

正常输出示例：
```
Loading DiT...
DiT loaded
Denoising 3 steps, H=32...
  step 1/3: 120s (total 120s)
  step 2/3: 119s (total 239s)
  step 3/3: 120s (total 359s)
VkGEMM: 0 Vulkan, 1362 CPU calls
Loading VAE...
Decoding...
Saved: /sdcard/anima_on_android/output/phone_first.png
TOTAL: 3 steps, 375s (125s/step), 256x256
```

## 目录结构

```
src/          DiT, LLMAdapter, WanVAE, scheduler
scripts/      phone_pipeline, precompute_context, DiffSynth baseline
vulkan/       GLSL compute shaders, SPIR-V, NDK build scripts, vk_ops
models/       模型权重（占位，实际从原路径读取）
output/       生成图片输出
```

## Vulkan GPU 加速状态

**C++ 引擎 `libdit_vk.so`**：28-block DiT forward **13.06s/步**（vs CPU 120s, 9.2×）。

**已验证**：GEMM、LayerNorm、RMSNorm、SiLU、ScaleShift、self+cross+MLP blocks (skip-attention)。

**已实现待验证**：GPU AdaLN（benchmark 通过，完整管线验证未完成）。

**已探索暂回退**：Attention / RoPE 集成——遇到 Adreno 多 dispatch 并行执行导致的同步问题（A shader 隔离正确、管线中 binding 错乱）。详见 `STATUS.md`。

## 致谢

- 模型：[circlestone-labs/Anima](https://huggingface.co/circlestone-labs/Anima)
- 代码改编自：ComfyUI ([GPL-3.0](https://github.com/comfyanonymous/ComfyUI))、[DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio) (Apache 2.0)、NVIDIA Cosmos (Apache 2.0)、Wan-Video VAE (Apache 2.0)
- Vulkan GEMM shader 参考：[ncnn](https://github.com/Tencent/ncnn) (BSD-3-Clause) by Tencent — innerproduct pack4 + fp16 vectorization 设计思路
- 开发辅助：DeepSeek V4 Pro + Claude Code

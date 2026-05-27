# Anima 项目状态摘要 (2026-05-27 晚间更新)

## 我们在做什么
在 Snapdragon 8+ Gen1 手机上运行 Anima 动漫风格图像生成模型 (2B DiT)。

## 工作区
- **主仓库**: `D:\AI\anima_phone\` — vulkan/ (GLSL+SPIR-V+C++引擎), src/ (DiT/VAE), scripts/ (管线+测试)
- **手机**: `/sdcard/anima_on_android/` — 运行时脚本、权重、输出
- **编译**: `D:\android_vk\build_android.bat\` (NDK + Vulkan)
- **NDK**: `D:\android-ndk-r27d-windows\`
- **Vulkan SDK**: `D:\Vulkan_SDK\`
- **模型**: `D:\AI\手坤的anima\models\` (diffusion, text_encoder, vae, lora)

## 速度演进

| 方案 | 速度 | vs CPU | 状态 |
|------|------|--------|------|
| CPU only (亮屏) | 120s/步 | 1× | ✅ |
| Python ctypes per-layer Vulkan | 56s/步 | 2.1× | ✅ |
| C++ 引擎 self-attn+MLP only | 9.9s/步 | 12× | ✅ |
| C++ 引擎 +cross-attn | 13.3s/步 | 9× | ✅ |
| **C++ 引擎 +GPU AdaLN** | **13.06s/步** | **9.2×** | **✅ 当前** |
| C++ 引擎 +RoPE+Attention (目标) | ~15s/步 | 8× | ⏳ 开发中 |

## 当前状态 (2026-05-27 晚间): C++ 引擎重写成功

**libdit_vk.so v2**：完全重写，架构改为 per-block cmd buffer（28 个，每个 16 dispatches），回避了旧版 monolithic cmd buffer 的 Adreno submit 失败问题。

### 已验证通过的链路 (PyTorch reference 对比)

| 测试项 | max_err | 说明 |
|--------|---------|------|
| GEMM (单次) | 0.004 | fp16 精度极限 |
| LayerNorm | 0.004 | |
| SiLU (MLP 路径) | 0.031 | 8192-dim fp16 累积 |
| ScaleShift (AdaLN apply) | 0.008 | |
| LN+4×GEMM barrier chain | 0.004 | 内存屏障正确 |
| Self-attn full (LN+AdaLN+QKV+norms+O+gate) | 2.5 | 值 ~294, 2.6 ULP |
| MLP (LN+fc1+SiLU+fc2+gate) | 0.031 | |
| **Block 0 真实 pipeline 输入** | **0.75** | mean/std 完全匹配 |
| 28 blocks benchmark | — | 9.9s/步, 448 dispatches |

### 引擎架构

```
dit_init(weight_bin, spv_dir)
  → 加载 567 个 per-tensor Vulkan buffer (3.9GB)
  → 创建 8 个 shader pipeline
  → 分配 28 个 cmd buffer + I/O buffer

dit_init_all_blocks()
  → 每个 cmd buffer 录制 1 个 block (16 dispatches)
  → bcBuf 共享 (18MB, 9 个 AdaLN 分量)

dit_forward_28blocks(x, adaln_all, out)
  → for i in 0..27:
      上传 block i 的 AdaLN → bcBuf
      上传 x → xBuf
      submit cmd[i] → wait fence
      复制 outBuf → xBuf (下一 block 输入)
    → 9.9s total
```

### Shader 验证状态

| Shader | 独立验证 | Block 内集成 |
|--------|---------|-------------|
| GEMM (gemm_fp16) | ✅ | ✅ |
| LayerNorm | ✅ | ✅ |
| RMSNorm | ✅ | ✅ |
| SiLU | ✅ | ✅ |
| ScaleShift | ✅ | ✅ |
| Broadcast | ❌ 未测 | — |
| Attention | ❌ 未测 | — |
| RoPE | ❌ 未测 | — |

### 未完成

- **RoPE + Attention** ⚠️ 已尝试（commit b729aca），发现两个问题：
  1. Attention shader 的 K/V layout 假设错误（原为 batch-head-token，实际 batch-token-head），导致格子 artifact。已定位修复但 SPIR-V 加载了旧文件
  2. Cross-attn Q_proj/Q_norm 用错权重（用了 self 的），已定位修复
  3. Workgroup 粒度过细（8192+16384/block），53s/步 → 需优化 attention shader
  4. 已回退到 92ee6b2，保留 skip-attention
- x_embedder, t_embedder, final_layer C++ 移植
- VAE decode（暂留 PyTorch）
- 端到端 phone_pipeline 出图（pipeline_cpp.py 已可用但出图有格子）

### 关键技术发现

- **Adreno 单 cmd buffer 上限 ~64 dispatches**：超过后 vkQueueSubmit 失败
- **单 buffer 上限 < 3.9GB**：weight buffer 分配失败，改 per-tensor buffer 解决
- **fp16 溢出于 block 23**：随机输入下残差累积突破 65504
- **验证策略**：PyTorch dump → 卸载 → C++ 对比
- **Attention Q/K/V layout**：(batch, token, head) 而非 (batch, head, token)，shader KV 访问需 stride by n_heads
- **Attention workgroup 过细**：8192 wg/self-attn → 53s/步，需合并 query rows 到一个 wg

---

## 历史记录

<details>
<summary>2026-05-27 上午：C++ 引擎回顾 & 路线修正 (已推翻)</summary>

旧 libdit_vk.so 编译成功但从未产出非零输出。排查方向包括 descriptor pool 生命周期、command buffer 大小限制（784 dispatches）、Adreno 驱动 bug。当时结论是"放弃独立引擎路线"，改为 Python ctypes + 扩展 libvk_gemm.so。

**事后分析**：根因不是驱动 bug，而是 784 dispatches 超过了 Adreno 单 cmd buffer 限制。改为 28×16 的 per-block cmd buffer 架构后解决。
</details>

<details>
<summary>2026-05-26 晚间：性能剖析 & 路线决策</summary>

- GEMM shader 只有 14 GFLOPS (0.75% GPU 利用率)
- 优化无效项：批量提交、Type 6 HOST_COHERENT、shared memory tiling
- ncnn pack4 + f16vec4+dot → 149 GFLOPS (10.6×)
- Dispatch swap bug 定位：gemm.comp row/col 映射跟 dispatch X/Y 对调
</details>

<details>
<summary>2026-05-25：早期里程碑</summary>

- PC DiffSynth baseline ✅
- 手机 CPU 管线: 120s/步 ✅
- VAE grid bug 修复 (latent mean/std 归一化) ✅
- Vulkan GEMM standalone 验证 ✅
- C++ 引擎 skeleton (旧版, 后来重写) ✅
</details>

## 手机端关键文件

- `/data/local/tmp/libdit_vk.so` — C++ 引擎
- `/data/local/tmp/libvk_gemm.so` — 旧 GEMM .so (Python ctypes 用)
- `/data/local/tmp/diffusion_weights.bin` — 3.9GB 权重 (567 tensors)
- `/data/local/tmp/*.spv` — 10 个 SPIR-V shader
- `/sdcard/anima_on_android/scripts/phone_pipeline.py` — 端到端管线
- `/sdcard/anima_on_android/models/` — 权重 .pt 文件 + context

## 快速恢复命令

```bash
# 手机 ADB
adb connect 192.168.0.104:5555  (WiFi)
MSYS_NO_PATHCONV=1 adb shell ...  (防止 Git Bash 路径转换)

# 手机跑 Python 管线
adb shell "su -c 'taskset f0 /data/data/com.termux/files/usr/bin/python -u /sdcard/anima_on_android/scripts/phone_pipeline.py'"
adb logcat -d -s "DiT_VK:*"  # 看 C++ 引擎日志

# GLSL → SPIR-V 编译
D:\Vulkan_SDK\Bin\glslangValidator.exe -V shader.comp -o shader.spv

# NDK 编译 libdit_vk.so (一行版)
set NDK=D:\android-ndk-r27d-windows\android-ndk-r27d
set TC=%NDK%\toolchains\llvm\prebuilt\windows-x86_64
"%TC%\bin\clang++.exe" --target=aarch64-none-linux-android28 --sysroot="%TC%\sysroot" -O2 -std=c++17 -fPIC -shared -I"D:\Vulkan_SDK\Include" -o libdit_vk.so dit_engine.cpp -Wl,-z,max-page-size=16384 -Wl,-z,common-page-size=16384 -Wl,--no-rosegment -llog -landroid -lvulkan -L"%TC%\sysroot\usr\lib\aarch64-linux-android\28" -static-libstdc++

# 或者直接跑 build_dit.bat（含 shader 编译 + .so 编译）
cd D:\AI\anima_phone\vulkan && build_dit.bat

# 推送文件到手机
MSYS_NO_PATHCONV=1 adb push D:/AI/anima_phone/vulkan/libdit_vk.so /data/local/tmp/
MSYS_NO_PATHCONV=1 adb push D:/AI/anima_phone/vulkan/*.spv /data/local/tmp/
MSYS_NO_PATHCONV=1 adb push D:/AI/anima_phone/scripts/test_*.py /sdcard/anima_on_android/scripts/

# 权重导出 (PC 端, PyTorch .pt → raw .bin)
python D:/AI/anima_phone/scripts/export_weights.py D:/AI/anima_phone/models/diffusion_weights_fp16.pt D:/AI/anima_phone/models/diffusion_weights.bin

# 运行测试 (手机端)
adb shell "su -c 'taskset f0 /data/data/com.termux/files/usr/bin/python -u /sdcard/anima_on_android/scripts/test_dit_engine.py'"
```

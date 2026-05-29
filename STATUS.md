# Anima 项目状态摘要 (2026-05-29 更新)

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
| **phone_pipeline.py (HybridOps)** | **50s/歩** | **2.4×** | **✅ 管线级正确 (81KB PNG)** |
| C++ 引擎 self-attn+MLP only | 9.9s/步 | 12× | ✅ |
| C++ 引擎 +cross-attn | 13.3s/步 | 9× | ✅ |
| **C++ 引擎 +GPU AdaLN** | **13.06s/步** | **9.2×** | benchmark 通过, 管线未验证 |
| C++ 引擎 +RoPE+Attention (目标) | ~15s/步 | 8× | ⏳ 开发中 |

注：2026-05-28 去掉了 `taskset f0` 绑定大核，配合 Scene 调度器优化，HybridOps 稳定在 50s/步。之前 56-57s 是单纯 taskset 限制 4 核的结果。

## C++ 引擎状态 (libdit_vk.so v2)

完全重写，架构改为 per-block cmd buffer（28 个，每个 16 dispatches），回避了旧版 monolithic cmd buffer 的 Adreno submit 失败问题。

⚠️ **C++ 引擎已知缺陷**：
- Attention 是 skip 模式（V→O），非真正 QK^T+softmax
- RoPE 未集成到 block recording

## 当前状态 (2026-05-30 更新): 自注意力 + 跨注意力 GPU 化完成

**管线速度**: 77s/步 (息屏+省电调度)

### 已注入的 GPU 模块（全部管线验证通过）

| 模块 | 引擎 | 每步调用 | 精度 | 注入方式 |
|------|------|---------|------|---------|
| **GEMM** | libvk_gemm.so | 281 | max_err ~0.01 | HybridLinear |
| **AdaLN** | libdit_vk.so | 28 blocks | — | PrecomputedAdaLN |
| **LayerNorm** | libdit_vk.so FP32 | 85 | max_err 2e-6 | HybridLayerNorm |
| **RMSNorm** | libdit_vk.so FP16 | 84 | max_err 0.004 | HybridRMSNorm |
| **GELU** | libdit_vk.so FP16 | 28 | max_err 0.0017 | post-init replace |
| **Self-attention** | libdit_vk.so 3-pass | 28 | — | monkey-patch compute_attn |
| **Cross-attention** | libdit_vk.so 3-pass batched | 28 | max_err 0.001 | monkey-patch compute_attn |
| **t_embedder** | libdit_vk.so C++ CPU | 1 | max_err 6e-5 | dit_compute_timestep |

### 描述符池体系（最终方案）

| 池 | 用途 | 手动 free | 步间 reset | 容量 |
|---|------|----------|-----------|------|
| **descPool** | init 时 AdaLN 预录（仅一次） | 否 | 永不 | maxSets=6000 |
| **stepPool** | 运行时 LN/RMS/GELU/AdaLN/attn 录制 | 否 | 每步 vkResetDescriptorPool | maxSets=3000 |

AdaLN 重录时 `dit_adaln_one_block` 临时 swap `g_vk.descPool → g_vk.stepPool`，录完换回。

### Cross-attention 修复 (2026-05-30)

**根因（疑似）**: Adreno 730 可能存在 250ms TDR 看门狗（Qualcomm 文档提及）。单 dispatch 若 GPU 执行时间超过此阈值，驱动强制复位 → `VK_ERROR_DEVICE_LOST`。但我们的实测环境 GPU 底频 510MHz，全计算量理论上不应超时，且错误模式是 `max_err=0.28` 部分值错误而非全零/随机垃圾，更接近 GPU 内部资源冲突（L1 cache eviction 或 barrier 同步竞争）而非硬复位。确切根因待确认。

**修复**: `dit_run_attention` 内部将 Q 维度拆分为 `batch_q = 64` 的批次（每批 `64*16 = 1024 WG`），逐批 submit+wait。每批 dispatch 在 TDR 窗口内完成。K/V 上传一次复用，Q 分批上传。

| 参数 | 值 |
|------|-----|
| batch_q | 64 (1024 WG) |
| 每 attn 调用 submit 数 | 8 (=512/64) |
| max_err (cross-attn) | 0.0013 |
| 跨步稳定性 | ✅ 多次管线验证通过 |

### Adreno 730 已知限制 & 疑似问题 (2026-05-30)

| 发现 | 说明 | 确定性 |
|------|------|--------|
| **TDR 250ms** | Qualcomm 文档提及看门狗阈值。单 dispatch 超时可能触发 DEVICE_LOST。但 510MHz 底频下计算量理论上不会超时 | ⚠️ 疑似 |
| **WG 软上限** | `maxComputeWorkGroupCount=65535` 是理论值。实测大 M_kv+大 WG 数+密集 barrier 的组合触发不稳定 | ✅ 实测 |
| **并行 dispatch binding confusion** | 同一 cmd buffer 内多 dispatch 可并行执行，不同 descriptor 可能混淆 | ✅ 已踩坑 |
| **部分错误模式** | ROWS=8 时 max_err=0.28，非全零/随机，更像 cache eviction 或 barrier race 而非 TDR 硬复位 | ❓ 待确认 |
| **官方手册** | 500 系手册 74 页 (本地)。7XX 系 docs.qualcomm.com 按平台代号查 | — |

### GEMM 单实例合并实验 (2026-05-30，未成功)

**目标**：把 GEMM 从 libvk_gemm.so 迁入 libdit_vk.so（单 Vulkan 实例），消两实例竞争。

**实测**（4 次逐级测试）：

| 测试 | 改动 | 结果 |
|------|------|------|
| 基线 | 双实例（当前架构） | 63s/步 ✅ |
| 测试 1 | +32MB buffer, dit_run_gemm, 共享 fence+cBuf | 137s, GEMM 全 CPU |
| 测试 2 | 测试 1 + 独立 GEMM fence | 135s, 同上 |
| 测试 3 | 测试 1 + 独立 GEMM fence + 独立 cmdBuf | 133s, 同上 |
| 烟雾测试 | 测试 3 + libvk_gemm.so 作为 idle 实例加载 | 不变, attn #1 开始全挂 |

**现象**：只要 libdit_vk.so 额外分配 GEMM buffer（哪怕仅 32MB），第二个 attention 调用就失败（`attn #1 batch 0/8 failed` → 后续全 CPU）。不是 fence/cmdBuf 共享的问题。

**当前判断**：根因未定位。怀疑与 Vulkan buffer 分配后 GPU 内存布局变化有关，可能纯属技巧问题而非架构缺陷。单实例路线暂搁置。

**验证过的**：
- `dit_run_gemm` 逻辑正确（新旧 GEMM 比特一致）
- 双实例架构稳定（5+ 次管线验证）

### 剩余 PyTorch CPU 项

| 模块 | 原因 |
|------|------|
| **RoPE** | 56 次 ~3s，仍在 compute_qkv 内 |
| **final_layer** | 1 次，可忽略 |

### 速度分析 (2026-05-30)

**基线**：63s/步（今晚实测，息屏+省电调度）

| 组件 | 每步耗时 | 状态 | 下一步 |
|------|---------|------|--------|
| **GEMM** (281 次) | ~25s | ✅ libvk_gemm.so | 预录进 block（内部） |
| **Self-attention** (28 blocks) | ~8s | ✅ 3-pass Vulkan | — |
| **Cross-attention** (28 blocks) | ~8s | ✅ 3-pass Vulkan | — |
| **RoPE** (56 次) | ~3s | ❌ CPU | GPU 化 |
| **final_layer** | <1s | PyTorch GEMM | C++ |
| 其他（LN/RMS/GELU/AdaLN/t_emb） | <2s | ✅ | — |
| Python/Vulkan submit 开销 | ~17s | — | 预录消 submit |
| **总计** | **~63s/步** | | **目标 ~40s（预录后）** |

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

## 2026-05-27 Attention 集成教训

Adreno 730 上同一 cmd buffer 内多个 compute dispatch 可并行执行（Vulkan 规范允许）。尝试了 VkMemoryBarrier、VkBufferMemoryBarrier、hard sync（submit+wait+新 cmd buffer）均无法修复 dispatch 间 binding 错乱。

Attention shader 三种实现：3-pass（fp16 重算累积误差大）、共享内存（隔离正确管线失败）、极简（太慢）。

后续策略：在 phone_pipeline.py HybridOps 框架上逐模块用 Vulkan 替换 PyTorch，每次替换必须与 PyTorch 做元素级对齐（max_err < 1），不可仅凭端到端结果判断。

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

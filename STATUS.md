# Anima 项目状态摘要 (2026-06-02 晚间)

## 我们在做什么
在 Snapdragon 8+ Gen1 手机上跑 Anima DiT 2B 模型。

## 当前状态（一句话）
**anima_rt 轻量推理库 v0.3！** 完成「不造轮子，只搬运」策略转型：从 PyTorch 源码抠出 FlashAttention (tiled online softmax)、RMSNorm (sum/N 公式)、GEMM (dlopen OpenBLAS) 等 kernel，全部公式级对齐 PT。出图 77,272 字节（PC 白盒 73,840，差 4.6%）。剩余未替换：Shell Linear、RoPE。VAE 暂留 PyTorch。

---

## 2026-06-02 全天：全后端搬运策略转型 + 误差系统化分析

### 核心策略变更
**不再手搓 kernel → 从 PT 源码抠，或 dlopen PT 同款依赖 (OpenBLAS)。**
手写 kernel 和 PT 的累加顺序/公式不同 → 误差无法消除。
正确做法：PT 用什么依赖我们就用什么，PT 用什么公式我们就搬什么公式。

### 成果

#### 1. SDPA: Math → Flash Attention (最大突破)
- **问题**：之前手搓的 math backend (matmul→softmax→matmul) 单次 err=0.99，出图 23KB（全黑）
- **根因**：PT 2.11 默认走 flash attention (tiled online softmax)，和 math backend 是不同数值路径
- **解决**：从 PT `FlashAttentionKernel.cpp` 抠 tiled online softmax 算法
- **结果**：Flash attention vs PT `F.sdpa` err=**4.77e-07** ✅，出图回到正常范围

#### 2. RMSNorm: Welford → PT sum/N 公式
- **问题**：手搓的 Welford RMS，err=9.5e-06
- **根因**：PT 的 RMSNorm 不是 dedicated kernel，是复合 `rsqrt(mean(x²)+eps)*weight`，用简单 sum/N
- **解决**：替换为 PT 的 exact 公式 (sum(x²)/N)
- **结果**：err 从 9.5e-06 → **6.68e-06** ✅

#### 3. SiLU 接线
- **问题**：`silu_kernel.h` 已存在且 bit-exact，但 `_patch_model_anima_rt` 没 patch `nn.SiLU`
- **解决**：补上 `AnimaRTSiLU` + `_patch_sequential_anima_rt` 处理 `nn.Sequential` 内的 SiLU
- **结果**：113 次/步全走 anima_rt，bit-exact ✅

#### 4. GEMM backend: dlopen OpenBLAS
- 手机上有 `libopenblas.so` (10MB)，`sbgemm_` 符号可用
- 实现 dlopen → `sbgemm_` (BF16×BF16→FP32) + 纯 C++ fallback
- FP32 纯 C++ err=4.77e-07，BF16 sbgemm_ err=4.77e-07 ✅

#### 5. Vulkan GEMM 误差根因分析
- **根因**：FMA 非结合律。shader 单 accumulator 串行链 vs OpenBLAS 多 accumulator 树归约
- **实验**：1acc (77,685) → 4acc (77,272) → 8acc (77,965)
- **结论**：调整 GLSL accumulator 数量只能有限改善，无法匹配 OpenBLAS 的特定 blocking。CPU/GPU 的 FMA 硬件舍入方向厂商差异无法消除
- **策略**：GEMM 未来走 QNN NPU（高通 NPU BF16 矩阵乘法是其核心卖点）；其他 op 可安全上 Vulkan（累加规模小，有 LN/Softmax 自压缩效应）

#### 6. P-1 PT 误差地板基准
- PT CPU FP32 vs CUDA BF16：最终 v_cond max_err = **0.0495**
- 块内误差在 b12 爆炸（×85），但 final_layer 压缩回来
- 这是理论下限：不可能比 PT 自身跨后端差异更小

### 当前管线 op 路由 & 误差

| Op | 次数/步 | 后端 | err vs PT | 搬运状态 |
|---|---------|------|-----------|---------|
| Block GEMM | ~280 | Vulkan | 0.0044 | 🔴 FMA 非结合律 |
| Shell Linear | ~174 | PT nn.Linear | 0 | ❌ 未替换 |
| Flash Attn | 56 | anima_rt CPU | 4.77e-07 | ✅ 搬 PT |
| SiLU | 113 | anima_rt CPU | **0** | ✅ 搬 PT |
| GELU | 28 | anima_rt CPU | 4.8e-07 | ✅ 搬 PT |
| LayerNorm | 85 | anima_rt CPU | 3.3e-06 | ✅ 搬 PT |
| RMSNorm | 113 | anima_rt CPU | 6.7e-06 | ✅ 搬 PT（刚改） |
| Softmax | 56 (独立) | anima_rt CPU | 2.2e-08 | ✅ 搬 PT |
| RoPE | 28 | PT | 0 | ❌ 未替换 |
| VAE | 1 | PT | — | ❌ 不碰 |

### 出图演进（今天）

| 版本 | 大小 | 说明 |
|------|------|------|
| PC 白盒 | 73,840 | 干净基准 |
| anima_rt v0.2 (之前) | 77,056 | math SDPA + Welford RMS |
| math SDPA (Path 3) | 23,304 | PT 公式验证，全黑 |
| Flash SDPA + 1acc | 77,722 | 搬到 flash，回到正常 |
| Flash SDPA + PT RMS | 77,685 | RMSNorm 搬完 |
| Flash SDPA + **4acc** | **77,272** | Vulkan GEMM 最佳调参 |
| Flash SDPA + 8acc | 77,965 | 过度优化反而倒退 |

### Vulkan 上各 op 风险判断

| Op | 累加规模 | 有自压缩？ | Vulkan 风险 |
|----|---------|----------|------------|
| SiLU/GELU | 1 element | N/A | ✅ 安全 |
| Softmax | 256~512 | ✅ softmax | ✅ 安全 |
| LN/RMSNorm | 2048 | ✅ /std | ✅ 安全 |
| FlashAttn | 256×128 | ✅ softmax | 🟡 中等（内部有 GEMM） |
| GEMM | 2048~8192 | 无 | 🔴 唯一高风险 |

### 关键教训
- **BF16 加载**：`.view(torch.bfloat16)` 做位重解释，`.to(torch.bfloat16)` 做值转换——用错权重全毁
- **FMA 非结合律**：GPU 和 CPU 的 FMA 硬件差异是物理壁垒，GEMM 无法 vulkan→PT 对齐
- **搬运 > 手搓**：今天无一例外——PT 的实现在精度和速度上都优于手写

---

## 自研轮子 vs 未替换清单 (2026-06-02 凌晨)

### ✅ 已替换（anima_rt + BF16 Vulkan GEMM）

| DiT 算子 | 每步次数 | 实现 | 来源 | 手机 err vs PT |
|----------|---------|------|------|---------------|
| **GEMM** (block 内) | 280 | Vulkan `gemm_bf16.comp` | BF16 权重 + FP32 FMA 累加 | 0.0044 (v_cond) |
| **GEMM** (AdaLN/head/tail) | 174 | Vulkan `gemm_bf16.comp` | 同上 | — |
| **LayerNorm** | 85 | `layernorm_kernel.h` | 抠自 PT `cpu/layer_norm_kernel.cpp` | 3.3e-06 |
| **RMSNorm** | 113 | `rmsnorm_kernel.h` | 自写 dedicated kernel | 9.5e-06 |
| **GELU** | 28 | `gelu_kernel.h` | 抠自 PT `cpu/Gelu.h` (exact erf) | 4.8e-07 |
| **SiLU** | 86 | `silu_kernel.h` | 抠自 PT `cpu/Activation.cpp` | **0** (bit-exact!) |
| **Softmax** | 56 | `softmax_kernel.h` | 抠自 PT `cpu/SoftMaxKernel.cpp` | 2.2e-08 |
| **Welford** | LN/RMS 内部 | `welford.h` | 简化自 PT `cpu/moments_utils.h` | — |

### ❌ 未替换（仍在 PyTorch / Vulkan 旧 shader）

| 算子 | 每步次数 | 还走 | 原因 |
|------|---------|------|------|
| **Attention SDPA** | 56 | PyTorch `F.scaled_dot_product_attention` | 多后端 dispatch 复杂，需 CPU→Vulkan→QNN 三后端 |
| **RoPE** | 28 | PyTorch `apply_rotary_pos_emb` | 纯 cos/sin 乘加，精度无损，非瓶颈 |
| **AdaLN modulation** | 84 | PyTorch `SiLU→Linear→Linear` | SiLU 已替换，小 Linear 走 Vulkan GEMM |
| **x_embedder** | 1 | PyTorch PatchEmbed | 单次调用，非瓶颈 |
| **t_embedder** | 1 | PyTorch TimestepEmbedding | 单次调用，SiLU 已替换 |
| **final_layer** | 1 | PyTorch FinalLayer | 单次调用 |
| **VAE decoder** | 1 | PyTorch WanVAE | 独立模块，3D conv 复杂 |

### 管线精度演进

| 版本 | 图片大小 | v_cond max_err vs PC | 说明 |
|------|---------|---------------------|------|
| PC 基准 (RTX 3060) | 73,840 | 0 | PyTorch CUDA |
| C++ v2 fp32 best | 75,000 | ~5 | 6 bug 修复后，手写 kernel 终止 |
| **anima_rt + BF16 GEMM** | **77,056** | **0.0044** | 今晚成果 |
| anima_rt + FP16 GEMM | 77,942 | 0.0059 | 昨晚 |
| HybridOps fp16 | 81,000 | — | 原始管线 |

## 2026-06-02 凌晨：anima_rt 轻量推理库 v0.1 ✅

### 成果
- **复制 hybridops → anima_rt**，对副本动刀
- **从 PyTorch 源码抠出 6 个 kernel**:
  | Kernel | 来源 | 手机 max_err vs PT |
  |--------|------|-------------------|
  | GELU | PT cpu/Gelu.h | 4.77e-07 (1 ULP) |
  | SiLU | PT cpu/Activation.cpp | **0.00e+00** (bit-exact!) |
  | LayerNorm | PT cpu/layer_norm_kernel.cpp | 3.34e-06 (14 ULP) |
  | RMSNorm | 自写 dedicated kernel | 9.54e-06 (40 ULP) |
  | Softmax | PT cpu/SoftMaxKernel.cpp | 2.24e-08 (<0.1 ULP) |
- **`libanima_rt.so`**: 454KB (Android), 18KB (host)
- **类型策略**: BF16 权重 + FP32 计算（不做 FP16），预留 INT8/BF16 对接 NPU
- **手机管线出图**: **77,942 字节**，无 NaN，v_cond 正常

### 误差 vs C++ v2 的本质区别
C++ v2 后来确实转了 FP32（shader 内部 fp32 累加+FMA），但**仍因并行归约求和顺序 ≠ PT 而失败**。即使都是 FP32，`(a+b)+(c+d) ≠ ((a+b)+c)+d`。
anima_rt 的标量 Welford 更接近 PT 的串行路径 → 误差 10⁻⁶ 级（FP32 的 ULP），而 v2 shader 的并行归约误差在 10⁻² 级 → 28 层累积后放大。

### 误差解释
| 来源 | 量级 | 原因 |
|------|------|------|
| GELU (1 ULP) | 2-5e-07 | NDK libm `erff` ≠ PT 的 libm |
| SiLU (0) | **0** | NDK `expf` 恰好和 PT 逐位一致 |
| LN/RMS (14-40 ULP) | 3e-06~1e-05 | 标量 Welford vs PT 向量化 cascade-sum 的累加顺序不同 |
| Softmax | 2e-08 | 算法结构简单，几乎无差异 |

### vs 基准对比
| 版本 | 图片大小 |
|------|---------|
| PC 基准 (RTX 3060) | 73,840 |
| C++ v2 fp32 best | 75,000 |
| **anima_rt (今晚)** | **77,942** |
| HybridOps fp16 | 81,000 |

### 架构
```
anima_rt/
  include/
    anima_tensor.h       ← 最小化 Tensor (FP32/BF16/INT8)
    anima_backend.h      ← 函数指针表 (CPU/Vulkan/QNN 可插拔)
    cpu/                 ← 抠自 PT 的 6 个 kernel headers + 3 stub
  src/                   ← tensor + backend + C API (3 文件)
  scripts/               ← Python ctypes 绑定 + 验证脚本
  libanima_rt.so         ← 454KB NDK 交叉编译产物
```

### 下一步
- GEMM: dlopen OpenBLAS `sbgemm_` 替代 PyTorch Linear
- Vulkan: 只加速 GEMM（已验证 149 GFLOPS），其余全走 anima_rt CPU kernel
- QNN NPU: BF16 直接对接高通 NPU

## 2026-06-01 晚间~凌晨：v2 fp32 管线大规模 debug（终止）

### 修复的 6 个 bug

| # | Bug | 修复位置 |
|---|-----|---------|
| 1 | loraBuf 布局 `[M,3D]` vs `[3,M,D]` | `dit_compute_timestep` 上传前转置 |
| 2 | CPU scale+1 flat index（i≥D 读错 batch） | `seg_adaln_cpu` per-batch 索引 |
| 3 | SiLU 顺序：PT 是 SiLU→Linear→Linear（无中间 SiLU），错误加了 SiLU #2 | **已 revert** |
| 4 | final_layer lora flat index（同 #2） | `head_tail_ops.h:final_layer` per-batch |
| 5 | `record_attn_3pass` sub-batch buffer offset bug | batch_q=M_q 消除分批 |
| 6 | GELU tanh 近似 → exact erf | `gelu_fp32.comp` 重写 |

### 验证通过（逐 op vs PT）

| 模块 | max_err | 
|------|---------|
| bcBuf (AdaLN 9 slot) | **0.000015** |
| g_nBuf (LN+调制) | **0.000092** |
| Q/K/V GEMM | **0.0003~0.0007** |
| RMSNorm | **0.000004** |
| RoPE | magnitude ratio=1.0 |
| Block 0 SA → CX → MLP 全链路 | SA=0.0015, CX=0.0000, MLP=0.05 |
| Block 27 C++ vs PT range | 基本一致 (`[-441k,10k]` vs `[-440k,10k]`) |

### 出图演进

| 版本 | 大小 | 说明 |
|------|------|------|
| v2 原始 (sin/cos fix) | 85KB | 有 lora/scale+1/SiLU 多个 bug |
| +所有修复 | **121KB** | 6 bug 修复 + GELU exact erf |

### 为什么终止

1. **Smoke test 只比 GPU vs CPU 不比 PT** — 自洽不代表正确
2. **每轮迭代 6 分钟** — 调试效率太低
3. **Python RoPE 实现有 bug** — PT 参考自己不准，丢失信任锚
4. **GEMM/GELU/softmax 的浮点累积差异** — 手写 shader 和 PT 的累加顺序不同，无法消除
5. **AI 生成的代码**（`cpu_gemm_bf16`、shader）没有经过 PyTorch 级别的测试和 review

### 新方向

从 PyTorch 源码 (`aten/src/ATen/native/cpu/`) 抠出 DiT 所需算子的 C++ kernel：
- `gelu_kernel.cpp`、`layer_norm_kernel.cpp`、`softmax_kernel.cpp`
- `Linear.cpp`（GEMM）、`Activation.cpp`（SiLU）
- 替换 malloc/tensor 抽象为轻量实现
- 此基础上做 Vulkan compute shader 和 QNN NPU 加速

### 关键教训
- 不要手写 kernel
- Smoke test 必须包含 PT 参考
- 对比脚本本身也可能是 bug 来源（SiLU 顺序反了两次）
- `adb pull` 目录会在目标下创建嵌套子目录

### 成果（白天）
- **定位并修复 head_tail_ops.h sin/cos 顺序 bug**：PyTorch `Timesteps.forward()` 是 `torch.cat([sin_emb, cos_emb], dim=-1)`（sin 在前），C++ 写成了 cos 在前。这个 bug 导致 t_emb/adanaln_lora 完全错误（max_err=329.9），是之前 1e+31 溢出和 123KB 出图的根因。
- **系统性验证框架**：`scripts/replica/` 新建 7 个测试脚本，覆盖 CPU vs CUDA 基线、逐 op 复刻验证、shader 算法模拟、RoPE 逐行对比、head/tail E2E
- **全部 12 个 shader 验证通过**：GEMM、LN、RMSNorm、GELU、SiLU、ScaleShift、Gate、Broadcast、RoPE、attn_qkt、attn_softmax、attn_out 均与 PyTorch fp32 误差 < 1.5× CPU-vs-CUDA 基线
- **RoPE 确认正确**：逐行对比 `VideoRopePosition3DEmb`，3D 维度分割、NTK 缩放公式、频率布局全部一致
- **出图**：3 步 75KB（PC 参考 74KB），是目前最接近参考的手机管线
- **速度**：~100s/步（fp32 带宽 + 515MHz 底频），锁 912MHz + GEMM 优化后预期 30-40s/步

### 代码改动（仅一行）
`vulkan/head_tail_ops.h:213-214` — sin 和 cos 对调：
```cpp
// 修复前（错误 — PyTorch 是 [sin|cos]，这里是 [cos|sin]）
emb[m * D + i] = cosf(val);
emb[m * D + half + i] = sinf(val);

// 修复后
emb[m * D + i] = sinf(val);
emb[m * D + half + i] = cosf(val);
```

### 验证脚本
| 脚本 | 用途 |
|------|------|
| `scripts/replica/bench_cpu_cuda.py` | CPU vs CUDA 基线测量（13 ops） |
| `scripts/replica/test_t_embed.py` | sin/cos bug 复现 & 验证 |
| `scripts/replica/test_head_tail_e2e.py` | head/tail 全链路 E2E |
| `scripts/replica/verify_shaders.py` | 12 shader 算法 vs PyTorch |

## 出图演进

| 版本 | 大小 | 改动 |
|------|------|------|
| C++ v2 (sin/cos bug) | 123,000 | ❌ |
| HybridOps (fp16) | 81,000 | — |
| **v2 fp32 fixed** | **75,000** | **sin/cos 修复** |
| PC 基准 (RTX 3060) | **73,840** | 干净 |
|------|------|------|
| 原始 | 118,681 | — |
| GEMM FMA + Welford LN | 118,614 | 同前 |
| +MLP fc2×1/4 | 114,056 | -4% |
| +MLP fc2×1/8 | 109,109 | -8% |
| +SA×1/4 + CX×1/4 + MLP×1/8 | **105,537** | **-11%** |
| PC 基准 (RTX 3060) | **73,840** | 干净 |

## 终极目标（已调整）
~~bit-exact=0~~ → **出图大小匹配 PC 基准 74KB**。bit-exact 已放弃：Vulkan/CUDA FMA 硬件指令舍入方向不同，无法消除。实际走混合精度路线（缩放 gate residual 压低 FP16 溢出）。

## 2026-06-01 凌晨：hybridops 管线 BF16 直读 safetensors ✅

### 成果
- **手机直接读 `.safetensors`**（BF16, 3.9GB），无需桌面端预处理
- **新 C++ Vulkan 引擎**：`hybridops/vulkan/libhybrid_engine.so`（~500KB）
  - 加载时一次性 BF16→FP16 转换存入 Vulkan buffer
  - 按名 GEMM: `vk_run_gemm(weight_name, x, out, M, N, K)`
  - 内置 LN (FP32)、RMSNorm (FP16)、GELU (FP16)
- **PyTorch 不持有 block GEMM 权重**：DummyOps 创建模型（零 weight 内存）→ patch_shell_linear → load_state_dict → patch_block_layers → VulkanGemmLinear
- **内存**：稳态 ~3.9GB VmRSS，峰值 ~5.6GB（加载阶段 mmap + Vulkan buffer 共存）
- **速度**：3 步 142s (47s/步)，840 GEMM 全部 Vulkan（0 CPU fallback）
- **正确性**：LN max_err 0.000002，RMS max_err 0.004，v_cond 无 NaN

### 关键文件
| 文件 | 说明 |
|------|------|
| `hybridops/vulkan/hybrid_engine.cpp` | ~360 行，Vulkan 引擎 |
| `hybridops/vulkan/build_engine.bat` | NDK 编译脚本 |
| `hybridops/scripts/vk_ops.py` | VulkanGemmLinear (无 weight Parameter)、DummyLinear、patch 函数 |
| `hybridops/scripts/phone_pipeline.py` | SafetensorsReader、DummyOps 加载流程 |

### 编译 & 推送
```bash
# 编译
cd D:\AI\anima_phone\hybridops\vulkan && build_engine.bat

# 推送
MSYS_NO_PATHCONV=1 adb push hybridops/vulkan/libhybrid_engine.so /data/local/tmp/
MSYS_NO_PATHCONV=1 adb push hybridops/vulkan/*.spv /data/local/tmp/
MSYS_NO_PATHCONV=1 adb push hybridops/scripts/vk_ops.py /sdcard/anima_on_android/scripts/
MSYS_NO_PATHCONV=1 adb push hybridops/scripts/phone_pipeline.py /sdcard/anima_on_android/scripts/

# 运行
adb shell "su -c 'taskset f0 /data/data/com.termux/files/usr/bin/python -u -B /sdcard/anima_on_android/scripts/phone_pipeline.py'"
```

### 已知问题
- 加载阶段峰值 ~5.6GB（safetensors mmap 页缓存 + Vulkan buffer 共存），可能触发 OOM
- BLAS bad memory unallocation 警告（VAE decode 时，不影响出图）
- GPU 底频 515MHz 下 dispatch 可能超 TDR（锁 912MHz 可解但 max_gpuclk 只读）

---

## ⚠️ 快速提醒（每次会话重启后先看这个）
- **ADB 连接**：WiFi: `adb connect 192.168.0.104:5555`。设备 ID: `87cca7ec`。文件推送需 `MSYS_NO_PATHCONV=1` 防止 Git Bash 路径转换。
- **VK_ERROR_DEVICE_LOST → 拉满 GPU 频率 912MHz**。低频下 dispatch 超 TDR 250ms→驱动杀进程。`adb shell "su -c 'echo 912000000 > /sys/class/kgsl/kgsl-3d0/max_gpuclk'"`。
- **PC 参考图基准**：`output/whitebox/pc_ref_whitebox.png`，**73,840 字节** = 干净。HybridOps 出图 81KB。当前 C++ 出图 105KB（SA×1/4 + CX×1/4 + MLP×1/8 缩放后）。
- **HybridOps 管线**：`hybridops/` 目录保存了原始 57s/步 HybridOps 管线（2026-05-25），用于对比和合并参考。
- **核心对比框架**：`scripts/pc_whitebox_ref.py`（WSL2 运行，`source /home/riorg/miniconda3/etc/profile.d/conda.sh && conda activate /home/riorg/anima-work/.conda`）。白盒逐 op 复现 C++ 引擎。`--compare output/cmp` 对比手机 dump。
- **手机 dump 脚本**：`scripts/phone_dump_blocks.py`。用法：先 `adb push` + `adb shell` 运行，再 `adb pull` 结果，再 PC `--compare`。
- **C++ 引擎关键知识**：
  - lora 以 `[3, M, D]`（component-major）存储在 `g_loraBuf`，AdaLN shader 按 shift@0/scale@M*D*2/gate@2*M*D*2 读。dump 脚本已做 `reshape(3,M,D).transpose(1,0,2)` 转为 `[M,3D]` 再保存 .npy。
  - t_emb 用 C++ 自己的 `dit_compute_timestep`（sin→RMSNorm→SiLU@w1→@w2），不是 PyTorch 的 t_embedder。PC 白盒已复现，验证一致。
  - `dit_init_adaln_only` 用 `load_weights`（全部 685 个 tensor，4.18GB），不是 `load_adaln_weights`。
- **权重必须一致**：手机 `diffusion_weights.bin` 和 PC `diffusion_weights_fp16.pt` 必须来自同一源。用 `scripts/export_weights.py`（会 strip `net.` 前缀）从 .pt 生成 .bin。
- **真实管线对比**：PC `gen_real_inputs.py` 生成 x_emb/t_emb/ctx → 手机 `run_realpipe_phone.py` 用同输入跑 → 拉回对比。Block 0 max_err=5.29（fp16 精度天花板）。
- **合并目标**：去掉 PyTorch 权重加载（small.pt），C++ 全包 x_embedder + blocks + final_layer。可参考 `hybridops/` 的管线结构。

---

## 2026-05-31 晚间会话：白盒对比 + lora 布局 + AdaLN 修复

### 成果概览

| 发现 | 根因 | 修复 | 效果 |
|------|------|------|------|
| **lora 不一致 (max_err=70.7)** | C++ 存 `[3,M,D]`（component-major），phone dump 按 `[M,3D]` 误读 | dump 脚本 `reshape(3,M,D).transpose(1,0,2)` | lora max_err→**0.016** ✓ |
| **Q/K RMSNorm 偏差 (max_err=3.59)** | PC 白盒 AdaLN 的 SiLU 写成了 `SiLU(emb @ w1)` 应为 `SiLU(emb) @ w1` | 修正 `compute_adaln_block` | Q_norm max_err→**0.027** ✓ |
| **C++ `load_adaln_weights` 丢失 GEMM 权重** | 只加载 AdaLN+t_embedder（~471MB），Block 计算需 GEMM 权重 | 还原为 `load_weights`（全部 685 tensor） | Block 输出正常 |
| **`.bin` 与 `.pt` 权重不同源** | 手机 .bin (May 26) 和 PC .pt (May 31) 来自不同时间转换 | 重新 `export_weights.py` (.pt→.bin) + 推送 | 权重一致 ✓ |

### 当前精度状态（vs PC 白盒，synthetic input seed=12345 sigma=1.0）

| 对比 | max_err | 结论 |
|------|---------|------|
| x, ctx 输入 | **0.0000** | ✅ 完全相同 |
| t_emb | **0.00006** | ✅ FP16 精度极限 |
| lora | **0.016** | ✅ FP16 精度极限 |
| Q after RMSNorm | **0.027** | ✅ 8192 行中仅 2 行 >0.02 |
| K after RMSNorm | **0.027** | ✅ |
| Attn output | **0.41** | ⚠️ attention shader 数值差异 |
| SA residual | **1.12** | ⚠️ O_proj+gate 累积 |
| MLP residual (Block 0 输出) | **4.02** | ⚠️ GELU+fc2+gate 累积 |
| Phone 3-step 出图 | **118,681 字节** | ❌ PC 基准 73,840 = 干净 |

### Block 0 全链路精度链（最终状态 2026-05-31）

```
输入 x, ctx         max_err=0.0000  ✓
  → t_emb, lora     max_err=0.00006/0.016  ✓
  → AdaLN modulate  max_err<0.02  ✓
  → LN + ScaleShift max_err<0.02  ✓
  → QKV GEMM        max_err<0.02  ✓
  → RMSNorm Q/K     max_err=0.027  ✓
  → RoPE Q/K        max_err=0.027  ✓  ← 新验证！
  → V (raw)         max_err=0.039  ✓  ← 修复捕获时机
  → QK^T scores     — (captured post-softmax, 不可比)
  → Softmax         max_err=0.008  ✓
  → AV (attn_out shader)  max_err=0.001  ✓✓✓ ← 三 shader 只引入 0.001！
  → O_proj GEMM     max_err<0.02  ✓
  → gate+residual   → SA residual max_err=1.12  ⚠️
  → cross-attn      → CX residual max_err=1.14  ⚠️
  → MLP (fc1→GELU→fc2→gate) → Block 0 输出 max_err=4.02  ⚠️
```

**结论**：attention 的三个 shader (attn_qkt, attn_softmax, attn_out) 合计只引入 max_err=0.001——不是 bug！Block 0 的 4.02 误差是 20+ 个 op 串行 FP16 累积+gate 乘法的结果。O_proj+gate、cross-attn、MLP 路径各贡献 1-2 误差，逐 block 放大。

**下步方向**：
1. 在真实管线输入（非 synthetic）下重跑对比——真实 latent 有空间结构，attention softmax 更平滑，FP16 精度损失更小
2. 或者直接跑管线出图看效果——当前 118KB 可能需要调采样参数而非修 shader

### 工具链

```bash
# PC 白盒（WSL2）
cd /mnt/d/AI/anima_phone
python scripts/pc_whitebox_ref.py [--image-only] [--compare output/cmp]

# 手机 dump
# 1. 推送
MSYS_NO_PATHCONV=1 adb push scripts/phone_dump_blocks.py /sdcard/anima_on_android/scripts/
# 2. 运行（先锁 GPU 频率！）
adb shell "su -c 'taskset f0 /data/data/com.termux/files/usr/bin/python /sdcard/anima_on_android/scripts/phone_dump_blocks.py'"
# 3. 拉取
MSYS_NO_PATHCONV=1 adb pull /sdcard/anima_on_android/output/cmp/ output/cmp/
# 4. 对比
python scripts/pc_whitebox_ref.py --compare output/cmp

# 权重更新
python scripts/export_weights.py models/diffusion_weights_fp16.pt models/diffusion_weights.bin
MSYS_NO_PATHCONV=1 adb push models/diffusion_weights.bin /data/local/tmp/

# 编译 C++ 引擎
"D:/android-ndk-r27d-windows/android-ndk-r27d/toolchains/llvm/prebuilt/windows-x86_64/bin/clang++.exe" --target=aarch64-none-linux-android28 --sysroot="D:/android-ndk-r27d-windows/android-ndk-r27d/toolchains/llvm/prebuilt/windows-x86_64/sysroot" -O2 -std=c++17 -fPIC -shared -I"D:/Vulkan_SDK/Include" -o "D:/AI/anima_phone/vulkan/libdit_vk.so" "D:/AI/anima_phone/vulkan/dit_engine.cpp" -Wl,-z,max-page-size=16384 -Wl,-z,common-page-size=16384 -Wl,--no-rosegment -llog -landroid -lvulkan -L"D:/android-ndk-r27d-windows/android-ndk-r27d/toolchains/llvm/prebuilt/windows-x86_64/sysroot/usr/lib/aarch64-linux-android/28" -static-libstdc++
MSYS_NO_PATHCONV=1 adb push vulkan/libdit_vk.so /data/local/tmp/
```

### 3-step 出图

```bash
adb shell "su -c 'taskset f0 /data/data/com.termux/files/usr/bin/python -u -B /sdcard/anima_on_android/scripts/phone_pipeline.py'"
# 输出：/sdcard/anima_on_android/output/phone_first.png
```

### Shader 精度分析 (2026-05-31 深夜) & 真实管线最终结果

PC 同条件（real latent seed=6666, C++ 风格 t_emb/lora, C++ 风格 RoPE）白盒 vs 手机 C++ 引擎（lora bug 已修）：

| Block | C++ | PyTorch白盒 | max_err |
|-------|-----|------------|---------|
| 0 | [-22.6, 23.1] | [-22.6, 23.4] | **5.29** |
| 1 | [-3202, 3204] | [-3212, 3198] | 147 |

逐 shader 内部精度检查：

| Shader | I/O | 内部累加 | 与 PyTorch 一致？ |
|--------|-----|----------|-------------------|
| gemm_fp16 | fp16 | 改为 **fp32 fma()** | ✅ 数学一致 |
| layernorm_fp16 | fp16 | **fp32** | ✅ 数学一致 |
| rms_norm_fp16 | fp16 | **fp32** | ✅ 数学一致 |
| silu_fp16 | fp16 | **fp32** | ✅ 数学一致 |
| scale_shift_fp16 | fp16 | fp16 逐元素 | ✅ 无累加 |

**权重验证**：`.bin` 与 `.pt` 中权重 bit-exact 一致（0/4,194,304 差异）。

**Block 0 逐 op 精度链（真实输入）**：
```
Q after RMSNorm   max_err=0.039  (53% elements differ)  ← 第一个 bit 不同的 op
K after RMSNorm   max_err=0.031
V raw             max_err=0.063
Attn output       max_err=1.07   (误差放大 30×)
O_proj GEMM       max_err=0.59
SA residual       max_err=3.65
→ Block 0 输出     max_err=5.29
```

**根因确认**：PyTorch FP16 两次 run 完全 deterministic（max_err=0.0000），C++ 误差 = 0.039 不是 FP16 宿命。所有 shader 都已经是 fp32 内部计算、FMA 累加、权重 bit-exact——残余来自**并行 reduction 的求和顺序**与 PyTorch CUDA kernel 的舍入差异。即使都是 fp32，`(a+b)+c ≠ a+(b+c)`。

### 真实管线 + 缩放实验 (2026-06-01 凌晨)

用真实 latent 输入，C++ 风格 t_emb/lora/RoPE，白盒对比 Block 0 max_err=5.29（纯 fp16 累积）。

AdaLN 误判：shift/scale/gate 一度认为"全错"——根原是测试脚本用 sin_emb 而非 t_emb 做 SiLU。修正后 AdaLN max_err=0.03，确认正确。

Gate 缩放实验（防止后层 fp16 溢出）：

| SA | CX | MLP | 图大小 | vs 原始 |
|----|----|------|--------|------|
| 1 | 1 | 1 | 118,681 | — |
| 1 | 1 | 1/4 | 114,056 | -4% |
| 1 | 1 | 1/8 | 109,109 | -8% |
| 1/4 | 1/4 | 1/8 | **105,537** | **-11%** |
| 1/2 | 1/2 | 1/8 | 106,755 | 倒退 |

**结论**：SA×1/4 + CX×1/4 + MLP×1/8 最优。边际递减已触 fp16 天花板（FP16 最大 65504，后层值域频繁触及）。下步方向：BF16 权重→fp32 计算，用 BF16 的动态范围（同 fp32）替代 FP16 精度。
- GEMM: fp32 fma() ✅
- LN: Welford 算法 ✅
- RMSNorm: fp32 内部 ✅
- SiLU: fp32 内部 ✅
- AdaLN: shift/scale/gate max_err=0.03 ✅
- t_emb/lora: max_err=0.00006/0.016 ✅
- 权重: bit-exact ✅

残余 max_err 链（纯 fp16 精度，非 bug）:
```
Modulated(LN*scale+shift) max_err=0.19
Q_raw(GEMM) max_err=0.08
RMSNorm(Q_raw) max_err=0.004
→ Block 0 max_err=5.29
```

bit-exact=0 的剩余障碍：GLSL 编译器生成的 fp32 指令序列与 CUDA nvcc 不完全相同（fma 融合、寄存器分配、指令调度层面）。逐个对比已到瓶颈。

---

## FP16 精度知识库（来自调研）

### Vulkan vs CUDA 核心差异
- **Vulkan FMA 不可控**：CUDA 有 `-fmad=false` 关闭快速数学优化，Vulkan/GLSL 没有等价选项。同样写 `fma()`，CUDA 和 Vulkan 硬件实现舍入方向可能不同（每个 op 差 0.5 ULP）。
- **跨后端不一致是常态**：GGML 实测 CUDA FP16 vs CPU FP32 结果差 721.5 vs 720.0。不是 bug——是 fp16 物理特性。
- **FP16 极限**：超过 ~1000 次加法，FP16 吞没新的贡献值（10-bit 尾数不足）。2048 元素点积 = 2048 次 add，必然有舍入。

### DiT 推理的混合精度策略（Draw Things 团队经验）
- **LN → FP32**：LayerNorm 允许激活值自由缩放，值域常超 FP16 范围。
- **Block 内部 FP16**：大多数 op 在 fp16 下行得通。
- **MLP down-projection 加缩放**：1/4 或 1/8 保守缩放因子防止溢出。
- **gate 回来时→FP32**：MLP GEMM 后上采样回 fp32 做 gate + residual。
- 注：此策略在 M1/M2（Metal）上验证，Adreno 适配需实测。

### 硬件事实
- Adreno 730: FP16 ~5.5 TFLOPS, FP32 ~2.8 TFLOPS（2× 差）。但实际 fp16 vs fp32 执行行为高度依赖驱动和运算类型。
- SPIR-V 优化器（spirv-opt）激进 FMA 折叠在 Adreno 上可能导致性能倒退。

---

## 2026-05-31 晚间：per-step recording 正确性修复（旧记录，保留参考）

### Bug 修复

| Bug | 根因 | 修复 | 效果 |
|-----|------|------|------|
| MLP 激活函数错误 | `predict2.GPT2FeedForward` 用 `nn.GELU()`，C++ engine 错用 SiLU | `record_segment_mlp` 改用 `dispatch_gelu` | Block 0 MLP [-8.3,17.3]→[-6.4,16.9] |
| Self/cross-attn 跨 batch 混合 | flat attention 不分 batch，batch 0 query attend 到 batch 1 key | `record_attn_3pass` 加 `q_base_off/kv_base_off`，attention 按 M 分片 | dispatch 数减半，91s→46s |
| RoPE pipeline 未加载 | `create_adaln_pipelines` 缺少 `CP(rope,...)`，`dispatch_rope` 绑定 VK_NULL_HANDLE → GPU hang（第一次侥幸通过，killall 后永久损坏） | 加 `CP(rope, 3, sizeof(PC_Rope))` 到 adaln pipelines；rope freqs 改 per-step buffer（避免 killall 残留） | RoPE 正确执行，killall -9 后恢复正常 |

### 精度状态（vs PyTorch FP16 with RoPE）

| 对比 | max_err | 结论 |
|------|---------|------|
| PyTorch FP16 自身两次 run | **0.0000** | GPU FP16 完全 deterministic |
| C++ vs PyTorch Block 0 | **5.5** | ⚠️ 不是 FP16 精度问题，存在系统偏差 |
| C++ vs PyTorch Block 1 | **192.5** | 逐层放大约 35× |

**已验证正确的模块**（独立验证 max_err < 0.02）：GEMM, LN, RMSNorm, SiLU, GELU, ScaleShift, attn_qkt, attn_softmax, attn_out, RoPE

**2026-05-31 晚间修复**: PC 白盒 AdaLN SiLU 顺序修正（`SiLU(emb) @ w1` 而非 `SiLU(emb @ w1)`）后，Q/K RMSNorm max_err 从 3.59 → 0.027。Block 0 误差从 7.34 → 4.02。Block 1 从 800 → 51.88。剩余误差来自 attention shader 的 FP16 数值精度（非结构性 bug）。

**lora 布局发现**: C++ 引擎以 `[3, M, D]` 存储 lora（component-major），PyTorch 以 `[M, 3D]`（batch-major）。phone dump 已修正解读。t_emb/lora 确认与 PC 完全一致（max_err=0.016/0.00006）。

**管线**：54s/步，出图 119KB 有噪声（正常应该 70-80KB）

### 下步（清空上下文后）

1. **全面逐层对比**：Block 0 SA→CX→MLP 三段 vs PyTorch（带 RoPE）
2. **MLP drill-down**：如果 MLP 段偏差最大，进一步对比 fc1 GEMM → GELU → fc2 GEMM
3. **GEMM 单步对齐**：对偏差最大的 GEMM，同输入下逐元素对比 C++ vs `F.linear`
4. **定位到具体 shader/权重**，修复后跑集成管线验证出图

---

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
| **C++ 引擎 28-block 预录 (skip-attn)** | **~25s/步** | **~4.8×** | ✅ 管线验证无 NaN (26°C) |
| **C++ 引擎 +空 cmd_attn (56 submit/步)** | **~25s/步** | **~4.8×** | ✅ RESET flag 修复 |
| C++ 引擎 +真实 attention (目标) | ~30-40s/步 | 3-4× | ⏳ 开发中 (descPool 阈值问题) |

注：HybridOps 管线已被 C++ 引擎替代。当前管线：PyTorch 轻量壳 (~306MB) + C++ 引擎 (3.9GB Vulkan)。56 submit/步 (28 cmd[i] + 28 cmd_attn[i])。

## C++ 引擎状态 (libdit_vk.so v3 — 2026-05-30 重写)

**架构变更**：
- 全量权重加载：`load_weights` 流式加载 567 tensor (3.9GB)，无 CPU 临时 buffer，不 OOM
- 28 block 全量预录：`record_one_block` × 28 (63 dispatch/block, 1764 dispatch total)
- `dit_forward_nblocks`：逐 block submit cmd[i] + fence wait + chain 输出
- FP16 LayerNorm：pre-recorded 用 `layer_norm_f16` (FP16 I/O)，per-call 保持 `layer_norm` (FP32 I/O)

**已修复的关键 Bug**：
- **FP32 LN shader + FP16 buffer = NaN**：`layer_norm.spv` 是 FP32 shader (4字节/元素)，但 pre-recorded block 绑的是 FP16 buffer (2字节/元素)。shader 读 4 字节当 float，实际是两个 fp16 拼在一起 → 垃圾值 → 全 NaN。修复：加 `layer_norm_f16` pipeline 加载 `layernorm_fp16.spv`，`dispatch_layernorm` 改用 FP16 版本。
- **Buffer 尺寸 bug**：g_t1/g_tQ/g_tK/g_tV 原来按 M=2 分配 (AdaLN only)，改为 MS=512 分配 (full block recording)

**验证状态**：
- ✅ 3.9GB 流式加载 (7s init)
- ✅ 28 block 预录成功 (1764 dispatches)
- ✅ 单 LN dispatch 输出 clean (min=-2.1 max=4.1 nan=0)
- ✅ 1-block 输出 clean (min=-758 max=750 nan=0)
- ✅ 28-block 输出 clean (min=-65504 max=47744 nan=0)
- ❌ skip-attention (V→O) — 值合法但语义不对，不出正图

**⚠️ Attention 状态**：
- Self-attn + cross-attn 是 **skip 模式** (`V @ O_proj`)，跳过了 QK^T+softmax+SV
- RoPE 未集成到 block recording
- Per-call attention (3-pass Vulkan) 仍然正常工作，但每步 534 submit → 慢

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

### 引擎架构 (当前 v3)

```
dit_init_adaln_only(weight_bin, spv_dir)
  → load_weights: 流式加载 567 个 per-tensor Vulkan buffer (3.9GB, 无临时 CPU buf)
  → 创建 10+1 个 shader pipeline (含 layer_norm_f16)
  → 分配 28 个 cmd buffer + 24 个 I/O buffer

record_one_block × 28
  → 每个 cmd[i] 录制: AdaLN×3(36) + self-attn(10) + cross-attn(10) + MLP(7) = 63 dispatches
  → self/cross-attn 目前是 skip 模式 (V→O_proj)

dit_forward_nblocks(x, t_emb, ctx, out, nblocks)
  → 上传 x → g_xBuf, ctx → g_ctxBuf (t_emb 已有)
  → for i in 0..nblocks:
      submit cmd[i] → wait fence
      复制 g_outBuf → g_xBuf (chain)
    → 下载 g_outBuf → out
  → 28 submit/步, ~25s/步
```

### Attention 集成 (3b) — 2026-05-30 晚间结论

**旧假设全部推翻。阻塞不是 descPool 阈值，是 Adreno TDR 看门狗。**

#### 二分定位 descPool 阈值 → 结论：不存在

| 测试 | descPool2 sets | cmd[0] submit | 结论 |
|------|---------------|---------------|------|
| 14 blocks attention | 336 | ✅ | |
| 21 blocks | 504 | ✅ | |
| 25 blocks | 600 | ✅ | |
| **28 blocks (全量)** | **672** | ✅ | 之前失败是其他 bug，不是阈值 |

总 descriptor sets: 1764 (descPool) + 672 (descPool2) = 2436，cmd[0] submit 正常。
之前 commit `8a17806` 报告的 "672 sets 导致 cmd[0] submit 失败" 实际是被 3b-iv smoke test
(48 dispatch on g_lnCmdBuf) 污染了 GPU 状态。已用 `#if 0` 禁用该测试。

#### 真正瓶颈：TDR 看门狗

Adreno 730 的单次 `vkQueueSubmit` 执行时间有看门狗超时。超时 → `VK_ERROR_DEVICE_LOST`(-4)。
关键证据：

| GPU 频率 | 单 block 执行时间(63 dispatch) | 能跑几个 block | 
|----------|------------------------------|---------------|
| 510 MHz (底频) | ~2.9s | 0-2 |
| 645 MHz | ~2.3s | ~10 |
| **912 MHz (锁定超频)** | ~1.6s | **28 (全过)** |

规律：频率越低→执行越慢→越早触发 TDR。频率完全决定了崩溃的早晚。

**这意味着 pre-record 大块 dispatch 的架构对 Adreno 这类移动 GPU 是反模式**——正确的做法是每次 submit
只包含少量 dispatch（控制在几百 ms 以内），避免触发看门狗。详见"下步路线"。

#### 三种架构对比

| 方案 | submit/步 | dispatch/cmd | TDR 风险 | 状态 |
|------|-----------|-------------|---------|------|
| separate cmd_attn | 56 | cmd=63, attn=24 | ⚠️ 看频率 | 912MHz 过 |
| **merge 进 cmd[i]** | **28** | 73 (batch_q=256) | ⚠️ 看频率 | 912MHz **待测** |
| per-call (HybridOps) | 534 | 3 | ✅ 安全 | 太慢 |

#### Shader 优化：batch_q 动态化

`record_attn_3pass` 的 `batch_q` 从硬编码 64 → 自适应（按 target WG=2048~4096 计算）:
- batch_q=256: 2 批 × 3 pass = 6 dispatch/attention（vs 原来 8 批 × 3 = 24）
- block dispatch 从 109 降到 73 (61 block + 6 self + 6 cross)
- 73 dispatch 在 510MHz 下仍超 TDR，912MHz 待测

**待补模块**（短期 912MHz merge / 中期 per-step recording）:

| 模块 | 优先级 | 说明 |
|------|--------|------|
| **Self-attn + Cross-attn 真实计算** | 🔴 最高 | 替换 skip-attn (V→O_proj) → 出正图 |
| **RoPE GPU** | 🟡 中 | 56 次 ~3s/步，可参考 ET 的 per-texel shader |
| **final_layer** | 🟢 低 | 1 次 <1s，PyTorch 暂时够用 |
| **VAE decoder** | ⚪ 远期 | 完全独立模块 |

**PyTorch 轻量化策略**：`num_blocks=0` — PyTorch 只创建 x_embedder + t_embedder + final_layer (~20MB)，不加载 28 block 权重 (~3.7GB)。C++ 引擎扛全部 block 推理。

### Shader 验证状态

| Shader | 独立验证 | Block 内集成 |
|--------|---------|-------------|
| GEMM (gemm_fp16) | ✅ | ✅ |
| LayerNorm (FP16) | ✅ | ✅ (FP16 fix) |
| RMSNorm | ✅ | ✅ |
| SiLU | ✅ | ✅ |
| ScaleShift | ✅ | ✅ |
| Attn QK^T | ✅ | 🚧 cmd_attn |
| Attn Softmax | ✅ | 🚧 cmd_attn |
| Attn AV | ✅ | 🚧 cmd_attn |
| Broadcast | — | ✅ |
| RoPE | — | ❌ |

### 未完成

- **Cross-attn Q_proj 用错权重**：`record_one_block` 的 cross-attn 段用了 `self_attn.q_proj.weight` 而非 `cross_attn.q_proj.weight`（skip-attn 路径也存在此 bug，但不影响 skip 正确性，真 attention 需修复）
- **Self-attn + Cross-attn 真实计算**：已用 `record_attn_3pass` + `batch_q=256` 实现 merge 版（73 dispatch/cmd），912MHz 待管线验证
- RoPE GPU（per-texel shader，参考 ET `apply_rotary_emb_interleaved.glsl`）
- x_embedder, t_embedder, final_layer C++ 移植
- VAE decode（暂留 PyTorch）
- 端到端 phone_pipeline 出正图

### 下步路线 (2026-05-31 凌晨)

**per-step recording 架构已落地**（`dit_forward_step`，140 submit/步，5 段/block），管线能跑到出图但图片是雪花。

cross-attn 的 Q_proj + Q_norm 权重 bug 已修，但 **attention 计算（QK^T/softmax/AV shader）疑似有数值问题**，C++ 输出值域 -47328~+23616（fp16 极限），28 层累积后偏差爆炸。

**🔴 最高优先：PC PyTorch vs C++ 逐 block 精度对比**
- 手机导出同一组输入 → PC 用同一份 FP16 权重跑 PyTorch forward → 逐 block 对比
- PC 端用 cos=1/sin=0 零频 RoPE 对齐 C++ 的 no-RoPE 行为
- 从第一个偏差 block drill down 到具体 shader

---

## 2026-05-31 晚间：白盒对比框架

核心工具 `scripts/pc_whitebox_ref.py` — 逐 op 复现 C++ 引擎（非 block.forward 黑盒），用法见上方"快速提醒"。输出目录 `output/whitebox/`：block_*_pt.npy（28 block 输出）、b0/intermediates/（Block 0 每步中间量）、pc_ref_whitebox.png（3-step 参考图 73,840 字节）。

> 详细架构、对比流程、使用示例已整合到上方"2026-05-31 晚间会话：白盒对比 + lora 布局 + AdaLN 修复"章节，此处不再重复。
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

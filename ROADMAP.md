# Anima Snapdragon Android Port Roadmap

## 0. Final goal

The goal is not to simply run ComfyUI. The goal is to split Anima into a reproducible, inspectable, convertible inference pipeline that can eventually run on a Snapdragon Android phone.

Target device:

- Redmi K50 Ultra / Xiaomi 12T Pro
- Snapdragon 8+ Gen 1
- 12 GB RAM / 256 GB storage
- Android 12 / MIUI 13
- Magisk root available

Preferred final direction:

1. First reproduce Anima inference on desktop/WSL with a minimal independent Python pipeline.
2. Then identify which submodules can be exported or rewritten for mobile.
3. Prefer Snapdragon NPU if practical.
4. Accept CPU/GPU/NPU hybrid execution if full NPU coverage is unrealistic.
5. Optimize for the turbo LoRA path first because it reduces the generation workload substantially.

## 1. Collaboration and safety rules

This project follows the safety-word workflow from the repository `CLAUDE.md`:

- Before modifying files, running commands, creating environments, or producing irreversible output, discuss and align first.
- Only proceed with real actions after the user says the safety word: `开始` or `动手吧`.
- Before the safety word, only provide explanations, proposals, pseudocode, or action drafts.
- Avoid broad destructive permissions. Prefer read-only inspection and scoped writes inside the agreed workspace.

## 2. Current workspace and paths

WSL workspace:

```text
/home/riorg/anima-work
```

Conda environment:

```text
/home/riorg/anima-work/.conda
```

Activate it with:

```bash
. "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate /home/riorg/anima-work/.conda
```

Important rule: use path-based conda environments with `conda -p`, not named environments with `conda -n`, especially on Windows, because the C drive is low on space.

Model asset repository on Windows:

```text
D:\AI\手坤的anima
```

Same model assets from WSL:

```text
/mnt/d/AI/手坤的anima/models
```

Reference ComfyUI package, read-only:

```text
E:\sbkuake\ComfyUI-aki-v2\ComfyUI-aki-v2
```

Local encoder reference, read-only:

```text
D:\AI\prompts\画师串小助手\engines
```

Do not modify the Windows ComfyUI package while building the independent reproduction. Use it only as a reference.

## 3. Confirmed model assets

Known files in the local model repository:

```text
models/diffusion_model/anima-base-v1.0.safetensors
models/text_encoder/qwen_3_06b_base.safetensors
models/vae/qwen_image_vae.safetensors
models/lora/anima-turbo-lora-v0.1.safetensors
```

The first reproduction should reference these files from `/mnt/d/AI/手坤的anima/models`. Do not copy multi-GB model files into the WSL workspace unless there is a measured reason to do so.

## 4. Confirmed Anima architecture facts

### 4.1 Diffusion model

Anima is not an SD1.5 or SDXL UNet. It is a Cosmos Predict2 / DiT-style model.

Confirmed diffusion checkpoint facts:

- File: `anima-base-v1.0.safetensors`
- Size: about 3.9 GB
- Top-level checkpoint prefix: `net`
- Tensor dtype: BF16
- Transformer blocks: 28
- Main model hidden width: 2048
- Cross-attention context width: 1024
- Uses `net.llm_adapter`
- Uses flow-style sampling in ComfyUI, not classic EPS or v-pred UNet behavior

Representative tensor shapes observed:

```text
net.blocks.0.cross_attn.k_proj.weight      [2048, 1024]
net.blocks.0.cross_attn.v_proj.weight      [2048, 1024]
net.blocks.0.cross_attn.q_proj.weight      [2048, 2048]
net.blocks.0.self_attn.q_proj.weight       [2048, 2048]
net.blocks.0.mlp.layer1.weight             [8192, 2048]
net.blocks.0.mlp.layer2.weight             [2048, 8192]
net.t_embedding_norm.weight                [2048]
net.final_layer.linear.weight              [64, 2048]
```

### 4.2 Text encoder

The text encoder is a Qwen3 0.6B-like encoder using Qwen tokenizer files and Qwen2 tokenizer class.

Confirmed text encoder checkpoint facts:

- File: `qwen_3_06b_base.safetensors`
- Size: about 1.2 GB
- Tensor dtype: BF16
- Hidden size: 1024
- Layers: 28
- Attention heads: 16
- KV heads: 8
- Head dim: 128
- Vocabulary size: 151936
- Pad token id used by the local code: 151643
- Tokenizer class: `Qwen2Tokenizer`

Representative tensor shapes observed:

```text
model.embed_tokens.weight                  [151936, 1024]
model.layers.0.self_attn.q_proj.weight     [2048, 1024]
model.layers.0.self_attn.k_proj.weight     [1024, 1024]
model.layers.0.self_attn.v_proj.weight     [1024, 1024]
model.layers.0.self_attn.o_proj.weight     [1024, 2048]
model.layers.0.mlp.gate_proj.weight        [3072, 1024]
model.layers.0.mlp.up_proj.weight          [3072, 1024]
model.layers.0.mlp.down_proj.weight        [1024, 3072]
```

Important caveat:

The local vector-library encoder mean-pools Qwen hidden states into a 1024-dimensional normalized vector. That is useful for vector search, but it is not the true generation conditioning path.

For generation, Anima uses sequence hidden states plus token ids/weights that feed the diffusion model's `LLMAdapter`.

### 4.3 Tokenization and LLMAdapter path

ComfyUI's native Anima tokenizer produces both:

- `qwen3_06b` token weights for Qwen hidden states
- `t5xxl` token ids and weights

This does not mean Anima uses a full T5XXL text encoder. The T5 tokenizer ids are embedded inside Anima's diffusion-side `LLMAdapter`.

Important ComfyUI behavior:

1. Qwen text encoder produces hidden states with width 1024.
2. T5 tokenizer ids and weights are attached into the conditioning metadata.
3. `model_base.Anima.extra_conds()` either preprocesses the text embeds during inference mode or passes `t5xxl_ids` and `t5xxl_weights` into the diffusion model.
4. `comfy.ldm.anima.model.Anima.forward()` calls `preprocess_text_embeds()` when `t5xxl_ids` exists.
5. `LLMAdapter` combines Qwen hidden states and T5 token ids to produce the final cross-attention context.

Expected adapted context target:

```text
[B, 512, 1024]
```

The independent reproduction must reproduce this path, not just a mean-pooled text vector.

### 4.4 VAE and latent format

Anima uses the Qwen Image VAE, not the usual SDXL VAE.

Confirmed VAE facts:

- File: `qwen_image_vae.safetensors`
- Size: about 243 MB
- Tensor dtype: BF16
- Uses 3D convolution-style weights
- ComfyUI latent format: `Wan21`
- Latent channels: 16
- Latent dimensions: 3

Likely image latent shape:

```text
[B, 16, 1, H/8, W/8]
```

Representative tensor shapes observed:

```text
encoder.conv1.weight       [96, 3, 3, 3, 3]
decoder.conv1.weight       [384, 16, 3, 3, 3]
decoder.head.2.weight      [3, 96, 3, 3, 3]
conv1.weight               [32, 32, 1, 1, 1]
conv2.weight               [16, 16, 1, 1, 1]
```

### 4.5 Turbo LoRA

The turbo LoRA is already present locally.

Confirmed LoRA facts:

- File: `anima-turbo-lora-v0.1.safetensors`
- Size: about 143 MB
- Tensor dtype: BF16
- LoRA rank: 32
- Top prefix: `diffusion_model`
- Targets diffusion model blocks and `llm_adapter` blocks
- Does not target the text encoder or VAE

Practical implication:

The turbo path is a strong candidate for the first mobile target because it allows much lower sampling cost. Offline LoRA merge should be considered before export/conversion so the mobile runtime does not need dynamic LoRA application.

## 5. Known ComfyUI generation settings

Normal Anima preview workflow observed:

```text
steps: 30
cfg: 4
sampler: euler_ancestral
scheduler: simple
```

Turbo workflow observed:

```text
LoRA: anima-turbo-lora-v0.1.safetensors
LoRA strength_model: 1
steps: 12
cfg: 1
sampler: er_sde
scheduler: beta57
denoise: 1
latent: 1024x1024 batch 1
VAE: qwen_image_vae.safetensors
CLIP/text encoder: qwen_3_06b_base.safetensors
```

Important unresolved issue:

The workflow mentions `beta57`, but the inspected ComfyUI source showed the standard `beta` scheduler handler and did not reveal `beta57`. Possible explanations:

- old saved workflow name
- frontend alias
- extension-provided scheduler
- removed alias
- dynamic mapping not found yet

Do not assume `beta57` is understood until it is located or experimentally mapped.

## 6. Completed work so far

- Created repository `CLAUDE.md`.
- User added strict collaboration and safety-word rules to `CLAUDE.md`.
- Confirmed the current repository is a model-asset repository, not a source-code repository.
- Found local Anima model assets.
- Inspected safetensors headers for diffusion model, text encoder, VAE, and turbo LoRA.
- Read local Anima text encoder reference code.
- Read native ComfyUI Anima implementation from the user's ComfyUI package.
- Confirmed true generation conditioning is sequence/context-based, not mean-pooled vector-based.
- Confirmed Anima is a Cosmos/DiT model, not SDXL UNet.
- Confirmed WSL distro and Windows-to-WSL path mapping.
- Created WSL workspace:

```text
/home/riorg/anima-work
```

- Created path-based conda environment:

```text
/home/riorg/anima-work/.conda
```

- Verified environment Python:

```text
/home/riorg/anima-work/.conda/bin/python
Python 3.11.15
```

## 7. Current core judgment

Do not start with Android or QNN conversion immediately.

Reason:

- Anima is not SDXL, so LocalDream can only provide broad engineering inspiration.
- The model path is custom: Qwen hidden states plus T5 tokenizer ids plus diffusion-side `LLMAdapter`.
- If text conditioning is wrong, converted mobile outputs will fail silently or degrade badly.
- If DiT forward shape assumptions are wrong, ONNX/QNN/export errors will be hard to diagnose.
- If LoRA is not merged or represented correctly, turbo inference will not match the ComfyUI turbo workflow.

Therefore the immediate priority is:

1. Build a minimal independent WSL reproduction.
2. Verify every submodule with shapes and small test inputs.
3. Reproduce text conditioning and diffusion forward before any mobile conversion.
4. Use the turbo route first.

## 8. Phased plan

### Phase 1: Asset probe

Goal:

Verify that the WSL environment can reliably access all model files and inspect their safetensors metadata.

Inputs:

```text
/mnt/d/AI/手坤的anima/models/diffusion_model/anima-base-v1.0.safetensors
/mnt/d/AI/手坤的anima/models/text_encoder/qwen_3_06b_base.safetensors
/mnt/d/AI/手坤的anima/models/vae/qwen_image_vae.safetensors
/mnt/d/AI/手坤的anima/models/lora/anima-turbo-lora-v0.1.safetensors
```

Output:

A small script that prints:

- file existence
- file size
- safetensors tensor count
- dtype summary
- key prefix summary
- representative shapes

Acceptance criteria:

- Script runs inside `/home/riorg/anima-work/.conda`.
- Script does not load full multi-GB tensors into memory.
- Script does not modify model files.
- Output is stable enough to use as a reference for later steps.

Do not do in this phase:

- no model conversion
- no full checkpoint loading
- no Android work
- no copying model files

### Phase 2: Minimal Python skeleton

Goal:

Create a small independent Python project in `/home/riorg/anima-work` that can run probes and later host the reproduction code.

Expected contents:

```text
/home/riorg/anima-work/scripts/...
/home/riorg/anima-work/src/...
/home/riorg/anima-work/ROADMAP.md
```

Acceptance criteria:

- Scripts run from the WSL conda environment.
- Paths are configurable or centralized.
- Windows model assets are referenced read-only.
- No changes are made to the Windows ComfyUI package.

Do not over-engineer this phase. A small script-based layout is enough.

### Phase 3: Text encoder reproduction

Goal:

Reproduce Qwen text tokenization and hidden-state encoding.

Inputs:

- Qwen tokenizer files from the known tokenizer reference
- `qwen_3_06b_base.safetensors`
- a short test prompt

Expected output:

```text
Qwen hidden states: [B, seq, 1024]
attention mask or equivalent mask data
```

Acceptance criteria:

- Bare tokenization path is understood.
- No chat template is used for Anima generation unless later evidence contradicts this.
- Output sequence hidden states are available, not just mean-pooled vectors.
- Dtype/device behavior is explicit.

Do not do in this phase:

- do not treat the mean-pooled vector-library output as generation conditioning
- do not assume Alibaba's embedding model behavior is identical without verification

### Phase 4: LLMAdapter reproduction

Goal:

Reproduce the diffusion-side `LLMAdapter` behavior that converts Qwen sequence hidden states plus T5 token ids into Anima cross-attention context.

Inputs:

- Qwen hidden states `[B, seq, 1024]`
- T5 tokenizer ids
- T5 token weights, if present
- `llm_adapter` weights from the diffusion checkpoint

Expected output:

```text
adapted context: [B, 512, 1024]
```

Acceptance criteria:

- Output shape matches ComfyUI behavior.
- Padding to 512 tokens is reproduced.
- T5 weights are applied in the same place as ComfyUI.
- The implementation can load the relevant `llm_adapter` weights.

### Phase 5: DiT single-step forward

Goal:

Run one diffusion model forward pass with controlled dummy inputs.

Inputs:

- Random latent with expected Anima latent shape
- Timestep tensor
- Adapted context `[B, 512, 1024]`
- diffusion checkpoint weights

Expected output:

A tensor with the expected latent/model output shape.

Acceptance criteria:

- Main DiT model can instantiate.
- Weight keys map correctly.
- Forward pass works for at least a small controlled shape.
- Output shape matches expectations.

This phase is about shape and architecture alignment, not image quality.

### Phase 6: Turbo LoRA merge

Goal:

Merge `anima-turbo-lora-v0.1.safetensors` into the diffusion model offline or create a deterministic merged representation.

Inputs:

- base diffusion checkpoint
- turbo LoRA checkpoint

Expected output:

A merged diffusion model state dict or reproducible merge script.

Acceptance criteria:

- LoRA rank and target mappings are understood.
- Merged model loads into the same diffusion architecture.
- Single-step forward output shape remains valid.
- Text encoder and VAE are not incorrectly modified.

Do not do in this phase:

- do not apply LoRA to Qwen text encoder
- do not apply LoRA to VAE

### Phase 7: Sampling reproduction

Goal:

Reproduce enough of the denoising loop to generate a latent using the turbo route.

Target turbo settings:

```text
steps: 12
cfg: 1
sampler: er_sde
scheduler: beta or resolved beta57 equivalent
denoise: 1
```

Acceptance criteria:

- Denoising loop runs end-to-end on desktop/WSL.
- Scheduler behavior is explicitly chosen and documented.
- `beta57` is resolved, mapped, or replaced with a justified approximation.
- Latent output can be passed to VAE decode.

### Phase 8: VAE decode reproduction

Goal:

Decode an Anima latent into an image using Qwen Image VAE.

Acceptance criteria:

- VAE loads.
- Latent shape is correct.
- Decode produces image-shaped output.
- Memory usage is observed.

### Phase 9: Mobile export investigation

Goal:

Choose the first realistic mobile runtime path.

Candidates:

- Qualcomm QNN / SNPE style route
- ONNX Runtime Mobile with NNAPI or QNN EP
- ExecuTorch
- custom native runtime for selected submodules
- hybrid approach with some modules on CPU/GPU and some on NPU

Acceptance criteria:

- Decide which submodule to port first.
- Identify unsupported operators early.
- Estimate memory pressure.
- Decide quantization strategy for each component.

Likely first submodule candidates:

1. text encoder, if Qwen3 0.6B can be quantized/handled
2. VAE decode, if 3D conv support is acceptable
3. DiT block subset or full DiT, if attention/MLP/rope path can be represented

### Phase 10: Android prototype

Goal:

Run at least one verified Anima submodule on the target phone before attempting the entire pipeline.

Acceptance criteria:

- A small Android binary/app/script can load test inputs.
- The selected submodule runs on-device.
- Output is numerically or structurally compared against WSL output.
- Memory and runtime are measured.

Only after this should the full pipeline be assembled on Android.

## 9. Risks and open problems

### Hardware and memory

- 12 GB phone RAM may be tight for a 2B DiT plus Qwen text encoder plus VAE.
- Android memory pressure may be worse than desktop memory estimates.
- Snapdragon 8+ Gen 1 NPU operator support may not cover all required ops.
- CPU/GPU fallback may be necessary.

### Model architecture

- DiT/Cosmos path is less standard than SDXL UNet mobile pipelines.
- RoPE, 3D latent handling, flow sampling, and custom LLMAdapter may complicate export.
- Qwen text encoder could dominate memory unless quantized or cached.

### Precision and quantization

- Source weights are BF16.
- Mobile path may require FP16, INT8, INT4, or mixed precision.
- Aggressive quantization could hurt prompt adherence or image quality.
- Turbo LoRA already trades quality/adherence for speed, so further quantization needs careful testing.

### Scheduler and sampling

- `beta57` is unresolved.
- `er_sde` behavior must be reproduced or approximated correctly.
- Flow model sampling differs from classic SD EPS/v-pred assumptions.

### VAE

- Qwen Image VAE uses 3D conv-style weights.
- Mobile runtime support and memory behavior need early testing.

### Engineering

- Avoid modifying the Windows ComfyUI package.
- Avoid copying large checkpoints unless performance requires it.
- Keep every step reproducible with scripts and shape logs.
- Do not jump to Android export before desktop forward alignment.

## 10. Decision principles

- Build the smallest verifiable loop at each stage.
- Preserve shape logs and important intermediate tensor metadata.
- Prefer scripts that can be rerun over manual notebook-only experiments.
- Treat ComfyUI as the reference implementation, not the working tree to mutate.
- Use WSL for conversion/reproduction work.
- Use path-based conda environments with `conda -p`.
- Keep model assets read-only until a specific conversion or merge step is approved.
- Optimize the turbo route first.
- Only pursue the quality/full route after the turbo route is understood.

## 11. Immediate next steps

The next concrete step is Phase 1.

Recommended next actions after approval:

1. Activate the WSL conda environment.
2. Install minimal dependencies:

```text
numpy
safetensors
transformers
torch
```

3. Create a small asset-probe script in `/home/riorg/anima-work`.
4. Probe files under:

```text
/mnt/d/AI/手坤的anima/models
```

5. Print a stable summary of tensor counts, dtypes, key prefixes, and representative shapes.
6. Use that output as the baseline for later loading and conversion code.

Do not install large or mobile-specific toolchains yet. Do not start QNN/Android work yet.

## 12. Current status (updated 2026-06-01 daytime)

### Vulkan v2 fp32 管线达标 ✅ (2026-06-01 白天)

- **bug 定位**：`head_tail_ops.h:213` sin/cos 顺序反了（PyTorch `[sin|cos]`，C++ 写成 `[cos|sin]`），导致 t_emb/adanaln_lora max_err=329.9 → GEMM 1e+31 溢出 → 出图 123KB
- **修复**：一行代码，sin 和 cos 对调
- **验证**：系统性 4-phase 验证（CPU vs CUDA 基线 → 逐函数 test → 12 shader 算法模拟 → RoPE 逐行对比），全部通过
- **出图**：3 步 **75KB**（PC 参考 74KB），是目前最接近参考的手机管线 ✅
- **速度**：~100s/步（fp32 + 515MHz 底频），HybridOps 47s/步（fp16）仍是速度基线
- **代码**：`vulkan/dit_engine_v2.cpp` (~1180行) + `vulkan/head_tail_ops.h` (C++ CPU head/tail)，12 fp32 shader
- **验证脚本**：`scripts/replica/` 下 7 个测试脚本

### HybridOps safetensors 直读 ✅ (2026-06-01 凌晨)

`hybridops/` 已实现：
- 手机直读 `.safetensors`（BF16），加载时一次性 BF16→FP16 存入 Vulkan
- 新 Vulkan 引擎 `libhybrid_engine.so` (~500KB): GEMM/LN/RMSNorm/GELU per-call dispatch
- PyTorch 不再持有 block 权重 (~200MB shell), 稳态内存 ~3.9GB
- 详见 `STATUS.md` 和 `hybridops/scripts/phone_pipeline.py`

### C++ Vulkan Engine v2 — REWRITTEN & WORKING

- **libdit_vk.so**: ~1000 lines, 8 shader pipelines, 28 per-block cmd buffers
- **Speed**: 9.9s/step (28 blocks, self-attn+MLP), 12× vs CPU 120s, 5.7× vs Python ctypes 56s
- **Correctness**: Block 0 with real pipeline inputs — max_err=0.75 vs PyTorch, mean/std matching
- **Architecture**: Per-tensor weight buffers (567 tensors, 3.9GB), shared bcBuf (18MB), Type 6 unified memory

### Key technical findings

- Adreno single cmd buffer limit ~64 dispatches (submit fails beyond)
- Single VkBuffer allocation fails at 3.9GB (per-tensor buffers solve)
- fp16 overflow risk at block 23+ with out-of-distribution inputs (real pipeline inputs should be fine)
- Old engine's "all zeros" was vkQueueSubmit failure, not driver bug
- Validation strategy: PyTorch dump → unload → C++ compare (avoid dual 3.9GB OOM)

### Completed

- ✅ Phase 10c: Phone e2e pipeline — 3-step 256×256 via Python CPU/ctypes
- ✅ Vulkan GEMM 149 GFLOPS (f16vec4+dot shader)
- ✅ C++ engine v2: init, weight loading, 8 pipelines, 28-block recording, forward
- ✅ Per-shader validation: GEMM, LayerNorm, RMSNorm, SiLU, ScaleShift
- ✅ Multi-dispatch barrier chain validation: LN+4×GEMM, full self-attn block
- ✅ Full block (self-attn+MLP) with real pipeline inputs validated

### Pending

- VAE decode (stay PyTorch for now)
- Lock GPU 912MHz + GEMM shader optimization → target <30s/step
- APK packaging
- QNN NPU exploration

### Completed (2026-06-01 daytime)

- ✅ sin/cos bug fix (head_tail_ops.h) — root cause of 123KB output
- ✅ 12 shader algorithm verification vs PyTorch (all within 1.5× CPU-vs-CUDA baseline)
- ✅ RoPE 3D dimension split verified correct (matches VideoRopePosition3DEmb line-by-line)
- ✅ Systematic verification framework (scripts/replica/)
- ✅ End-to-end pipeline: 75KB output image ✅

**The "driver bug" was wrong.** All prior "inf/nan/.so garbage" diagnoses were parameter pollution (tile=4 or shape=4×4 triggered shader degeneration). Binary baseline re-verified clean. The real bugs found and fixed:

1. **Dispatch swap bug** (gemm.comp): `gl_GlobalInvocationID` row/col mapped opposite to dispatch X/Y axes. M=N时隐藏, M≠N时columns被截断输出0. Fixed by swapping 2 lines in shader. 281 GEMM calls all max_err<0.02 ✅

2. **GEMM shader performance**: clock_gettime internal timing revealed 98% of 178s VK wall time is in GPU submit+fence. Our generic tiled GEMM shader achieves only **14 GFLOPS on Adreno 730** — **0.75% GPU utilization** (1,843 GFLOPS theoretical fp16 peak).

**Optimizations tried and their results:**

| Optimization | Expected | Actual | Conclusion |
|---|---|---|---|
| Cached Vulkan resources | ~30s | 2s | alloc not bottleneck |
| fp16 direct path (no f2h/h2f) | ~20s | 3s | CPU conversion not bottleneck |
| Batch submission (281→140 submit) | ~40s | -2s | submit/fence not bottleneck |
| Type 6 HOST_COHERENT (no DMA) | ~150s | 0s | DMA not bottleneck |

**Native fp16 shader (f16vec4 + dot)** based on ncnn approach: 149 GFLOPS (10.6× improvement, 8.1% GPU util). Per-GEMM time dropped from 627ms→100ms (K=8192). Estimated step time: ~54s (2.2× faster than CPU 120s).

**Current speed**: Vulkan 220s/step, CPU 120s/step. Vulkan still slower than CPU. New shader integration in progress.

亮屏 vs 息屏: 120s vs 550s (MIUI throttles CPU by 4.5× when screen off). Fixed with `screen_off_timeout=30min`.

### Vulkan GPU acceleration (2026-05-26)

**Correctness**: Dispatch swap bug fixed. All GEMM calls max_err<0.02. 3-step pipeline output clean ✅

**Speed**: 220s/step (Vulkan) vs 120s/step (CPU). Vulkan is slower because generic tiled GEMM shader is 0.75% efficient on Adreno 730.

**New fp16 shader** (f16vec4 + dot, based on ncnn): achieves 149 GFLOPS (8.1% GPU util). Integration into libvk_gemm.so in progress. Expected step time ~54s.

**Phone pipeline velocity**:
- CPU-only: 120s/step (256×256) ✅
- Python ctypes per-layer Vulkan: 56s/step ✅
- **C++ engine v2 (28 blocks): 9.9s/step** ✅ (self-attn+MLP only)
- C++ engine (est. +cross-attn+attention): ~15-20s/step ⏳

### WSL workspace

```text
/home/riorg/anima-work/
  src/anima_min/
    predict2.py             # DiT
    llm_adapter.py          # LLMAdapter
    flow_match.py           # FlowMatchScheduler
    wan_vae.py              # WanVAE (has grid bug, use DiffSynth VAE instead)
    beta57_ersde.py         # er_sde + beta57 scheduler (buggy)
    position_embedding.py   # RoPE
    vae_ops.py              # VAE ops
  scripts/
    run_diffsynth_final.py  # ✅ Verified: DiffSynth native pipeline
    pipeline_e2e.py         # Self-built e2e pipeline (grid bug)
    export_cpu.py           # DiT CPU export for phone
    export_dit_int4.py      # DiT INT8/INT4 export (abandoned)
    decode_one.py           # Latent → PNG decoder
    probe_assets.py         # Asset probe
  output/
    base.png, turbo.png     # DiffSynth verified outputs
    anima_dit_cpu.pt2       # Exported DiT (FP16, dynamic shapes)
  ROADMAP.md
```

### Phone workspace

```text
/sdcard/anima_on_android/
  models/
    diffusion_weights_fp16.pt  (3.7GB)
    qwen_weights_fp16.pt       (1.2GB)
    vae_weights_fp16.pt        (243MB)
    context_cond.pt / context_uncond.pt  (pre-computed)
  src/   predict2.py, llm_adapter.py, position_embedding.py, wan_vae.py, vae_ops.py
  scripts/  phone_pipeline.py (e2e: DiT denoising + VAE decode)
  output/
```

### Decision principles (reaffirmed)

- DiffSynth-Studio (Apache 2.0) is the verified reference pipeline.
- Self-built pipeline (pipeline_e2e.py) has grid bug — deferred, not abandoned.
- Mobile: native PyTorch on Termux, CPU-only for now; GPU acceleration deferred.
- er_sde + beta57: deferred until mobile pipeline is stable.

## 13. Known issues

### WanVAE grid bug ✅ FIXED (2026-05-25)

**Root cause**: Our `wan_vae.py` decode() was missing latent normalization. DiffSynth's WanVideoVAE applies per-channel mean/std scaling before decode: `z = z * (1/std) + mean`. Our port skipped this step.

**Fix**: Added 16-channel `latent_mean` and `latent_std` buffers (hardcoded values from DiffSynth source) to WanVAE.__init__, applied in decode().

**Verification**: Same latent decoded by our fixed VAE vs DiffSynth VAE now produces identical statistics at both 256×256 and 1024×1024.

### Self-built pipeline (e2e) — deferred

Same VAE root cause. Text encoder + LLMAdapter + DiT + scheduler are all verified correct. Pipeline retained for future debugging.

### er_sde + beta57 (deferred)

Anima model card's recommended sampler. Our implementation produces incorrect results. DiffSynth's `FlowMatchScheduler` (Euler + Z-Image) substitutes.

### Phone pipeline status (2026-05-26 evening)

| Config | Speed | Image | Notes |
|--------|-------|-------|-------|
| CPU only | 120s/step | ✅ Clean (256×256) | Baseline |
| Vulkan (old generic shader) | 220s/step | ✅ Clean | dispatch swap fixed, but 0.75% GPU util |
| Vulkan (new fp16 shader) | ~54s/step (est.) | TBD | 8.1% GPU util, integration in progress |

**Decision**: Vulkan path is correct now (dispatch swap fixed). Performance bottleneck is GEMM shader efficiency, not driver bugs. New native fp16 shader (f16vec4+dot) achieves 10.6× GPU throughput improvement. Continue optimizing shader and plan C++ DiT engine for next phase.

### Phone CPU performance note

亮屏 vs 息屏: 120s vs 550s (MIUI throttles 4.5× when screen off). Fixed with `screen_off_timeout=30min`.

---

## 14. Next phases (DiffSynth-based)

### Phase 9: DiffSynth baseline verification ✅ (2026-05-25)

**Final result**: Both base and turbo confirmed working via DiffSynth native pipeline (DiffSynth scheduler + DiffSynth VAE) on RTX 3060 12GB.

**Files**:
- Base: `scripts/run_diffsynth_final.py` → `output/base.png` ✅
- Turbo: `scripts/run_diffsynth_final.py` → `output/turbo.png` ✅

**Confirmed baselines**:

| | Base | Turbo |
|---|---|---|
| Steps | 30 | 12 |
| CFG | 5.0 | 1.0 |
| LoRA | ❌ | ✅ (508 keys merged) |
| Scheduler | Z-Image (Euler) | Z-Image (Euler) |
| VAE | DiffSynth native | DiffSynth native |
| GPU | 3060 12G | 3060 12G |
| Result | ✅ Clean | ✅ Clean |

**Key findings**:
- Our WanVAE port has a subtle bug (grid artifacts) — use DiffSynth native VAE for verification
- Euler + Z-Image is correct and verified; our er_sde + beta57 implementation is buggy
- Self-built pipeline (Phase 7) abandoned; DiffSynth is the reference

**Deferred**: er_sde + beta57 — Anima model card's recommended sampler, user's preferred style. Fix after mobile pipeline is stable. Known issue: our er_sde step function produces wrong results; DiffSynth's `FlowMatchScheduler` (Euler) is a working substitute.

### Phase 10: Mobile deployment ✅ (in progress, 2026-05-25)

#### 10a: DiT export exploration

- **ExecuTorch + torchao**: `torch.export` + `to_edge` + `to_executorch` succeeded on PC; `.pte` file generated. But `.pte` not loadable on phone (torch version incompatibility).
- **INT8 quantization**: `Int8WeightOnlyConfig` worked on PC but phone torchao incompatible (deprecated ops).
- **INT4 quantization**: Blocked by `mslk` dependency.
- **Conclusion**: Abandon ExecuTorch export path for now. Use native PyTorch on phone.

#### 10b: Phone DiT native inference ✅

- Phone workspace: `/sdcard/anima_on_android/`
- Termux PyTorch 2.11 + einops installed
- DiT 2B FP16 (3.7GB) loads and runs successfully via `torch.load`
- Test results (28-block DiT, single forward):
  - H=8 (64 tokens): 9.0s ✅
  - H=16 (256 tokens): 32.8s ✅
  - H=24 (576 tokens): 63.8s ✅
  - H=32 (1024 tokens): 88.1s ✅
- Larger sizes OOM on CPU (attention matrix too large)

#### 10c: Phone end-to-end pipeline ⏳

- Pre-computed LLMAdapter context on PC (bypassed text encoder issue)
- Deployed: DiT 3.7GB + Qwen context 1MB + VAE 243MB
- Script: 5 steps, 256×256 (H=32), Z-Image scheduler, CFG=5
- Running on phone with CPU affinity locked to big cores (A710×3 + X2)
- Expected: ~7 minutes for 5 steps

#### 10d: Mobile GPU acceleration ✅ explored, active development

- **Vulkan GEMM**: GLSL shaders compiled, SPIR-V→Android NDK cross-compiled, `.so` + Python ctypes integration working
- **Standalone test**: GEMM 64³ on Adreno 730 → 7.8× speedup, max_err=0.0001 ✅
- **Pipeline integration**: 281 GEMM calls per step, all correct after dispatch swap fix ✅
- **Performance**: Original generic tiled shader only 14 GFLOPS (0.75% GPU utilization). New native fp16 shader (f16vec4+dot, based on ncnn approach) achieves 149 GFLOPS (10.6×).
- **Conclusion**: "Driver bug" theory was wrong — all prior garbage outputs were parameter pollution. Real bottleneck is shader efficiency on Adreno's scalar/TBDR architecture.
- **Next**: Integrate new shader into .so, then plan C++ DiT engine (keep data on GPU, avoid per-layer Python↔Vulkan boundary).

### Phase 11: Mobile optimization (planned → partially in progress)

**11a**: ✅ Vulkan GEMM shader done — f16vec4+dot achieves 149 GFLOPS (10.6× vs old, 8.1% GPU util). Integrated into libvk_gemm.so + phone pipeline (56s/step).

**11a+**: ✅ Shared memory tiling tested and rejected — Adreno 730 barriers too expensive, 2× slower than sequential global reads.

**11b**: ✅ **C++ DiT engine v2** — libdit_vk.so (~600KB). 28 per-block cmd buffers, GPU AdaLN, 13.06s/step (9.2× vs CPU). Verified: GEMM, LN, RMSNorm, SiLU, ScaleShift, self+cross+MLP blocks with skip-attention. Pending: RoPE+Attention (shader layout fixed, workgroup too many), x_embedder/t_embedder/final_layer.

**11c**: **补全 DiT forward** — 策略：在 HybridOps 框架上逐模块用 Vulkan 替换 PyTorch，每次替换后管线级验证。

**当前进度 (2026-05-29)**:

### AdaLN GPU 注入 ✅ (phone_pipeline.py + libdit_vk.so)

**方法**: PrecomputedAdaLN(nn.Module) 子类替换 `block.adaln_modulation_*`，不用 monkey-patch。

**架构**: `dit_init_adaln_only()` 轻量加载 173 个 AdaLN tensor (390MB)，预录制 28 cmd buffer。每步上传 t_emb+lora → submit → read back → 注入。

**验证**: 86KB PNG, 与纯 PyTorch 一致。AdaLN 开销 <1s/步。

### LayerNorm GPU 注入 ✅ (phone_pipeline.py + libdit_vk.so)

**shader**: FP32 I/O（`layernorm_fp32.comp`），消除 FP16 量化误差。max_err 0.000002。

**教训**: 先用了 libvk_hybrid.so（独立 Vulkan 实例）→ 出图雪花。排查排除了三实例互抢、shader 精度问题，最终定位到 `vk_hybrid_download` 的 HOST_COHERENT 缓存一致性问题。**每个 op 类型必须归入同一个 Vulkan 实例（libdit_vk.so）**，不引入额外实例。

**架构**: `dit_run_layernorm(in, out, M, D, eps)` — 每次调用现场录制 1 dispatch，用独立 cmd buffer（g_lnCmdBuf），不覆盖 AdaLN 预录缓冲区。

### 速度分析 (2026-05-30 更新)

| 组件 | 每步耗时 | 加速状态 | 可节省 |
|------|---------|---------|--------|
| **GEMM** (281 次) | ~25s | ✅ HybridLinear (libvk_gemm.so) | libvk_gemm.so 内预录 |
| **Self-attention** (28 blocks) | ~8s | ✅ 3-pass Vulkan batched | — |
| **Cross-attention** (28 blocks) | ~8s | ✅ 3-pass Vulkan batched | — |
| **LayerNorm** (85 次) | <1s | ✅ libdit_vk.so FP32 | — |
| **RMSNorm** (84 次) | <1s | ✅ libdit_vk.so FP16 | — |
| **GELU** (28 次) | <1s | ✅ libdit_vk.so FP16 | — |
| **AdaLN** (28 blocks) | <1s | ✅ pre-recorded + pool swap | — |
| **t_embedder** | <1s | ✅ C++ CPU | — |
| **RoPE** (56 次) | ~3s | ❌ CPU | GPU 化 -3s |
| **final_layer** | <1s | PyTorch GEMM | C++ |
| 其他（numpy/Python 开销） | ~5s | — | — |
| **总计** | **~63s/步** | | **目标 ~40s（预录后）** |

### Cross-attention 修复 (2026-05-30)

**根因（疑似）**: Adreno 730 可能从 250ms TDR 看门狗，但 510MHz 底频下不应超时。错误模式 (max_err=0.28 部分错误) 更接近 GPU 内部资源竞争（L1 cache eviction / barrier 同步 / wave scheduler 竞争）。确切根因待确认。`maxComputeWorkGroupCount=65535` 是理论值，实际 WG 软上限由未尽文档化的内部限制决定。

**修复**: 内部将 Q 拆为 batch_q=64 (1024 WG) 的分批，每批独立 submit+wait，保证每批 dispatch <250ms。M_kv=1024 时 batch=64 × 16 = 1024 WG 安全，M_kv=512 时也可全量 dispatch。

### Adreno 730 已知 & 疑似限制 (2026-05-30)

| 限制 | 说明 | 确定性 |
|------|------|--------|
| **TDR 250ms** | Qualcomm 文档提及，但 510MHz 底频下纯计算不应超时 | ⚠️ 疑似 |
| **WG 软上限** | 大 M_kv+大 WG+密集 barrier 的组合可复现触发不稳定 | ✅ 实测 |
| **并行 dispatch binding confusion** | 同 cmd buffer 内多 dispatch 可并行，不同 descriptor 可能混淆 | ✅ 已踩坑 |
| **部分错误模式** | ROWS=8 时 max_err=0.28，非全零/随机，更像 cache/barrier race | ❓ 待确认 |
| **官方手册** | Adreno 530 手册 74 页 (本地)。7XX docs.qualcomm.com 按平台代号查 | — |

### 下步路线 (2026-05-30 更新)

1. **GEMM 预录进 libvk_gemm.so block** — 保持双实例，在各自的 .so 内做预录消 submit。libvk_gemm.so 侧：权重预加载 + 28 block GEMM 预录，287 submit → 28。单实例合并已验证路线不通（根因未定位，暂搁置）
2. **RoPE GPU** — 56 次调用省 ~3s/步
3. **Attention 入 block 预录制** — shader 改造（8 行/WG 方向，WG 上限待解决）
4. **最终目标** — 双实例各自 28 submit/step，GPU 交叉执行

**参考来源**: 本地 clone 的 ExecuTorch 仓库 `D:\AI\手坤的anima\参考\sdpa找不到了，只能克隆了\executorch\`，重点参考其 Vulkan backend 的 GLSL shader 实现。官方用模板系统（`${}` 宏）生成 shader，我们直接读展开后的逻辑。与自研 shader 的逐模块对比结论：

| 模块 | 决策 | 理由 |
|------|------|------|
| **GEMM** | ✅ 保留 | f16vec4+dot + 无 shared memory 恰好是 Adreno 最优解；ET 的 shared memory tiling 在 Adreno 上 2× 更慢（已验证） |
| **SiLU** | ✅ 保留 | 简单逐元素 op，不会出错 |
| **ScaleShift** | 🔧 融合进 LN | 省一次 dispatch，不再独立存在 |
| **RoPE** | ❌ 重写 | 当前每行 1 线程，并行度极低；改为 ET 的 per-texel dispatch + interleaved `[r0,i0,r1,i1]` layout |
| **LayerNorm** | 🔧 改造 | 加 affine (weight+bias) 融合；workgroup size 从 256 调到 64（省 shared memory） |
| **Attention** | ❌ 拆 3 重写 | 单 shader 做 QK^T+softmax+SV 寄存器炸、barrier 链太复杂、Adreno 并行 dispatch 出错。改为 ET 的 3-shader 拆分 |

**Attention 拆分策略**（参考 `sdpa_compute_attn_weights_coop.glsl` + `sdpa_attn_weights_softmax.glsl` + `sdpa_compute_out_coop.glsl`）：

```
Q·K^T (compute_attn_weights)  →  中间 buffer: attn_weights [B, H, S, C]
softmax (attn_weights_softmax) →  中间 buffer: attn_softmax [B, H, S, C]  
softmax(A)·V (compute_out)    →  最终 output [B, H, S, D]
```

3 个 dispatch 独立验证，中间结果可 dump 对比 PyTorch。每个 shader 用 cooperative reduction (64 workers) 分摊 head_dim/context_len 维度。

**RoPE 重写要点**（参考 `apply_rotary_emb_interleaved.glsl`）：
- 输入改为 interleaved layout: `[cos0, sin0, cos1, sin1, ...]`（当前存 4 值/对有冗余）
- Dispatch per-texel（4 元素/线程），不再 per-row 单线程
- Freqs 改为 flat buffer 索引，不依赖 rank

**LayerNorm 改造要点**（参考 `native_layer_norm_buffer.glsl`）：
- 加 weight/bias affine 参数
- workgroup size 从 256 → 64（Adreno shared memory 稀缺）
- 保留现有 3-pass tree reduction 结构（与 ET 一致）

- **约束**：每个替换模块必须与 PyTorch 做元素级对齐验证（max_err < 1），不可仅靠 pipeline 端到端结果判断。新写的 C++ 代码需仔细检查与 torch 计算的一致性。

**11d**: VAE decoder → Qualcomm QNN for NPU (project plan priority item)

**11e**: Text encoder on-device — investigate TinyLlama-style small encoder or pre-computed context caching

**11f**: Full pipeline at higher resolutions (1024×1024, up to 2048×1024 for wallpapers)
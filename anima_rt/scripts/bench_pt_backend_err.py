"""Benchmark: PyTorch CPU vs CUDA intrinsic error for full DiT forward (head + 28 blocks + tail).

Purpose: Establish the THEORETICAL ERROR FLOOR before we migrate ops out of PyTorch.
If PT CPU vs PT CUDA already has max_err X, we can't possibly beat X with our own library.

KEY: We use BF16 weights (native from safetensors) + FP32 compute, matching the phone's
"类似20系显卡的bf16内存fp32计算" approach.  FP16 compute overflows (x_embedder weights up to 48640).

Usage (WSL):
  cd /mnt/d/AI/anima_phone
  source /home/riorg/miniconda3/etc/profile.d/conda.sh
  conda activate /home/riorg/anima-work/.conda
  python anima_rt/scripts/bench_pt_backend_err.py
"""

import sys, time, json, struct, mmap, os
import torch
import torch.nn as nn
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import predict2
from predict2 import MiniTrainDIT

# ── Default PyTorch operations factory ──
class DefaultOps:
    Linear = nn.Linear
    RMSNorm = nn.RMSNorm
    LayerNorm = nn.LayerNorm
    Embedding = nn.Embedding

# ═══════════════════════════════════════════════════════════
# Load safetensors  (BF16 native → FP32 for BC-safe compute)
# ═══════════════════════════════════════════════════════════
SAFETENSORS = "/mnt/d/AI/手坤的anima/models/diffusion_model/anima-base-v1.0.safetensors"
CONTEXT_FILE = "/mnt/d/AI/anima_phone/anima_rt/models/context_cond.pt"

print(f"Loading: {SAFETENSORS}")

class SafetensorsReader:
    def __init__(self, path):
        with open(path, 'rb') as f:
            header_len = struct.unpack('<Q', f.read(8))[0]
            header = json.loads(f.read(header_len).decode('utf-8'))
        self.header = header
        self.data_start = 8 + header_len
        self._mmap = None; self._file = None

    def keys(self): return list(self.header.keys())

    def get_tensor(self, key):
        if self._mmap is None:
            self._file = open(SAFETENSORS, 'rb')
            self._mmap = mmap.mmap(self._file.fileno(), 0, access=mmap.ACCESS_READ)
        info = self.header[key]
        off = self.data_start + info['data_offsets'][0]
        end = self.data_start + info['data_offsets'][1]
        buf = self._mmap[off:end]
        dtype_str = info['dtype']
        np_dtype = np.uint16 if dtype_str in ('BF16', 'F16') else np.float32
        return torch.from_numpy(
            np.frombuffer(buf, dtype=np_dtype).copy()
        ).reshape(info['shape'])

    def close(self):
        if self._mmap: self._mmap.close(); self._file.close()

st = SafetensorsReader(SAFETENSORS)

all_keys = st.keys()
tensor_keys = [k for k in all_keys if k != "__metadata__"]
PREFIX = ""
if tensor_keys and '.' in tensor_keys[0]:
    first_part = tensor_keys[0].split('.')[0]
    if first_part not in ('blocks', 'x_embedder', 't_embedder', 'final_layer',
                          't_embedding_norm', 'pos_embedder', 'llm_adapter'):
        PREFIX = first_part + '.'
        print(f"  Detected prefix: '{PREFIX}'")

def strip_prefix(k):
    return k[len(PREFIX):] if PREFIX and k.startswith(PREFIX) else k

# Build state_dict — BF16 → FP32 (safe, no overflow)
state_dict = {}
for key in all_keys:
    if key == "__metadata__": continue
    clean_key = strip_prefix(key)
    data = st.get_tensor(key)
    info = st.header[key]
    if info['dtype'] == 'BF16':
        data = data.view(torch.bfloat16).to(torch.float32)
    elif info['dtype'] == 'F16':
        data = data.view(torch.float16).to(torch.float32)
    elif info['dtype'] == 'F32':
        data = data.to(torch.float32)
    state_dict[clean_key] = data

st.close()
print(f"  Loaded {len(state_dict)} tensors (FP32)")

# ═══════════════════════════════════════════════════════════
# Create model (FP32 — matches "bf16内存fp32计算" strategy)
# ═══════════════════════════════════════════════════════════
config = dict(
    max_img_h=240, max_img_w=240, max_frames=128,
    in_channels=16, out_channels=16,
    patch_spatial=2, patch_temporal=1,
    concat_padding_mask=True,
    model_channels=2048, num_blocks=28, num_heads=16,
    mlp_ratio=4.0, crossattn_emb_channels=1024,
    pos_emb_cls="rope3d", pos_emb_learnable=True,
    pos_emb_interpolation="crop",
    min_fps=1, max_fps=30,
    use_adaln_lora=True, adaln_lora_dim=256,
    rope_h_extrapolation_ratio=4.0, rope_w_extrapolation_ratio=4.0,
    rope_t_extrapolation_ratio=1.0,
    extra_per_block_abs_pos_emb=False,
    rope_enable_fps_modulation=False,
)

print("Creating DiT model (FP32)...")
dit = MiniTrainDIT(**config, device="cpu", dtype=torch.float32,
                   operations=DefaultOps)
dit.load_state_dict(state_dict, strict=False)
dit.eval()
del state_dict
print("  Model loaded")

# ═══════════════════════════════════════════════════════════
# Register hooks to capture per-block outputs
# ═══════════════════════════════════════════════════════════
block_outputs = []

def make_hook(idx):
    def hook(module, input, output):
        block_outputs.append(("b{:02d}".format(idx), output.detach().float().cpu()))
    return hook

for i, block in enumerate(dit.blocks):
    block.register_forward_hook(make_hook(i))

# ═══════════════════════════════════════════════════════════
# Prepare inputs (deterministic, matching phone pipeline)
# ═══════════════════════════════════════════════════════════
print("Preparing inputs...")
SEED = 6666
SIGMA = 1.0
H = 32  # 256×256 image

gen = torch.Generator(device="cpu").manual_seed(SEED)
x = torch.randn(1, 16, H, H, generator=gen).float()  # [B, C, H, W]
x = x.unsqueeze(2)  # [B, C, 1, H, W] — add temporal dim
t = torch.tensor([[SIGMA]], dtype=torch.float32)  # [B, 1]

# Load or generate context
if os.path.exists(CONTEXT_FILE):
    ctx = torch.load(CONTEXT_FILE, weights_only=True).float()
    if ctx.dim() == 2:
        ctx = ctx.unsqueeze(0)
else:
    print("  [WARN] No context file, using random context")
    ctx = torch.randn(1, 512, 1024, generator=gen).float()

print(f"  x: {x.shape}, t: {t.shape}, ctx: {ctx.shape}")

# ═══════════════════════════════════════════════════════════
# Run on CPU
# ═══════════════════════════════════════════════════════════
print("\n=== CPU Run ===")
block_outputs.clear()

dit_cpu = dit.cpu()
x_cpu = x.cpu(); t_cpu = t.cpu(); ctx_cpu = ctx.cpu()

t0 = time.time()
with torch.no_grad():
    v_cpu = dit_cpu(x_cpu, t_cpu, ctx_cpu)
t_cpu_time = time.time() - t0

print(f"  Forward: {t_cpu_time:.1f}s")
print(f"  v_cond: [{v_cpu.min():.6f}, {v_cpu.max():.6f}] nan={torch.isnan(v_cpu).sum().item()}")

cpu_blocks = {name: val for name, val in block_outputs}
v_cpu_f32 = v_cpu.detach().float().cpu()

if torch.isnan(v_cpu_f32).any():
    print("  FATAL: NaN in CPU output — cannot continue")
    for name, val in list(cpu_blocks.items())[:5]:
        print(f"  {name}: [{val.min():.6f}, {val.max():.6f}] nan={torch.isnan(val).any()}")
    sys.exit(1)

# ═══════════════════════════════════════════════════════════
# Run on CUDA
# ═══════════════════════════════════════════════════════════
if torch.cuda.is_available():
    print("\n=== CUDA Run (BF16 model to fit 8GB VRAM) ===")
    block_outputs.clear()

    # Free CPU model first to avoid OOM
    del dit_cpu, dit
    torch.cuda.empty_cache()

    # Create fresh model in BF16 (fits in 8GB VRAM: ~4GB weights + activations)
    # Reload state dict in BF16
    st2 = SafetensorsReader(SAFETENSORS)
    sd_bf16 = {}
    for key in all_keys:
        if key == "__metadata__": continue
        clean_key = strip_prefix(key)
        data = st2.get_tensor(key)
        info = st2.header[key]
        if info['dtype'] == 'BF16':
            data = data.view(torch.bfloat16)
        elif info['dtype'] == 'F16':
            data = data.view(torch.float16).to(torch.bfloat16)
        elif info['dtype'] == 'F32':
            data = data.to(torch.bfloat16)
        sd_bf16[clean_key] = data
    st2.close()

    dit_bf16 = MiniTrainDIT(**config, device="cpu", dtype=torch.bfloat16, operations=DefaultOps)
    dit_bf16.load_state_dict(sd_bf16, strict=False)
    dit_bf16.eval()
    del sd_bf16

    # Re-register hooks
    for i, block in enumerate(dit_bf16.blocks):
        block.register_forward_hook(make_hook(i))

    dit_cuda = dit_bf16.cuda()
    del dit_bf16
    x_cuda = x.to(torch.bfloat16).cuda()
    t_cuda = t.to(torch.bfloat16).cuda()
    ctx_cuda = ctx.to(torch.bfloat16).cuda()
    torch.cuda.synchronize()

    t0 = time.time()
    with torch.no_grad():
        v_cuda = dit_cuda(x_cuda, t_cuda, ctx_cuda)
    torch.cuda.synchronize()
    t_cuda_time = time.time() - t0

    print(f"  Forward: {t_cuda_time:.1f}s")
    print(f"  v_cond: [{v_cuda.min():.6f}, {v_cuda.max():.6f}] nan={torch.isnan(v_cuda).sum().item()}")

    cuda_blocks = {name: val.cpu() for name, val in block_outputs}
    v_cuda_f32 = v_cuda.detach().float().cpu()

    torch.cuda.empty_cache()

    # ═══════════════════════════════════════════════════════════
    # Compare: CPU vs CUDA
    # ═══════════════════════════════════════════════════════════
    print("\n=== CPU vs CUDA Comparison ===")
    print(f"{'Stage':<8} {'CPU_range':<32} {'CUDA_range':<32} {'max_err':<12} {'mean_err':<12}")
    print("-" * 100)

    max_errs = []
    for name in sorted(cpu_blocks.keys(), key=lambda x: int(x[1:])):
        if name not in cuda_blocks: continue
        a = cpu_blocks[name]; b = cuda_blocks[name]
        diff = (a - b).abs()
        max_errs.append(diff.max().item())
        print(f"{name:<8} [{a.min():.4f},{a.max():.4f}]         [{b.min():.4f},{b.max():.4f}]         {diff.max().item():.4e}    {diff.mean().item():.4e}")

    diff_v = (v_cpu_f32 - v_cuda_f32).abs()
    print(f"{'v_cond':<8} [{v_cpu_f32.min():.4f},{v_cpu_f32.max():.4f}]         [{v_cuda_f32.min():.4f},{v_cuda_f32.max():.4f}]         {diff_v.max().item():.4e}    {diff_v.mean().item():.4e}")

    print(f"\nSummary:")
    print(f"  peak block error: b{max_errs.index(max(max_errs)):02d} = {max(max_errs):.4e}")
    print(f"  final v_cond max_err = {diff_v.max().item():.4e}  ← ERROR FLOOR (PT CPU vs CUDA intrinsic)")

    # Error growth curve
    print(f"\nError growth across 28 blocks:")
    prev = 0
    for i, e in enumerate(max_errs):
        ratio = f" x{e/prev:.1f}" if prev > 1e-10 else ""
        label = f"b{i:02d}:{e:.2e}"
        if torch.isinf(torch.tensor(e)):
            label += "(INF!)"
        print(f"  {label}{ratio}")
        prev = e

    print(f"\nDone. CPU (FP32): {t_cpu_time:.1f}s  CUDA (BF16): {t_cuda_time:.1f}s")
else:
    print("\n[SKIP] CUDA not available — only ran CPU")
    print(f"Done. CPU: {t_cpu_time:.1f}s")

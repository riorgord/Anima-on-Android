"""PC-side comparison: load phone inputs, run PyTorch DiT, compare v_cond.
Usage (WSL):
  conda activate /home/riorg/anima-work/.conda
  python compare_vcond_pc.py
"""
import sys, time, struct, json, mmap, gc
import torch, numpy as np

# Path to predict2 model
sys.path.insert(0, "/mnt/d/AI/anima_phone/hybridops/src")
import predict2
import torch.nn as nn

# Use PyTorch's native nn ops (like predict2's default)
class TorchOps:
    Linear = nn.Linear
    RMSNorm = nn.RMSNorm
    LayerNorm = nn.LayerNorm
    Embedding = nn.Embedding

DTYPE = torch.float16
DEV = "cuda" if torch.cuda.is_available() else "cpu"
CMPDIR = "/mnt/d/AI/anima_phone/output/cmp/cmp"  # adb pull nested
SAFETENSORS = "/mnt/d/AI/手坤的anima/models/diffusion_model/anima-base-v1.0.safetensors"

print(f"Device: {DEV}")

# ── Load phone inputs ──
x_b = torch.from_numpy(np.load(f"{CMPDIR}/x_input.npy")).to(DEV).to(DTYPE)
ts_b = torch.from_numpy(np.load(f"{CMPDIR}/ts_input.npy")).to(DEV).to(DTYPE)
ctx = torch.from_numpy(np.load(f"{CMPDIR}/ctx_input.npy")).to(DEV).to(DTYPE)
print(f"Loaded inputs: x={list(x_b.shape)} ts={ts_b} ctx={list(ctx.shape)}")

# ── Load safetensors weights ──
class SafetensorsReader:
    def __init__(self, path):
        self._path = path
        with open(path, 'rb') as f:
            header_len = struct.unpack('<Q', f.read(8))[0]
            header = json.loads(f.read(header_len).decode('utf-8'))
        self.header = header
        self.data_start = 8 + header_len
        self._mmap = None; self._file = None
    def keys(self): return list(self.header.keys())
    def get_tensor(self, key):
        if self._mmap is None:
            self._file = open(self._path, 'rb')
            self._mmap = mmap.mmap(self._file.fileno(), 0, access=mmap.ACCESS_READ)
        info = self.header[key]
        off = self.data_start + info['data_offsets'][0]
        end = self.data_start + info['data_offsets'][1]
        buf = self._mmap[off:end]
        raw_dtype = self.header[key]['dtype']
        shape = info['shape']
        if raw_dtype == 'BF16':
            return torch.from_numpy(np.frombuffer(buf, dtype=np.uint16).copy()).view(torch.bfloat16).to(DTYPE).reshape(shape)
        elif raw_dtype == 'F16':
            return torch.from_numpy(np.frombuffer(buf, dtype=np.uint16).copy()).view(torch.float16).to(DTYPE).reshape(shape)
        else:
            return torch.from_numpy(np.frombuffer(buf, dtype=np.float32).copy()).to(DTYPE).reshape(shape)
    def close(self):
        if self._mmap: self._mmap.close(); self._file.close()

print(f"Loading weights from {SAFETENSORS}...")
st = SafetensorsReader(SAFETENSORS)
PREFIX = "net."
all_keys = [k for k in st.keys() if k != "__metadata__"]
state_dict = {}
for k in all_keys:
    clean = k[len(PREFIX):] if k.startswith(PREFIX) else k
    if clean.startswith("llm_adapter"): continue  # not part of DiT model
    state_dict[clean] = st.get_tensor(k)
st.close()
print(f"  Loaded {len(state_dict)} tensors")

# ── Create DiT model (pure PyTorch) ──
print("Creating DiT (pure PyTorch)...")
config = dict(max_img_h=240, max_img_w=240, max_frames=128, in_channels=16,
    out_channels=16, patch_spatial=2, patch_temporal=1, concat_padding_mask=True,
    model_channels=2048, num_blocks=28, num_heads=16, mlp_ratio=4.0,
    crossattn_emb_channels=1024, pos_emb_cls="rope3d", pos_emb_learnable=True,
    pos_emb_interpolation="crop", min_fps=1, max_fps=30,
    use_adaln_lora=True, adaln_lora_dim=256,
    rope_h_extrapolation_ratio=4.0, rope_w_extrapolation_ratio=4.0,
    rope_t_extrapolation_ratio=1.0, extra_per_block_abs_pos_emb=False,
    rope_enable_fps_modulation=False)

dit = predict2.MiniTrainDIT(**config, device=DEV, dtype=DTYPE, operations=TorchOps)
dit.load_state_dict(state_dict, strict=False)
dit.eval()
del state_dict; gc.collect()
if DEV == "cuda": torch.cuda.empty_cache()
print("Model ready")

# ── Run forward ──
print("Running DiT forward...")
t0 = time.time()
with torch.no_grad():
    v_b = dit(x_b, ts_b, ctx)
dt = time.time() - t0
print(f"Forward done: {dt:.1f}s")

v_cond_pt = v_b[0:1].float().cpu().numpy()
v_uncond_pt = v_b[1:2].float().cpu().numpy()
print(f"PT v_cond: range=[{v_cond_pt.min():.4f}, {v_cond_pt.max():.4f}] nan={np.isnan(v_cond_pt).sum()}")

# ── Compare with phone output ──
v_cond_phone = np.load(f"{CMPDIR}/v_cond_phone.npy")
v_uncond_phone = np.load(f"{CMPDIR}/v_uncond_phone.npy")

print(f"\n{'='*60}")
print("COMPARISON: Phone (anima_rt + Vulkan GEMM) vs PC PyTorch")
print(f"{'='*60}")

def compare(label, phone, pc):
    diff = np.abs(phone.astype(np.float64) - pc.astype(np.float64))
    max_err = diff.max()
    mean_err = diff.mean()
    rel_err = diff / (np.abs(pc.astype(np.float64)) + 1e-8)
    max_rel = rel_err.max()
    n_bad_1pct = int((rel_err > 0.01).sum())
    print(f"  {label}: max_err={max_err:.6f} mean_err={mean_err:.6f} max_rel={max_rel:.4f} n_rel>1%={n_bad_1pct}/{diff.size}")

compare("v_cond", v_cond_phone, v_cond_pt)
compare("v_uncond", v_uncond_phone, v_uncond_pt)

# ── Check if error is spatially structured (green lines = specific channels/locations) ──
diff_cond = v_cond_phone.astype(np.float64) - v_cond_pt.astype(np.float64)
# v_cond shape: [1, 16, 1, 32, 32]
# Check per-channel errors
for c in range(min(16, diff_cond.shape[1])):
    ch_diff = diff_cond[0, c, 0]
    print(f"  ch{c:2d}: max={np.abs(ch_diff).max():.4f} mean={np.abs(ch_diff).mean():.4f} std={ch_diff.std():.4f}")

# Check spatial error pattern (is it uniform or structured?)
spatial_diff = np.abs(diff_cond[0, :, 0]).mean(axis=0)  # avg over channels → [H,W]
print(f"\n  Spatial error pattern: mean={spatial_diff.mean():.4f} H-std={spatial_diff.std(axis=0).mean():.4f} W-std={spatial_diff.std(axis=1).mean():.4f}")

"""Phone-side: run DiT forward with anima_rt norms, dump v_cond + block outputs.
Uses real pipeline inputs (not synthetic), same as phone_pipeline_anima_rt.py.
Purpose: compare against PC PyTorch reference to find where the green-line error comes from.

Usage:
  adb push scripts/dump_dit_vcond.py /sdcard/anima_on_android/scripts/
  adb push scripts/anima_rt_ops.py /sdcard/anima_on_android/scripts/
  adb shell "su -c 'taskset f0 /data/data/com.termux/files/usr/bin/python -u /sdcard/anima_on_android/scripts/dump_dit_vcond.py'"
  adb pull /sdcard/anima_on_android/output/cmp/ output/cmp/
"""
import sys, os, time, gc, json, struct, mmap, ctypes
sys.path.insert(0, "/sdcard/anima_on_android/src")
import torch, numpy as np
import predict2
sys.path.insert(0, "/sdcard/anima_on_android/scripts")
import vk_ops
import anima_rt_ops

DEV = "cpu"
DTYPE = torch.float16
SEED = 12345
H = 32  # 256×256
SIGMA = 1.0  # match PC reference

OUTDIR = "/sdcard/anima_on_android/output/cmp"
os.makedirs(OUTDIR, exist_ok=True)

# ── Patch nn.GELU after model creation ──
def _patch_model_anima_rt(model):
    import torch.nn as nn
    for name, child in list(model.named_children()):
        if isinstance(child, nn.LayerNorm):
            new = anima_rt_ops.AnimaRTLayerNorm(
                child.normalized_shape, eps=child.eps,
                elementwise_affine=child.elementwise_affine,
                dtype=child.weight.dtype if child.weight is not None else torch.float16,
                device=child.weight.device if child.weight is not None else DEV)
            if child.weight is not None: new.weight.data.copy_(child.weight.data)
            if child.bias is not None:   new.bias.data.copy_(child.bias.data)
            setattr(model, name, new)
        elif isinstance(child, nn.RMSNorm):
            new = anima_rt_ops.AnimaRTRMSNorm(
                child.normalized_shape, eps=child.eps,
                dtype=child.weight.dtype, device=child.weight.device)
            new.weight.data.copy_(child.weight.data)
            setattr(model, name, new)
        elif isinstance(child, nn.GELU):
            setattr(model, name, anima_rt_ops.AnimaRTGELU())
        else:
            _patch_model_anima_rt(child)

# ── Safetensors reader ──
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
    def dtype_code(self, key):
        d = self.header[key]['dtype']
        return {'BF16': 2, 'F16': 1, 'F32': 0}.get(d, 2)
    def numpy_dtype(self, key):
        return np.uint16 if self.header[key]['dtype'] in ('BF16', 'F16') else np.float32
    def get_tensor(self, key):
        if self._mmap is None:
            self._file = open(self._path, 'rb')
            self._mmap = mmap.mmap(self._file.fileno(), 0, access=mmap.ACCESS_READ)
        info = self.header[key]
        off = self.data_start + info['data_offsets'][0]
        end = self.data_start + info['data_offsets'][1]
        return np.frombuffer(self._mmap[off:end], dtype=self.numpy_dtype(key)).reshape(info['shape'])
    def close(self):
        if self._mmap: self._mmap.close(); self._file.close()

# ═══════════════════════════════════════════════════════════════
print("Init Vulkan engine (GEMM only)...")
if not vk_ops._lib.vk_engine_init():
    print("FATAL: vk_engine_init failed"); sys.exit(1)

SAFETENSORS = "/sdcard/anima_on_android/models/diffusion.safetensors"
print(f"Loading: {SAFETENSORS}")
st = SafetensorsReader(SAFETENSORS)

# Detect prefix
PREFIX = ""
tensor_keys = [k for k in st.keys() if k != "__metadata__"]
if tensor_keys and '.' in tensor_keys[0]:
    first_part = tensor_keys[0].split('.')[0]
    if first_part not in ('blocks', 'x_embedder', 't_embedder', 'final_layer',
                          't_embedding_norm', 'pos_embedder', 'llm_adapter'):
        PREFIX = first_part + '.'

def strip(k): return k[len(PREFIX):] if PREFIX and k.startswith(PREFIX) else k

shell_sd = {}
for key in st.keys():
    if key == "__metadata__": continue
    clean = strip(key)
    if vk_ops.is_block_gemm_key(clean):
        data = st.get_tensor(key)
        shape = list(data.shape)
        shape_arr = (ctypes.c_int * len(shape))(*shape)
        vk_ops._lib.vk_weight_add(clean.encode(), data.ctypes.data, st.dtype_code(key), shape_arr, len(shape))
        del data
    else:
        data = st.get_tensor(key)
        raw = st.header[key]['dtype']
        if raw == 'BF16': tensor = torch.from_numpy(data.copy()).view(torch.bfloat16).to(DTYPE)
        elif raw == 'F16': tensor = torch.from_numpy(data.copy()).view(torch.float16)
        else: tensor = torch.from_numpy(data.copy()).to(DTYPE)
        shell_sd[clean] = tensor
        del data
st.close(); gc.collect()

vk_ops._lib.vk_engine_finalize()
print("Vulkan engine ready")

# ═══════════════════════════════════════════════════════════════
print("Creating DiT model...")
config = dict(max_img_h=240, max_img_w=240, max_frames=128, in_channels=16,
    out_channels=16, patch_spatial=2, patch_temporal=1, concat_padding_mask=True,
    model_channels=2048, num_blocks=28, num_heads=16, mlp_ratio=4.0,
    crossattn_emb_channels=1024, pos_emb_cls="rope3d", pos_emb_learnable=True,
    pos_emb_interpolation="crop", min_fps=1, max_fps=30,
    use_adaln_lora=True, adaln_lora_dim=256,
    rope_h_extrapolation_ratio=4.0, rope_w_extrapolation_ratio=4.0,
    rope_t_extrapolation_ratio=1.0, extra_per_block_abs_pos_emb=False,
    rope_enable_fps_modulation=False)

dit = predict2.MiniTrainDIT(**config, device=DEV, dtype=DTYPE, operations=vk_ops.DummyOps)
vk_ops.patch_shell_linear(dit)
dit.load_state_dict(shell_sd, strict=False)
dit.eval()
del shell_sd; gc.collect()

vk_ops.patch_block_layers(dit)  # block Linear → VulkanGemmLinear
_patch_model_anima_rt(dit)       # LN/RMS/GELU → AnimaRT
print("Model ready (GEMM=Vulkan, norms/GELU=anima_rt)")

# ═══════════════════════════════════════════════════════════════
# Generate same inputs as PC reference (seed=12345, sigma=1.0)
# Real pipeline inputs: CFG batch=2, H=32, real latent distribution
rng = np.random.RandomState(SEED)
x_latent = torch.from_numpy(rng.randn(1, 16, 1, H, H).astype(np.float32)).to(DEV).to(DTYPE)
ts = torch.tensor([SIGMA], dtype=DTYPE)

# Generate context — use same method as gen_real_inputs would
ctx = torch.from_numpy(rng.randn(2, 512, 1024).astype(np.float32) * 0.02).to(DEV).to(DTYPE)

# CFG: duplicate latent, provide cond+uncond context
x_b = x_latent.repeat(2, 1, 1, 1, 1)
ts_b = ts.repeat(2)

print(f"x_b: {x_b.shape} ts_b: {ts_b} ctx: {ctx.shape}")

# Save inputs for PC comparison
np.save(f"{OUTDIR}/x_input.npy", x_b.cpu().float().numpy())
np.save(f"{OUTDIR}/ts_input.npy", ts_b.cpu().float().numpy())
np.save(f"{OUTDIR}/ctx_input.npy", ctx.cpu().float().numpy())
print("Inputs saved")

# ═══════════════════════════════════════════════════════════════
# Run 1 forward step + dump outputs
print("Running DiT forward...")
vk_ops._lib.vk_reset_pool()
t0 = time.time()

with torch.no_grad():
    v_b = dit(x_b, ts_b, ctx)

dt = time.time() - t0
print(f"Forward done: {dt:.0f}s")

# Dump final v_cond + v_uncond
v_cond = v_b[0:1].float().cpu().numpy()
v_uncond = v_b[1:2].float().cpu().numpy()
np.save(f"{OUTDIR}/v_cond_phone.npy", v_cond)
np.save(f"{OUTDIR}/v_uncond_phone.npy", v_uncond)
print(f"v_cond: shape={v_cond.shape} range=[{v_cond.min():.4f}, {v_cond.max():.4f}] nan={np.isnan(v_cond).sum()}")
print(f"v_uncond: shape={v_uncond.shape} range=[{v_uncond.min():.4f}, {v_uncond.max():.4f}] nan={np.isnan(v_uncond).sum()}")

# Also save the VAE latent (used by the pipeline)
x_cfg = v_uncond + 5.0 * (v_cond - v_uncond)
print(f"v_cfg: range=[{x_cfg.min():.4f}, {x_cfg.max():.4f}]")

vk_ops._lib.vk_engine_destroy()
del dit; gc.collect()
print("Done. Files in:", OUTDIR)

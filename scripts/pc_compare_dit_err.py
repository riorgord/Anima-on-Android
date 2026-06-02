"""PC err comparison: patched DiT vs pure PT (single model, sequential).
Usage (WSL2):
  source /home/riorg/miniconda3/etc/profile.d/conda.sh
  conda activate /home/riorg/anima-work/.conda
  python /mnt/d/AI/anima_phone/scripts/pc_compare_dit_err.py
"""
import sys, math, types, time, os
import torch, torch.nn as nn, numpy as np
sys.path.insert(0, "/mnt/d/AI/anima_phone/src")
import predict2, safetensors.torch as st

# ═══════════════════════════════════════════
# Load anima_rt host .so
# ═══════════════════════════════════════════
import ctypes as ct
_lib = ct.CDLL("/mnt/d/AI/anima_phone/anima_rt/libanima_rt_host.so")
_lib.anima_rt_init.restype = ct.c_bool
_lib.anima_rt_run_gelu.argtypes = [ct.c_void_p, ct.c_void_p, ct.c_int]
_lib.anima_rt_run_gelu.restype = ct.c_bool
_lib.anima_rt_run_silu.argtypes = [ct.c_void_p, ct.c_void_p, ct.c_int]
_lib.anima_rt_run_silu.restype = ct.c_bool
_lib.anima_rt_run_layernorm.argtypes = [ct.c_void_p, ct.c_void_p, ct.c_int, ct.c_int, ct.c_float]
_lib.anima_rt_run_layernorm.restype = ct.c_bool
_lib.anima_rt_run_rmsnorm.argtypes = [ct.c_void_p, ct.c_void_p, ct.c_void_p, ct.c_int, ct.c_int, ct.c_float]
_lib.anima_rt_run_rmsnorm.restype = ct.c_bool
_lib.anima_rt_run_sdpa_flash.argtypes = [ct.c_void_p]*4 + [ct.c_int]*4 + [ct.c_float, ct.c_bool]
_lib.anima_rt_run_sdpa_flash.restype = ct.c_bool
assert _lib.anima_rt_init(), "host .so init failed"

# ═══════════════════════════════════════════
# anima_rt ops
# ═══════════════════════════════════════════
class AnimaRTLayerNorm(nn.LayerNorm):
    def forward(self, x):
        if self.elementwise_affine:
            return torch.nn.functional.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        *batch, D = x.shape; M = int(np.prod(batch)) if batch else 1
        x_f32 = x.reshape(M, D).float().cpu().numpy()
        out_buf = np.zeros((M, D), dtype=np.float32)
        ok = _lib.anima_rt_run_layernorm(x_f32.ctypes.data, out_buf.ctypes.data, M, D, ct.c_float(self.eps))
        if not ok: return torch.nn.functional.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        out = torch.from_numpy(out_buf).to(device=x.device, dtype=x.dtype)
        return out.reshape(*batch, D) if batch else out.squeeze(0)

class AnimaRTRMSNorm(nn.RMSNorm):
    def forward(self, x):
        *batch, D = x.shape; M = int(np.prod(batch)) if batch else 1
        x_f32 = x.reshape(M, D).float().cpu().numpy()
        w_f32 = self.weight.detach().cpu().float().numpy().copy()
        out_buf = np.zeros((M, D), dtype=np.float32)
        ok = _lib.anima_rt_run_rmsnorm(x_f32.ctypes.data, w_f32.ctypes.data, out_buf.ctypes.data, M, D, ct.c_float(self.eps))
        if not ok: return torch.nn.functional.rms_norm(x, self.normalized_shape, self.weight, self.eps)
        out = torch.from_numpy(out_buf).to(device=x.device, dtype=x.dtype)
        return out.reshape(*batch, D) if batch else out.squeeze(0)

class AnimaRTGELU(nn.GELU):
    def forward(self, x):
        flat = x.reshape(-1).float().cpu().numpy()
        out_buf = np.zeros(len(flat), dtype=np.float32)
        _lib.anima_rt_run_gelu(flat.ctypes.data, out_buf.ctypes.data, len(flat))
        return torch.from_numpy(out_buf).to(device=x.device, dtype=x.dtype).reshape(x.shape)

class AnimaRTSiLU(nn.SiLU):
    def forward(self, x):
        flat = x.reshape(-1).float().cpu().numpy()
        out_buf = np.zeros(len(flat), dtype=np.float32)
        _lib.anima_rt_run_silu(flat.ctypes.data, out_buf.ctypes.data, len(flat))
        return torch.from_numpy(out_buf).to(device=x.device, dtype=x.dtype).reshape(x.shape)

def anima_rt_sdpa(q, k, v):
    B, H, S_q, D = q.shape; S_kv = k.shape[2]
    q_f32 = q.reshape(B*H, S_q, D).float().cpu().numpy().astype(np.float32)
    k_f32 = k.reshape(B*H, S_kv, D).float().cpu().numpy().astype(np.float32)
    v_f32 = v.reshape(B*H, S_kv, D).float().cpu().numpy().astype(np.float32)
    out_buf = np.zeros((B*H, S_q, D), dtype=np.float32)
    scale = 1.0 / np.sqrt(D)
    _lib.anima_rt_run_sdpa_flash(q_f32.ctypes.data, k_f32.ctypes.data, v_f32.ctypes.data,
                                   out_buf.ctypes.data, B*H, S_q, S_kv, D,
                                   ct.c_float(scale), ct.c_bool(False))
    return torch.from_numpy(out_buf).to(device=q.device, dtype=q.dtype).reshape(B, H, S_q, D)

# ═══════════════════════════════════════════
# Patch functions
# ═══════════════════════════════════════════
def patch_model_anima_rt(model):
    def _patch_sequential(seq):
        for i, child in enumerate(seq):
            if isinstance(child, nn.SiLU): seq[i] = AnimaRTSiLU()
            elif isinstance(child, nn.Sequential): _patch_sequential(child)
    for name, child in list(model.named_children()):
        if isinstance(child, nn.LayerNorm):
            new = AnimaRTLayerNorm(child.normalized_shape, eps=child.eps,
                                   elementwise_affine=child.elementwise_affine,
                                   dtype=child.weight.dtype if child.weight is not None else torch.float16)
            if child.weight is not None: new.weight.data.copy_(child.weight.data)
            if child.bias is not None: new.bias.data.copy_(child.bias.data)
            setattr(model, name, new)
        elif isinstance(child, nn.RMSNorm):
            new = AnimaRTRMSNorm(child.normalized_shape, eps=child.eps, dtype=child.weight.dtype)
            new.weight.data.copy_(child.weight.data); setattr(model, name, new)
        elif isinstance(child, nn.GELU): setattr(model, name, AnimaRTGELU())
        elif isinstance(child, nn.SiLU): setattr(model, name, AnimaRTSiLU())
        elif isinstance(child, nn.Sequential): _patch_sequential(child)
        else: patch_model_anima_rt(child)

def numpy_rope(t, freqs):
    t_np = t.float().cpu().numpy(); f_np = freqs.float().cpu().numpy()
    half_D = t_np.shape[-1] // 2; t_shape = t_np.shape
    t_ = np.expand_dims(np.moveaxis(t_np.reshape(*t_shape[:-1], 2, half_D), -2, -1), -2)
    t_out = f_np[..., 0] * t_[..., 0] + f_np[..., 1] * t_[..., 1]
    return torch.from_numpy(np.moveaxis(t_out, -1, -2).reshape(*t_shape)).to(device=t.device, dtype=t.dtype)

def numpy_timesteps_forward(self, timesteps_B_T):
    ts_np = timesteps_B_T.float().cpu().numpy()
    B = ts_np.shape[0]; T = 1 if ts_np.ndim < 2 else ts_np.shape[1]
    timesteps = ts_np.reshape(-1); half_dim = self.num_channels // 2
    exponent = -math.log(10000) * np.arange(half_dim, dtype=np.float32) / (half_dim - 0.0)
    emb = np.exp(exponent); emb = timesteps[:, None] * emb[None, :]
    emb = np.concatenate([np.cos(emb), np.sin(emb)], axis=-1).reshape(B, T, -1)
    return torch.from_numpy(emb).to(device=timesteps_B_T.device, dtype=timesteps_B_T.dtype)


# ═══════════════════════════════════════════
# Main
# ═══════════════════════════════════════════
DEV = "cpu"; DTYPE = torch.float16
SAFETENSORS = "/mnt/d/AI/手坤的anima/models/diffusion_model/anima-base-v1.0.safetensors"

print("Loading safetensors...")
t0 = time.time()
raw = st.load_file(SAFETENSORS, device="cpu")
sd = {}
for k, v in raw.items():
    if k.startswith("net."): sd[k[4:]] = v.to(DTYPE)
    elif k != "__metadata__": sd[k] = v.to(DTYPE)
del raw
print(f"  {len(sd)} keys ({time.time()-t0:.0f}s)")

config = dict(max_img_h=240, max_img_w=240, max_frames=128, in_channels=16,
    out_channels=16, patch_spatial=2, patch_temporal=1, concat_padding_mask=True,
    model_channels=2048, num_blocks=28, num_heads=16, mlp_ratio=4.0,
    crossattn_emb_channels=1024, pos_emb_cls="rope3d", pos_emb_learnable=True,
    pos_emb_interpolation="crop", min_fps=1, max_fps=30,
    use_adaln_lora=True, adaln_lora_dim=256,
    rope_h_extrapolation_ratio=4.0, rope_w_extrapolation_ratio=4.0,
    rope_t_extrapolation_ratio=1.0, extra_per_block_abs_pos_emb=False,
    rope_enable_fps_modulation=False)

dit = predict2.MiniTrainDIT(**config, device=DEV, dtype=DTYPE, operations=nn)
dit.load_state_dict(sd, strict=False); dit.eval()
print(f"  Model ready ({time.time()-t0:.0f}s)")

# Fixed inputs
torch.manual_seed(42)
x = torch.randn(1, 16, 1, 32, 32, dtype=DTYPE, device=DEV)
sigma = torch.tensor([1.0], dtype=torch.float32, device=DEV)
ctx = torch.randn(1, 512, 1024, dtype=DTYPE, device=DEV)

# ═══════════ PT reference ═══════════
print("\nRunning PT reference...")
t0 = time.time()
with torch.no_grad():
    v_pt = dit(x, sigma, ctx)
print(f"  PT: {time.time()-t0:.0f}s  v_cond=[{v_pt.min():.4f},{v_pt.max():.4f}]")

# Save PT output
v_pt_cpu = v_pt.float().cpu().clone()
del v_pt

# ═══════════ Patch & run ═══════════
print("\nPatching model...")
_saved_rope = predict2.apply_rotary_pos_emb
_saved_sdpa = predict2._scaled_dot_product_attention
predict2.apply_rotary_pos_emb = numpy_rope
def _patched_sdpa(q,k,v,heads,skip_reshape=False,**kw):
    out = anima_rt_sdpa(q,k,v)  # [B, H, S_q, D]
    if skip_reshape:
        return out.transpose(1,2).reshape(q.shape[0], -1, heads * q.shape[-1])
    return out.reshape(q.shape[0], q.shape[2], heads * q.shape[-1])
predict2._scaled_dot_product_attention = _patched_sdpa
dit.t_embedder[0].forward = types.MethodType(numpy_timesteps_forward, dit.t_embedder[0])
patch_model_anima_rt(dit)
print("  Patched")

print("Running patched forward...")
t0 = time.time()
with torch.no_grad():
    v_patched = dit(x, sigma, ctx)
print(f"  Patched: {time.time()-t0:.0f}s  v_cond=[{v_patched.min():.4f},{v_patched.max():.4f}]")

v_patched_cpu = v_patched.float().cpu()
del v_patched

# ═══════════ Compare ═══════════
err = (v_patched_cpu - v_pt_cpu).abs()
max_err = err.max().item(); mean_err = err.mean().item()
print(f"\n  max_err:  {max_err:.6f}")
print(f"  mean_err: {mean_err:.6f}")

BASELINE = 0.0495
print(f"\n  Baseline (PT CPU vs CUDA): {BASELINE:.4f}")
print(f"  1.5x threshold: {BASELINE*1.5:.4f}")
print(f"  RESULT: {'PASS' if max_err < BASELINE*1.5 else 'FAIL'}")

"""Shared utilities for replica verification scripts.
Usage: from scripts.replica.common import *
WSL path: /mnt/d/AI/anima_phone/scripts/replica/common.py
"""
import os, sys, json, time, struct
import numpy as np
import torch
import torch.nn.functional as F

# ── Paths ──
SF_PATH = "/mnt/d/AI/手坤的anima/models/diffusion_model/anima-base-v1.0.safetensors"
REPLICA_DIR = "/mnt/d/AI/anima_phone/scripts/replica"

# ── Model constants ──
D = 2048
CTXD = 1024
NCTX = 512
NH = 16
HD = 128
MLP_HIDDEN = 8192
ADALN_LORA_DIM = 256
M, S = 2, 256
MS = M * S
D3 = 3 * D
HEAD_DIM = HD

# ── Weight loading ──
def load_weights(sf_path=None, device="cpu", dtype=torch.float32):
    """Load safetensors, strip net. prefix, convert to dtype."""
    import safetensors.torch
    sf = sf_path or SF_PATH
    t0 = time.time()
    st = safetensors.torch.load_file(sf, device=device)
    sd = {}
    for k, v in st.items():
        nk = k[4:] if k.startswith("net.") else k
        if v.dtype == torch.bfloat16:
            sd[nk] = v.to(dtype)
        else:
            sd[nk] = v.to(dtype)
    del st
    print(f"  Loaded {len(sd)} tensors in {time.time()-t0:.1f}s")
    return sd

def get_weight(sd, name, device="cpu", dtype=torch.float32):
    """Get a single weight tensor, converting if needed."""
    t = sd[name]
    if t.dtype != dtype:
        t = t.to(dtype)
    if t.device != torch.device(device):
        t = t.to(device)
    return t

# ── Error metrics ──
def compare(a, b, label="", rtol=1e-5, atol=1e-8):
    """Compare two numpy arrays. Returns dict of metrics."""
    a = np.asarray(a, dtype=np.float32).ravel()
    b = np.asarray(b, dtype=np.float32).ravel()
    assert a.shape == b.shape, f"Shape mismatch: {a.shape} vs {b.shape}"
    diff = np.abs(a - b)
    finite = np.isfinite(a) & np.isfinite(b) & np.isfinite(diff)
    if not finite.all():
        n_bad = (~finite).sum()
        print(f"  WARNING: {n_bad} non-finite values in {label}")
        a = a[finite]; b = b[finite]; diff = diff[finite]
    if len(a) == 0:
        return {"label": label, "max_err": float('nan'), "mean_err": float('nan'),
                "n": 0, "a_range": [float('nan')]*2, "b_range": [float('nan')]*2}
    return {
        "label": label,
        "max_err": float(diff.max()),
        "mean_err": float(diff.mean()),
        "n": len(a),
        "a_range": [float(a.min()), float(a.max())],
        "b_range": [float(b.min()), float(b.max())],
    }

# ── Test input generation ──
def rand_input(*shape, seed=12345, device="cpu", dtype=torch.float32):
    """Generate random input with fixed seed."""
    gen = torch.Generator(device=device).manual_seed(seed)
    return torch.randn(*shape, generator=gen, device=device, dtype=dtype)

def rand_bf16_weight(*shape, seed=42, device="cpu"):
    """Generate random BF16-like weight tensor (values in range [-1,1])."""
    gen = torch.Generator(device=device).manual_seed(seed)
    return torch.randn(*shape, generator=gen, device=device, dtype=torch.float32) * 0.1

# ── DiT op implementations (PyTorch reference, matching C++ engine semantics) ──

def pt_t_embed_sin_cos(sigma, M_val=M, D_val=D):
    """Replicate C++ dit_engine_v2.cpp sin/cos embedding.
    PyTorch Timesteps.forward: torch.cat([sin, cos], dim=-1)
    C++ head_tail_ops.h (BUG: swapped): [cos, sin]
    This function returns the PYTHON/CORRECT version: [sin, cos].
    """
    half = D_val // 2
    emb = torch.zeros(M_val, D_val)
    for m in range(M_val):
        s = sigma[m] if isinstance(sigma, (list, tuple)) else sigma
        s = float(s)
        for i in range(half):
            exponent = -np.log(10000.0) * i / half
            freq = np.exp(exponent)
            val = s * freq
            emb[m, i] = np.cos(val)          # cos second half — BUG in C++
            emb[m, half + i] = np.sin(val)   # sin first half — BUG in C++
    return emb

def pt_timesteps(sigma, M_val=M, D_val=D):
    """PyTorch-correct Timesteps: [sin | cos] (not [cos | sin])."""
    if isinstance(sigma, (int, float)):
        sigma_t = torch.tensor([sigma, sigma], dtype=torch.float32).unsqueeze(1)  # [M, 1]
    else:
        sigma_t = torch.as_tensor(sigma, dtype=torch.float32)
        if sigma_t.dim() == 0:
            sigma_t = sigma_t.unsqueeze(0)
        if sigma_t.dim() == 1:
            sigma_t = sigma_t.unsqueeze(1)  # [M, 1]
    ts = sigma_t
    half = D_val // 2
    device = ts.device
    exponent = -np.log(10000.0) * torch.arange(half, dtype=torch.float32, device=device) / half
    emb = ts.float() * torch.exp(exponent)  # [M, half]
    sin_emb = torch.sin(emb)
    cos_emb = torch.cos(emb)
    return torch.cat([sin_emb, cos_emb], dim=-1)  # [M, D] — sin first!

def pt_layernorm(x, rows, elems, eps=1e-6):
    """LayerNorm without affine parameters (matching C++ layernorm)."""
    return F.layer_norm(x.reshape(rows, elems), (elems,), None, None, eps).reshape(-1)

def pt_rmsnorm(x, rows, elems, weight_bf16=None, eps=1e-6):
    """RMSNorm with optional weight (matching C++ rms_norm_fp32)."""
    y = x.reshape(rows, elems)
    rms = torch.rsqrt((y * y).mean(-1, keepdim=True) + eps)
    y = y * rms
    if weight_bf16 is not None:
        w = weight_bf16.to(torch.float32).reshape(elems)
        y = y * w
    return y.reshape(-1)

def pt_gelu(x):
    """GELU using PyTorch's implementation."""
    return F.gelu(x.reshape(-1)).reshape(x.shape)

def pt_silu(x):
    """SiLU using PyTorch's implementation."""
    return F.silu(x.reshape(-1)).reshape(x.shape)

def pt_softmax_lastdim(x, M_q, M_kv, H):
    """Safe softmax over last dim, matching C++ 3-pass algorithm."""
    y = x.reshape(-1, M_kv)
    return F.softmax(y, dim=-1).reshape(M_q, H, M_kv)

def pt_rope_rotate(q_or_k, freqs, N, head_dim):
    """Apply RoPE rotation matching rope_fp32.comp.
    q_or_k: [N, head_dim]
    freqs: [N, half_dim, 4] = [cos, -sin, sin, cos] per pair
    """
    half = head_dim // 2
    out = torch.zeros_like(q_or_k)
    for i in range(half):
        a = q_or_k[:, 2*i]
        b = q_or_k[:, 2*i+1]
        c  = freqs[:, i, 0]   # cos
        ms = freqs[:, i, 1]   # -sin
        s  = freqs[:, i, 2]   # sin
        mc = freqs[:, i, 3]   # cos
        out[:, 2*i]   = c * a + ms * b
        out[:, 2*i+1] = s * a + mc * b
    return out

def pt_adaln_chain(t_emb, sd, block_idx, module_name, device="cpu", dtype=torch.float32):
    """Replicate C++ adaln_gpu: SiLU → Linear(256) → Linear(3D) + lora → chunk.
    Returns: (shift, scale, gate) each [M, D]
    """
    prefix = f"blocks.{block_idx}.adaln_modulation_{module_name}"
    w0 = get_weight(sd, f"{prefix}.1.weight", device, dtype)  # [256, D]
    w2 = get_weight(sd, f"{prefix}.2.weight", device, dtype)  # [3*D, 256]

    h = pt_silu(t_emb)  # SiLU
    h = F.linear(h, w0)  # [M, 256]
    out = F.linear(h, w2)  # [M, 3*D]
    shift, scale, gate = out.chunk(3, dim=-1)
    scale = scale + 1.0
    return shift, scale, gate  # each [M, D]

# ── Serialization ──
def save_npy(path, arr):
    """Save as .npy file."""
    np.save(path, np.asarray(arr, dtype=np.float32))

def load_npy(path):
    """Load .npy file as float32 numpy array."""
    return np.load(path).astype(np.float32)

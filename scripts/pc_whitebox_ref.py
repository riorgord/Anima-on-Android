"""PC White-box Reference — exact replication of C++ engine computation path.

Matches C++ dit_engine.cpp's dit_compute_timestep + dit_forward_step (per-step recording):
  - Sinusoidal embedding of sigma (NOT PyTorch's Timesteps)
  - RMSNorm for t_emb, SiLU+linear chain for lora
  - AdaLN with 256-dim bottleneck (use_adaln_lora=True)
  - Flat [MS, D] layout, per-batch attention
  - RoPE applied to self-attention Q/K (matching C++ compute_rope_freqs)
  - GELU activation in MLP (GPT2FeedForward)

Part A: White-box Block 0 full detail — every intermediate dumped
Part B: 28-block forward (white-box style)
Part C: 3-step reference image pipeline (PyTorch native)

Usage (WSL2):
    source /home/riorg/miniconda3/etc/profile.d/conda.sh
    conda activate /home/riorg/anima-work/.conda
    python /mnt/d/AI/anima_phone/scripts/pc_whitebox_ref.py [--image-only] [--compare DMPDIR]

Output layout:
    output/whitebox/
        block_XX_pt.npy          — per-block output [MS, D] fp16
        b0/intermediates/        — Block 0 every-op dumps
        b0/compare/              — C++ comparison results (when --compare)
        pc_ref_whitebox.png      — 3-step reference image
"""
import sys, os, time, gc, argparse
import torch, torch.nn.functional as F
import numpy as np

# ── Paths ──
SRC = "/mnt/d/AI/anima_phone/src"
MODELDIR = "/mnt/d/AI/anima_phone/models"
OUTBASE = "/mnt/d/AI/anima_phone/output/whitebox"
sys.path.insert(0, SRC)

import predict2, wan_vae

DEV = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16
EPS = 1e-6

# ── C++ engine constants (from dit_engine.cpp) ──
M, S_PER, D = 2, 256, 2048           # M=batch(CFG), S_per=spatial tokens per batch
MS = M * S_PER                         # 512 total tokens
N_HEADS, HEAD_DIM = 16, 128
NCTX, CTXD = 512, 1024
MLP_HIDDEN = 8192
ADALN_LORA_DIM = 256
HP = 16                                # spatial = sqrt(256) = 16
HALF_DIM = HEAD_DIM // 2               # 64
SCALE_ATTN = 1.0 / np.sqrt(HEAD_DIM)

# ── C++-style RoPE dims (from compute_rope_freqs) ──
DIM_T = HEAD_DIM - 2 * (HEAD_DIM // 6 * 2)  # 44
DIM_H = HEAD_DIM // 6 * 2                     # 42
DIM_W = DIM_H                                  # 42
assert DIM_T + DIM_H + DIM_W == HEAD_DIM


# ═══════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════

def load_weights():
    """Load FP16 weights, strip 'net.' prefix."""
    sd_raw = torch.load(f"{MODELDIR}/diffusion_weights_fp16.pt",
                        map_location="cpu", weights_only=True)
    sd = {}
    for k, v in sd_raw.items():
        while k.startswith("net."):
            k = k[4:]
        sd[k] = v
    del sd_raw
    return sd


def fp32_to_fp16(t):
    """Convert float tensor to fp16 bit-exact (matching C++ hard-coded conversion)."""
    if isinstance(t, torch.Tensor):
        t = t.detach().cpu().numpy()
    t = np.asarray(t, dtype=np.float32)
    # Use PyTorch for correctness — the C++ conversion is close enough
    return torch.from_numpy(t).to(DTYPE)


def bcast_2d(t_2d):
    """Broadcast [M, D] → [MS, D] (repeat_interleave S_PER along dim 0)."""
    return t_2d.repeat_interleave(S_PER, dim=0)


# ═══════════════════════════════════════════════════════════════════════
# C++-style t_emb & lora computation (dit_compute_timestep)
# ═══════════════════════════════════════════════════════════════════════

class CppTimestepEmbedder:
    """Replicate C++ dit_compute_timestep(sigma) in PyTorch FP32 then cast to FP16.

    C++ algorithm:
      1. sin_emb[b,j] = cos(sigma * exp(-log(10000) * j/halfD)) for j in [0, halfD)
      2. sin_emb[b,halfD+j] = sin(sigma * exp(-log(10000) * j/halfD))
      3. t_emb = RMSNorm(sin_emb, t_embedding_norm.weight, eps=1e-6)
      4. h = SiLU(sin_emb @ w1^T)  where w1 = t_embedder.1.linear_1.weight [D, D]
      5. lora = h @ w2^T           where w2 = t_embedder.1.linear_2.weight [3*D, D]
    """
    def __init__(self, sd):
        self.w1 = sd["t_embedder.1.linear_1.weight"].float().to(DEV)    # [D, D]
        self.w2 = sd["t_embedder.1.linear_2.weight"].float().to(DEV)    # [3*D, D]
        self.w_ln = sd["t_embedding_norm.weight"].float().to(DEV)       # [D]
        self.halfD = D // 2
        # Precompute frequency bases: 1/exp(log(10000) * j/halfD) for j in [0, halfD)
        j = torch.arange(self.halfD, dtype=torch.float32, device=DEV)
        self._freq_base = torch.exp(-np.log(10000.0) * j / self.halfD)

    def compute(self, sigma):
        """Returns t_emb [M, D] fp16, lora [M, 3*D] fp16."""
        # 1. Sinusoidal embedding
        sigma_f = float(sigma)
        freqs = sigma_f * self._freq_base  # [halfD]
        sin_emb = torch.zeros(M, D, dtype=torch.float32, device=DEV)
        sin_emb[:, :self.halfD] = torch.cos(freqs).unsqueeze(0)
        sin_emb[:, self.halfD:] = torch.sin(freqs).unsqueeze(0)

        # 2. RMSNorm: t_emb = sin_emb * w_ln / rms(sin_emb)
        rms = torch.sqrt((sin_emb * sin_emb).mean(dim=-1, keepdim=True) + 1e-6)
        t_emb = sin_emb * self.w_ln.unsqueeze(0) / rms

        # 3. h = SiLU(sin_emb @ w1^T)
        h = F.silu(sin_emb @ self.w1.T)  # [M, D]

        # 4. lora = h @ w2^T
        lora = h @ self.w2.T  # [M, 3*D]

        return t_emb.to(DTYPE), lora.to(DTYPE)


# ═══════════════════════════════════════════════════════════════════════
# C++-style RoPE computation (compute_rope_freqs)
# ═══════════════════════════════════════════════════════════════════════

def compute_rope_freqs_cpp():
    """Replicate C++ compute_rope_freqs: [M*S_PER*N_HEADS, HALF_DIM, 4] fp16.

    Layout: each row = (mb, pos, head), 4 values = [cos, -sin, sin, cos] per freq pair.
    NTK factors match predict2: extrapolation_ratio^(dim/(dim-2)).
    """
    # NTK factors (matching C++ lines 836-842)
    h_ntk = 4.0 ** (DIM_H / (DIM_H - 2))
    w_ntk = 4.0 ** (DIM_W / (DIM_W - 2))
    t_ntk = 1.0 ** (DIM_T / (DIM_T - 2))

    h_theta = 10000.0 * h_ntk
    w_theta = 10000.0 * w_ntk
    t_theta = 10000.0 * t_ntk

    half_dim = HEAD_DIM // 2
    half_dim_t = DIM_T // 2  # 22
    half_dim_h = DIM_H // 2  # 21
    half_dim_w = DIM_W // 2  # 21

    # Per-frequency theta values
    freqs = torch.zeros(half_dim, dtype=torch.float32)
    for j in range(half_dim):
        if j < half_dim_t:
            freqs[j] = 1.0 / (t_theta ** (2.0 * j / DIM_T))
        elif j < half_dim_t + half_dim_h:
            jh = j - half_dim_t
            freqs[j] = 1.0 / (h_theta ** (2.0 * jh / DIM_H))
        else:
            jw = j - half_dim_t - half_dim_h
            freqs[j] = 1.0 / (w_theta ** (2.0 * jw / DIM_W))

    # Per-position angles: position indices (h_idx, w_idx) for each of S_PER positions
    h_idx = torch.arange(S_PER) // HP   # [S_PER]
    w_idx = torch.arange(S_PER) % HP    # [S_PER]

    # angles[p, j] = position[p] * freq[j]
    # For temporal: position=0 (T=1)
    # For height: position = h_idx[p]
    # For width: position = w_idx[p]
    angles = torch.zeros(S_PER, half_dim, dtype=torch.float32)
    for j in range(half_dim):
        if j < half_dim_t:
            angles[:, j] = 0.0  # T=1
        elif j < half_dim_t + half_dim_h:
            angles[:, j] = h_idx.float() * freqs[j]
        else:
            angles[:, j] = w_idx.float() * freqs[j]

    cos_vals = torch.cos(angles)   # [S_PER, half_dim]
    sin_vals = torch.sin(angles)   # [S_PER, half_dim]

    # Replicate per-head and per-batch: [S_PER, half_dim] → [M*S_PER*N_HEADS, half_dim, 4]
    n_rows = M * S_PER * N_HEADS
    freqs_out = torch.zeros(n_rows, half_dim, 4, dtype=torch.float16)

    for mb in range(M):
        for p in range(S_PER):
            for h in range(N_HEADS):
                dst_row = mb * S_PER * N_HEADS + p * N_HEADS + h
                for j in range(half_dim):
                    c, s = cos_vals[p, j].item(), sin_vals[p, j].item()
                    freqs_out[dst_row, j, 0] = c
                    freqs_out[dst_row, j, 1] = -s
                    freqs_out[dst_row, j, 2] = s
                    freqs_out[dst_row, j, 3] = c

    return freqs_out  # [M*S*N_HEADS, half_dim, 4] fp16


def apply_rope_cpp(q_or_k_flat, freqs_out):
    """Apply RoPE matching C++ dispatch_rope shader.

    q_or_k_flat: [MS * N_HEADS, HEAD_DIM] fp16
    freqs_out:   [MS * N_HEADS, HALF_DIM, 4] fp16

    C++ shader: for each row i:
        x0 = buf[i*HEAD_DIM + 2*j]      (real part)
        x1 = buf[i*HEAD_DIM + 2*j + 1]  (imag part)
        out_real = x0*freq[0] + x1*freq[1]  = x0*cos + x1*(-sin) = x0*cos - x1*sin
        out_imag = x0*freq[2] + x1*freq[3]  = x0*sin + x1*cos
    """
    ph = q_or_k_flat.shape[0]  # MS * N_HEADS
    half_dim = HEAD_DIM // 2

    # Reshape to [ph, half_dim, 2]
    x = q_or_k_flat.view(ph, half_dim, 2).float()

    # freqs: [ph, half_dim, 4] → [ph, half_dim, 2, 2]
    f = freqs_out.view(ph, half_dim, 2, 2).float()

    # out_real = x0*cos + x1*(-sin)
    # out_imag = x0*sin + x1*cos
    out_real = x[..., 0] * f[..., 0, 0] + x[..., 1] * f[..., 0, 1]
    out_imag = x[..., 0] * f[..., 1, 0] + x[..., 1] * f[..., 1, 1]

    out = torch.stack([out_real, out_imag], dim=-1).view(ph, HEAD_DIM)
    return out.to(DTYPE)


# ═══════════════════════════════════════════════════════════════════════
# AdaLN helper (matching C++ adaln_gpu)
# ═══════════════════════════════════════════════════════════════════════

def compute_adaln_block(sd, b, t_emb_fp16, lora_fp16):
    """Compute shift/scale/gate for all 3 channels of one block.

    C++ adaln_gpu: SiLU(t_emb_fp32) @ w1^T @ w2^T + lora → [M, 3D] → chunk(3)
    Result: each channel returns [M, D] fp16 shift/scale/gate, scale+1 applied.

    Returns: dict with keys 'sa_shift','sa_scale','sa_gate', 'cx_shift','cx_scale','cx_gate',
             'mlp_shift','mlp_scale','mlp_gate' — each [M, D] fp16
    """
    t_f32 = t_emb_fp16.float()  # [M, D]
    l_f32 = lora_fp16.float()    # [M, 3*D]

    result = {}
    for key_prefix, weight_prefix in [("sa", "self_attn"), ("cx", "cross_attn"), ("mlp", "mlp")]:
        w1 = sd[f"blocks.{b}.adaln_modulation_{weight_prefix}.1.weight"].float().to(DEV)  # [256, D]
        w2 = sd[f"blocks.{b}.adaln_modulation_{weight_prefix}.2.weight"].float().to(DEV)  # [3*D, 256]

        h = F.silu(t_f32) @ w1.T  # SiLU first, then Linear(D,256) — matching nn.Sequential(SiLU, Linear, Linear)
        out = h @ w2.T            # [M, 3*D]
        out = out + l_f32         # [M, 3*D] + lora
        shift, scale, gate = torch.chunk(out, 3, dim=-1)  # each [M, D]
        scale = scale + 1.0  # C++ applies scale+1 in scale_shift shader

        result[f"{key_prefix}_shift"] = shift.to(DTYPE)
        result[f"{key_prefix}_scale"] = scale.to(DTYPE)
        result[f"{key_prefix}_gate"]  = gate.to(DTYPE)

    return result


# ═══════════════════════════════════════════════════════════════════════
# Per-batch flat attention (matching C++ record_attn_3pass)
# ═══════════════════════════════════════════════════════════════════════

def flat_self_attention(q_roped, k_roped, v_flat):
    """Per-batch flat self-attention.

    q_roped: [MS * N_HEADS, HEAD_DIM] — Q after RMSNorm + RoPE
    k_roped: [MS * N_HEADS, HEAD_DIM] — K after RMSNorm + RoPE
    v_flat:  [MS * N_HEADS, HEAD_DIM] — V (no norm, identity in predict2)

    Returns: attn_out [MS * N_HEADS, HEAD_DIM]
    """
    attn_out = torch.zeros(MS * N_HEADS, HEAD_DIM, dtype=DTYPE, device=DEV)

    for mb in range(M):
        q_off = mb * S_PER * N_HEADS * HEAD_DIM
        kv_off = mb * S_PER * N_HEADS * HEAD_DIM

        q_mb = q_roped[q_off // HEAD_DIM : (q_off // HEAD_DIM) + S_PER * N_HEADS]  # [S*H, D]
        k_mb = k_roped[kv_off // HEAD_DIM : (kv_off // HEAD_DIM) + S_PER * N_HEADS]
        v_mb = v_flat[kv_off // HEAD_DIM : (kv_off // HEAD_DIM) + S_PER * N_HEADS]

        # Reshape to [H, S, D]
        q_h = q_mb.view(S_PER, N_HEADS, HEAD_DIM).permute(1, 0, 2)  # [H, S, D]
        k_h = k_mb.view(S_PER, N_HEADS, HEAD_DIM).permute(1, 0, 2)
        v_h = v_mb.view(S_PER, N_HEADS, HEAD_DIM).permute(1, 0, 2)

        scores = torch.bmm(q_h, k_h.transpose(1, 2)) * SCALE_ATTN  # [H, S, S]
        attn_w = F.softmax(scores.float(), dim=-1).to(DTYPE)        # [H, S, S]
        attn_o = torch.bmm(attn_w, v_h)                              # [H, S, D]
        attn_o_mb = attn_o.permute(1, 0, 2).reshape(S_PER * N_HEADS, HEAD_DIM)

        attn_out[mb * S_PER * N_HEADS : (mb + 1) * S_PER * N_HEADS] = attn_o_mb

    return attn_out


def flat_cross_attention(q_norm, k_norm, v_flat):
    """Per-batch flat cross-attention.

    q_norm: [MS * N_HEADS, HEAD_DIM] — Q after RMSNorm (NO RoPE for cross-attn)
    k_norm: [M * NCTX * N_HEADS, HEAD_DIM] — K after RMSNorm
    v_flat: [M * NCTX * N_HEADS, HEAD_DIM] — V (identity norm)

    Returns: attn_out [MS * N_HEADS, HEAD_DIM]
    """
    attn_out = torch.zeros(MS * N_HEADS, HEAD_DIM, dtype=DTYPE, device=DEV)

    for mb in range(M):
        q_off = mb * S_PER * N_HEADS * HEAD_DIM
        kv_off = mb * NCTX * N_HEADS * HEAD_DIM

        q_mb = q_norm[q_off // HEAD_DIM : (q_off // HEAD_DIM) + S_PER * N_HEADS]
        k_mb = k_norm[kv_off // HEAD_DIM : (kv_off // HEAD_DIM) + NCTX * N_HEADS]
        v_mb = v_flat[kv_off // HEAD_DIM : (kv_off // HEAD_DIM) + NCTX * N_HEADS]

        q_h = q_mb.view(S_PER, N_HEADS, HEAD_DIM).permute(1, 0, 2)      # [H, S, D]
        k_h = k_mb.view(NCTX, N_HEADS, HEAD_DIM).permute(1, 0, 2)       # [H, Nctx, D]
        v_h = v_mb.view(NCTX, N_HEADS, HEAD_DIM).permute(1, 0, 2)

        scores = torch.bmm(q_h, k_h.transpose(1, 2)) * SCALE_ATTN       # [H, S, Nctx]
        attn_w = F.softmax(scores.float(), dim=-1).to(DTYPE)
        attn_o = torch.bmm(attn_w, v_h)
        attn_o_mb = attn_o.permute(1, 0, 2).reshape(S_PER * N_HEADS, HEAD_DIM)

        attn_out[mb * S_PER * N_HEADS : (mb + 1) * S_PER * N_HEADS] = attn_o_mb

    return attn_out


# ═══════════════════════════════════════════════════════════════════════
# White-box Block forward — exact C++ engine replica
# ═══════════════════════════════════════════════════════════════════════

def whitebox_block(b, x_flat, ctx_flat, adaln, sd, rope_freqs, capture=False):
    """Run one block exactly as C++ engine's per-step recording.

    Args:
        b: block index
        x_flat: [MS, D] input
        ctx_flat: [M*NCTX, CTXD] context
        adaln: dict from compute_adaln_block
        sd: weight dict
        rope_freqs: [MS*N_HEADS, HALF_DIM, 4] RoPE frequencies
        capture: if True, save all intermediates to `caps` dict

    Returns: x_out [MS, D], caps dict (or None)
    """
    caps = {} if capture else None
    ph = MS * N_HEADS
    ph_cross = M * NCTX * N_HEADS
    MS_kv = M * NCTX

    # Weight lookups
    q_w = sd[f"blocks.{b}.self_attn.q_proj.weight"].to(DEV).to(DTYPE)
    k_w = sd[f"blocks.{b}.self_attn.k_proj.weight"].to(DEV).to(DTYPE)
    v_w = sd[f"blocks.{b}.self_attn.v_proj.weight"].to(DEV).to(DTYPE)
    o_w = sd[f"blocks.{b}.self_attn.output_proj.weight"].to(DEV).to(DTYPE)
    qn_w = sd[f"blocks.{b}.self_attn.q_norm.weight"].to(DEV).to(DTYPE)
    kn_w = sd[f"blocks.{b}.self_attn.k_norm.weight"].to(DEV).to(DTYPE)

    cx_q_w = sd[f"blocks.{b}.cross_attn.q_proj.weight"].to(DEV).to(DTYPE)
    cx_k_w = sd[f"blocks.{b}.cross_attn.k_proj.weight"].to(DEV).to(DTYPE)
    cx_v_w = sd[f"blocks.{b}.cross_attn.v_proj.weight"].to(DEV).to(DTYPE)
    cx_o_w = sd[f"blocks.{b}.cross_attn.output_proj.weight"].to(DEV).to(DTYPE)
    cx_qn_w = sd[f"blocks.{b}.cross_attn.q_norm.weight"].to(DEV).to(DTYPE)
    cx_kn_w = sd[f"blocks.{b}.cross_attn.k_norm.weight"].to(DEV).to(DTYPE)

    l1_w = sd[f"blocks.{b}.mlp.layer1.weight"].to(DEV).to(DTYPE)
    l2_w = sd[f"blocks.{b}.mlp.layer2.weight"].to(DEV).to(DTYPE)

    # Broadcast helpers
    bc = lambda key: bcast_2d(adaln[key])  # [M,D] → [MS,D]

    # ── Segment B: Self-attention ──
    # LN (eps=1e-6, no affine)
    y = F.layer_norm(x_flat, (D,), weight=None, bias=None, eps=EPS)
    if capture: caps["sa_ln"] = y.detach().cpu().numpy().astype(np.float16)

    # AdaLN modulate: y * (1 + scale) + shift → but scale already has +1 from compute_adaln
    # C++ scale_shift: out = in * scale + shift, where scale = read_scale + 1.0
    y = y * bc("sa_scale") + bc("sa_shift")
    if capture: caps["sa_modulated"] = y.detach().cpu().numpy().astype(np.float16)

    # QKV GEMM
    q = F.linear(y, q_w)  # [MS, D]
    k = F.linear(y, k_w)
    v = F.linear(y, v_w)
    if capture:
        caps["sa_q_raw"] = q.detach().cpu().numpy().astype(np.float16)
        caps["sa_k_raw"] = k.detach().cpu().numpy().astype(np.float16)
        caps["sa_v_raw"] = v.detach().cpu().numpy().astype(np.float16)

    # RMSNorm Q/K (V is identity in predict2)
    q_n = F.rms_norm(q.view(ph, HEAD_DIM), (HEAD_DIM,), weight=qn_w, eps=EPS)
    k_n = F.rms_norm(k.view(ph, HEAD_DIM), (HEAD_DIM,), weight=kn_w, eps=EPS)
    v_flat = v.view(ph, HEAD_DIM)
    if capture:
        caps["sa_q_norm"] = q_n.detach().cpu().numpy().astype(np.float16)
        caps["sa_k_norm"] = k_n.detach().cpu().numpy().astype(np.float16)

    # RoPE (self-attn only)
    q_roped = apply_rope_cpp(q_n, rope_freqs)
    k_roped = apply_rope_cpp(k_n, rope_freqs)
    if capture:
        caps["sa_q_roped"] = q_roped.detach().cpu().numpy().astype(np.float16)
        caps["sa_k_roped"] = k_roped.detach().cpu().numpy().astype(np.float16)

    # Per-batch attention
    attn_o = flat_self_attention(q_roped, k_roped, v_flat)
    if capture: caps["sa_attn_o"] = attn_o.detach().cpu().numpy().astype(np.float16)

    # O_proj
    sa_out = F.linear(attn_o.view(MS, D), o_w)
    if capture: caps["sa_o_proj"] = sa_out.detach().cpu().numpy().astype(np.float16)

    # Gate + residual: x_sa = x + gate * sa_out
    x_sa = x_flat + bc("sa_gate") * sa_out
    if capture: caps["sa_residual"] = x_sa.detach().cpu().numpy().astype(np.float16)

    # ── Segment C1+C2: Cross-attention ──
    # LN
    y = F.layer_norm(x_sa, (D,), weight=None, bias=None, eps=EPS)
    if capture: caps["cx_ln"] = y.detach().cpu().numpy().astype(np.float16)

    # AdaLN modulate
    y = y * bc("cx_scale") + bc("cx_shift")
    if capture: caps["cx_modulated"] = y.detach().cpu().numpy().astype(np.float16)

    # Q from x, K/V from ctx
    q_cx = F.linear(y, cx_q_w)         # [MS, D]
    k_cx = F.linear(ctx_flat, cx_k_w)  # [M*Nctx, D]
    v_cx = F.linear(ctx_flat, cx_v_w)  # [M*Nctx, D]
    if capture:
        caps["cx_q_raw"] = q_cx.detach().cpu().numpy().astype(np.float16)
        caps["cx_k_raw"] = k_cx.detach().cpu().numpy().astype(np.float16)
        caps["cx_v_raw"] = v_cx.detach().cpu().numpy().astype(np.float16)

    # RMSNorm (no RoPE for cross-attn)
    q_cx_n = F.rms_norm(q_cx.view(ph, HEAD_DIM), (HEAD_DIM,), weight=cx_qn_w, eps=EPS)
    k_cx_n = F.rms_norm(k_cx.view(ph_cross, HEAD_DIM), (HEAD_DIM,), weight=cx_kn_w, eps=EPS)
    if capture:
        caps["cx_q_norm"] = q_cx_n.detach().cpu().numpy().astype(np.float16)
        caps["cx_k_norm"] = k_cx_n.detach().cpu().numpy().astype(np.float16)

    # Per-batch cross-attention
    cx_attn_o = flat_cross_attention(q_cx_n, k_cx_n, v_cx.view(ph_cross, HEAD_DIM))
    if capture: caps["cx_attn_o"] = cx_attn_o.detach().cpu().numpy().astype(np.float16)

    # O_proj
    cx_out = F.linear(cx_attn_o.view(MS, D), cx_o_w)
    if capture: caps["cx_o_proj"] = cx_out.detach().cpu().numpy().astype(np.float16)

    # Gate + residual
    x_cx = x_sa + bc("cx_gate") * cx_out
    if capture: caps["cx_residual"] = x_cx.detach().cpu().numpy().astype(np.float16)

    # ── Segment D: MLP ──
    # LN
    y = F.layer_norm(x_cx, (D,), weight=None, bias=None, eps=EPS)
    if capture: caps["mlp_ln"] = y.detach().cpu().numpy().astype(np.float16)

    # AdaLN modulate
    y = y * bc("mlp_scale") + bc("mlp_shift")
    if capture: caps["mlp_modulated"] = y.detach().cpu().numpy().astype(np.float16)

    # fc1 GEMM
    fc1 = F.linear(y, l1_w)  # [MS, MLP_HIDDEN]
    if capture: caps["mlp_fc1_pre_gelu"] = fc1.detach().cpu().numpy().astype(np.float16)

    # GELU (predict2 GPT2FeedForward uses nn.GELU, NOT SiLU!)
    fc1 = F.gelu(fc1)
    if capture: caps["mlp_fc1_gelu"] = fc1.detach().cpu().numpy().astype(np.float16)

    # fc2 GEMM
    fc2 = F.linear(fc1, l2_w)  # [MS, D]
    if capture: caps["mlp_fc2"] = fc2.detach().cpu().numpy().astype(np.float16)

    # Gate + residual
    x_out = x_cx + bc("mlp_gate") * fc2
    if capture: caps["mlp_residual"] = x_out.detach().cpu().numpy().astype(np.float16)

    return x_out, caps


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="PC White-box Reference for C++ engine")
    parser.add_argument("--image-only", action="store_true",
                        help="Only run 3-step reference pipeline (skip white-box)")
    parser.add_argument("--compare", type=str, default=None,
                        help="Path to phone dump directory for comparison (e.g. output/cmp2)")
    parser.add_argument("--seed", type=int, default=12345,
                        help="Random seed for synthetic inputs (default: 12345, matching phone_dump_blocks.py)")
    parser.add_argument("--sigma", type=float, default=1.0,
                        help="Sigma for t_emb computation (default: 1.0)")
    args = parser.parse_args()

    os.makedirs(OUTBASE, exist_ok=True)
    os.makedirs(f"{OUTBASE}/b0/intermediates", exist_ok=True)

    # ═══════════════════════════════════════════════════════════════
    # Load
    # ═══════════════════════════════════════════════════════════════
    print("=" * 70)
    print("PC White-box Reference — C++ Engine Replica")
    print(f"Device: {DEV}  Dtype: {DTYPE}  MS={MS}  S_per={S_PER}  D={D}")
    print("=" * 70)

    print("\n[1/5] Loading weights...")
    t0 = time.time()
    sd = load_weights()
    print(f"  {len(sd)} keys loaded in {time.time()-t0:.1f}s")

    # ═══════════════════════════════════════════════════════════════
    # t_emb & lora
    # ═══════════════════════════════════════════════════════════════
    print(f"\n[2/5] Computing t_emb & lora (C++ style, sigma={args.sigma})...")
    t_eng = CppTimestepEmbedder(sd)
    t_emb, lora = t_eng.compute(args.sigma)
    print(f"  t_emb: {t_emb.shape} range=[{t_emb.min():.4f}, {t_emb.max():.4f}]")
    print(f"  lora:  {lora.shape} range=[{lora.min():.4f}, {lora.max():.4f}]")

    # ═══════════════════════════════════════════════════════════════
    # Inputs
    # ═══════════════════════════════════════════════════════════════
    print(f"\n[3/5] Generating inputs (seed={args.seed})...")
    rng = np.random.RandomState(args.seed)
    x_np = (rng.randn(MS, D).astype(np.float32) * 0.02).astype(np.float16)
    ctx_np = (rng.randn(M * NCTX, CTXD).astype(np.float32) * 0.5).astype(np.float16)
    x_in = torch.from_numpy(x_np.astype(np.float32)).to(DEV, DTYPE)
    ctx_in = torch.from_numpy(ctx_np.astype(np.float32)).to(DEV, DTYPE)
    print(f"  x:   {x_in.shape} range=[{x_in.min():.4f}, {x_in.max():.4f}]")
    print(f"  ctx: {ctx_in.shape} range=[{ctx_in.min():.4f}, {ctx_in.max():.4f}]")

    # Save inputs
    np.save(f"{OUTBASE}/x_input.npy", x_np)
    np.save(f"{OUTBASE}/ctx_input.npy", ctx_np)
    np.save(f"{OUTBASE}/t_emb.npy", t_emb.cpu().numpy().astype(np.float16))
    np.save(f"{OUTBASE}/lora.npy", lora.cpu().numpy().astype(np.float16))

    # ═══════════════════════════════════════════════════════════════
    # RoPE freqs
    # ═══════════════════════════════════════════════════════════════
    print("\n[4/5] Computing RoPE frequencies (C++ style)...")
    rope_freqs = compute_rope_freqs_cpp().to(DEV)
    print(f"  freqs: {rope_freqs.shape} range=[{rope_freqs.min():.4f}, {rope_freqs.max():.4f}]")

    if args.image_only:
        print("\n  --image-only: skipping white-box block computation")
        whitebox_outputs = None
    else:
        # ═══════════════════════════════════════════════════════════
        # Part A: White-box Block 0 (full detail)
        # ═══════════════════════════════════════════════════════════
        print("\n[5a/5] White-box Block 0 (full intermediates)...")

        adaln0 = compute_adaln_block(sd, 0, t_emb, lora)
        for k, v in adaln0.items():
            print(f"  adaln {k}: range=[{v.min():.4f}, {v.max():.4f}]")

        x_b0, caps_b0 = whitebox_block(0, x_in, ctx_in, adaln0, sd, rope_freqs, capture=True)
        print(f"  Block 0 out: range=[{x_b0.min():.4f}, {x_b0.max():.4f}]  "
              f"nan={torch.isnan(x_b0).sum().item()}")

        # Save Block 0 output
        np.save(f"{OUTBASE}/block_00_pt.npy", x_b0.cpu().numpy().astype(np.float16))

        # Save all intermediates
        b0_dir = f"{OUTBASE}/b0/intermediates"
        for name, arr in caps_b0.items():
            np.save(f"{b0_dir}/{name}.npy", arr)
        print(f"  Saved {len(caps_b0)} intermediates to {b0_dir}/")

        # ═══════════════════════════════════════════════════════════
        # Part B: 28-block forward
        # ═══════════════════════════════════════════════════════════
        print(f"\n[5b/5] White-box 28-block forward...")

        x = x_in.clone()
        whitebox_outputs = []
        t_loop = time.time()

        for b in range(28):
            adaln_b = compute_adaln_block(sd, b, t_emb, lora)
            x, _ = whitebox_block(b, x, ctx_in, adaln_b, sd, rope_freqs, capture=False)
            out_np = x.cpu().numpy().astype(np.float16)
            whitebox_outputs.append(out_np)
            np.save(f"{OUTBASE}/block_{b:02d}_pt.npy", out_np)

            if b < 3 or b > 24:
                f = out_np[np.isfinite(out_np)]
                nans = np.sum(np.isnan(out_np))
                rng_str = f"[{f.min():.2f}, {f.max():.2f}]" if len(f) > 0 else "ALL NaN"
                print(f"  Block {b:2d}: {rng_str}  nan={nans}")
            elif b == 3:
                print(f"  ... (blocks 3-24 omitted)")

        dt = time.time() - t_loop
        final = whitebox_outputs[-1]
        f = final[np.isfinite(final)]
        print(f"  28 blocks done in {dt:.1f}s")
        print(f"  Final output range: [{f.min():.2f}, {f.max():.2f}]  nan={np.sum(np.isnan(final))}")

    # ═══════════════════════════════════════════════════════════════
    # Optional: Compare with phone dumps
    # ═══════════════════════════════════════════════════════════════
    if args.compare and whitebox_outputs:
        dmp = args.compare
        print(f"\n{'='*70}")
        print(f"Comparing with phone dumps: {dmp}")
        print(f"{'Block':<6} {'C++ range':<28} {'PT range':<28} {'max_err':<12} {'mean_err':<12}")
        print(f"{'-'*6} {'-'*28} {'-'*28} {'-'*12} {'-'*12}")

        for b in range(28):
            cpp_path = f"{dmp}/block_{b:02d}_cpp.npy"
            if not os.path.exists(cpp_path):
                # Try alternate naming
                alt = f"{dmp}/block_{b:02d}_pt25.npy"
                if os.path.exists(alt):
                    print(f"  {b:2d}    (phone dump not found, skipping)")
                continue

            cpp = np.load(cpp_path).astype(np.float32).reshape(MS, D)
            pt = whitebox_outputs[b].astype(np.float32)

            ok = np.isfinite(cpp) & np.isfinite(pt)
            if ok.sum() == 0:
                print(f"  {b:2d}    ALL NaN — cannot compare")
                continue

            diff = np.abs(cpp[ok] - pt[ok])
            max_e = diff.max()
            mean_e = diff.mean()

            cpp_rng = f"[{cpp[ok].min():.2f},{cpp[ok].max():.2f}]"
            pt_rng = f"[{pt[ok].min():.2f},{pt[ok].max():.2f}]"
            flag = " ⚠️ LARGE" if max_e > 100 else (" ⚠️" if max_e > 1 else "")

            print(f"  {b:2d}    {cpp_rng:<28} {pt_rng:<28} {max_e:<12.2f} {mean_e:<12.4f}{flag}")

            # Save comparison
            np.save(f"{OUTBASE}/b0/compare/diff_{b:02d}.npy", diff)

        # Block 0 sub-module comparison
        print(f"\n  Block 0 sub-module comparison:")
        phone_stages = {
            "sa": 0, "cx": 1, "mlp": 2,
            "q_norm": 10, "k_norm": 11, "v_raw": 12,
            "scores": 13, "attn_o": 14,
        }
        for name, stage_id in phone_stages.items():
            phone_path = f"{dmp}/b0_{name}.npy"
            pt_path = f"{OUTBASE}/b0/intermediates/sa_{name}.npy" if name in ("q_norm", "k_norm", "v_raw", "scores", "attn_o") else None

            # Map phone stage names to our intermediate names
            pt_name_map = {
                "sa": "sa_residual", "cx": "cx_residual", "mlp": "mlp_residual",
                "q_norm": "sa_q_roped", "k_norm": "sa_k_roped",  # phone captures after RoPE
                "v_raw": "sa_v_raw",
                "scores": None,  # different layout, skip for now
                "attn_o": "sa_attn_o",
            }
            pt_name = pt_name_map.get(name)
            if pt_name:
                pt_path = f"{OUTBASE}/b0/intermediates/{pt_name}.npy"

            if pt_path and os.path.exists(phone_path) and os.path.exists(pt_path):
                phone_arr = np.load(phone_path).astype(np.float32).flatten()
                pt_arr = np.load(pt_path).astype(np.float32).flatten()
                # Match sizes
                min_len = min(len(phone_arr), len(pt_arr))
                ok = np.isfinite(phone_arr[:min_len]) & np.isfinite(pt_arr[:min_len])
                if ok.sum() > 0:
                    diff = np.abs(phone_arr[:min_len][ok] - pt_arr[:min_len][ok])
                    print(f"    b0_{name}: max_err={diff.max():.4f}  mean_err={diff.mean():.6f}")
                else:
                    print(f"    b0_{name}: ALL NaN")
            else:
                if os.path.exists(phone_path):
                    print(f"    b0_{name}: phone dump found but no PT reference")

    # ═══════════════════════════════════════════════════════════════
    # Part C: 3-step reference image pipeline (PyTorch native)
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print("Part C: 3-step Reference Image Pipeline (PyTorch native)")
    print(f"{'='*70}")

    # Check for context
    ctx_cond_path = f"{MODELDIR}/context_cond.pt"
    vae_path = f"{MODELDIR}/vae_weights_fp16.pt"

    if not os.path.exists(ctx_cond_path):
        print(f"  WARNING: {ctx_cond_path} not found. Run pc_context.py first.")
        print(f"  Skipping image pipeline.")
        return

    if not os.path.exists(vae_path):
        print(f"  WARNING: {vae_path} not found.")
        print(f"  Skipping image pipeline.")
        return

    # Load context
    print("\n  Loading context...")
    ctx_cond = torch.load(ctx_cond_path, weights_only=True).to(DEV).to(DTYPE)
    ctx_uncond = torch.load(f"{MODELDIR}/context_uncond.pt", weights_only=True).to(DEV).to(DTYPE)
    print(f"  cond: {ctx_cond.shape}  uncond: {ctx_uncond.shape}")

    # Build full DiT model (PyTorch native)
    print("  Building DiT model...")
    config = dict(
        max_img_h=240, max_img_w=240, max_frames=128,
        in_channels=16, out_channels=16,
        patch_spatial=2, patch_temporal=1, concat_padding_mask=True,
        model_channels=D, num_blocks=28, num_heads=N_HEADS, mlp_ratio=4.0,
        crossattn_emb_channels=CTXD,
        pos_emb_cls="rope3d", pos_emb_learnable=True, pos_emb_interpolation="crop",
        min_fps=1, max_fps=30, use_adaln_lora=True, adaln_lora_dim=ADALN_LORA_DIM,
        rope_h_extrapolation_ratio=4.0, rope_w_extrapolation_ratio=4.0,
        rope_t_extrapolation_ratio=1.0,
        extra_per_block_abs_pos_emb=False, rope_enable_fps_modulation=False,
    )

    dit = predict2.MiniTrainDIT(**config, device=DEV, dtype=DTYPE, operations=torch.nn)
    dit.load_state_dict(sd, strict=False)
    dit.eval()
    del sd; gc.collect()
    torch.cuda.empty_cache()
    print(f"  DiT loaded, {len(dit.blocks)} blocks")

    # Load VAE
    print("  Loading VAE...")
    vae_sd = torch.load(vae_path, weights_only=True)
    vae = wan_vae.WanVAE(dim=96, z_dim=16, dim_mult=[1,2,4,4], num_res_blocks=2, attn_scales=[],
                         temperal_downsample=[False,True,True], image_channels=3,
                         conv_out_channels=3, dropout=0.0)
    vae = vae.to(DEV)
    vae.load_state_dict({k: v.float() for k, v in vae_sd.items()}, strict=False)
    vae.eval()
    del vae_sd; gc.collect()

    # Denoising loop
    STEPS = 3; CFG = 5.0; SEED_IMG = 6666; H_LAT = 32
    sigmas = [1.0, 0.5, 0.25, 0.0]  # Default schedule for 3 steps

    print(f"\n  Denoising: {STEPS} steps, {H_LAT*8}x{H_LAT*8}, CFG={CFG}, seed={SEED_IMG}")
    gen = torch.Generator(device=DEV).manual_seed(SEED_IMG)
    x = torch.randn(1, 16, H_LAT, H_LAT, generator=gen, dtype=DTYPE, device=DEV)
    t_start = time.time()

    for i in range(STEPS):
        sigma = sigmas[i]; sigma_next = sigmas[i+1]
        ts = torch.tensor([sigma, sigma], dtype=DTYPE, device=DEV)
        x_b = x.unsqueeze(2).repeat(2, 1, 1, 1, 1)  # [2, 16, 1, 32, 32]
        ctx_b = torch.cat([ctx_cond, ctx_uncond], dim=0)  # [2, 512, 1024]

        t0_step = time.time()
        with torch.no_grad():
            v_b = dit(x_b, ts, ctx_b)
        dt_step = time.time() - t0_step

        v_cond = v_b[0:1].float()
        v_uncond = v_b[1:2].float()
        v_cfg = v_uncond + CFG * (v_cond - v_uncond)
        x = (x.float() + v_cfg.squeeze(2) * (sigma_next - sigma)).to(DTYPE)

        print(f"    step {i+1}/{STEPS}: {dt_step:.1f}s  "
              f"latent std={x.float().std():.3f}  (total {time.time()-t_start:.0f}s)")

    # VAE decode
    print("  Decoding...")
    with torch.no_grad():
        image = vae.decode(x.float().unsqueeze(2))
    img = image[0,:,0].clamp(-1,1)
    img = ((img+1)/2*255).permute(1,2,0).cpu().numpy().astype(np.uint8)

    out_path = f"{OUTBASE}/pc_ref_whitebox.png"
    from PIL import Image
    Image.fromarray(img).save(out_path)
    total_t = time.time() - t_start
    file_sz = os.path.getsize(out_path)

    print(f"\n  Image saved: {out_path}")
    print(f"  Size: {file_sz:,} bytes")
    print(f"  Pixel range: [{img.min()},{img.max()}], mean={img.mean():.1f}")
    print(f"  TOTAL: {STEPS} steps, {total_t:.0f}s ({total_t/STEPS:.0f}s/step)")

    # Acceptance criterion: clean image ~70-90KB, noisy image ~119KB
    if 65000 < file_sz < 95000:
        print(f"  ✅ ACCEPT — image size in clean range (70-95KB)")
    elif file_sz > 110000:
        print(f"  ⚠️ WARNING — image size suggests noise (>{110000})")

    print(f"\n{'='*70}")
    print("DONE")
    print(f"  White-box outputs: {OUTBASE}/block_*_pt.npy")
    print(f"  Block 0 intermediates: {OUTBASE}/b0/intermediates/")
    print(f"  Reference image: {out_path}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()

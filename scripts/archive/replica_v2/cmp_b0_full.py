"""Block 0 full comparison WITH RoPE — self-attn, cross-attn, MLP."""
import numpy as np, torch, torch.nn.functional as F, safetensors.torch, struct, math, os

CMP = '/mnt/d/AI/anima_phone/output/cmp_v2'
SF = '/mnt/d/AI/手坤的anima/models/diffusion_model/anima-base-v1.0.safetensors'
M, S, D, NH, HD = 2, 256, 2048, 16, 128
MS = M*S; S_per = MS//M; ph = MS*NH

# Load captures
def load_npy(name, shape=None):
    path = f'{CMP}/{name}'
    if not os.path.exists(path):
        print(f"  SKIP: {name} not found")
        return None
    d = np.load(path)
    if shape: d = d.reshape(shape)
    return torch.from_numpy(d)

b0_x   = load_npy('b0_x', (MS, D))
b0_q   = load_npy('b0_q', (ph, HD))
b0_k   = load_npy('b0_k', (ph, HD))
b0_v   = load_npy('b0_v', (ph, HD))
b0_qn  = load_npy('b0_qn', (ph, HD))
b0_kn  = load_npy('b0_kn', (ph, HD))
b0_qr  = load_npy('b0_qr', (ph, HD))
b0_kr  = load_npy('b0_kr', (ph, HD))
b0_attn_o = load_npy('b0_attn_o', (ph, HD))
b0_oproj  = load_npy('b0_oproj', (MS, D))
b0_sa  = load_npy('b0_sa', (MS, D))
b0_cx  = load_npy('b0_cx', (MS, D))
b0_mlp = load_npy('b0_mlp', (MS, D))
b0_temb = load_npy('b0_temb', (M, D))
b0_lora = load_npy('b0_lora', (M, 3*D))
b0_bcbuf = load_npy('b0_bcbuf', (9, MS, D))
b0_nbuf  = load_npy('b0_nbuf', (MS, D))

# Load weights
st = safetensors.torch.load_file(SF, device='cpu')
sd = {k[4:] if k.startswith('net.') else k: v.to(torch.float32) for k, v in st.items()}
del st

# AdaLN with captured lora and t_emb
def pt_adaln(emb, lora, module):
    w0 = sd[f'blocks.0.adaln_modulation_{module}.1.weight']
    w2 = sd[f'blocks.0.adaln_modulation_{module}.2.weight']
    h = F.linear(F.silu(emb), w0); h = F.silu(h); out = F.linear(h, w2) + lora
    s, sc, g = out.chunk(3, dim=-1)
    return s, sc + 1.0, g

# Compute RoPE frequencies (replicate C++ compute_rope_freqs)
def compute_rope_freqs(S_val=S, head_dim=HD, n_heads=NH, M_val=M):
    T_val = 1
    dim_h = head_dim // 6 * 2  # 42
    dim_w = dim_h               # 42
    dim_t = head_dim - 2 * dim_h  # 44
    half_dim = head_dim // 2     # 64

    h_ntk = 4.0 ** (dim_h / (dim_h - 2))
    w_ntk = 4.0 ** (dim_w / (dim_w - 2))
    t_ntk = 1.0 ** (dim_t / (dim_t - 2))

    h_theta = 10000.0 * h_ntk
    w_theta = 10000.0 * w_ntk
    t_theta = 10000.0 * t_ntk

    H_grid = int(np.sqrt(S_val))
    W_grid = H_grid

    pos_freqs = np.zeros((S_val, half_dim, 4), dtype=np.float32)
    for p in range(S_val):
        h_idx = p // W_grid
        w_idx = p % W_grid
        t_idx = 0
        for j in range(half_dim):
            if j < dim_t / 2:
                freq = 1.0 / (t_theta ** (2.0 * j / dim_t))
                angle = t_idx * freq; cos_val, sin_val = np.cos(angle), np.sin(angle)
            elif j < dim_t / 2 + dim_h / 2:
                jh = j - dim_t / 2
                freq = 1.0 / (h_theta ** (2.0 * jh / dim_h))
                angle = h_idx * freq; cos_val, sin_val = np.cos(angle), np.sin(angle)
            elif j < dim_t / 2 + dim_h / 2 + dim_w / 2:
                jw = j - dim_t / 2 - dim_h / 2
                freq = 1.0 / (w_theta ** (2.0 * jw / dim_w))
                angle = w_idx * freq; cos_val, sin_val = np.cos(angle), np.sin(angle)
            else:
                jr = j - dim_t / 2 - dim_h / 2 - dim_w / 2
                freq = 1.0 / (t_theta ** (2.0 * jr / dim_t))
                angle = t_idx * freq; cos_val, sin_val = np.cos(angle), np.sin(angle)
            pos_freqs[p, j, 0] = cos_val
            pos_freqs[p, j, 1] = -sin_val
            pos_freqs[p, j, 2] = sin_val
            pos_freqs[p, j, 3] = cos_val

    n_rows = M_val * S_val * n_heads
    freqs = np.zeros((n_rows, half_dim, 4), dtype=np.float32)
    for mb in range(M_val):
        for p in range(S_val):
            for h in range(n_heads):
                dst_row = mb * S_val * n_heads + p * n_heads + h
                freqs[dst_row] = pos_freqs[p]
    return torch.from_numpy(freqs)

# Apply RoPE (replicate rope shader logic)
def apply_rope(t, rope_freqs):
    """t: [N, head_dim], rope_freqs: [N, half_dim, 4]"""
    N, hd = t.shape
    half = hd // 2
    out = t.clone()
    t_reshaped = t.reshape(N, half, 2)
    for i in range(N):
        for j in range(half):
            x, y = t_reshaped[i, j, 0].item(), t_reshaped[i, j, 1].item()
            c, ns, s, c2 = rope_freqs[i, j]  # cos, -sin, sin, cos
            out[i, j] = x * c + y * ns      # x*cos + y*(-sin)
            out[i, half + j] = x * s + y * c2  # x*sin + y*cos
    return out

print("=== C++ value ranges ===")
for name, t in [("Q_norm", b0_qn), ("K_norm", b0_kn), ("Q_rope", b0_qr), ("K_rope", b0_kr),
                 ("Attn_out", b0_attn_o), ("O_proj", b0_oproj), ("SA", b0_sa), ("CX", b0_cx), ("MLP", b0_mlp)]:
    if t is None: continue
    print(f"  {name:12s}: [{t.min():.4f}, {t.max():.4f}] NaN={torch.isnan(t).any().item()}")

# ── PT Self-attention with RoPE ──
print("\n=== Self-attention with RoPE ===")
print(f"  DEBUG: b0_temb type={type(b0_temb)}, b0_lora type={type(b0_lora)}")
if b0_temb is None: print("  FATAL: b0_temb is None!"); exit(1)
if b0_lora is None: print("  FATAL: b0_lora is None!"); exit(1)

# Compute bcBuf from captured t_emb/lora
shift_s, scale_s, gate_s = pt_adaln(b0_temb, b0_lora, 'self_attn')

# Verify bcBuf matches
bc_pt_scale = scale_s.repeat_interleave(S_per, dim=0)
bc_pt_shift = shift_s.repeat_interleave(S_per, dim=0)
bc_pt_gate  = gate_s.repeat_interleave(S_per, dim=0)
print(f"  bcBuf SA scale+1: max_err={(b0_bcbuf[0] - bc_pt_scale).abs().max():.6f}")
print(f"  bcBuf SA shift:   max_err={(b0_bcbuf[1] - bc_pt_shift).abs().max():.6f}")
print(f"  bcBuf SA gate:    max_err={(b0_bcbuf[2] - bc_pt_gate).abs().max():.6f}")

# LN + modulate
x_5d = b0_x.reshape(M, 1, 16, 16, D)
ln_s = F.layer_norm(x_5d, (D,), None, None, 1e-6)
mod_s = ln_s * scale_s.reshape(M, 1, 1, 1, D) + shift_s.reshape(M, 1, 1, 1, D)
mod_f = mod_s.reshape(MS, D)

# Q/K/V GEMM
w_q = sd['blocks.0.self_attn.q_proj.weight']
w_k = sd['blocks.0.self_attn.k_proj.weight']
w_v = sd['blocks.0.self_attn.v_proj.weight']
w_o = sd['blocks.0.self_attn.output_proj.weight']
w_qn = sd['blocks.0.self_attn.q_norm.weight']
w_kn = sd['blocks.0.self_attn.k_norm.weight']

q_pt = F.linear(mod_f, w_q).reshape(ph, HD)
k_pt = F.linear(mod_f, w_k).reshape(ph, HD)
v_pt = F.linear(mod_f, w_v).reshape(ph, HD)

print(f"  Q GEMM: max_err={(b0_q - q_pt).abs().max():.6f}")
print(f"  K GEMM: max_err={(b0_k - k_pt).abs().max():.6f}")
print(f"  V GEMM: max_err={(b0_v - v_pt).abs().max():.6f}")

# RMSNorm
q_n_pt = F.rms_norm(q_pt, (HD,), w_qn, 1e-6)
k_n_pt = F.rms_norm(k_pt, (HD,), w_kn, 1e-6)
print(f"  Q_norm: max_err={(b0_qn - q_n_pt).abs().max():.6f}")
print(f"  K_norm: max_err={(b0_kn - k_n_pt).abs().max():.6f}")

# RoPE
rope_freqs = compute_rope_freqs()
q_r_pt = apply_rope(q_n_pt, rope_freqs)
k_r_pt = apply_rope(k_n_pt, rope_freqs)
print(f"  Q_rope: max_err={(b0_qr - q_r_pt).abs().max():.6f}")
print(f"  K_rope: max_err={(b0_kr - k_r_pt).abs().max():.6f}")
if (b0_qr - q_r_pt).abs().max() > 0.01:
    print(f"    ⚠️ RoPE mismatch! Checking per-head...")
    for h in range(NH):
        qr_err = (b0_qr.reshape(MS, NH, HD)[:, h, :] - q_r_pt.reshape(MS, NH, HD)[:, h, :]).abs().max()
        if qr_err > 1.0:
            print(f"    Head {h}: Q_rope max_err={qr_err:.4f}")

# Attention
sc = 1.0 / np.sqrt(HD)
attn_o_pt = torch.zeros(ph, HD)
for mb in range(M):
    qi = q_r_pt[mb*S_per*NH:(mb+1)*S_per*NH].reshape(S_per, NH, HD).permute(1, 0, 2)  # [NH, S, HD]
    ki = k_r_pt[mb*S_per*NH:(mb+1)*S_per*NH].reshape(S_per, NH, HD).permute(1, 0, 2)  # [NH, S, HD]
    vi = v_pt.reshape(MS, NH, HD)[mb*S_per:(mb+1)*S_per].permute(1, 0, 2)              # [NH, S, HD]
    scores = torch.bmm(qi, ki.transpose(1, 2)) * sc
    aw = F.softmax(scores, dim=-1)
    ao = torch.bmm(aw, vi).permute(1, 0, 2).reshape(S_per*NH, HD)
    attn_o_pt[mb*S_per*NH:(mb+1)*S_per*NH] = ao

print(f"\n  Attn out (with RoPE): max_err={(b0_attn_o - attn_o_pt).abs().max():.6f}")

# O_proj
oproj_pt = F.linear(attn_o_pt.reshape(MS, D), w_o)
print(f"  O_proj: max_err={(b0_oproj - oproj_pt).abs().max():.6f}")

# SA residual
sa_pt = b0_x + gate_s.repeat_interleave(S_per, dim=0) * oproj_pt
print(f"  SA residual: max_err={(b0_sa - sa_pt).abs().max():.6f}")

if (b0_sa - sa_pt).abs().max() > 1.0:
    print(f"    ⚠️ SA residual MISMATCH!")
    # Check gate and residual separately
    gate_err = (b0_bcbuf[2] - bc_pt_gate).abs().max()
    print(f"    Gate error: {gate_err:.6f}")
    resid_change_cpp = b0_sa - b0_x
    resid_change_pt = sa_pt - b0_x
    print(f"    SA change range: C++[{resid_change_cpp.min():.4f},{resid_change_cpp.max():.4f}] PT[{resid_change_pt.min():.4f},{resid_change_pt.max():.4f}]")

print("\n=== Done ===")

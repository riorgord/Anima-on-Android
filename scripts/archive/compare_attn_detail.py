"""Drill-down: compare C++ attention intermediates vs PyTorch bmm/softmax/bmm.

Step 1: Q_roped, K_roped, V_raw (after RoPE, before attention)
Step 2: QK^T scores (attn_qkt shader output)
Step 3: softmax output (attn_softmax shader output)
Step 4: AV output (attn_out shader output)

Usage (WSL2):
    python /mnt/d/AI/anima_phone/scripts/compare_attn_detail.py
"""
import sys, numpy as np

CMP = "/mnt/d/AI/anima_phone/output/cmp"
REF = "/mnt/d/AI/anima_phone/output/whitebox"

M, S, D = 2, 256, 2048; MS = M*S; NH = 16; HD = 128; S_per = MS // M

# Load phone attention intermediates
ph = {}
for name in ["b0_q_norm", "b0_k_norm", "b0_v_raw", "b0_q_roped", "b0_k_roped",
             "b0_scores", "b0_attn_o"]:
    ph[name] = np.load(f"{CMP}/{name}.npy").astype(np.float32)

# Load PC whitebox intermediates (pre-RoPE Q/K, V)
pt = {}
for name in ["sa_q_norm", "sa_k_norm", "sa_v_raw", "sa_q_roped", "sa_k_roped",
             "sa_attn_o"]:
    try:
        pt[name] = np.load(f"{REF}/b0/intermediates/{name}.npy").astype(np.float32)
    except:
        print(f"  WARNING: {name} not found in PT intermediates")
        pt[name] = None

# Load PC RoPE freqs (computed in whitebox)
# Recompute RoPE here
import numpy as np
DIM_T = HD - 2 * (HD // 6 * 2)  # 44
DIM_H = HD // 6 * 2               # 42
DIM_W = DIM_H                      # 42
HP = int(np.sqrt(S_per))          # 16

h_ntk = 4.0 ** (DIM_H / (DIM_H - 2))
w_ntk = 4.0 ** (DIM_W / (DIM_W - 2))
t_ntk = 1.0 ** (DIM_T / (DIM_T - 2))
h_theta = 10000.0 * h_ntk
w_theta = 10000.0 * w_ntk
t_theta = 10000.0 * t_ntk

half_dim = HD // 2
half_dim_t = DIM_T // 2
half_dim_h = DIM_H // 2
half_dim_w = DIM_W // 2

freqs = np.zeros(half_dim, dtype=np.float32)
for j in range(half_dim):
    if j < half_dim_t:
        freqs[j] = 1.0 / (t_theta ** (2.0 * j / DIM_T))
    elif j < half_dim_t + half_dim_h:
        jh = j - half_dim_t
        freqs[j] = 1.0 / (h_theta ** (2.0 * jh / DIM_H))
    else:
        jw = j - half_dim_t - half_dim_h
        freqs[j] = 1.0 / (w_theta ** (2.0 * jw / DIM_W))

h_idx = np.arange(S_per) // HP
w_idx = np.arange(S_per) % HP
angles = np.zeros((S_per, half_dim), dtype=np.float32)
for j in range(half_dim):
    if j < half_dim_t:
        angles[:, j] = 0.0
    elif j < half_dim_t + half_dim_h:
        angles[:, j] = h_idx * freqs[j]
    else:
        angles[:, j] = w_idx * freqs[j]

cos_vals = np.cos(angles)
sin_vals = np.sin(angles)

def apply_rope_np(q_or_k_flat):
    """Apply RoPE matching C++ shader: x0*cos - x1*sin, x0*sin + x1*cos."""
    ph_ = q_or_k_flat.shape[0]  # MS * NH
    x = q_or_k_flat.reshape(ph_, half_dim, 2)
    out = np.zeros_like(x)
    for p in range(S_per):
        for h in range(NH):
            row = M*S_per*NH  # wait, need to map correctly
    # Use vectorized approach
    q_or_k_reshaped = q_or_k_flat.reshape(MS, NH, HD)
    out = np.zeros_like(q_or_k_reshaped)
    for b in range(M):
        for p in range(S_per):
            cos = cos_vals[p]  # [half_dim]
            sin = sin_vals[p]  # [half_dim]
            for h in range(NH):
                idx = b * S_per * NH + p * NH + h
                row_data = q_or_k_flat[idx * HD : (idx+1) * HD].reshape(half_dim, 2)
                # Apply: [real, imag] -> [real*cos - imag*sin, real*sin + imag*cos]
                new_real = row_data[:, 0] * cos - row_data[:, 1] * sin
                new_imag = row_data[:, 0] * sin + row_data[:, 1] * cos
                out[b, p, h, 0::2] = new_real.reshape(-1, 1)
                out[b, p, h, 1::2] = new_imag.reshape(-1, 1)
    return out.reshape(MS * NH, HD)

# ── Step 1: Compare Q_roped and K_roped ──
print("=" * 70)
print("Step 1: Q_roped / K_roped / V_raw comparison")
print("=" * 70)

for name_ph, name_pt, label in [("b0_v_raw", "sa_v_raw", "V raw"),
                                  ("b0_q_roped", "sa_q_roped", "Q roped"),
                                  ("b0_k_roped", "sa_k_roped", "K roped")]:
    if name_ph not in ph or name_pt not in pt or pt[name_pt] is None:
        print(f"  {label}: SKIP (missing data)")
        continue
    ph_arr = ph[name_ph].flatten()
    pt_arr = pt[name_pt].flatten()
    mi = min(len(ph_arr), len(pt_arr))
    ok = np.isfinite(ph_arr[:mi]) & np.isfinite(pt_arr[:mi])
    if ok.sum() == 0:
        print(f"  {label}: ALL NaN")
        continue
    diff = np.abs(ph_arr[:mi][ok] - pt_arr[:mi][ok])
    print(f"  {label}: max_err={diff.max():.4f} mean_err={diff.mean():.6f}  "
          f"ph_range=[{ph_arr[ok].min():.3f},{ph_arr[ok].max():.3f}]  "
          f"pt_range=[{pt_arr[ok].min():.3f},{pt_arr[ok].max():.3f}]")

# ── Step 2: Attention scores (QK^T) ──
print("\n" + "=" * 70)
print("Step 2: QK^T scores comparison")
print("=" * 70)

scale = 1.0 / np.sqrt(HD)

# Phone scores: captured as [per-batch S*H*S] flat
# Layout: row-major [S, H, S] per batch
scores_b0_raw = ph["b0_scores"].reshape(S_per, NH, S_per)  # batch 0 only
scores_b0 = scores_b0_raw.transpose(1, 0, 2)  # [H, S_per, S_per]

# Compute PC scores from Q_roped and K_roped
# Use phone Q_roped and K_roped (to isolate QK^T from earlier errors)
q_roped_all = ph["b0_q_roped"].reshape(MS * NH, HD)
k_roped_all = ph["b0_k_roped"].reshape(MS * NH, HD)

# Batch 0 self-attention
q_b0 = q_roped_all[:S_per * NH].reshape(S_per, NH, HD).transpose(1, 0, 2)  # [H, S, D]
k_b0 = k_roped_all[:S_per * NH].reshape(S_per, NH, HD).transpose(1, 0, 2)  # [H, S, D]

# QK^T: [H, S, S]
scores_pc = np.zeros((NH, S_per, S_per), dtype=np.float32)
for h in range(NH):
    scores_pc[h] = q_b0[h] @ k_b0[h].T * scale

print(f"  Phone scores range: [{scores_b0.min():.4f}, {scores_b0.max():.4f}]")
print(f"  PC scores range:    [{scores_pc.min():.4f}, {scores_pc.max():.4f}]")
s_diff = np.abs(scores_b0.flatten() - scores_pc.flatten())
ok = np.isfinite(scores_b0.flatten()) & np.isfinite(scores_pc.flatten())
print(f"  scores max_err={s_diff[ok].max():.4f} mean_err={s_diff[ok].mean():.6f}")

# ── Step 3: Softmax ──
print("\n" + "=" * 70)
print("Step 3: Softmax comparison")
print("=" * 70)

# Phone softmax: stored in scores buffer after softmax
# PC: softmax over last dim (key positions)
# Use fp32 softmax for ground truth
scores_pc_f32 = scores_pc.astype(np.float64)
scores_pc_f32 -= scores_pc_f32.max(axis=-1, keepdims=True)  # stabilize
softmax_pc = np.exp(scores_pc_f32) / np.exp(scores_pc_f32).sum(axis=-1, keepdims=True)

print(f"  Phone softmax range: [{scores_b0.min():.4f}, {scores_b0.max():.4f}]")
print(f"  PC softmax range:    [{softmax_pc.min():.4f}, {softmax_pc.max():.4f}]")
sm_diff = np.abs(scores_b0.flatten() - softmax_pc.astype(np.float32).flatten())
ok = np.isfinite(scores_b0.flatten()) & np.isfinite(softmax_pc.flatten())
print(f"  softmax max_err={sm_diff[ok].max():.4f} mean_err={sm_diff[ok].mean():.6f}")

# ── Step 4: AV output ──
print("\n" + "=" * 70)
print("Step 4: AV (attention output) comparison")
print("=" * 70)

attn_o_all = ph["b0_attn_o"].reshape(MS * NH, HD)
# Batch 0
attn_o_b0 = attn_o_all[:S_per * NH].reshape(S_per, NH, HD).transpose(1, 0, 2)  # [H, S, D]

v_b0 = ph["b0_v_raw"].reshape(MS * NH, HD)[:S_per * NH].reshape(S_per, NH, HD).transpose(1, 0, 2)  # [H, S, D]

# PC AV using phone softmax and phone V
attn_o_pc = np.zeros((NH, S_per, HD), dtype=np.float32)
for h in range(NH):
    attn_o_pc[h] = scores_b0[h].astype(np.float64) @ v_b0[h].astype(np.float64)

print(f"  Phone attn_o range: [{attn_o_b0.min():.4f}, {attn_o_b0.max():.4f}]")
print(f"  PC attn_o range:    [{attn_o_pc.min():.4f}, {attn_o_pc.max():.4f}]")
ao_diff = np.abs(attn_o_b0.flatten() - attn_o_pc.astype(np.float32).flatten())
ok = np.isfinite(attn_o_b0.flatten()) & np.isfinite(attn_o_pc.flatten())
print(f"  attn_o max_err={ao_diff[ok].max():.4f} mean_err={ao_diff[ok].mean():.6f}")

print("\nDone.")

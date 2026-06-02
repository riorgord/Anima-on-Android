"""Block 0 detailed comparison: C++ GPU intermediates vs PyTorch reference."""
import numpy as np, torch, torch.nn.functional as F, safetensors.torch

CMP = '/mnt/d/AI/anima_phone/output/cmp_v2'
SF = '/mnt/d/AI/手坤的anima/models/diffusion_model/anima-base-v1.0.safetensors'
M, S, D, NH, HD = 2, 256, 2048, 16, 128
MS = M*S; S_per = MS//M; ph = MS*NH

# Load all C++ captures
b0_x   = torch.from_numpy(np.load(f'{CMP}/b0_x.npy').reshape(MS, D))
b0_q   = torch.from_numpy(np.load(f'{CMP}/b0_q.npy').reshape(ph, HD))
b0_k   = torch.from_numpy(np.load(f'{CMP}/b0_k.npy').reshape(ph, HD))
b0_v   = torch.from_numpy(np.load(f'{CMP}/b0_v.npy').reshape(ph, HD))
b0_qn  = torch.from_numpy(np.load(f'{CMP}/b0_qn.npy').reshape(ph, HD))
b0_kn  = torch.from_numpy(np.load(f'{CMP}/b0_kn.npy').reshape(ph, HD))
b0_qr  = torch.from_numpy(np.load(f'{CMP}/b0_qr.npy').reshape(ph, HD))
b0_kr  = torch.from_numpy(np.load(f'{CMP}/b0_kr.npy').reshape(ph, HD))
b0_attn_o = torch.from_numpy(np.load(f'{CMP}/b0_attn_o.npy').reshape(ph, HD))
b0_oproj = torch.from_numpy(np.load(f'{CMP}/b0_oproj.npy').reshape(MS, D))
b0_sa  = torch.from_numpy(np.load(f'{CMP}/b0_sa.npy').reshape(MS, D))
b0_lora = torch.from_numpy(np.load(f'{CMP}/b0_lora.npy').reshape(M, 3*D))

print("=== C++ value ranges ===")
for name, t in [("Q",b0_q),("K",b0_k),("V",b0_v),("Q_norm",b0_qn),("K_norm",b0_kn),
    ("Q_rope",b0_qr),("K_rope",b0_kr),("Attn_out",b0_attn_o),("O_proj",b0_oproj),("SA",b0_sa)]:
    print(f"  {name:12s}: [{t.min():.4f}, {t.max():.4f}] NaN={torch.isnan(t).any().item()}")

# Load weights
st = safetensors.torch.load_file(SF, device='cpu')
sd = {k[4:] if k.startswith('net.') else k: v.to(torch.float32) for k, v in st.items()}
del st

# Build PT reference with same inputs
def pt_adaln(emb, lora, block_idx, module):
    prefix = f'blocks.{block_idx}.adaln_modulation_{module}'
    w0 = sd[f'{prefix}.1.weight']; w2 = sd[f'{prefix}.2.weight']
    # AdaLN Sequential: SiLU → Linear(D→256) → Linear(256→3D) (no activation between stacked Linears)
    h = F.linear(F.silu(emb), w0); out = F.linear(h, w2) + lora
    shift, scale, gate = out.chunk(3, dim=-1)
    return shift, scale + 1.0, gate

# Use CAPTURED t_emb (post-RMSNorm, same as what C++ passes to blocks)
t_emb_pt = torch.from_numpy(np.load(f'{CMP}/b0_temb.npy').reshape(M, D))

# AdaLN using C++ lora
shift_s, scale_s, gate_s = pt_adaln(t_emb_pt, b0_lora, 0, 'self_attn')

# Block 0 self-attention (NO RoPE first, to isolate)
x_5d = b0_x.reshape(M, 1, 16, 16, D)
ln_s = F.layer_norm(x_5d, (D,), None, None, 1e-6)
mod_s = ln_s * scale_s.reshape(M, 1, 1, 1, D) + shift_s.reshape(M, 1, 1, 1, D)
mod_f = mod_s.reshape(MS, D)

w_q = sd['blocks.0.self_attn.q_proj.weight']
w_k = sd['blocks.0.self_attn.k_proj.weight']
w_v = sd['blocks.0.self_attn.v_proj.weight']
w_o = sd['blocks.0.self_attn.output_proj.weight']
w_qn = sd['blocks.0.self_attn.q_norm.weight']
w_kn = sd['blocks.0.self_attn.k_norm.weight']

# PT Q/K/V
q_pt = F.linear(mod_f, w_q).reshape(ph, HD)
k_pt = F.linear(mod_f, w_k).reshape(ph, HD)
v_pt = F.linear(mod_f, w_v).reshape(ph, HD)

print("\n=== Op-by-op comparison (C++ vs PT, pre-RoPE) ===")
for name, cpp, pt in [
    ("Q (GEMM)", b0_q, q_pt),
    ("K (GEMM)", b0_k, k_pt),
    ("V (GEMM)", b0_v, v_pt),
]:
    diff = (cpp - pt).abs()
    print(f"  {name:15s}: max_err={diff.max():.6f}, mean_err={diff.mean():.6f}")

# Q/K RMSNorm
q_n_pt = F.rms_norm(q_pt, (HD,), w_qn, 1e-6)
k_n_pt = F.rms_norm(k_pt, (HD,), w_kn, 1e-6)
for name, cpp, pt in [
    ("Q after RMSNorm", b0_qn, q_n_pt),
    ("K after RMSNorm", b0_kn, k_n_pt),
]:
    diff = (cpp - pt).abs()
    print(f"  {name:15s}: max_err={diff.max():.6f}, mean_err={diff.mean():.6f}")

# Q/K after RoPE — compare C++ RoPE vs identity (no RoPE)
# Since C++ applies RoPE and we don't, expect differences
diff_qr = (b0_qr - q_n_pt).abs()
diff_kr = (b0_kr - k_n_pt).abs()
print(f"\n  C++ RoPE vs no-RoPE: Q max_err={diff_qr.max():.4f}, K max_err={diff_kr.max():.4f}")
print(f"  (Expected: RoPE changes values — this shows RoPE IS being applied)")

# Check if RoPE changes are reasonable (not all-zero or all-same)
qr_diff_from_qn = (b0_qr - b0_qn).abs()
kr_diff_from_kn = (b0_kr - b0_kn).abs()
print(f"  RoPE delta Q: max={qr_diff_from_qn.max():.4f}, mean={qr_diff_from_qn.mean():.4f}")
print(f"  RoPE delta K: max={kr_diff_from_kn.max():.4f}, mean={kr_diff_from_kn.mean():.4f}")
if qr_diff_from_qn.max() < 1e-6:
    print("  ⚠️ WARNING: RoPE appears to be NO-OP (no change from pre-RoPE)!")
elif qr_diff_from_qn.max() > 1e6:
    print("  ⚠️ WARNING: RoPE changes are extremely large — possible bug!")
else:
    print("  ✓ RoPE changes look reasonable")

# Attention output (NO RoPE comparison)
attn_o_pt = torch.zeros(ph, HD)
sc = 1.0 / np.sqrt(HD)
for mb in range(M):
    qi = q_n_pt[mb*S_per*NH:(mb+1)*S_per*NH].reshape(S_per, NH, HD).permute(1, 0, 2)
    ki = k_n_pt[mb*S_per*NH:(mb+1)*S_per*NH].reshape(S_per, NH, HD).permute(1, 0, 2)
    vi = v_pt.reshape(MS, NH, HD)[mb*S_per:(mb+1)*S_per].permute(1, 0, 2)
    scores = torch.bmm(qi, ki.transpose(1, 2)) * sc
    aw = F.softmax(scores, dim=-1)
    ao = torch.bmm(aw, vi).permute(1, 0, 2).reshape(S_per*NH, HD)
    attn_o_pt[mb*S_per*NH:(mb+1)*S_per*NH] = ao

diff_attn = (b0_attn_o - attn_o_pt).abs()
print(f"\n  Attn out (C++ w/RoPE vs PT no-RoPE): max_err={diff_attn.max():.4f}, mean_err={diff_attn.mean():.4f}")

# O_proj
oproj_pt = F.linear(attn_o_pt.reshape(MS, D), w_o)
diff_oproj = (b0_oproj - oproj_pt).abs()
print(f"  O_proj (C++ w/RoPE vs PT no-RoPE): max_err={diff_oproj.max():.4f}, mean_err={diff_oproj.mean():.4f}")

# SA residual
sa_pt = b0_x + gate_s.repeat_interleave(S_per, dim=0) * oproj_pt
diff_sa = (b0_sa - sa_pt).abs()
print(f"  SA residual: max_err={diff_sa.max():.4f}, mean_err={diff_sa.mean():.4f}")

print("\n=== Key checks ===")
print(f"  Q/K/V: {'MATCH' if all([(b0_q-q_pt).abs().max()<1e-3,(b0_k-k_pt).abs().max()<1e-3]) else 'MISMATCH'}")
print(f"  Q_norm/K_norm: {'MATCH' if (b0_qn-q_n_pt).abs().max()<1e-3 else 'MISMATCH'}")
print(f"  RoPE active: {'YES' if (b0_qr-b0_qn).abs().max()>1e-6 else 'NO — BUG!'}")
print(f"  RoPE values: {'OK' if 0.01 < (b0_qr-b0_qn).abs().max() < 100 else 'CHECK'}")

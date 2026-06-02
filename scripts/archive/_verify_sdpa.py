"""Verify: anima_rt SDPA vs PT manual SDPA (matmul→softmax→matmul)."""
import ctypes, torch, numpy as np

lib = ctypes.CDLL("/data/local/tmp/libanima_rt.so")
lib.anima_rt_init.restype = ctypes.c_bool
lib.anima_rt_run_sdpa.argtypes = [ctypes.c_void_p]*4 + [ctypes.c_int]*4 + [ctypes.c_float, ctypes.c_bool]
lib.anima_rt_run_sdpa.restype = ctypes.c_bool
assert lib.anima_rt_init()

# Test with realistic shapes (self-attention: B=2, H=16, S=256, D=128)
B, H, S, D = 2, 16, 256, 128
print(f"Self-attn: B={B} H={H} S={S} D={D}")

gen = torch.Generator().manual_seed(42)
q_pt = torch.randn(B, H, S, D, generator=gen, dtype=torch.float32)
k_pt = torch.randn(B, H, S, D, generator=gen, dtype=torch.float32)
v_pt = torch.randn(B, H, S, D, generator=gen, dtype=torch.float32)

# ── Our SDPA ──
q_bh = q_pt.reshape(B*H, S, D).contiguous().numpy()
k_bh = k_pt.reshape(B*H, S, D).contiguous().numpy()
v_bh = v_pt.reshape(B*H, S, D).contiguous().numpy()
out_np = np.zeros((B*H, S, D), dtype=np.float32)

scale = 1.0/np.sqrt(D)
ok = lib.anima_rt_run_sdpa(
    q_bh.ctypes.data, k_bh.ctypes.data, v_bh.ctypes.data, out_np.ctypes.data,
    B*H, S, S, D, ctypes.c_float(scale), ctypes.c_bool(False))
print(f"anima_rt_sdpa: {ok}  range=[{out_np.min():.4f},{out_np.max():.4f}]")

# ── PT manual SDPA (same formula: scale→matmul→softmax→matmul) ──
q_s = q_pt * scale
k_s = k_pt * scale
q_bh_pt = q_s.reshape(B*H, S, D)
k_bh_pt = k_s.reshape(B*H, S, D)
v_bh_pt = v_pt.reshape(B*H, S, D)

# Manual per-head matmul (BH heads, each [S,D] @ [D,S])
attn = torch.zeros(B*H, S, S, dtype=torch.float32)
for h in range(B*H):
    attn[h] = q_bh_pt[h] @ k_bh_pt[h].T  # [S,S]
attn_sm = torch.softmax(attn, dim=-1)
out_pt = torch.zeros(B*H, S, D, dtype=torch.float32)
for h in range(B*H):
    out_pt[h] = attn_sm[h] @ v_bh_pt[h]  # [S,D]

out_pt_np = out_pt.numpy()
err = np.abs(out_np - out_pt_np).max()
print(f"max_err vs PT manual: {err:.2e}")
print(f"mean_err: {np.abs(out_np - out_pt_np).mean():.2e}")

# Check for NaN/Inf
print(f"nan={np.isnan(out_np).any()} inf={np.isinf(out_np).any()}")

# Cross-attention test (S_kv=512, different from S_q=256)
print(f"\nCross-attn: S_q={S} S_kv={512}")
kv_pt = torch.randn(B, H, 512, D, generator=gen, dtype=torch.float32)
kv_bh = kv_pt.reshape(B*H, 512, D).contiguous().numpy()
out2_np = np.zeros((B*H, S, D), dtype=np.float32)

ok = lib.anima_rt_run_sdpa(
    q_bh.ctypes.data, kv_bh.ctypes.data, kv_bh.ctypes.data, out2_np.ctypes.data,
    B*H, S, 512, D, ctypes.c_float(scale), ctypes.c_bool(False))
print(f"cross-attn sdpa: {ok}  range=[{out2_np.min():.4f},{out2_np.max():.4f}]")

# PT manual cross-attn
q_s2 = q_pt * scale
k_s2 = kv_pt * scale
attn2 = torch.zeros(B*H, S, 512)
for h in range(B*H):
    attn2[h] = q_s2.reshape(B*H, S, D)[h] @ k_s2.reshape(B*H, 512, D)[h].T
attn2_sm = torch.softmax(attn2, dim=-1)
out2_pt = torch.zeros(B*H, S, D)
for h in range(B*H):
    out2_pt[h] = attn2_sm[h] @ kv_pt.reshape(B*H, 512, D)[h]

err2 = np.abs(out2_np - out2_pt.numpy()).max()
print(f"max_err vs PT manual: {err2:.2e}")

print("\nDone!")

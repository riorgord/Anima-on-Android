"""Test fixed SDPA C function (attn@V now manual matmul)."""
import ctypes, torch, numpy as np

lib = ctypes.CDLL("/data/local/tmp/libanima_rt.so")
lib.anima_rt_init.restype = ctypes.c_bool
lib.anima_rt_run_sdpa.argtypes = [ctypes.c_void_p]*4 + [ctypes.c_int]*4 + [ctypes.c_float, ctypes.c_bool]
lib.anima_rt_run_sdpa.restype = ctypes.c_bool
assert lib.anima_rt_init()

B, H, S, D = 2, 16, 256, 128
gen = torch.Generator().manual_seed(42)
q = torch.randn(B, H, S, D, generator=gen).float()
k = torch.randn(B, H, S, D, generator=gen).float()
v = torch.randn(B, H, S, D, generator=gen).float()

# Our SDPA
q_bh = np.ascontiguousarray(q.reshape(B*H, S, D).numpy()).astype(np.float32)
k_bh = np.ascontiguousarray(k.reshape(B*H, S, D).numpy()).astype(np.float32)
v_bh = np.ascontiguousarray(v.reshape(B*H, S, D).numpy()).astype(np.float32)
out = np.zeros((B*H, S, D), dtype=np.float32)
scale = np.float32(1.0 / np.sqrt(D))
ok = lib.anima_rt_run_sdpa(
    q_bh.ctypes.data, k_bh.ctypes.data, v_bh.ctypes.data, out.ctypes.data,
    B*H, S, S, D, ctypes.c_float(scale), ctypes.c_bool(False))
print(f"SDPA ok={ok}  range=[{out.min():.4f},{out.max():.4f}]  nan={np.isnan(out).any()}")

# PT manual: scale→matmul→softmax→matmul
scale_f = float(scale)
qs = (q * scale_f).reshape(B*H, S, D)
ks = (k * scale_f).reshape(B*H, S, D)
vs = v.reshape(B*H, S, D)

attn = torch.zeros(B*H, S, S)
for h in range(B*H):
    attn[h] = qs[h] @ ks[h].T
attn_sm = torch.softmax(attn, dim=-1)
out_pt = torch.zeros(B*H, S, D)
for h in range(B*H):
    out_pt[h] = attn_sm[h] @ vs[h]

err = np.abs(out - out_pt.numpy()).max()
print(f"max_err vs PT manual: {err:.2e}")

# Also compare vs F.sdpa if available
try:
    q_bhsd = q.permute(0, 2, 1, 3)  # [B,S,H,D] -> [B,H,S,D]
    k_bhsd = k.permute(0, 2, 1, 3)
    v_bhsd = v.permute(0, 2, 1, 3)
    out_sdpa = torch.nn.functional.scaled_dot_product_attention(q_bhsd, k_bhsd, v_bhsd)
    out_sdpa_np = out_sdpa.reshape(B*H, S, D).numpy()
    err2 = np.abs(out - out_sdpa_np).max()
    print(f"max_err vs F.sdpa: {err2:.2e}")
except Exception as e:
    print(f"F.sdpa not available: {e}")

print("Done!")

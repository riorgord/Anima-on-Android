"""Verify: isolate GEMM vs Softmax error in SDPA pipeline."""
import ctypes, torch, numpy as np

lib = ctypes.CDLL("/data/local/tmp/libanima_rt.so")
lib.anima_rt_init.restype = ctypes.c_bool
lib.anima_rt_run_gemm_fp32.argtypes = [ctypes.c_void_p]*3 + [ctypes.c_int]*3
lib.anima_rt_run_gemm_fp32.restype = ctypes.c_bool
lib.anima_rt_run_softmax.argtypes = [ctypes.c_void_p]*2 + [ctypes.c_int]*2
lib.anima_rt_run_softmax.restype = ctypes.c_bool
assert lib.anima_rt_init()

B, H, S, D = 2, 16, 256, 128
gen = torch.Generator().manual_seed(42)
q = torch.randn(B, H, S, D, generator=gen, dtype=torch.float32)
k = torch.randn(B, H, S, D, generator=gen, dtype=torch.float32)
v = torch.randn(B, H, S, D, generator=gen, dtype=torch.float32)

scale = 1.0/np.sqrt(D)
h = 0  # first head

# Use contiguous numpy copies to avoid GC issues
q0 = np.ascontiguousarray(q.reshape(B*H, S, D)[h].numpy() * scale).astype(np.float32)
k0 = np.ascontiguousarray(k.reshape(B*H, S, D)[h].numpy() * scale).astype(np.float32)
v0 = np.ascontiguousarray(v.reshape(B*H, S, D)[h].numpy()).astype(np.float32)

# Step 1: GEMM Q@K^T
attn_our = np.zeros((S, S), dtype=np.float32)
ok = lib.anima_rt_run_gemm_fp32(q0.ctypes.data, k0.ctypes.data, attn_our.ctypes.data, S, S, D)
attn_pt = torch.from_numpy(q0) @ torch.from_numpy(k0).T
err1 = np.abs(attn_our - attn_pt.numpy()).max()
print(f"Step 1 — GEMM Q@K^T: max_err={err1:.2e}  range_our=[{attn_our.min():.4f},{attn_our.max():.4f}] range_pt=[{attn_pt.min():.4f},{attn_pt.max():.4f}]")

# Step 2: Softmax (using PT's GEMM output as input to isolate softmax error)
sm_in = attn_pt.float().numpy().astype(np.float32)
sm_our = np.zeros((S, S), dtype=np.float32)
ok = lib.anima_rt_run_softmax(sm_in.ctypes.data, sm_our.ctypes.data, S, S)
sm_pt = torch.softmax(torch.from_numpy(sm_in), dim=-1)
err2 = np.abs(sm_our - sm_pt.numpy()).max()
print(f"Step 2 — Softmax: max_err={err2:.2e}  range_our=[{sm_our.min():.6f},{sm_our.max():.6f}]")

# Step 3: Second GEMM attn@V (using PT's softmax output to isolate GEMM error)
attn_pt_np = sm_pt.float().numpy()
out_our = np.zeros((S, D), dtype=np.float32)
ok = lib.anima_rt_run_gemm_fp32(attn_pt_np.ctypes.data, v0.ctypes.data, out_our.ctypes.data, S, D, S)
out_pt = torch.from_numpy(attn_pt_np) @ torch.from_numpy(v0)
err3 = np.abs(out_our - out_pt.numpy()).max()
print(f"Step 3 — GEMM attn@V: max_err={err3:.2e}  range_our=[{out_our.min():.4f},{out_our.max():.4f}]")

# End-to-end with all our ops
attn_all = np.zeros((S, S), dtype=np.float32)
ok = lib.anima_rt_run_gemm_fp32(q0.ctypes.data, k0.ctypes.data, attn_all.ctypes.data, S, S, D)
sm_all = np.zeros((S, S), dtype=np.float32)
ok = lib.anima_rt_run_softmax(attn_all.ctypes.data, sm_all.ctypes.data, S, S)
out_all = np.zeros((S, D), dtype=np.float32)
ok = lib.anima_rt_run_gemm_fp32(sm_all.ctypes.data, v0.ctypes.data, out_all.ctypes.data, S, D, S)

out_pt_full = torch.softmax(torch.from_numpy(q0) @ torch.from_numpy(k0).T, dim=-1) @ torch.from_numpy(v0)
err_e2e = np.abs(out_all - out_pt_full.numpy()).max()
print(f"\nE2E (our GEMM+Softmax+GEMM): max_err={err_e2e:.2e}")
print(f"Error budget: GEMM={err1:.1e} + Softmax={err2:.1e} + GEMM2={err3:.1e} → cumulative")

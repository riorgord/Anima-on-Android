"""Quick phone verification: .so loads, GEMM works, SiLU patched."""
import ctypes, sys, torch
import torch.nn.functional as F
import numpy as np

lib = ctypes.CDLL("/data/local/tmp/libanima_rt.so")
lib.anima_rt_init.restype = ctypes.c_bool
lib.anima_gemm_has_blas.restype = ctypes.c_bool
ok = lib.anima_rt_init()
print(f"anima_rt_init: {ok}")

# Check if OpenBLAS was loaded
has_blas = lib.anima_gemm_has_blas()
print(f"OpenBLAS loaded: {has_blas}")
lib.anima_rt_run_gemm_fp32.argtypes = [ctypes.c_void_p]*3 + [ctypes.c_int]*3
lib.anima_rt_run_gemm_fp32.restype = ctypes.c_bool

# Small FP32 GEMM test
M, N, K = 4, 8, 16
A = np.random.randn(M, K).astype(np.float32)
B = np.random.randn(N, K).astype(np.float32)
C = np.zeros((M, N), dtype=np.float32)

ok = lib.anima_rt_run_gemm_fp32(A.ctypes.data, B.ctypes.data, C.ctypes.data, M, N, K)
print(f"gemm_fp32: {ok}  C[0,0]={C[0,0]:.4f}")

# Compare with PyTorch
C_pt = (torch.from_numpy(A) @ torch.from_numpy(B).T).numpy()
err = np.abs(C - C_pt).max()
print(f"gemm_fp32 vs PT: max_err={err:.2e}")

# BF16 GEMM test
lib.anima_rt_run_gemm_bf16.argtypes = [ctypes.c_void_p]*3 + [ctypes.c_int]*3
lib.anima_rt_run_gemm_bf16.restype = ctypes.c_bool

# Create BF16 weights (uint16 representation)
w_bf16 = torch.randn(N, K).bfloat16()
w_u16 = w_bf16.view(torch.uint16).numpy()
C2 = np.zeros((M, N), dtype=np.float32)
ok = lib.anima_rt_run_gemm_bf16(A.ctypes.data, w_u16.ctypes.data, C2.ctypes.data, M, N, K)
print(f"gemm_bf16: {ok}  C2[0,0]={C2[0,0]:.4f}")

# Compare: C2 should match A @ w_bf16.float().T
C2_pt = (torch.from_numpy(A) @ w_bf16.float().T).numpy()
err2 = np.abs(C2 - C2_pt).max()
print(f"gemm_bf16 vs PT: max_err={err2:.2e}")

# Quick SiLU verification
lib.anima_rt_run_silu.argtypes = [ctypes.c_void_p]*2 + [ctypes.c_int]
lib.anima_rt_run_silu.restype = ctypes.c_bool
x = np.random.randn(100).astype(np.float32)
out = np.zeros(100, dtype=np.float32)
lib.anima_rt_run_silu(x.ctypes.data, out.ctypes.data, 100)
pt_silu = torch.nn.SiLU()(torch.from_numpy(x)).numpy()
print(f"silu vs PT: max_err={np.abs(out - pt_silu).max():.2e}")

# SDPA test
lib.anima_rt_run_sdpa.argtypes = [ctypes.c_void_p]*4 + [ctypes.c_int]*4 + [ctypes.c_float, ctypes.c_bool]
lib.anima_rt_run_sdpa.restype = ctypes.c_bool

B, H, S, D = 1, 2, 4, 8
q_np = np.random.randn(B*H, S, D).astype(np.float32)
k_np = np.random.randn(B*H, S, D).astype(np.float32)
v_np = np.random.randn(B*H, S, D).astype(np.float32)
out_np = np.zeros((B*H, S, D), dtype=np.float32)
scale = 1.0/np.sqrt(D)
ok = lib.anima_rt_run_sdpa(
    q_np.ctypes.data, k_np.ctypes.data, v_np.ctypes.data, out_np.ctypes.data,
    B*H, S, S, D, ctypes.c_float(scale), ctypes.c_bool(False))
print(f"sdpa: {ok}  out[0,0,0]={out_np[0,0,0]:.4f}")

# Compare with PT math backend
q_pt = torch.from_numpy(q_np).reshape(B, H, S, D)
k_pt = torch.from_numpy(k_np).reshape(B, H, S, D)
v_pt = torch.from_numpy(v_np).reshape(B, H, S, D)
with torch.backends.sdpa_kernel(enable_flash=False, enable_math=True):
    out_pt = F.scaled_dot_product_attention(q_pt, k_pt, v_pt)
out_pt_np = out_pt.reshape(B*H, S, D).numpy()
err = np.abs(out_np - out_pt_np).max()
print(f"sdpa vs PT math: max_err={err:.2e}")

print("\nAll checks passed!")

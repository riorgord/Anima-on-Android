"""Verify: our flash attention vs PT default F.sdpa."""
import ctypes, torch, numpy as np, math

lib = ctypes.CDLL("/data/local/tmp/libanima_rt.so")
lib.anima_rt_init.restype = ctypes.c_bool
lib.anima_rt_run_sdpa_flash.argtypes = [ctypes.c_void_p]*4 + [ctypes.c_int]*4 + [ctypes.c_float, ctypes.c_bool]
lib.anima_rt_run_sdpa_flash.restype = ctypes.c_bool
assert lib.anima_rt_init()

B,H,S,D = 2,16,256,128
torch.manual_seed(42)
q = torch.randn(B,H,S,D).float()
k = torch.randn(B,H,S,D).float()
v = torch.randn(B,H,S,D).float()

# Our flash
q_bh = np.ascontiguousarray(q.reshape(B*H,S,D).numpy()).astype(np.float32)
k_bh = np.ascontiguousarray(k.reshape(B*H,S,D).numpy()).astype(np.float32)
v_bh = np.ascontiguousarray(v.reshape(B*H,S,D).numpy()).astype(np.float32)
out = np.zeros((B*H,S,D), dtype=np.float32)
scale = float(1.0/math.sqrt(D))
ok = lib.anima_rt_run_sdpa_flash(
    q_bh.ctypes.data, k_bh.ctypes.data, v_bh.ctypes.data, out.ctypes.data,
    B*H, S, S, D, ctypes.c_float(scale), ctypes.c_bool(False))
print(f"Flash ok={ok} range=[{out.min():.4f},{out.max():.4f}] nan={np.isnan(out).any()}")

# PT default
out_pt = torch.nn.functional.scaled_dot_product_attention(q,k,v)
out_pt_np = out_pt.reshape(B*H,S,D).numpy()
err = np.abs(out - out_pt_np).max()
print(f"Flash vs PT default: max_err={err:.2e}")

# Also compare vs PT math for reference
qs = q * scale; ks = k * scale
out_math = (torch.softmax(qs @ ks.transpose(-2,-1), dim=-1) @ v).reshape(B*H,S,D).numpy()
err_math = np.abs(out - out_math).max()
print(f"Flash vs PT math: max_err={err_math:.2e}")
print(f"PT default vs PT math: max_err={np.abs(out_pt_np - out_math).max():.2e}")

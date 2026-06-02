"""Compare PT-exact RMSNorm vs hand-built Welford RMSNorm vs PT nn.RMSNorm."""
import ctypes, torch, numpy as np

lib = ctypes.CDLL("/data/local/tmp/libanima_rt.so")
lib.anima_rt_init.restype = ctypes.c_bool
lib.anima_rt_run_rmsnorm.argtypes = [ctypes.c_void_p]*3 + [ctypes.c_int]*2 + [ctypes.c_float]
lib.anima_rt_run_rmsnorm.restype = ctypes.c_bool
assert lib.anima_rt_init()

# Test with realistic dims and values
torch.manual_seed(12345)
M, D = 256, 2048  # typical RMSNorm usage in DiT
w = torch.randn(D).float()

# Test 1: small values
x = torch.randn(M, D).float()
x_np = x.numpy().astype(np.float32)
w_np = w.numpy().astype(np.float32)
out = np.zeros((M, D), dtype=np.float32)

eps = 1e-6
ok = lib.anima_rt_run_rmsnorm(x_np.ctypes.data, w_np.ctypes.data, out.ctypes.data, M, D, ctypes.c_float(eps))
out_pt = torch.nn.functional.rms_norm(x, [D], weight=w, eps=eps)
err = np.abs(out - out_pt.numpy()).max()
print(f"Normal values: max_err={err:.2e}")

# Test 2: larger values (closer to what DiT produces internally)
x_big = torch.randn(M, D).float() * 10
x_big_np = x_big.numpy().astype(np.float32)
out2 = np.zeros((M, D), dtype=np.float32)
ok = lib.anima_rt_run_rmsnorm(x_big_np.ctypes.data, w_np.ctypes.data, out2.ctypes.data, M, D, ctypes.c_float(eps))
out2_pt = torch.nn.functional.rms_norm(x_big, [D], weight=w, eps=eps)
err2 = np.abs(out2 - out2_pt.numpy()).max()
print(f"Large values (x10): max_err={err2:.2e}")

# Test 3: extreme values (DiT residual can be large)
x_ext = torch.randn(M, D).float() * 100
x_ext_np = x_ext.numpy().astype(np.float32)
out3 = np.zeros((M, D), dtype=np.float32)
ok = lib.anima_rt_run_rmsnorm(x_ext_np.ctypes.data, w_np.ctypes.data, out3.ctypes.data, M, D, ctypes.c_float(eps))
out3_pt = torch.nn.functional.rms_norm(x_ext, [D], weight=w, eps=eps)
err3 = np.abs(out3 - out3_pt.numpy()).max()
print(f"Extreme values (x100): max_err={err3:.2e}")

# Compare: was 9.5e-06 before the fix (Welford-based)
print(f"\nPrevious Welford RMSNorm: 9.5e-06")
print(f"New PT-exact RMSNorm: {err:.2e}")
if err < 9.5e-06:
    print("IMPROVED!")
else:
    print("Worse or same")

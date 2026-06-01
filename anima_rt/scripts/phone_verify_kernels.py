"""Phone-side kernel verification — compare libanima_rt.so vs PyTorch.
Push and run:
  MSYS_NO_PATHCONV=1 adb push scripts/phone_verify_kernels.py /sdcard/anima_on_android/scripts/
  adb shell "su -c 'taskset f0 /data/data/com.termux/files/usr/bin/python /sdcard/anima_on_android/scripts/phone_verify_kernels.py'"
"""
import ctypes as ct
import numpy as np
import torch
import torch.nn.functional as F
import os, sys

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

_lib = ct.CDLL("/data/local/tmp/libanima_rt.so")
_lib.anima_rt_init.restype = ct.c_bool
_lib.anima_rt_run_gelu.argtypes = [ct.c_void_p, ct.c_void_p, ct.c_int]
_lib.anima_rt_run_gelu.restype = ct.c_bool
_lib.anima_rt_run_silu.argtypes = [ct.c_void_p, ct.c_void_p, ct.c_int]
_lib.anima_rt_run_silu.restype = ct.c_bool
_lib.anima_rt_run_layernorm.argtypes = [ct.c_void_p, ct.c_void_p, ct.c_int, ct.c_int, ct.c_float]
_lib.anima_rt_run_layernorm.restype = ct.c_bool
_lib.anima_rt_run_rmsnorm.argtypes = [ct.c_void_p, ct.c_void_p, ct.c_void_p, ct.c_int, ct.c_int, ct.c_float]
_lib.anima_rt_run_rmsnorm.restype = ct.c_bool
_lib.anima_rt_run_softmax.argtypes = [ct.c_void_p, ct.c_void_p, ct.c_int, ct.c_int]
_lib.anima_rt_run_softmax.restype = ct.c_bool

assert _lib.anima_rt_init(), "anima_rt_init failed"
print("anima_rt_init OK")

PASS = 0
FAIL = 0

def check(name, our, ref, tol=1e-5):
    global PASS, FAIL
    diff = np.abs(our.astype(np.float64) - ref.astype(np.float64))
    max_err = diff.max()
    mean_err = diff.mean()
    if max_err <= tol:
        PASS += 1
        print(f"  [PASS] {name}: max_err={max_err:.2e} mean_err={mean_err:.2e}")
    else:
        FAIL += 1
        n_bad = int((diff > tol).sum())
        print(f"  [FAIL] {name}: max_err={max_err:.6e} mean_err={mean_err:.6e} n_bad={n_bad}/{diff.size}")

# ── GELU ──
N = 1024
x = np.random.randn(N).astype(np.float32)
ref = F.gelu(torch.from_numpy(x)).numpy()
out = np.zeros(N, dtype=np.float32)
assert _lib.anima_rt_run_gelu(x.ctypes.data, out.ctypes.data, N)
check("GELU N=1024", out, ref, tol=5e-7)

# ── SiLU ──
ref_s = F.silu(torch.from_numpy(x)).numpy()
out_s = np.zeros(N, dtype=np.float32)
assert _lib.anima_rt_run_silu(x.ctypes.data, out_s.ctypes.data, N)
check("SiLU N=1024", out_s, ref_s, tol=5e-7)

# ── LayerNorm ──
M, D = 64, 2048
x_ln = np.random.randn(M, D).astype(np.float32)
ref_ln = F.layer_norm(torch.from_numpy(x_ln), [D], None, None, 1e-6).numpy()
out_ln = np.zeros((M, D), dtype=np.float32)
assert _lib.anima_rt_run_layernorm(x_ln.ctypes.data, out_ln.ctypes.data, M, D, 1e-6)
check("LayerNorm M=64 D=2048", out_ln, ref_ln, tol=5e-6)

# ── RMSNorm ──
w_rms = np.random.randn(D).astype(np.float32)
ref_rms = F.rms_norm(torch.from_numpy(x_ln), [D], torch.from_numpy(w_rms), 1e-6).numpy()
out_rms = np.zeros((M, D), dtype=np.float32)
_lib.anima_rt_run_rmsnorm(x_ln.ctypes.data, w_rms.ctypes.data, out_rms.ctypes.data, M, D, 1e-6)
check("RMSNorm M=64 D=2048", out_rms, ref_rms, tol=1e-5)

# ── Softmax ──
outer, dim = 16, 256
x_sm = np.random.randn(outer, dim).astype(np.float32)
ref_sm = F.softmax(torch.from_numpy(x_sm), dim=-1).numpy()
out_sm = np.zeros((outer, dim), dtype=np.float32)
assert _lib.anima_rt_run_softmax(x_sm.ctypes.data, out_sm.ctypes.data, outer, dim)
check("Softmax outer=16 dim=256", out_sm, ref_sm, tol=1e-7)

# ── Summary ──
print(f"\nResults: {PASS} PASS, {FAIL} FAIL on phone")
_lib.anima_rt_destroy()
if FAIL > 0:
    sys.exit(1)

"""Compare AnimaRTSiLU (PT path) vs NumpyRT.run_silu (ND path) — same input."""
import sys, ctypes, numpy as np
sys.path.insert(0, "/sdcard/anima_on_android/scripts")
import torch, anima_rt_ops

_lib = anima_rt_ops._lib

test_in = np.random.randn(2048).astype(np.float32) * 2.0
print(f"Input: [{test_in.min():.4f},{test_in.max():.4f}]")

# PT: AnimaRTSiLU
pt_silu = anima_rt_ops.AnimaRTSiLU()
pt_in = torch.from_numpy(test_in)
pt_out = pt_silu.forward(pt_in)
pt_np = pt_out.float().cpu().numpy()

# ND: NumpyRT.run_silu (same C kernel, direct ctypes)
nd_out = np.zeros(len(test_in), dtype=np.float32)
_lib.anima_rt_run_silu(test_in.ctypes.data, nd_out.ctypes.data, len(test_in))

err = np.abs(nd_out - pt_np).max()
print(f"max_err: {err:.8f}")
print(f"PT first 5: {pt_np[:5]}")
print(f"ND first 5: {nd_out[:5]}")
print(f"RESULT: {'PASS' if err < 1e-7 else 'FAIL'}")

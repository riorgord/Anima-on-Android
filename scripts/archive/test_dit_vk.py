"""Test libdit_vk.so: load weights, run one DiT forward, compare with CPU"""
import sys, time, ctypes
sys.path.insert(0, "/sdcard/anima_on_android/scripts")

# Load .so
_lib = ctypes.CDLL("/data/local/tmp/libdit_vk.so")
_lib.dit_init.argtypes = [ctypes.c_char_p]
_lib.dit_init.restype = ctypes.c_bool
_lib.dit_forward.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int]
_lib.dit_forward.restype = ctypes.c_bool
_lib.dit_destroy.argtypes = []
_lib.dit_destroy.restype = None

print("dit_init...")
ok = _lib.dit_init(b"/data/local/tmp/diffusion_weights.bin")
if not ok:
    print("dit_init FAILED")
    sys.exit(1)
print("dit_init OK")

# Generate dummy inputs matching DiT shapes
import torch, numpy as np
M, H, W = 2, 32, 32  # CFG batch, latent spatial
S = (H//2) * (W//2)   # 256 tokens
D = 2048

latent = torch.randn(M, 16, H, W, dtype=torch.float16)
t_emb = torch.randn(M, D, dtype=torch.float16)
ctx_cond = torch.randn(1, 512, 1024, dtype=torch.float16)
ctx_uncond = torch.randn(1, 512, 1024, dtype=torch.float16)
output = torch.zeros(M, 16, H, W, dtype=torch.float16)

print(f"dit_forward M={M} H={H} W={W} S={S} D={D}...")
t0 = time.perf_counter()
ok = _lib.dit_forward(
    latent.numpy().ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),
    t_emb.numpy().ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),
    ctx_cond.numpy().ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),
    ctx_uncond.numpy().ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),
    output.numpy().ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),
    M, H, W)
elapsed = time.perf_counter() - t0
print(f"dit_forward: {elapsed:.1f}s  ok={ok}")

# Phase timings from C++
_lib.dit_get_timings_us.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
_init = ctypes.c_double(); _w = ctypes.c_double(); _blk = ctypes.c_double()
_lib.dit_get_timings_us(ctypes.byref(_init), ctypes.byref(_w), ctypes.byref(_blk))
print(f"Phases: init={_init.value*1e-3:.0f}ms weights={_w.value*1e-3:.0f}ms blocks={_blk.value*1e-3:.0f}ms")

_lib.dit_destroy()

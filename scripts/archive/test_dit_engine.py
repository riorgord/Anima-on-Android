"""Minimal test: libdit_vk.so — Vulkan init + LayerNorm validation"""
import ctypes, numpy as np, torch

_lib = ctypes.CDLL("/data/local/tmp/libdit_vk.so")
_lib.dit_init.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
_lib.dit_init.restype = ctypes.c_bool
_lib.dit_record_oneshot.argtypes = []
_lib.dit_record_oneshot.restype = ctypes.c_bool
_lib.dit_forward.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
    ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]
_lib.dit_forward.restype = ctypes.c_bool
_lib.dit_destroy.argtypes = []
_lib.dit_destroy.restype = None

print("dit_init (weightless)...")
ok = _lib.dit_init(b"", b"/data/local/tmp")
print(f"  init = {ok}")
if not ok:
    print("FAILED")
    import sys; sys.exit(1)

print("dit_record_oneshot (LayerNorm test)...")
ok = _lib.dit_record_oneshot()
print(f"  record = {ok}")
if not ok:
    print("FAILED")
    import sys; sys.exit(1)

# Test data: x [MS=512, D=2048]
MS, D, M = 512, 2048, 2
np.random.seed(42)
x = np.random.randn(MS, D).astype(np.float16)
t_emb = np.random.randn(M, D).astype(np.float16)
ctx = np.random.randn(M, 512, 1024).astype(np.float16)
out = np.zeros((MS, D), dtype=np.float16)

print(f"dit_forward: MS={MS} D={D}")
ok = _lib.dit_forward(
    x.ctypes.data_as(ctypes.c_void_p),
    t_emb.ctypes.data_as(ctypes.c_void_p),
    ctx.ctypes.data_as(ctypes.c_void_p),
    out.ctypes.data_as(ctypes.c_void_p),
    MS, D, M, 512, 1024)
print(f"  forward = {ok}")

# CPU reference: LayerNorm(x) — no affine params
x_t = torch.from_numpy(x.astype(np.float32))
ref = torch.nn.functional.layer_norm(x_t, (D,), weight=None, bias=None, eps=1e-6)
ref_np = ref.numpy().astype(np.float16)

err = np.abs(out.astype(np.float32) - ref_np.astype(np.float32)).max()
print(f"max_err = {err:.6f}")
print(f"out mean={out.astype(np.float32).mean():.4f} std={out.astype(np.float32).std():.4f}")
print(f"ref mean={ref_np.astype(np.float32).mean():.4f} std={ref_np.astype(np.float32).std():.4f}")
print(f"output non-zero: {(out != 0).sum()} / {out.size}")

_lib.dit_destroy()
print("DONE")

"""Minimal test: LN + AdaLN apply only (ScaleShift shader test)"""
import ctypes, numpy as np, torch, sys, time

_lib = ctypes.CDLL("/data/local/tmp/libdit_vk.so")
_lib.dit_init.argtypes = [ctypes.c_char_p, ctypes.c_char_p]; _lib.dit_init.restype = ctypes.c_bool
_lib.dit_record_adaln_only.argtypes = []; _lib.dit_record_adaln_only.restype = ctypes.c_bool
_lib.dit_write_buf.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_size_t]
_lib.dit_write_buf.restype = ctypes.c_bool
_lib.dit_forward.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
    ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]
_lib.dit_forward.restype = ctypes.c_bool

print("Init (weightless)...")
ok = _lib.dit_init(b"", b"/data/local/tmp")
print(f"  init = {ok}")
if not ok: sys.exit(1)

MS, D, M = 512, 2048, 2
np.random.seed(99)
x_t = torch.randn(MS, D, dtype=torch.float32)
scale_t = torch.randn(M, D, dtype=torch.float32) * 0.1 + 1.1  # small scale around 1.1
shift_t = torch.randn(M, D, dtype=torch.float32) * 0.1  # small shift

# Broadcast
scale_b = scale_t.repeat_interleave(MS//M, dim=0)
shift_b = shift_t.repeat_interleave(MS//M, dim=0)

# Upload concat [scale | shift] to bcBuf
n_elem = MS * D
concat = np.zeros(n_elem * 2, dtype=np.uint16)
concat[0*n_elem:1*n_elem] = scale_b.numpy().astype(np.float16).ravel().view(np.uint16)
concat[1*n_elem:2*n_elem] = shift_b.numpy().astype(np.float16).ravel().view(np.uint16)
_lib.dit_write_buf(4, concat.ctypes.data_as(ctypes.c_void_p), concat.nbytes)

# Upload x
x_np = x_t.numpy().astype(np.float16)
_lib.dit_write_buf(0, x_np.ctypes.data_as(ctypes.c_void_p), x_np.nbytes)

ok = _lib.dit_record_adaln_only()
print(f"  record = {ok}")

out_np = np.zeros((MS, D), dtype=np.float16)
t_np = np.zeros((M, D), dtype=np.float16); c_np = np.zeros((M, 512, 1024), dtype=np.float16)
ok = _lib.dit_forward(x_np.ctypes.data_as(ctypes.c_void_p), t_np.ctypes.data_as(ctypes.c_void_p),
    c_np.ctypes.data_as(ctypes.c_void_p), out_np.ctypes.data_as(ctypes.c_void_p),
    MS, D, M, 512, 1024)
print(f"  forward = {ok}")

# Reference: LN(x) * scale + shift
import torch.nn.functional as F
ln_x = F.layer_norm(x_t, (D,), weight=None, bias=None, eps=1e-6)
ref = ln_x * scale_b + shift_b
ref_np = ref.half().numpy()

err = np.abs(out_np.astype(np.float32) - ref_np.astype(np.float32)).max()
print(f"max_err = {err:.6f}")
print(f"out mean={out_np.astype(np.float32).mean():.4f} std={out_np.astype(np.float32).std():.4f}")
print(f"ref mean={ref_np.astype(np.float32).mean():.4f} std={ref_np.astype(np.float32).std():.4f}")
print("PASS" if err < 0.1 else "FAIL")
_lib.dit_destroy()

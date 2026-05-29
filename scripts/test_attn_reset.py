"""Test with stepPool reset between calls."""
import ctypes, numpy as np, torch
torch.manual_seed(42)

lib = ctypes.CDLL("/data/local/tmp/libdit_vk.so")
lib.dit_init_adaln_only.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
lib.dit_init_adaln_only.restype = ctypes.c_bool
lib.dit_run_attention.argtypes = [ctypes.c_void_p]*4 + [ctypes.c_int]*4 + [ctypes.c_float]
lib.dit_run_attention.restype = ctypes.c_bool
lib.dit_reset_step_pool.argtypes = []
lib.dit_reset_step_pool.restype = ctypes.c_bool

print("Init...")
ok = lib.dit_init_adaln_only(b"/data/local/tmp/diffusion_weights.bin", b"/data/local/tmp")
assert ok, "init failed"

M_q, M_kv, H, D = 512, 1024, 16, 128
scale = 1.0 / np.sqrt(D)
Q = torch.randn(M_q, H, D, dtype=torch.float16).numpy().view(np.uint16)
K = torch.randn(M_kv, H, D, dtype=torch.float16).numpy().view(np.uint16)
V = torch.randn(M_kv, H, D, dtype=torch.float16).numpy().view(np.uint16)
O = np.zeros(M_q * H * D, dtype=np.uint16)

for r in range(3):
    ok = lib.dit_run_attention(
        Q.ctypes.data_as(ctypes.c_void_p),
        K.ctypes.data_as(ctypes.c_void_p),
        V.ctypes.data_as(ctypes.c_void_p),
        O.ctypes.data_as(ctypes.c_void_p),
        M_q, M_kv, H, D, ctypes.c_float(scale))
    print(f"round {r}: {'OK' if ok else 'FAIL'}")
    if not ok:
        break
    lib.dit_reset_step_pool()
    print(f"  stepPool reset")

print("done")
lib.dit_destroy()

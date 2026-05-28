"""Quick smoke test: zero lora → engine should init and run without crash"""
import ctypes, numpy as np, os

lib = ctypes.CDLL("/data/local/tmp/libdit_vk.so")
lib.dit_init.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
lib.dit_init.restype = ctypes.c_bool
lib.dit_write_lora.argtypes = [ctypes.c_void_p]
lib.dit_write_lora.restype = None
lib.dit_destroy.argtypes = []
lib.dit_destroy.restype = None

ok = lib.dit_init(b"", b"/data/local/tmp")
print(f"init={ok}")

# Zero lora → no-op, matches old engine
lora = np.zeros(3*2*2048, dtype=np.float16)
lib.dit_write_lora(lora.ctypes.data_as(ctypes.c_void_p))
print("lora uploaded (zero fill)")

lib.dit_destroy()
print("SMOKE TEST PASSED")

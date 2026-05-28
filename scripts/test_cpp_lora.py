"""Verify C++ dit_compute_timestep matches PC torch pre-computed lora & t_emb"""
import ctypes, numpy as np

lib = ctypes.CDLL("/data/local/tmp/libdit_vk.so")
lib.dit_init.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
lib.dit_init.restype = ctypes.c_bool
lib.dit_compute_timestep.argtypes = [ctypes.c_float]
lib.dit_compute_timestep.restype = ctypes.c_bool
lib.dit_read_buf.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_size_t]
lib.dit_read_buf.restype = ctypes.c_bool
lib.dit_destroy.argtypes = []
lib.dit_destroy.restype = None

MS, D, M = 512, 2048, 2

print("init...")
ok = lib.dit_init(b"/data/local/tmp/diffusion_weights.bin", b"/data/local/tmp")
assert ok

# PC reference
pc_lora = np.fromfile("/sdcard/anima_on_android/output/lora_step0.bin", dtype=np.float16).reshape(3, M, D)
pc_temb = np.fromfile("/sdcard/anima_on_android/output/t_step0.bin", dtype=np.float16).reshape(M, D)

# C++ CPU compute
ok = lib.dit_compute_timestep(ctypes.c_float(1.0))
assert ok

# Read t_emb (buf_id=1)
cpp_temb = np.zeros(M*D, dtype=np.uint16)
lib.dit_read_buf(1, cpp_temb.ctypes.data_as(ctypes.c_void_p), cpp_temb.nbytes)
cpp_temb = cpp_temb.view(np.float16).reshape(M, D)

# Read lora (buf_id=8)
cpp_lora = np.zeros(3*M*D, dtype=np.uint16)
lib.dit_read_buf(8, cpp_lora.ctypes.data_as(ctypes.c_void_p), cpp_lora.nbytes)
cpp_lora = cpp_lora.view(np.float16).reshape(3, M, D)

# Compare t_emb
te = np.abs(cpp_temb.astype(np.float32) - pc_temb.astype(np.float32)).max()
print(f"t_emb max_err: {te:.6f}")
print(f"  PC[0,:8]:  {pc_temb[0,:8].astype(np.float32)}")
print(f"  CPP[0,:8]: {cpp_temb[0,:8].astype(np.float32)}")

# Compare lora
le_shift = np.abs(cpp_lora[0].astype(np.float32) - pc_lora[0].astype(np.float32)).max()
le_scale = np.abs(cpp_lora[1].astype(np.float32) - pc_lora[1].astype(np.float32)).max()
le_gate  = np.abs(cpp_lora[2].astype(np.float32) - pc_lora[2].astype(np.float32)).max()
print(f"lora max_err: shift={le_shift:.6f} scale={le_scale:.6f} gate={le_gate:.6f}")
print(f"  PC shift[0,:4]:  {pc_lora[0,0,:4].astype(np.float32)}")
print(f"  CPP shift[0,:4]: {cpp_lora[0,0,:4].astype(np.float32)}")

max_err = max(te, le_shift, le_scale, le_gate)
if max_err < 0.1:
    print(f"PASS — C++ CPU lora matches PC torch (max_err={max_err:.6f})")
else:
    print(f"CHECK — max_err={max_err:.6f}")

lib.dit_destroy()

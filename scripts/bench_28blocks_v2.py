"""Benchmark: 28-block DiT with per-block cmd buffers"""
import ctypes, numpy as np, time, sys
_lib=ctypes.CDLL("/data/local/tmp/libdit_vk.so")
_lib.dit_init.argtypes=[ctypes.c_char_p,ctypes.c_char_p]; _lib.dit_init.restype=ctypes.c_bool
_lib.dit_init_all_blocks.argtypes=[]; _lib.dit_init_all_blocks.restype=ctypes.c_bool
_lib.dit_forward_28blocks.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_void_p,
    ctypes.c_int,ctypes.c_int,ctypes.c_int]; _lib.dit_forward_28blocks.restype=ctypes.c_bool

print("Init...")
t0=time.time()
ok=_lib.dit_init(b"/data/local/tmp/diffusion_weights.bin",b"/data/local/tmp")
print(f"  init={ok} ({time.time()-t0:.1f}s)")
if not ok: sys.exit(1)

print("Recording 28 blocks (1 per cmd buf)...")
t0=time.time()
ok=_lib.dit_init_all_blocks()
print(f"  record={ok} ({time.time()-t0:.1f}s)")

MS,D,M=512,2048,2; n_elem=MS*D

# Pre-fill AdaLN data: 28×9×[MS,D] fp16 (just random for benchmark)
print("Preparing AdaLN data...")
adaln_all = np.random.randn(28 * 9 * n_elem).astype(np.float16)
x = np.random.randn(MS,D).astype(np.float16)
out = np.zeros((MS,D), dtype=np.float16)

# Warmup
print("Warmup...")
_lib.dit_forward_28blocks(x.ctypes.data_as(ctypes.c_void_p),
    adaln_all.ctypes.data_as(ctypes.c_void_p),
    out.ctypes.data_as(ctypes.c_void_p), MS, D, M)

# Benchmark
print("Benchmark...")
times = []
for r in range(3):
    t0 = time.time()
    ok = _lib.dit_forward_28blocks(x.ctypes.data_as(ctypes.c_void_p),
        adaln_all.ctypes.data_as(ctypes.c_void_p),
        out.ctypes.data_as(ctypes.c_void_p), MS, D, M)
    dt = time.time() - t0
    times.append(dt)
    print(f"  run {r}: {dt:.3f}s  ok={ok}")

avg = sum(times) / len(times)
print(f"\nAverage: {avg:.3f}s/step (28 blocks, {28*16} dispatches)")
print(f"Per block: {avg/28*1000:.1f}ms")
_lib.dit_destroy()

"""Benchmark 28 blocks with GPU-side AdaLN"""
import ctypes, numpy as np, time
_lib=ctypes.CDLL("/data/local/tmp/libdit_vk.so")
_lib.dit_init.argtypes=[ctypes.c_char_p,ctypes.c_char_p]; _lib.dit_init.restype=ctypes.c_bool
_lib.dit_init_all_blocks.argtypes=[]; _lib.dit_init_all_blocks.restype=ctypes.c_bool
_lib.dit_forward_28blocks.argtypes=[ctypes.c_void_p]*4+[ctypes.c_int]*5
_lib.dit_forward_28blocks.restype=ctypes.c_bool

print("Init...")
t0=time.time()
ok=_lib.dit_init(b"/data/local/tmp/diffusion_weights.bin",b"/data/local/tmp")
print(f"  init={ok} ({time.time()-t0:.1f}s)")
if not ok: exit()

print("Recording all 28 blocks (GPU AdaLN, ~53 dispatches/block)...")
t0=time.time()
ok=_lib.dit_init_all_blocks()
print(f"  record={ok} ({time.time()-t0:.1f}s)")

MS,D,M=512,2048,2
x=np.random.randn(MS,D).astype(np.float16)
t_emb=np.random.randn(M,D).astype(np.float16)
ctx=np.random.randn(M,512,1024).astype(np.float16)
out=np.zeros((MS,D),dtype=np.float16)

# Warmup
_lib.dit_forward_28blocks(x.ctypes.data_as(ctypes.c_void_p),t_emb.ctypes.data_as(ctypes.c_void_p),ctx.ctypes.data_as(ctypes.c_void_p),out.ctypes.data_as(ctypes.c_void_p),MS,D,M,512,1024)

# Benchmark
times=[]
for r in range(3):
    t0=time.time()
    _lib.dit_forward_28blocks(x.ctypes.data_as(ctypes.c_void_p),t_emb.ctypes.data_as(ctypes.c_void_p),ctx.ctypes.data_as(ctypes.c_void_p),out.ctypes.data_as(ctypes.c_void_p),MS,D,M,512,1024)
    dt=time.time()-t0
    times.append(dt)
    print(f"  run {r}: {dt:.3f}s")
avg=sum(times)/len(times)
ttl_disp = 28 * 53
print(f"Avg: {avg:.3f}s/step ({ttl_disp} dispatches, GPU AdaLN)")
print(f"Per block: {avg/28*1000:.1f}ms")
_lib.dit_destroy()

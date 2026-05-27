"""Benchmark: 28-block DiT forward with pre-uploaded AdaLN"""
import ctypes, numpy as np, time, sys
_lib=ctypes.CDLL("/data/local/tmp/libdit_vk.so")
_lib.dit_init.argtypes=[ctypes.c_char_p,ctypes.c_char_p]; _lib.dit_init.restype=ctypes.c_bool
_lib.dit_record_all_blocks.argtypes=[]; _lib.dit_record_all_blocks.restype=ctypes.c_bool
_lib.dit_write_buf.argtypes=[ctypes.c_int,ctypes.c_void_p,ctypes.c_size_t]; _lib.dit_write_buf.restype=ctypes.c_bool
_lib.dit_forward.argtypes=[ctypes.c_void_p]*9; _lib.dit_forward.restype=ctypes.c_bool

print("Init...")
t0=time.time()
ok=_lib.dit_init(b"/data/local/tmp/diffusion_weights.bin",b"/data/local/tmp")
print(f"  init={ok} ({time.time()-t0:.1f}s)")
if not ok: sys.exit(1)

MS,D,M=512,2048,2; n_elem=MS*D
# Skip bcBuf fill — benchmark only, AdaLN data doesn't affect GPU timing

print("Recording 28 blocks...")
t0=time.time()
ok=_lib.dit_record_all_blocks()
print(f"  record={ok} ({time.time()-t0:.1f}s)")
if not ok: sys.exit(1)

# Warmup
x=np.random.randn(MS,D).astype(np.float16); tn=np.zeros((M,D),dtype=np.float16); cn=np.zeros((M,512,1024),dtype=np.float16)
out=np.zeros((MS,D),dtype=np.float16)
_lib.dit_forward(x.ctypes.data_as(ctypes.c_void_p),tn.ctypes.data_as(ctypes.c_void_p),cn.ctypes.data_as(ctypes.c_void_p),out.ctypes.data_as(ctypes.c_void_p),MS,D,M,512,1024)

# Benchmark
print("Benchmarking 28-block forward...")
times=[]
for _ in range(3):
    t0=time.time()
    ok=_lib.dit_forward(x.ctypes.data_as(ctypes.c_void_p),tn.ctypes.data_as(ctypes.c_void_p),cn.ctypes.data_as(ctypes.c_void_p),out.ctypes.data_as(ctypes.c_void_p),MS,D,M,512,1024)
    dt=time.time()-t0
    times.append(dt)
    print(f"  {dt:.3f}s  ok={ok}")

avg=sum(times)/len(times)
print(f"\nAverage: {avg:.3f}s/step (28 blocks)")
print(f"Per block: {avg/28*1000:.1f}ms")
print(f"Output non-zero: {(out!=0).sum()}/{out.size}")
_lib.dit_destroy()

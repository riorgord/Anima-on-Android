"""Find max block count before Vulkan submit fails"""
import ctypes, numpy as np, time, sys
_lib=ctypes.CDLL("/data/local/tmp/libdit_vk.so")
_lib.dit_init.argtypes=[ctypes.c_char_p,ctypes.c_char_p]; _lib.dit_init.restype=ctypes.c_bool
_lib.dit_record_n_blocks.argtypes=[ctypes.c_int]; _lib.dit_record_n_blocks.restype=ctypes.c_bool
_lib.dit_forward.argtypes=[ctypes.c_void_p]*9; _lib.dit_forward.restype=ctypes.c_bool

print("Init...")
ok=_lib.dit_init(b"/data/local/tmp/diffusion_weights.bin",b"/data/local/tmp")
print(f"init={ok}")
if not ok: sys.exit(1)

MS,D,M=512,2048,2
x=np.random.randn(MS,D).astype(np.float16); tn=np.zeros((M,D),dtype=np.float16)
cn=np.zeros((M,512,1024),dtype=np.float16); out=np.zeros((MS,D),dtype=np.float16)

for n in [2,4,8,16,24,28]:
    ok=_lib.dit_record_n_blocks(n)
    t0=time.time()
    fwd=_lib.dit_forward(x.ctypes.data_as(ctypes.c_void_p),tn.ctypes.data_as(ctypes.c_void_p),cn.ctypes.data_as(ctypes.c_void_p),out.ctypes.data_as(ctypes.c_void_p),MS,D,M,512,1024)
    dt=time.time()-t0
    nz=(out!=0).sum()
    print(f"n={n:2d}  {dt:.3f}s  ok={fwd}  nonz={nz}")
_lib.dit_destroy()

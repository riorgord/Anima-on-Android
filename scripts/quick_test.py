"""Quick GPU AdaLN test — timing only, no PyTorch comparison"""
import ctypes, numpy as np, time
_lib=ctypes.CDLL("/data/local/tmp/libdit_vk.so")
_lib.dit_init.argtypes=[ctypes.c_char_p,ctypes.c_char_p]; _lib.dit_init.restype=ctypes.c_bool
_lib.dit_record_block_to.argtypes=[ctypes.c_int,ctypes.c_int]; _lib.dit_record_block_to.restype=ctypes.c_bool
_lib.dit_forward.argtypes=[ctypes.c_void_p]*9; _lib.dit_forward.restype=ctypes.c_bool
_lib.dit_write_buf.argtypes=[ctypes.c_int,ctypes.c_void_p,ctypes.c_size_t]; _lib.dit_write_buf.restype=ctypes.c_bool

print("Init..."); t0=time.time()
ok=_lib.dit_init(b"/data/local/tmp/diffusion_weights.bin",b"/data/local/tmp")
print(f"  init={ok} ({time.time()-t0:.1f}s)")
if not ok: exit()

MS,D,M=512,2048,2
x=np.random.randn(MS,D).astype(np.float16)
t_emb=np.random.randn(M,D).astype(np.float16)
ctx=np.random.randn(M,512,1024).astype(np.float16)
_lib.dit_write_buf(0,x.ctypes.data_as(ctypes.c_void_p),x.nbytes)
_lib.dit_write_buf(1,t_emb.ctypes.data_as(ctypes.c_void_p),t_emb.nbytes)
_lib.dit_write_buf(2,ctx.ctypes.data_as(ctypes.c_void_p),ctx.nbytes)

ok=_lib.dit_record_block_to(0,0)
print(f"  record={ok}")

out=np.zeros((MS,D),dtype=np.float16)
t0=time.time()
ok=_lib.dit_forward(x.ctypes.data_as(ctypes.c_void_p),t_emb.ctypes.data_as(ctypes.c_void_p),ctx.ctypes.data_as(ctypes.c_void_p),out.ctypes.data_as(ctypes.c_void_p),MS,D,M,512,1024)
print(f"  forward={ok} ({time.time()-t0:.3f}s)")
print(f"  out mean/std={out.astype(np.float32).mean():.3f}/{out.astype(np.float32).std():.3f}")
print(f"  non-zero: {(out!=0).sum()}/{out.size}")
_lib.dit_destroy()
print("DONE")

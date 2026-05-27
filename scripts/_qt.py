import ctypes, torch, numpy as np, sys, time
sys.path.insert(0, "/sdcard/anima_on_android/scripts")

_lib = ctypes.CDLL("/data/local/tmp/libdit_vk.so")
_lib.dit_init.argtypes = [ctypes.c_char_p]
_lib.dit_init.restype = ctypes.c_bool
_lib.dit_forward.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_void_p,ctypes.c_void_p,ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_int]
_lib.dit_forward.restype = ctypes.c_bool
ok = _lib.dit_init(b"/data/local/tmp/diffusion_weights.bin")
print("init:", ok)

MS,D,B,Nctx,CtxD = 512,2048,2,512,1024
x=torch.randn(MS,D,dtype=torch.float16)
t=torch.randn(B,D,dtype=torch.float16)
c=torch.randn(B,Nctx,CtxD,dtype=torch.float16)
o=torch.zeros(MS,D,dtype=torch.float16)

t0=time.perf_counter()
ok2=_lib.dit_forward(
    x.numpy().ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),
    t.numpy().ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),
    c.numpy().ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),
    o.numpy().ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),
    MS,D,B,Nctx,CtxD)
dt=time.perf_counter()-t0
print(f"fwd: {dt:.1f}s ok={ok2}")
print(f"mean={float(o.float().mean()):.4f} std={float(o.float().std()):.4f}")
print(f"non-zero: {(o != 0).sum().item()} / {MS*D}")
_lib.dit_destroy()

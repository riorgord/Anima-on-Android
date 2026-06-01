import ctypes, numpy as np
lib=ctypes.CDLL("/data/local/tmp/libdit_vk.so")
lib.dit_init_adaln_only.argtypes=[ctypes.c_char_p,ctypes.c_char_p]
lib.dit_init_adaln_only.restype=ctypes.c_bool
lib.dit_forward_nblocks.argtypes=[ctypes.c_void_p]*4+[ctypes.c_int]*6
lib.dit_forward_nblocks.restype=ctypes.c_bool
ok=lib.dit_init_adaln_only(b"/data/local/tmp/diffusion_weights.bin",b"/data/local/tmp")
print("init:",ok)
MS,M,D,Nctx,CtxD=512,2,2048,512,1024
rng=np.random.RandomState(42)
x=(rng.randn(MS*D).astype(np.float16)*0.1).view(np.uint16)
t=(rng.randn(M*D).astype(np.float16)*0.1).view(np.uint16)
c=(rng.randn(M*Nctx*CtxD).astype(np.float16)*0.1).view(np.uint16)
o=np.zeros(MS*D,dtype=np.uint16)
print("running 28 blocks...")
ok=lib.dit_forward_nblocks(
    x.ctypes.data_as(ctypes.c_void_p),
    t.ctypes.data_as(ctypes.c_void_p),
    c.ctypes.data_as(ctypes.c_void_p),
    o.ctypes.data_as(ctypes.c_void_p),
    MS,D,M,Nctx,CtxD,28)
print("forward 28:",ok)
lib.dit_destroy()

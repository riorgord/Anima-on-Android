"""Skip-attention pipeline with lora — C++ engine adaln_gpu (now with lora addition)."""
import ctypes, numpy as np, time

OUT="/sdcard/anima_on_android/output"
MS,D,M=512,2048,2; Nctx=512; CtxD=1024

_lib=ctypes.CDLL("/data/local/tmp/libdit_vk.so")
_lib.dit_init.argtypes=[ctypes.c_char_p,ctypes.c_char_p]; _lib.dit_init.restype=ctypes.c_bool
_lib.dit_init_all_blocks.argtypes=[]; _lib.dit_init_all_blocks.restype=ctypes.c_bool
_lib.dit_forward_28blocks.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_void_p,ctypes.c_void_p,
    ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_int]
_lib.dit_forward_28blocks.restype=ctypes.c_bool
_lib.dit_write_buf.argtypes=[ctypes.c_int,ctypes.c_void_p,ctypes.c_size_t]
_lib.dit_write_buf.restype=ctypes.c_bool
_lib.dit_write_lora.argtypes=[ctypes.c_void_p]
_lib.dit_write_lora.restype=None
_lib.dit_destroy.argtypes=[]
_lib.dit_destroy.restype=None

print("Init C++ engine...")
t0=time.time()
ok=_lib.dit_init(b"/data/local/tmp/diffusion_weights.bin",b"/data/local/tmp")
print(f"  init={ok} ({time.time()-t0:.1f}s)")

print("Recording 28 blocks...")
ok=_lib.dit_init_all_blocks()
print(f"  record={ok}")

# Upload ctx (constant across steps)
ctx=np.fromfile(f"{OUT}/ctx_stacked.bin",dtype=np.float16)
_lib.dit_write_buf(2,ctx.ctypes.data_as(ctypes.c_void_p),ctypes.c_size_t(ctx.nbytes))

x_cur=np.fromfile(f"{OUT}/x_init.bin",dtype=np.float16).reshape(MS,D)
out=np.zeros((MS,D),dtype=np.float16)
total_time=0

for step in range(3):
    sigma=[1.0,0.667,0.333][step]
    print(f"\nStep {step+1}/3 sigma={sigma:.3f}")

    t_emb=np.fromfile(f"{OUT}/t_step{step}.bin",dtype=np.float16).reshape(M,D)
    lora=np.fromfile(f"{OUT}/lora_step{step}.bin",dtype=np.float16)
    print(f"  lora uploaded: {lora.shape}")
    _lib.dit_write_lora(lora.ctypes.data_as(ctypes.c_void_p))

    t0=time.time()
    ok=_lib.dit_forward_28blocks(
        x_cur.ctypes.data_as(ctypes.c_void_p),
        t_emb.ctypes.data_as(ctypes.c_void_p),
        ctx.ctypes.data_as(ctypes.c_void_p),
        out.ctypes.data_as(ctypes.c_void_p),
        MS,D,M,Nctx,CtxD)
    dt=time.time()-t0
    total_time+=dt
    print(f"  C++: {dt:.3f}s ok={ok}")
    print(f"  mean/std = {out.astype(np.float32).mean():.4f}/{out.astype(np.float32).std():.4f}")
    x_cur=out.copy()

out.astype(np.float16).tofile(f"{OUT}/latent_lora_final.bin")
print(f"\nTotal: {total_time:.1f}s ({total_time/3:.1f}s/step)")
print("DONE — latent_lora_final.bin")
_lib.dit_destroy()

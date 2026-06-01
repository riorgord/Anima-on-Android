"""Validate GPU-side AdaLN: single block with C++ engine vs PyTorch"""
import ctypes, numpy as np, torch, torch.nn.functional as F, time, sys

_lib=ctypes.CDLL("/data/local/tmp/libdit_vk.so")
_lib.dit_init.argtypes=[ctypes.c_char_p,ctypes.c_char_p]; _lib.dit_init.restype=ctypes.c_bool
_lib.dit_record_block_to.argtypes=[ctypes.c_int,ctypes.c_int]; _lib.dit_record_block_to.restype=ctypes.c_bool
_lib.dit_forward.argtypes=[ctypes.c_void_p]*9; _lib.dit_forward.restype=ctypes.c_bool
_lib.dit_write_buf.argtypes=[ctypes.c_int,ctypes.c_void_p,ctypes.c_size_t]; _lib.dit_write_buf.restype=ctypes.c_bool

print("Init C++ engine...")
t0=time.time()
ok=_lib.dit_init(b"/data/local/tmp/diffusion_weights.bin",b"/data/local/tmp")
print(f"  init={ok} ({time.time()-t0:.1f}s)")
if not ok: sys.exit(1)

# Load dump data
OUT="/sdcard/anima_on_android/output/ref_dump"
x_in=np.load(f"{OUT}/x_in.npy")  # [MS, D] fp16
t_emb_in=np.load(f"{OUT}/t_emb.npy")  # [M, D] fp16
ctx=np.load(f"{OUT}/ctx.npy")  # [M, Nctx, CtxD] fp16
MS,D=x_in.shape; M=t_emb_in.shape[0]; Nctx,CtxD=ctx.shape[1],ctx.shape[2]

# Upload inputs
_lib.dit_write_buf(0, x_in.ctypes.data_as(ctypes.c_void_p), x_in.nbytes)
_lib.dit_write_buf(1, t_emb_in.ctypes.data_as(ctypes.c_void_p), t_emb_in.nbytes)
_lib.dit_write_buf(2, ctx.astype(np.float16).ctypes.data_as(ctypes.c_void_p), ctx.nbytes)

print("Recording block 0 (GPU AdaLN)...")
ok=_lib.dit_record_block_to(0, 0)  # block 0 → cmd[0]
print(f"  record={ok}")

out_np=np.zeros((MS,D),dtype=np.float16)
print("Running...")
t0=time.time()
ok=_lib.dit_forward(x_in.ctypes.data_as(ctypes.c_void_p),
    t_emb_in.ctypes.data_as(ctypes.c_void_p),
    ctx.astype(np.float16).ctypes.data_as(ctypes.c_void_p),
    out_np.ctypes.data_as(ctypes.c_void_p), MS, D, M, Nctx, CtxD)
print(f"  forward={ok} ({time.time()-t0:.3f}s)")

# PyTorch reference (same computation as C++: self+cross+mlp)
print("Computing reference...")
sd=torch.load("/sdcard/anima_on_android/models/diffusion_weights_fp16.pt",weights_only=True)
pfx="blocks.0."
x_t=torch.from_numpy(x_in.astype(np.float32))
t_emb_t=torch.from_numpy(t_emb_in.astype(np.float32))
ctx_t=torch.from_numpy(ctx.astype(np.float32))

def adaln_ref(emb, w1, w2):
    h=F.silu(emb); h=F.linear(h,w1.float()); h=F.linear(h,w2.float())
    sh,sc,ga=torch.chunk(h,3,-1)
    sc=sc+1.0; S=MS//M
    return (sc.repeat_interleave(S,0), sh.repeat_interleave(S,0), ga.repeat_interleave(S,0))

sc_s,sh_s,ga_s=adaln_ref(t_emb_t,
    sd[pfx+"adaln_modulation_self_attn.1.weight"],
    sd[pfx+"adaln_modulation_self_attn.2.weight"])
sc_c,sh_c,ga_c=adaln_ref(t_emb_t,
    sd[pfx+"adaln_modulation_cross_attn.1.weight"],
    sd[pfx+"adaln_modulation_cross_attn.2.weight"])
sc_m,sh_m,ga_m=adaln_ref(t_emb_t,
    sd[pfx+"adaln_modulation_mlp.1.weight"],
    sd[pfx+"adaln_modulation_mlp.2.weight"])

# Self-attn
ln=F.layer_norm(x_t,(D,),weight=None,bias=None,eps=1e-6); mod=ln*sc_s+sh_s
q=F.linear(mod,sd[pfx+"self_attn.q_proj.weight"].float())
v=F.linear(mod,sd[pfx+"self_attn.v_proj.weight"].float())
q=F.rms_norm(q.reshape(MS*16,128),(128,),weight=sd[pfx+"self_attn.q_norm.weight"].float(),eps=1e-6).reshape(MS,D)
o=F.linear(v,sd[pfx+"self_attn.output_proj.weight"].float()); x_t=x_t+ga_s*o

# Cross-attn
ln=F.layer_norm(x_t,(D,),weight=None,bias=None,eps=1e-6); mod=ln*sc_c+sh_c
q=F.linear(mod,sd[pfx+"self_attn.q_proj.weight"].float())
k=F.linear(ctx_t.reshape(1024,1024),sd[pfx+"cross_attn.k_proj.weight"].float())
v=F.linear(ctx_t.reshape(1024,1024),sd[pfx+"cross_attn.v_proj.weight"].float())
q=F.rms_norm(q.reshape(MS*16,128),(128,),weight=sd[pfx+"self_attn.q_norm.weight"].float(),eps=1e-6).reshape(MS,D)
k=F.rms_norm(k.reshape(1024*16,128),(128,),weight=sd[pfx+"cross_attn.k_norm.weight"].float(),eps=1e-6).reshape(1024,D)
o=F.linear(v[:MS],sd[pfx+"cross_attn.output_proj.weight"].float()); x_t=x_t+ga_c*o

# MLP
ln=F.layer_norm(x_t,(D,),weight=None,bias=None,eps=1e-6); mod=ln*sc_m+sh_m
h=F.linear(mod,sd[pfx+"mlp.layer1.weight"].float()); h=F.silu(h)
fc2=F.linear(h,sd[pfx+"mlp.layer2.weight"].float()); x_t=x_t+ga_m*fc2

ref=x_t.half().numpy(); del sd

err=np.abs(out_np.astype(np.float32)-ref.astype(np.float32)).max()
print(f"max_err={err:.5f}")
print(f"C++ mean/std={out_np.astype(np.float32).mean():.4f}/{out_np.astype(np.float32).std():.4f}")
print(f"REF mean/std={ref.astype(np.float32).mean():.4f}/{ref.astype(np.float32).std():.4f}")
nz=(out_np!=0).sum()
print(f"non-zero: {nz}/{out_np.size} ({100*nz/out_np.size:.1f}%)")
print("PASS" if err<10 else "FAIL")
_lib.dit_destroy()

"""Test block 0 with lora: CPU reference (with lora) vs C++ engine (pre-computed bcBuf)"""
import ctypes, numpy as np, torch, sys, time
_lib=ctypes.CDLL("/data/local/tmp/libdit_vk.so")
_lib.dit_init.argtypes=[ctypes.c_char_p,ctypes.c_char_p]; _lib.dit_init.restype=ctypes.c_bool
_lib.dit_record_block_full.argtypes=[ctypes.c_int]; _lib.dit_record_block_full.restype=ctypes.c_bool
_lib.dit_write_buf.argtypes=[ctypes.c_int,ctypes.c_void_p,ctypes.c_size_t]; _lib.dit_write_buf.restype=ctypes.c_bool
_lib.dit_forward.argtypes=[ctypes.c_void_p]*5 + [ctypes.c_int]*4; _lib.dit_forward.restype=ctypes.c_bool

print("Init...")
t0=time.time()
ok=_lib.dit_init(b"/data/local/tmp/diffusion_weights.bin",b"/data/local/tmp")
print(f"  init={ok} ({time.time()-t0:.1f}s)")
if not ok: sys.exit(1)

MS,D,M=512,2048,2; S=MS//M; torch.manual_seed(42)
x_t=torch.randn(MS,D,dtype=torch.float32)
t_emb=torch.randn(M,D,dtype=torch.float32)

# Load lora for reference
import os; import torch.nn.functional as F
lora_np = np.fromfile("/sdcard/anima_on_android/output/lora_step0.bin", dtype=np.float16).reshape(3, M, D)
lora_t = torch.from_numpy(lora_np.astype(np.float32))  # [3, M, D]
lora_cat = torch.cat([lora_t[0], lora_t[1], lora_t[2]], dim=-1)  # [M, 3D]
print(f"lora loaded: {lora_np.shape}")

# Load weights
print("Loading weights...")
sd=torch.load("/sdcard/anima_on_android/models/diffusion_weights_fp16.pt",weights_only=True,map_location="cpu")
pfx="blocks.0."
w_q=sd[pfx+"self_attn.q_proj.weight"]; w_v=sd[pfx+"self_attn.v_proj.weight"]
w_o=sd[pfx+"self_attn.output_proj.weight"]
w_qn=sd[pfx+"self_attn.q_norm.weight"].float()
wl1_s=sd[pfx+"adaln_modulation_self_attn.1.weight"].float()
wl2_s=sd[pfx+"adaln_modulation_self_attn.2.weight"].float()
w_l1=sd[pfx+"mlp.layer1.weight"]; w_l2=sd[pfx+"mlp.layer2.weight"]
wl1_m=sd[pfx+"adaln_modulation_mlp.1.weight"].float()
wl2_m=sd[pfx+"adaln_modulation_mlp.2.weight"].float()
del sd

def compute_adaln_with_lora(emb, w1, w2, lora_3d):
    """GPU adaln_gpu equivalent: SiLU→GEMM(D,256)→GEMM(256,3D)→+lora→scale+1→broadcast"""
    h=F.silu(emb.float())
    h=F.linear(h,w1)
    h=F.linear(h,w2)
    h = h + lora_3d      # <-- NEW: external lora addition
    shift,scale,gate=torch.chunk(h,3,dim=-1)
    scale_p1=scale+1.0
    scale_b=scale_p1.repeat_interleave(S,0)
    shift_b=shift.repeat_interleave(S,0)
    gate_b=gate.repeat_interleave(S,0)
    return scale_b, shift_b, gate_b

# Compute AdaLN with lora
scale_s, shift_s, gate_s = compute_adaln_with_lora(t_emb, wl1_s, wl2_s, lora_cat)
scale_m, shift_m, gate_m = compute_adaln_with_lora(t_emb, wl1_m, wl2_m, lora_cat)

# Upload AdaLN to bcBuf
n_elem=MS*D
concat=np.zeros(n_elem*6,dtype=np.uint16)
def pack(tensor, slot):
    concat[slot*n_elem:(slot+1)*n_elem]=tensor.numpy().astype(np.float16).ravel().view(np.uint16)
pack(scale_s,0); pack(shift_s,1); pack(gate_s,2)
pack(scale_m,3); pack(shift_m,4); pack(gate_m,5)
_lib.dit_write_buf(4, concat.ctypes.data_as(ctypes.c_void_p), concat.nbytes)

x_np=x_t.numpy().astype(np.float16)
_lib.dit_write_buf(0, x_np.ctypes.data_as(ctypes.c_void_p), x_np.nbytes)

print("Recording full block 0...")
ok=_lib.dit_record_block_full(0)
print(f"  record={ok}")

print("Running...")
out_np=np.zeros((MS,D),dtype=np.float16)
tn=np.zeros((M,D),dtype=np.float16); cn=np.zeros((M,512,1024),dtype=np.float16)
t0=time.time()
ok=_lib.dit_forward(x_np.ctypes.data_as(ctypes.c_void_p),tn.ctypes.data_as(ctypes.c_void_p),
    cn.ctypes.data_as(ctypes.c_void_p),out_np.ctypes.data_as(ctypes.c_void_p),MS,D,M,512,1024)
print(f"  forward={ok} ({time.time()-t0:.3f}s)")

# CPU reference with lora
print("Computing reference (with lora)...")
ln=F.layer_norm(x_t,(D,),weight=None,bias=None,eps=1e-6)
mod=ln*scale_s+shift_s
q=F.linear(mod,w_q.float()); v=F.linear(mod,w_v.float())
q=F.rms_norm(q.reshape(MS*16,128),(128,),weight=w_qn,eps=1e-6).reshape(MS,D)
o=F.linear(v,w_o.float())
x1=x_t+gate_s*o

ln2=F.layer_norm(x1,(D,),weight=None,bias=None,eps=1e-6)
mod2=ln2*scale_m+shift_m
h=F.linear(mod2,w_l1.float()); h=F.silu(h)
fc2=F.linear(h,w_l2.float())
ref=x1+gate_m*fc2

ref_np=ref.half().numpy()
err=np.abs(out_np.astype(np.float32)-ref_np.astype(np.float32)).max()
print(f"max_err={err:.5f}")
print(f"out mean/std={out_np.astype(np.float32).mean():.4f}/{out_np.astype(np.float32).std():.4f}")
print(f"ref mean/std={ref_np.astype(np.float32).mean():.4f}/{ref_np.astype(np.float32).std():.4f}")
nz=(out_np!=0).sum()
print(f"non-zero: {nz}/{out_np.size} ({100*nz/out_np.size:.1f}%)")

if err<5.0:
    print("PASS — C++ engine output matches CPU reference (with lora)")
else:
    print("FAIL")

_lib.dit_destroy()
print("DONE")

"""Debug: compare Vulkan output vs PyTorch element-by-element"""
import ctypes, numpy as np, torch, sys
_lib=ctypes.CDLL("/data/local/tmp/libdit_vk.so")
_lib.dit_init.argtypes=[ctypes.c_char_p,ctypes.c_char_p]; _lib.dit_init.restype=ctypes.c_bool
_lib.dit_write_buf.argtypes=[ctypes.c_int,ctypes.c_void_p,ctypes.c_size_t]; _lib.dit_write_buf.restype=ctypes.c_bool
_lib.dit_forward.argtypes=[ctypes.c_void_p]*9; _lib.dit_forward.restype=ctypes.c_bool
_lib.dit_record_self_attn_full.argtypes=[ctypes.c_int]; _lib.dit_record_self_attn_full.restype=ctypes.c_bool
ok=_lib.dit_init(b"/data/local/tmp/diffusion_weights.bin",b"/data/local/tmp")
if not ok: sys.exit(1)

MS,D,M=512,2048,2; torch.manual_seed(1234); x_t=torch.randn(MS,D,dtype=torch.float32)
sd=torch.load("/sdcard/anima_on_android/models/diffusion_weights_fp16.pt",weights_only=True,map_location="cpu")
pfx="blocks.0."
w_q=sd[pfx+"self_attn.q_proj.weight"]; w_v=sd[pfx+"self_attn.v_proj.weight"]
w_o=sd[pfx+"self_attn.output_proj.weight"]; w_qn=sd[pfx+"self_attn.q_norm.weight"].float()
w_l1=sd[pfx+"adaln_modulation_self_attn.1.weight"].float()
w_l2=sd[pfx+"adaln_modulation_self_attn.2.weight"].float()
del sd
t_emb=torch.randn(M,D,dtype=torch.float32)
import torch.nn.functional as F
h=F.silu(t_emb); h=F.linear(h,w_l1); h=F.linear(h,w_l2)
shift,scale,gate=torch.chunk(h,3,dim=-1)
scale_p1=scale+1.0
scale_b=scale_p1.repeat_interleave(MS//M,0); shift_b=shift.repeat_interleave(MS//M,0)
gate_b=gate.repeat_interleave(MS//M,0)
n_elem=MS*D; concat=np.zeros(n_elem*3,dtype=np.uint16)
concat[0*n_elem:1*n_elem]=scale_b.numpy().astype(np.float16).ravel().view(np.uint16)
concat[1*n_elem:2*n_elem]=shift_b.numpy().astype(np.float16).ravel().view(np.uint16)
concat[2*n_elem:3*n_elem]=gate_b.numpy().astype(np.float16).ravel().view(np.uint16)
_lib.dit_write_buf(4,concat.ctypes.data_as(ctypes.c_void_p),concat.nbytes)
_lib.dit_write_buf(0,x_t.numpy().astype(np.float16).ctypes.data_as(ctypes.c_void_p),n_elem*2)
ok=_lib.dit_record_self_attn_full(0)
out_np=np.zeros((MS,D),dtype=np.float16); tn=np.zeros((M,D),dtype=np.float16); cn=np.zeros((M,512,1024),dtype=np.float16)
ok=_lib.dit_forward(x_t.numpy().astype(np.float16).ctypes.data_as(ctypes.c_void_p),tn.ctypes.data_as(ctypes.c_void_p),cn.ctypes.data_as(ctypes.c_void_p),out_np.ctypes.data_as(ctypes.c_void_p),MS,D,M,512,1024)
# Reference
ln=F.layer_norm(x_t,(D,),weight=None,bias=None,eps=1e-6)
mod=ln*scale_b+shift_b
q=F.linear(mod,w_q.float()); v=F.linear(mod,w_v.float())
q_rs=q.reshape(MS*16,128)
q_hat=F.rms_norm(q_rs,(128,),weight=w_qn,eps=1e-6).reshape(MS,D)
o=F.linear(v,w_o.float())
ref=x_t+gate_b*o
ref_np=ref.half().numpy()
diff=np.abs(out_np.astype(np.float32)-ref_np.astype(np.float32))
print(f"max_err = {diff.max():.5f}")
print(f"out[:3,:5] =\n{out_np.astype(np.float32)[:3,:5]}")
print(f"ref[:3,:5] =\n{ref_np.astype(np.float32)[:3,:5]}")
print(f"diff[:3,:5] =\n{diff[:3,:5]}")
wi=np.unravel_index(diff.argmax(),diff.shape)
print(f"worst at {wi}: out={out_np.astype(np.float32)[wi]:.4f} ref={ref_np.astype(np.float32)[wi]:.4f}")
print(f"out mean/std={out_np.astype(np.float32).mean():.4f}/{out_np.astype(np.float32).std():.4f}")
print(f"ref mean/std={ref_np.astype(np.float32).mean():.4f}/{ref_np.astype(np.float32).std():.4f}")
_lib.dit_destroy()

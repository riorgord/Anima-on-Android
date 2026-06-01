"""Compare bcBuf (AdaLN parameters) C++ vs PyTorch.
Uses the NORMALIZED t_emb (what blocks actually receive)."""
import numpy as np, torch, torch.nn.functional as F, safetensors.torch
CMP='/mnt/d/AI/anima_phone/output/cmp_v2'
SF='/mnt/d/AI/手坤的anima/models/diffusion_model/anima-base-v1.0.safetensors'
M,S,D,NH,HD=2,256,2048,16,128; MS=M*S

b0_bcbuf=np.load(f'{CMP}/b0_bcbuf.npy').reshape(9,MS,D)
b0_lora=np.load(f'{CMP}/b0_lora.npy').reshape(M,3*D)

# Use CAPTURED t_emb (post-norm, same as what blocks receive)
b0_temb = torch.from_numpy(np.load(f'{CMP}/b0_temb.npy').reshape(M, D))

st=safetensors.torch.load_file(SF,device='cpu')
sd={k[4:] if k.startswith('net.') else k:v.to(torch.float32) for k,v in st.items()}
del st

def pt_adaln(emb,lora,module):
    w0=sd[f'blocks.0.adaln_modulation_{module}.1.weight']
    w2=sd[f'blocks.0.adaln_modulation_{module}.2.weight']
    h=F.silu(F.linear(emb,w0));out=F.linear(h,w2)+lora
    s,sc,g=out.chunk(3,dim=-1);return s,sc+1.0,g

S_per=MS//M
def bcast(t):
    return t.repeat_interleave(S_per,dim=0).numpy()

print("=== bcBuf comparison (C++ vs PT, norm'd t_emb) ===")
for idx, name, pt_val in [
    (0,"SA scale+1", bcast(pt_adaln(b0_temb,b0_lora,'self_attn')[1])),
    (1,"SA shift",   bcast(pt_adaln(b0_temb,b0_lora,'self_attn')[0])),
    (2,"SA gate",    bcast(pt_adaln(b0_temb,b0_lora,'self_attn')[2])),
    (3,"CX scale+1", bcast(pt_adaln(b0_temb,b0_lora,'cross_attn')[1])),
    (4,"CX shift",   bcast(pt_adaln(b0_temb,b0_lora,'cross_attn')[0])),
    (5,"CX gate",    bcast(pt_adaln(b0_temb,b0_lora,'cross_attn')[2])),
    (6,"MLP scale+1",bcast(pt_adaln(b0_temb,b0_lora,'mlp')[1])),
    (7,"MLP shift",  bcast(pt_adaln(b0_temb,b0_lora,'mlp')[0])),
    (8,"MLP gate",   bcast(pt_adaln(b0_temb,b0_lora,'mlp')[2])),
]:
    cpp=b0_bcbuf[idx].reshape(MS,D)
    diff=np.abs(cpp-pt_val)
    print(f"  {name:15s}: max_err={diff.max():.6f}, C++[{cpp.min():.4f},{cpp.max():.4f}], PT[{pt_val.min():.4f},{pt_val.max():.4f}]")

first_err=np.abs(b0_bcbuf[0].reshape(MS,D)-bcast(pt_adaln(b0_temb,b0_lora,'self_attn')[1])).max()
if first_err<1e-3:
    print(f"\nbcBuf MATCHES PT! (max_err={first_err:.6f})")
else:
    print(f"\nbcBuf MISMATCH (max_err={first_err:.6f})!")

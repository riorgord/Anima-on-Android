"""Compare bcBuf (AdaLN parameters) C++ vs PyTorch."""
import numpy as np, torch, torch.nn.functional as F, safetensors.torch
CMP='/mnt/d/AI/anima_phone/output/cmp_v2'
SF='/mnt/d/AI/手坤的anima/models/diffusion_model/anima-base-v1.0.safetensors'
M,S,D,NH,HD=2,256,2048,16,128; MS=M*S

b0_bcbuf=np.load(f'{CMP}/b0_bcbuf.npy').reshape(9,MS,D)
b0_nbuf=np.load(f'{CMP}/b0_nbuf.npy').reshape(MS,D)
b0_x=np.load(f'{CMP}/b0_x.npy').reshape(MS,D)
b0_lora=np.load(f'{CMP}/b0_lora.npy').reshape(M,3*D)

st=safetensors.torch.load_file(SF,device='cpu')
sd={k[4:] if k.startswith('net.') else k:v.to(torch.float32) for k,v in st.items()}
del st

# PT t_embed
sigma=1.0;half=D//2
emb_pt=torch.zeros(M,D)
for m in range(M):
    for i in range(half):
        v=sigma*np.exp(-np.log(10000.0)*i/half)
        emb_pt[m,i]=np.sin(v); emb_pt[m,half+i]=np.cos(v)

def pt_adaln(emb,lora,module):
    w0=sd[f'blocks.0.adaln_modulation_{module}.1.weight']
    w2=sd[f'blocks.0.adaln_modulation_{module}.2.weight']
    h=F.silu(F.linear(emb,w0));out=F.linear(h,w2)+lora
    s,sc,g=out.chunk(3,dim=-1);return s,sc+1.0,g

# PT bcBuf layout: [SA_scale, SA_shift, SA_gate, CX_scale, CX_shift, CX_gate, MLP_scale, MLP_shift, MLP_gate]
# C++ bcBuf layout: slot 0=SA_scale+1, 1=SA_shift, 2=SA_gate, 3=CX_scale+1, 4=CX_shift, 5=CX_gate, 6=MLP_scale+1, 7=MLP_shift, 8=MLP_gate

c_bc = b0_bcbuf  # [9, MS, D]

# PT bcBuf computation
shift_s,scale_s,gate_s=pt_adaln(emb_pt,b0_lora,'self_attn')
shift_c,scale_c,gate_c=pt_adaln(emb_pt,b0_lora,'cross_attn')
shift_m,scale_m,gate_m=pt_adaln(emb_pt,b0_lora,'mlp')

# Broadcast: PT shape is [M, D], C++ shape is [9, MS, D] with MS/M=S=256 repeats
S_per=MS//M
def bcast(t):
    return t.repeat_interleave(S_per,dim=0).numpy()  # [MS, D]

print("=== bcBuf comparison (C++ vs PT) ===")
for idx, name, pt_val in [
    (0,"SA scale+1", bcast(scale_s)),
    (1,"SA shift",   bcast(shift_s)),
    (2,"SA gate",    bcast(gate_s)),
    (3,"CX scale+1", bcast(scale_c)),
    (4,"CX shift",   bcast(shift_c)),
    (5,"CX gate",    bcast(gate_c)),
    (6,"MLP scale+1",bcast(scale_m)),
    (7,"MLP shift",  bcast(shift_m)),
    (8,"MLP gate",   bcast(gate_m)),
]:
    cpp=c_bc[idx].reshape(MS,D)
    diff=np.abs(cpp-pt_val)
    print(f"  {name:15s}: max_err={diff.max():.6f}, C++[{cpp.min():.4f},{cpp.max():.4f}], PT[{pt_val.min():.4f},{pt_val.max():.4f}]")

# Check if bcBuf matches
first_err=np.abs(c_bc[0].reshape(MS,D)-bcast(scale_s)).max()
if first_err<1e-3:
    print("\nbcBuf MATCHES PT!")
    print("  -> LN or scale_shift shader is broken")
else:
    print(f"\nbcBuf MISMATCH (max_err={first_err:.6f})!")
    print("  -> AdaLN computation (SiLU/GEMM/lora-add) is broken on GPU")

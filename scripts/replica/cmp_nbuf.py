"""Compare g_nBuf (LN+AdaLN output) C++ vs PyTorch."""
import numpy as np, torch, torch.nn.functional as F, safetensors.torch
CMP='/mnt/d/AI/anima_phone/output/cmp_v2'
SF='/mnt/d/AI/手坤的anima/models/diffusion_model/anima-base-v1.0.safetensors'
M,S,D,NH,HD=2,256,2048,16,128; MS=M*S

b0_x=torch.from_numpy(np.load(f'{CMP}/b0_x.npy').reshape(MS,D))
b0_nbuf=torch.from_numpy(np.load(f'{CMP}/b0_nbuf.npy').reshape(MS,D))
b0_q=torch.from_numpy(np.load(f'{CMP}/b0_q.npy').reshape(MS*NH,HD))
b0_lora=torch.from_numpy(np.load(f'{CMP}/b0_lora.npy').reshape(M,3*D))

st=safetensors.torch.load_file(SF,device='cpu')
sd={k[4:] if k.startswith('net.') else k:v.to(torch.float32) for k,v in st.items()}
del st

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

shift_s,scale_s,gate_s=pt_adaln(emb_pt,b0_lora,'self_attn')

# PT: LN -> AdaLN -> g_nBuf
x_5d=b0_x.reshape(M,1,16,16,D)
ln_s=F.layer_norm(x_5d,(D,),None,None,1e-6)
mod_s=ln_s*scale_s.reshape(M,1,1,1,D)+shift_s.reshape(M,1,1,1,D)
mod_f=mod_s.reshape(MS,D)

diff_nbuf=(b0_nbuf-mod_f).abs()
print(f'g_nBuf C++ vs PT: max_err={diff_nbuf.max():.6f}, mean_err={diff_nbuf.mean():.6f}')
print(f'  C++: [{b0_nbuf.min():.4f},{b0_nbuf.max():.4f}]')
print(f'  PT:  [{mod_f.min():.4f},{mod_f.max():.4f}]')

if diff_nbuf.max()<1e-3:
    print('g_nBuf MATCHES PT! Now checking Q...')
    w_q=sd['blocks.0.self_attn.q_proj.weight']
    q_pt=F.linear(mod_f,w_q).reshape(MS*NH,HD)
    diff_q=(b0_q-q_pt).abs()
    print(f'  Q C++ vs PT: max_err={diff_q.max():.6f}')
else:
    print('g_nBuf MISMATCH -> LN or AdaLN is broken on GPU!')
    # Check LN alone
    ln_pt=F.layer_norm(x_5d,(D,),None,None,1e-6).reshape(MS,D)
    print(f'  LN output PT range: [{ln_pt.min():.4f},{ln_pt.max():.4f}]')

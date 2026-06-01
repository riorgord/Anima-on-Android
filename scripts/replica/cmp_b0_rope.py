"""Compare C++ block 0 with PyTorch reference including RoPE."""
import numpy as np, torch, torch.nn.functional as F, safetensors.torch, sys
sys.path.insert(0, '/mnt/d/AI/anima_phone/hybridops/src')
import predict2

CMP = '/mnt/d/AI/anima_phone/output/cmp_v2'
SF = '/mnt/d/AI/手坤的anima/models/diffusion_model/anima-base-v1.0.safetensors'
M, S, D, NH, HD = 2, 256, 2048, 16, 128; MS=M*S; Sp=MS//M

b0_x = torch.from_numpy(np.load(f'{CMP}/b0_x.npy').reshape(MS, D))
b0_sa = torch.from_numpy(np.load(f'{CMP}/b0_sa.npy').reshape(MS, D))
b0_lora = torch.from_numpy(np.load(f'{CMP}/b0_lora.npy').reshape(M, 3*D))

class RefOps:
    Linear=torch.nn.Linear;LayerNorm=torch.nn.LayerNorm;RMSNorm=torch.nn.RMSNorm;GELU=torch.nn.GELU

st=safetensors.torch.load_file(SF,device='cpu')
sd={k[4:] if k.startswith('net.') else k:v.to(torch.float32) for k,v in st.items()}
del st

cfg=dict(max_img_h=240,max_img_w=240,max_frames=128,in_channels=16,out_channels=16,patch_spatial=2,
    patch_temporal=1,concat_padding_mask=True,model_channels=D,num_blocks=28,num_heads=NH,mlp_ratio=4.0,
    crossattn_emb_channels=1024,pos_emb_cls='rope3d',pos_emb_learnable=True,pos_emb_interpolation='crop',
    min_fps=1,max_fps=30,use_adaln_lora=True,adaln_lora_dim=256,rope_h_extrapolation_ratio=4.0,
    rope_w_extrapolation_ratio=4.0,rope_t_extrapolation_ratio=1.0,extra_per_block_abs_pos_emb=False,
    rope_enable_fps_modulation=False)
dit=predict2.MiniTrainDIT(**cfg,device='cpu',dtype=torch.float32,operations=RefOps)
dit.load_state_dict(sd,strict=False);dit.eval()

dummy=torch.zeros(M,1,16,16,D)
rope_pt=dit.pos_embedder(dummy,fps=None,device='cpu',dtype=torch.float32)
print(f'PT RoPE shape: {rope_pt.shape}')

# t_emb
sigma=1.0;half=D//2
emb_pt=torch.zeros(M,D)
for m in range(M):
    for i in range(half):
        v=sigma*np.exp(-np.log(10000.0)*i/half)
        emb_pt[m,i]=np.sin(v); emb_pt[m,half+i]=np.cos(v)

def pt_adaln(emb,lora,block_idx,module):
    w0=sd[f'blocks.{block_idx}.adaln_modulation_{module}.1.weight']
    w2=sd[f'blocks.{block_idx}.adaln_modulation_{module}.2.weight']
    h=F.silu(F.linear(emb,w0)); out=F.linear(h,w2)+lora
    s,sc,g=out.chunk(3,dim=-1); return s,sc+1.0,g

shift_s,scale_s,gate_s=pt_adaln(emb_pt,b0_lora,0,'self_attn')

x_5d=b0_x.reshape(M,1,16,16,D)
ln_s=F.layer_norm(x_5d,(D,),None,None,1e-6)
mod_s=ln_s*scale_s.reshape(M,1,1,1,D)+shift_s.reshape(M,1,1,1,D)
mod_f=mod_s.reshape(MS,D)

w_q=sd['blocks.0.self_attn.q_proj.weight']; w_k=sd['blocks.0.self_attn.k_proj.weight']
w_v=sd['blocks.0.self_attn.v_proj.weight']; w_o=sd['blocks.0.self_attn.output_proj.weight']
w_qn=sd['blocks.0.self_attn.q_norm.weight']; w_kn=sd['blocks.0.self_attn.k_norm.weight']

q=F.linear(mod_f,w_q);k=F.linear(mod_f,w_k);v=F.linear(mod_f,w_v)
q_n=F.rms_norm(q.reshape(MS*NH,HD),(HD,),w_qn,1e-6).reshape(MS,NH,HD)
k_n=F.rms_norm(k.reshape(MS*NH,HD),(HD,),w_kn,1e-6).reshape(MS,NH,HD)

# Apply PT RoPE
roped_q=np.zeros((MS,NH,HD),dtype=np.float32);roped_k=np.zeros((MS,NH,HD),dtype=np.float32)
for mb in range(M):
    qi=q_n[mb*Sp:(mb+1)*Sp]; ki=k_n[mb*Sp:(mb+1)*Sp]
    rqi=predict2.apply_rotary_pos_emb(qi.unsqueeze(0).unsqueeze(0), rope_pt.unsqueeze(0).unsqueeze(0))[0,0]
    rki=predict2.apply_rotary_pos_emb(ki.unsqueeze(0).unsqueeze(0), rope_pt.unsqueeze(0).unsqueeze(0))[0,0]
    roped_q[mb*Sp:(mb+1)*Sp]=rqi; roped_k[mb*Sp:(mb+1)*Sp]=rki

roped_q_f=torch.from_numpy(roped_q.reshape(MS*NH,HD))
roped_k_f=torch.from_numpy(roped_k.reshape(MS*NH,HD))

attn_o=torch.zeros(MS,D); sc=1.0/np.sqrt(HD)
for mb in range(M):
    qi=roped_q_f[mb*Sp*NH:(mb+1)*Sp*NH].reshape(Sp,NH,HD).permute(1,0,2)
    ki=roped_k_f[mb*Sp*NH:(mb+1)*Sp*NH].reshape(Sp,NH,HD).permute(1,0,2)
    vi=v.reshape(MS,NH,HD)[mb*Sp:(mb+1)*Sp].permute(1,0,2)
    scores=torch.bmm(qi,ki.transpose(1,2))*sc
    aw=F.softmax(scores,dim=-1)
    ao=torch.bmm(aw,vi).permute(1,0,2).reshape(Sp,D)
    attn_o[mb*Sp:mb*Sp+Sp]=ao

oproj=F.linear(attn_o,w_o)
sa_pt=b0_x+gate_s.repeat_interleave(Sp,dim=0)*oproj

diff_sa=(b0_sa-sa_pt).abs()
print(f'SA C++ vs PT (WITH RoPE): max_err={diff_sa.max():.4f}, mean_err={diff_sa.mean():.4f}')
print(f'PT SA range: [{sa_pt.min():.2f},{sa_pt.max():.2f}]')
print(f'C++ SA range: [{b0_sa.min():.2f},{b0_sa.max():.2f}]')

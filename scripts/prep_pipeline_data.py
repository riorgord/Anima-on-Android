"""WSL: Pre-compute minimal data for phone C++ engine 3-step pipeline.
Output: rope_flat.bin, ctx_stacked.bin, x_init.bin, t_step{0,1,2}.bin"""
import sys; sys.path.insert(0,"/mnt/d/AI/anima_phone/src")
import numpy as np, torch, os
import torch.nn.functional as F
from safetensors import safe_open

MS,D,M=512,2048,2; S=MS//M; NH=16; HD=128; H=16; W=16; T=1
OUT="/mnt/d/AI/anima_phone/output"
os.makedirs(OUT,exist_ok=True)

# ============================================================
# RoPE freqs (numpy — no model needed)
# ============================================================
def compute_rope_freqs():
    dim_h=HD//6*2; dim_w=dim_h; dim_t=HD-2*dim_h
    h_ntk=4.0**(dim_h/(dim_h-2)); w_ntk=4.0**(dim_w/(dim_w-2)); t_ntk=1.0**(dim_t/(dim_t-2))
    h_theta=10000.0*h_ntk; w_theta=10000.0*w_ntk; t_theta=10000.0*t_ntk
    dh=np.arange(0,dim_h,2,dtype=np.float32)[:dim_h//2]/dim_h
    dw=np.arange(0,dim_w,2,dtype=np.float32)[:dim_w//2]/dim_w
    dt=np.arange(0,dim_t,2,dtype=np.float32)[:dim_t//2]/dim_t
    hf=1.0/(h_theta**dh); wf=1.0/(w_theta**dw); tf=1.0/(t_theta**dt)
    sh=np.arange(H,dtype=np.float32); sw=np.arange(W,dtype=np.float32); st=np.arange(T,dtype=np.float32)
    hh=np.outer(sh,hf); hw=np.outer(sw,wf); ht=np.outer(st,tf)
    def emb(x): return np.stack([np.cos(x),-np.sin(x),np.sin(x),np.cos(x)],-1)
    eh=emb(hh); ew=emb(hw); et=emb(ht)
    ete=np.tile(et.reshape(T,1,1,dim_t//2,4),(1,H,W,1,1))
    ehe=np.tile(eh.reshape(1,H,1,dim_h//2,4),(T,1,W,1,1))
    ewe=np.tile(ew.reshape(1,1,W,dim_w//2,4),(T,H,1,1,1))
    em=np.concatenate([ete,ehe,ewe],-2)
    return em.reshape(T*H*W,HD//2,2,2).astype(np.float16)

rope_np=compute_rope_freqs()
rope_flat=np.zeros((MS*NH,HD//2,4),dtype=np.float16)
for b in range(M):
    for p in range(S):
        for h in range(NH):
            idx=(b*S+p)*NH+h
            rope_flat[idx,:,0]=rope_np[p,:,0,0]
            rope_flat[idx,:,1]=rope_np[p,:,0,1]
            rope_flat[idx,:,2]=rope_np[p,:,1,0]
            rope_flat[idx,:,3]=rope_np[p,:,1,1]
rope_flat.tofile(f"{OUT}/rope_flat.bin")
print(f"RoPE: {rope_flat.nbytes/1024:.0f}KB")

# ============================================================
# Load weights (just t_embedder, x_embedder)
# ============================================================
print("Loading weights...")
sd={}
with safe_open("/mnt/d/AI/手坤的anima/models/diffusion_model/anima-base-v1.0.safetensors",
               framework="pt",device="cpu") as f:
    for key in f.keys():
        sd[key[4:] if key.startswith("net.") else key]=f.get_tensor(key).float()
t_w1=sd["t_embedder.1.linear_1.weight"]; t_w2=sd["t_embedder.1.linear_2.weight"]
t_ln_w=sd["t_embedding_norm.weight"]
pe_w=sd["x_embedder.proj.1.weight"]  # [2048, 68]

# ============================================================
# Initial latent + x_embedder (PatchEmbed: Rearrange + Linear)
# ============================================================
torch.manual_seed(6666)
latent=torch.randn(1,16,32,32,dtype=torch.float32)
x_pad=torch.cat([latent,torch.zeros(1,1,32,32)],dim=1).unsqueeze(2).repeat(2,1,1,1,1)  # [2,17,1,32,32]
# Rearrange: "b c t (h m) (w n) -> b t h w (c m n)" with m=2,n=2
B_,C_,T_,H_,W_=x_pad.shape
x_rearr=x_pad.reshape(B_,C_,T_,H_//2,2,W_//2,2).permute(0,2,3,5,1,4,6).reshape(B_,T_,H_//2,W_//2,-1)  # [2,1,16,16,68]
x_flat=F.linear(x_rearr.float(),pe_w).reshape(MS,D)  # [2,1,16,16,2048] → [512,2048]
x_flat.numpy().astype(np.float16).tofile(f"{OUT}/x_init.bin")
print(f"x_flat: {x_flat.shape}")

# ============================================================
# Contexts
# ============================================================
ctx_cond=torch.load("/mnt/d/AI/anima_phone/models/context_cond.pt",weights_only=True,map_location="cpu").float()
ctx_uncond=torch.load("/mnt/d/AI/anima_phone/models/context_uncond.pt",weights_only=True,map_location="cpu").float()
ctx=torch.cat([ctx_cond.unsqueeze(0),ctx_uncond.unsqueeze(0)],dim=0)
ctx.numpy().astype(np.float16).tofile(f"{OUT}/ctx_stacked.bin")
print(f"ctx: {ctx.shape}")

# ============================================================
# t_emb for each sigma
# ============================================================
def make_t_emb(sigma_val):
    sigma=torch.tensor([sigma_val,sigma_val]).unsqueeze(1)
    half_dim=D//2
    exponent=-np.log(10000)*np.arange(half_dim,dtype=np.float32)/half_dim
    emb_val=sigma.float().numpy()*np.exp(exponent)
    emb_sincos=np.concatenate([np.cos(emb_val),np.sin(emb_val)],-1)
    t_input=torch.from_numpy(emb_sincos).float()
    _=F.linear(F.silu(F.linear(t_input,t_w1)),t_w2)
    return F.rms_norm(t_input.squeeze(1),(D,),weight=t_ln_w,eps=1e-6).numpy().astype(np.float16)

for step,s in enumerate([1.0,0.667,0.333]):
    te=make_t_emb(s)
    te.tofile(f"{OUT}/t_step{step}.bin")
    print(f"t_step{step} (sigma={s:.3f}): {te.shape}")

print("DONE — all data ready for phone")

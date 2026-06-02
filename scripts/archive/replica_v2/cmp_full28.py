"""Full 28-block PT forward pass, compare with C++ block 27 output."""
import numpy as np, torch, torch.nn.functional as F, safetensors.torch, os

CMP = '/mnt/d/AI/anima_phone/output/cmp_v2'
SF  = '/mnt/d/AI/手坤的anima/models/diffusion_model/anima-base-v1.0.safetensors'
CTX_DIR = '/mnt/d/AI/anima_phone/output'
M, S, D, NH, HD = 2, 256, 2048, 16, 128
Nctx, CtxD = 512, 1024
MS = M*S; S_per = MS//M; ph = MS*NH; ph_cross = M*Nctx*NH; MS_kv = M*Nctx

print("Loading captures...")
b0_x = torch.from_numpy(np.load(f'{CMP}/b0_x.npy').reshape(MS, D))
b0_temb = torch.from_numpy(np.load(f'{CMP}/b0_temb.npy').reshape(M, D))
b0_lora = torch.from_numpy(np.load(f'{CMP}/b0_lora.npy').reshape(M, 3*D))

ctx_cond  = torch.load(f'{CTX_DIR}/context_cond.pt', weights_only=True).float()
ctx_uncond = torch.load(f'{CTX_DIR}/context_uncond.pt', weights_only=True).float()
ctx = torch.stack([ctx_cond.reshape(1, Nctx, CtxD)[0], ctx_uncond.reshape(1, Nctx, CtxD)[0]], dim=0)
print(f"  x: {b0_x.shape}, t_emb: {b0_temb.shape}, lora: {b0_lora.shape}, ctx: {ctx.shape}")

print("Loading weights...")
st = safetensors.torch.load_file(SF, device='cpu')
sd = {k[4:] if k.startswith('net.') else k: v.to(torch.float32) for k, v in st.items()}
del st

# ── RoPE frequencies ──
def make_rope_freqs():
    dim_h, dim_w, dim_t = 42, 42, 44; half_dim = 64
    h_ntk=4.0**(dim_h/(dim_h-2)); w_ntk=4.0**(dim_w/(dim_w-2)); t_ntk=1.0**(dim_t/(dim_t-2))
    h_theta=10000.0*h_ntk; w_theta=10000.0*w_ntk; t_theta=10000.0*t_ntk
    H_grid=int(np.sqrt(S))
    pos_freqs=np.zeros((S,half_dim,4),dtype=np.float32)
    for p in range(S):
        h_idx=p//H_grid; w_idx=p%H_grid
        for j in range(half_dim):
            if j<dim_t//2:
                freq=1.0/(t_theta**(2.0*j/dim_t)); a=0.0; cv,sv=np.cos(a),np.sin(a)
            elif j<dim_t//2+dim_h//2:
                jh=j-dim_t//2; freq=1.0/(h_theta**(2.0*jh/dim_h)); a=h_idx*freq; cv,sv=np.cos(a),np.sin(a)
            elif j<dim_t//2+dim_h//2+dim_w//2:
                jw=j-dim_t//2-dim_h//2; freq=1.0/(w_theta**(2.0*jw/dim_w)); a=w_idx*freq; cv,sv=np.cos(a),np.sin(a)
            else:
                jr=j-dim_t//2-dim_h//2-dim_w//2; freq=1.0/(t_theta**(2.0*jr/dim_t)); a=0.0; cv,sv=np.cos(a),np.sin(a)
            pos_freqs[p,j,0]=cv; pos_freqs[p,j,1]=-sv; pos_freqs[p,j,2]=sv; pos_freqs[p,j,3]=cv
    freqs=np.zeros((M*S*NH,half_dim,4),dtype=np.float32)
    for mb in range(M):
        for p in range(S):
            for h in range(NH): freqs[mb*S*NH+p*NH+h]=pos_freqs[p]
    return freqs

rope_freqs = make_rope_freqs()

def apply_rope(t):
    """t: [N, head_dim]"""
    N,hd=t.shape; half=hd//2; out=t.clone()
    for i in range(N):
        for j in range(half):
            x,y=t[i,j].item(),t[i,half+j].item()
            c,ns,s,c2=rope_freqs[i,j,0].item(),rope_freqs[i,j,1].item(),rope_freqs[i,j,2].item(),rope_freqs[i,j,3].item()
            out[i,j]=x*c+y*ns; out[i,half+j]=x*s+y*c2
    return out

# ── AdaLN ──
def adaln(emb, lora, b_idx, module):
    w0 = sd[f'blocks.{b_idx}.adaln_modulation_{module}.1.weight']
    w2 = sd[f'blocks.{b_idx}.adaln_modulation_{module}.2.weight']
    h = F.linear(F.silu(emb), w0); out = F.linear(h, w2) + lora
    return out.chunk(3, dim=-1)

# ── Attention ──
sc = 1.0 / np.sqrt(HD)

def sdpa(q, k, v):
    scores = torch.bmm(q, k.transpose(1,2)) * sc
    return torch.bmm(F.softmax(scores, dim=-1), v)

def self_attn_block(x, emb, lora, b):
    shift, scale, gate = adaln(emb, lora, b, 'self_attn')
    w_q=sd[f'blocks.{b}.self_attn.q_proj.weight']; w_k=sd[f'blocks.{b}.self_attn.k_proj.weight']
    w_v=sd[f'blocks.{b}.self_attn.v_proj.weight']; w_o=sd[f'blocks.{b}.self_attn.output_proj.weight']
    w_qn=sd[f'blocks.{b}.self_attn.q_norm.weight']; w_kn=sd[f'blocks.{b}.self_attn.k_norm.weight']

    x_5d=x.reshape(M,1,16,16,D)
    ln=F.layer_norm(x_5d,(D,),None,None,1e-6).reshape(MS,D)
    mod=ln*(scale+1.0).repeat_interleave(S_per,dim=0)+shift.repeat_interleave(S_per,dim=0)

    q=F.linear(mod,w_q).reshape(ph,HD); k=F.linear(mod,w_k).reshape(ph,HD); v=F.linear(mod,w_v).reshape(ph,HD)
    q=F.rms_norm(q,(HD,),w_qn,1e-6); k=F.rms_norm(k,(HD,),w_kn,1e-6)
    q=apply_rope(q); k=apply_rope(k)

    attn_o=torch.zeros(ph,HD)
    for mb in range(M):
        qi=q[mb*S_per*NH:(mb+1)*S_per*NH].reshape(S_per,NH,HD).permute(1,0,2)
        ki=k[mb*S_per*NH:(mb+1)*S_per*NH].reshape(S_per,NH,HD).permute(1,0,2)
        vi=v.reshape(MS,NH,HD)[mb*S_per:(mb+1)*S_per].permute(1,0,2)
        ao=sdpa(qi,ki,vi).permute(1,0,2).reshape(S_per*NH,HD)
        attn_o[mb*S_per*NH:(mb+1)*S_per*NH]=ao
    oproj=F.linear(attn_o.reshape(MS,D),w_o)
    return x+gate.repeat_interleave(S_per,dim=0)*oproj

def cross_attn_block(x, ctx_in, emb, lora, b):
    shift, scale, gate = adaln(emb, lora, b, 'cross_attn')
    w_q=sd[f'blocks.{b}.cross_attn.q_proj.weight']; w_k=sd[f'blocks.{b}.cross_attn.k_proj.weight']
    w_v=sd[f'blocks.{b}.cross_attn.v_proj.weight']; w_o=sd[f'blocks.{b}.cross_attn.output_proj.weight']
    w_qn=sd[f'blocks.{b}.cross_attn.q_norm.weight']; w_kn=sd[f'blocks.{b}.cross_attn.k_norm.weight']

    x_5d=x.reshape(M,1,16,16,D)
    ln=F.layer_norm(x_5d,(D,),None,None,1e-6).reshape(MS,D)
    mod=ln*(scale+1.0).repeat_interleave(S_per,dim=0)+shift.repeat_interleave(S_per,dim=0)

    q=F.linear(mod,w_q).reshape(ph,HD)
    k=F.linear(ctx_in.reshape(MS_kv,CtxD),w_k).reshape(ph_cross,HD)
    v=F.linear(ctx_in.reshape(MS_kv,CtxD),w_v).reshape(ph_cross,HD)
    q=F.rms_norm(q,(HD,),w_qn,1e-6); k=F.rms_norm(k,(HD,),w_kn,1e-6)

    attn_o=torch.zeros(ph,HD)
    for mb in range(M):
        qi=q[mb*S_per*NH:(mb+1)*S_per*NH].reshape(S_per,NH,HD).permute(1,0,2)
        ki=k[mb*Nctx*NH:(mb+1)*Nctx*NH].reshape(Nctx,NH,HD).permute(1,0,2)
        vi=v.reshape(MS_kv,NH,HD)[mb*Nctx:(mb+1)*Nctx].permute(1,0,2)
        ao=sdpa(qi,ki,vi).permute(1,0,2).reshape(S_per*NH,HD)
        attn_o[mb*S_per*NH:(mb+1)*S_per*NH]=ao
    oproj=F.linear(attn_o.reshape(MS,D),w_o)
    return x+gate.repeat_interleave(S_per,dim=0)*oproj

def mlp_block(x, emb, lora, b):
    shift, scale, gate = adaln(emb, lora, b, 'mlp')
    w_fc1=sd[f'blocks.{b}.mlp.layer1.weight']; w_fc2=sd[f'blocks.{b}.mlp.layer2.weight']

    x_5d=x.reshape(M,1,16,16,D)
    ln=F.layer_norm(x_5d,(D,),None,None,1e-6).reshape(MS,D)
    mod=ln*(scale+1.0).repeat_interleave(S_per,dim=0)+shift.repeat_interleave(S_per,dim=0)
    fc1=F.linear(mod,w_fc1); fc2=F.linear(F.gelu(fc1),w_fc2)
    return x+gate.repeat_interleave(S_per,dim=0)*fc2

# ═══ Run 28 blocks ═══
print("\nRunning 28 blocks...")
x = b0_x.clone()
for b in range(28):
    x = self_attn_block(x, b0_temb, b0_lora, b)
    x = cross_attn_block(x, ctx, b0_temb, b0_lora, b)
    x = mlp_block(x, b0_temb, b0_lora, b)
    if b < 3 or b == 27:
        print(f"  Block {b:2d}: range=[{x.min():.4f},{x.max():.4f}]")

# Compare with C++ block 27 output
b27_cpp = torch.from_numpy(np.fromfile(f'{CMP}/b27_out.bin', dtype=np.float32)).reshape(MS, D)
diff = (x - b27_cpp).abs()
print(f"\n=== Block 27 comparison ===")
print(f"  PT  range: [{x.min():.4f},{x.max():.4f}]")
print(f"  C++ range: [{b27_cpp.min():.4f},{b27_cpp.max():.4f}]")
print(f"  max_err: {diff.max():.6f}")
print(f"  mean_err: {diff.mean():.6f}")
print(f"  NaN in PT: {torch.isnan(x).any().item()}")

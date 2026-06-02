"""Full Block 0 comparison: self-attn (w/RoPE) + cross-attn + MLP vs PT."""
import numpy as np, torch, torch.nn.functional as F, safetensors.torch, math, os

CMP = '/mnt/d/AI/anima_phone/output/cmp_v2'
SF  = '/mnt/d/AI/手坤的anima/models/diffusion_model/anima-base-v1.0.safetensors'
CTX_DIR = '/mnt/d/AI/anima_phone/output'
M, S, D, NH, HD = 2, 256, 2048, 16, 128
Nctx, CtxD = 512, 1024
MS = M*S; S_per = MS//M; ph = MS*NH; ph_cross = M*Nctx*NH

# ── Load captures ──
def load_npy(name, shape=None):
    d = np.load(f'{CMP}/{name}.npy')
    if shape: d = d.reshape(shape)
    return torch.from_numpy(d)

b0_x   = load_npy('b0_x', (MS, D))
b0_q   = load_npy('b0_q', (ph, HD))
b0_k   = load_npy('b0_k', (ph, HD))
b0_v   = load_npy('b0_v', (ph, HD))
b0_qn  = load_npy('b0_qn', (ph, HD))
b0_kn  = load_npy('b0_kn', (ph, HD))
b0_qr  = load_npy('b0_qr', (ph, HD))
b0_kr  = load_npy('b0_kr', (ph, HD))
b0_attn_o = load_npy('b0_attn_o', (ph, HD))
b0_oproj  = load_npy('b0_oproj', (MS, D))
b0_sa  = load_npy('b0_sa', (MS, D))
b0_cx  = load_npy('b0_cx', (MS, D))
b0_mlp = load_npy('b0_mlp', (MS, D))
b0_temb = load_npy('b0_temb', (M, D))
b0_lora = load_npy('b0_lora', (M, 3*D))
b0_bcbuf = load_npy('b0_bcbuf', (9, MS, D))
b0_nbuf  = load_npy('b0_nbuf', (MS, D))

# Load context
ctx_cond  = torch.load(f'{CTX_DIR}/context_cond.pt',  weights_only=True).float()
ctx_uncond = torch.load(f'{CTX_DIR}/context_uncond.pt', weights_only=True).float()
ctx = torch.stack([ctx_cond.reshape(1, Nctx, CtxD)[0], ctx_uncond.reshape(1, Nctx, CtxD)[0]], dim=0)
print(f"Context: shape={ctx.shape}, range=[{ctx.min():.4f},{ctx.max():.4f}]")

# Load weights
st = safetensors.torch.load_file(SF, device='cpu')
sd = {k[4:] if k.startswith('net.') else k: v.to(torch.float32) for k, v in st.items()}
del st

# ── PT adaln — SiLU → Linear → Linear ──
def pt_adaln(emb, lora, module):
    w0 = sd[f'blocks.0.adaln_modulation_{module}.1.weight']
    w2 = sd[f'blocks.0.adaln_modulation_{module}.2.weight']
    h = F.linear(F.silu(emb), w0); out = F.linear(h, w2) + lora
    return out.chunk(3, dim=-1)

shift_s, scale_s, gate_s = pt_adaln(b0_temb, b0_lora, 'self_attn')
shift_c, scale_c, gate_c = pt_adaln(b0_temb, b0_lora, 'cross_attn')
shift_m, scale_m, gate_m = pt_adaln(b0_temb, b0_lora, 'mlp')

# ── RoPE ──
def compute_rope_freqs():
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

def apply_rope(t, freqs):
    N,hd=t.shape; half=hd//2; out=t.clone()
    for i in range(N):
        for j in range(half):
            x,y=t[i,j].item(),t[i,half+j].item()
            c,ns,s,c2=freqs[i,j,0].item(),freqs[i,j,1].item(),freqs[i,j,2].item(),freqs[i,j,3].item()
            out[i,j]=x*c+y*ns; out[i,half+j]=x*s+y*c2
    return out

rope_freqs = compute_rope_freqs()

# ── Helper: scaled dot-product attention ──
def sdpa(q, k, v, scale):
    """q,k,v: [H, S_q, HD], [H, S_kv, HD], [H, S_kv, HD]"""
    scores = torch.bmm(q, k.transpose(1,2)) * scale
    attn_w = F.softmax(scores, dim=-1)
    return torch.bmm(attn_w, v)

# ═══════════════════════════════════════
# SELF-ATTENTION
# ═══════════════════════════════════════
x_5d = b0_x.reshape(M, 1, 16, 16, D)
ln_sa = F.layer_norm(x_5d, (D,), None, None, 1e-6).reshape(MS, D)
mod_sa = ln_sa * (scale_s+1.0).repeat_interleave(S_per, dim=0) + shift_s.repeat_interleave(S_per, dim=0)

w_q=sd['blocks.0.self_attn.q_proj.weight']; w_k=sd['blocks.0.self_attn.k_proj.weight']
w_v=sd['blocks.0.self_attn.v_proj.weight']; w_o=sd['blocks.0.self_attn.output_proj.weight']
w_qn=sd['blocks.0.self_attn.q_norm.weight']; w_kn=sd['blocks.0.self_attn.k_norm.weight']

q_pt = F.linear(mod_sa, w_q).reshape(ph, HD)
k_pt = F.linear(mod_sa, w_k).reshape(ph, HD)
v_pt = F.linear(mod_sa, w_v).reshape(ph, HD)
q_n_pt = F.rms_norm(q_pt, (HD,), w_qn, 1e-6)
k_n_pt = F.rms_norm(k_pt, (HD,), w_kn, 1e-6)
q_r_pt = apply_rope(q_n_pt, rope_freqs)
k_r_pt = apply_rope(k_n_pt, rope_freqs)

sc=1.0/np.sqrt(HD)
attn_o_pt = torch.zeros(ph, HD)
for mb in range(M):
    qi = q_r_pt[mb*S_per*NH:(mb+1)*S_per*NH].reshape(S_per,NH,HD).permute(1,0,2)
    ki = k_r_pt[mb*S_per*NH:(mb+1)*S_per*NH].reshape(S_per,NH,HD).permute(1,0,2)
    vi = v_pt.reshape(MS,NH,HD)[mb*S_per:(mb+1)*S_per].permute(1,0,2)
    ao = sdpa(qi,ki,vi,sc).permute(1,0,2).reshape(S_per*NH,HD)
    attn_o_pt[mb*S_per*NH:(mb+1)*S_per*NH] = ao
oproj_pt = F.linear(attn_o_pt.reshape(MS,D), w_o)
sa_pt = b0_x + gate_s.repeat_interleave(S_per,dim=0) * oproj_pt

print(f"=== Self-attention ===")
print(f"  Attn out: max_err={(b0_attn_o-attn_o_pt).abs().max():.4f}")
print(f"  O_proj:   max_err={(b0_oproj-oproj_pt).abs().max():.4f}")
print(f"  SA resid: max_err={(b0_sa-sa_pt).abs().max():.4f}")
print(f"  C++ SA range: [{b0_sa.min():.4f},{b0_sa.max():.4f}]")
print(f"  PT  SA range: [{sa_pt.min():.4f},{sa_pt.max():.4f}]")

# ═══════════════════════════════════════
# CROSS-ATTENTION
# ═══════════════════════════════════════
ln_cx = F.layer_norm(sa_pt.reshape(M,1,16,16,D), (D,), None, None, 1e-6).reshape(MS,D)
mod_cx = ln_cx * (scale_c+1.0).repeat_interleave(S_per,dim=0) + shift_c.repeat_interleave(S_per,dim=0)

w_cq=sd['blocks.0.cross_attn.q_proj.weight']; w_ck=sd['blocks.0.cross_attn.k_proj.weight']
w_cv=sd['blocks.0.cross_attn.v_proj.weight']; w_co=sd['blocks.0.cross_attn.output_proj.weight']
w_cqn=sd['blocks.0.cross_attn.q_norm.weight']; w_ckn=sd['blocks.0.cross_attn.k_norm.weight']

cq_pt = F.linear(mod_cx, w_cq).reshape(ph, HD)        # [MS*NH, HD]
ck_pt = F.linear(ctx.reshape(MS_kv:=M*Nctx, CtxD), w_ck).reshape(ph_cross, HD)
cv_pt = F.linear(ctx.reshape(MS_kv, CtxD), w_cv).reshape(ph_cross, HD)

cq_n_pt = F.rms_norm(cq_pt, (HD,), w_cqn, 1e-6)
ck_n_pt = F.rms_norm(ck_pt, (HD,), w_ckn, 1e-6)

cx_attn_o_pt = torch.zeros(ph, HD)
for mb in range(M):
    qi = cq_n_pt[mb*S_per*NH:(mb+1)*S_per*NH].reshape(S_per,NH,HD).permute(1,0,2)
    ki = ck_n_pt[mb*Nctx*NH:(mb+1)*Nctx*NH].reshape(Nctx,NH,HD).permute(1,0,2)
    vi = cv_pt.reshape(MS_kv,NH,HD)[mb*Nctx:(mb+1)*Nctx].permute(1,0,2)
    ao = sdpa(qi,ki,vi,sc).permute(1,0,2).reshape(S_per*NH,HD)
    cx_attn_o_pt[mb*S_per*NH:(mb+1)*S_per*NH] = ao

cx_oproj_pt = F.linear(cx_attn_o_pt.reshape(MS,D), w_co)
cx_pt = sa_pt + gate_c.repeat_interleave(S_per,dim=0) * cx_oproj_pt

print(f"\n=== Cross-attention ===")
print(f"  PT CX range: [{cx_pt.min():.4f},{cx_pt.max():.4f}]")
print(f"  C++ CX range: [{b0_cx.min():.4f},{b0_cx.max():.4f}]")
print(f"  CX resid max_err: {(b0_cx-cx_pt).abs().max():.4f}")

# ═══════════════════════════════════════
# MLP
# ═══════════════════════════════════════
ln_mlp = F.layer_norm(cx_pt.reshape(M,1,16,16,D), (D,), None, None, 1e-6).reshape(MS,D)
mod_mlp = ln_mlp * (scale_m+1.0).repeat_interleave(S_per,dim=0) + shift_m.repeat_interleave(S_per,dim=0)

w_fc1=sd['blocks.0.mlp.layer1.weight']; w_fc2=sd['blocks.0.mlp.layer2.weight']
fc1_pt = F.linear(mod_mlp, w_fc1)  # [MS, 8192]
gelu_pt = F.gelu(fc1_pt)
fc2_pt = F.linear(gelu_pt, w_fc2)  # [MS, 2048]
mlp_pt = cx_pt + gate_m.repeat_interleave(S_per,dim=0) * fc2_pt

print(f"\n=== MLP ===")
print(f"  PT  MLP range: [{mlp_pt.min():.4f},{mlp_pt.max():.4f}]")
print(f"  C++ MLP range: [{b0_mlp.min():.4f},{b0_mlp.max():.4f}]")
print(f"  MLP max_err: {(b0_mlp-mlp_pt).abs().max():.4f}")

# ═══════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════
print(f"\n=== Block 0 Summary ===")
print(f"  Input x range:        [{b0_x.min():.4f},{b0_x.max():.4f}]")
print(f"  SA residual err:      {(b0_sa-sa_pt).abs().max():.4f}")
print(f"  CX residual err:      {(b0_cx-cx_pt).abs().max():.4f}")
print(f"  MLP (block 0 out) err: {(b0_mlp-mlp_pt).abs().max():.4f}")

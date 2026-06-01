"""Compare C++ O_proj output vs PyTorch bit-exact."""
import numpy as np, torch, torch.nn.functional as F, sys
sys.path.insert(0, '/mnt/d/AI/anima_phone/src')

DEV = 'cuda'; DTYPE = torch.float16
M,S,D = 2,256,2048; MS = M*S; NH=16; HD=128; HALF_DIM=HD//2

RPDIR = '/mnt/d/AI/anima_phone/output/realpipe'

# Load inputs
x_np = np.load(f'{RPDIR}/x_flat.npy').astype(np.float16)
x = torch.from_numpy(x_np.astype(np.float32)).to(DEV, DTYPE)

# Load weights (same as C++ engine)
sd_r = torch.load('/mnt/d/AI/anima_phone/models/diffusion_weights_fp16.pt',
                  weights_only=True, map_location='cpu')
sd = {}
for k,v in sd_r.items():
    ck = k
    while ck.startswith('net.'): ck = ck[4:]
    sd[ck] = v
del sd_r

# C++ style t_emb & lora
w1 = sd['t_embedder.1.linear_1.weight'].float().to(DEV)
w2 = sd['t_embedder.1.linear_2.weight'].float().to(DEV)
w_ln = sd['t_embedding_norm.weight'].float().to(DEV)
halfD = D//2
j = torch.arange(halfD, dtype=torch.float32, device=DEV)
freqs = torch.exp(-torch.log(torch.tensor(10000.0)) * j / halfD)
sin_emb = torch.zeros(2, D, dtype=torch.float32, device=DEV)
sin_emb[:, :halfD] = torch.cos(freqs).unsqueeze(0)
sin_emb[:, halfD:] = torch.sin(freqs).unsqueeze(0)
rms = torch.sqrt((sin_emb*sin_emb).mean(-1, keepdim=True) + 1e-6)
t_emb = (sin_emb * w_ln.unsqueeze(0) / rms).to(DTYPE)
lora = (F.silu(sin_emb @ w1.T) @ w2.T).to(DTYPE)

# AdaLN block 0 self-attn
w1a = sd['blocks.0.adaln_modulation_self_attn.1.weight'].float().to(DEV)
w2a = sd['blocks.0.adaln_modulation_self_attn.2.weight'].float().to(DEV)
h_adaln = F.silu(t_emb.float()) @ w1a.T
out_adaln = h_adaln @ w2a.T + lora.float()
shift_sa, scale_sa, gate_sa = torch.chunk(out_adaln, 3, dim=-1)
scale_sa = scale_sa + 1.0
bc = lambda t: t.repeat_interleave(256, dim=0)

# LN + AdaLN
y = F.layer_norm(x, (D,), weight=None, bias=None, eps=1e-6)
y = y * bc(scale_sa.to(DTYPE)) + bc(shift_sa.to(DTYPE))

# QKV GEMM
qw = sd['blocks.0.self_attn.q_proj.weight'].to(DEV).to(DTYPE)
kw = sd['blocks.0.self_attn.k_proj.weight'].to(DEV).to(DTYPE)
vw = sd['blocks.0.self_attn.v_proj.weight'].to(DEV).to(DTYPE)
qnw = sd['blocks.0.self_attn.q_norm.weight'].to(DEV).to(DTYPE)
knw = sd['blocks.0.self_attn.k_norm.weight'].to(DEV).to(DTYPE)
ow = sd['blocks.0.self_attn.output_proj.weight'].to(DEV).to(DTYPE)

q = F.linear(y, qw); k = F.linear(y, kw); v = F.linear(y, vw)

# RMSNorm Q/K
ph = MS * NH
qn = F.rms_norm(q.view(ph, HD), (HD,), weight=qnw, eps=1e-6)
kn = F.rms_norm(k.view(ph, HD), (HD,), weight=knw, eps=1e-6)
vf = v.view(ph, HD)

# Load phone intermediates for comparison
CMP = '/mnt/d/AI/anima_phone/output/cmp'
ph_q = np.load(f'{CMP}/b0_q_norm.npy').astype(np.float32)
diff_q = np.abs(ph_q.flatten() - qn.cpu().numpy().astype(np.float32).flatten())
print(f'Q_norm phone vs PT: max_err={diff_q.max():.4f} mean_err={diff_q.mean():.6f}')

# RoPE (C++ style)
from compare_realpipe import compute_rope_freqs_cpp, apply_rope_cpp
rope_freqs = compute_rope_freqs_cpp()
q_roped = apply_rope_cpp(qn, rope_freqs)
k_roped = apply_rope_cpp(kn, rope_freqs)

# Self-attention (per-batch)
SCALE = 1.0 / np.sqrt(HD)
attn_o = torch.zeros(ph, HD, dtype=DTYPE, device=DEV)
for mb in range(M):
    base = mb * S * NH
    qmb = q_roped[base:base+S*NH].view(S, NH, HD).permute(1,0,2)
    kmb = k_roped[base:base+S*NH].view(S, NH, HD).permute(1,0,2)
    vmb = vf[base:base+S*NH].view(S, NH, HD).permute(1,0,2)
    scores = torch.bmm(qmb, kmb.transpose(1,2)) * SCALE
    attn_w = F.softmax(scores.float(), dim=-1).to(DTYPE)
    ao = torch.bmm(attn_w, vmb).permute(1,0,2).reshape(S*NH, HD)
    attn_o[base:base+S*NH] = ao

# O_proj
o_proj = F.linear(attn_o.view(MS, D), ow)

# Compare with phone O_proj
cpp_o = np.load(f'{RPDIR}/b0_o_proj_cpp.npy').astype(np.float32).reshape(MS, D)
pt_o = o_proj.cpu().numpy().astype(np.float32)
diff_o = np.abs(cpp_o.flatten() - pt_o.flatten())
print(f'\nO_proj phone vs PT: max_err={diff_o.max():.6f} mean_err={diff_o.mean():.8f}')

if diff_o.max() == 0:
    print('✓ O_proj is BIT-EXACT!')
else:
    idx = np.argmax(diff_o)
    r = idx // D; c = idx % D
    print(f'  FIRST DIFFERENCE at ({r},{c}): phone={cpp_o[r,c]:.8f} pt={pt_o[r,c]:.8f} diff={diff_o.max():.8f}')
    # Count how many elements differ
    n_diff = (diff_o > 0).sum()
    print(f'  Total different elements: {n_diff}/{MS*D} ({100*n_diff/(MS*D):.4f}%)')

    # Also check SA residual
    sa_cpp = np.load(f'{CMP}/b0_sa.npy').astype(np.float32).reshape(MS, D)
    gate_fp16 = bc(gate_sa.to(DTYPE))
    sa_pt = (x + gate_fp16 * o_proj).cpu().numpy().astype(np.float32)
    diff_sa = np.abs(sa_cpp.flatten() - sa_pt.flatten())
    print(f'\nSA residual phone vs PT: max_err={diff_sa.max():.4f} mean_err={diff_sa.mean():.6f}')

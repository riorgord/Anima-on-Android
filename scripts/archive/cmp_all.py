"""Bit-exact comparison: phone C++ intermediates vs PC whitebox (real inputs)."""
import numpy as np, torch, torch.nn.functional as F, sys
sys.path.insert(0, '/mnt/d/AI/anima_phone/src')

DEV = 'cuda'; DTYPE = torch.float16
M,S,D = 2,256,2048; MS = M*S; NH=16; HD=128; HALF_DIM=HD//2; HP=16

RPDIR = '/mnt/d/AI/anima_phone/output/realpipe'

# Load inputs
x_np = np.load(f'{RPDIR}/x_flat.npy').astype(np.float16)
ctx_np = np.load(f'{RPDIR}/ctx_flat.npy').astype(np.float16)
x = torch.from_numpy(x_np.astype(np.float32)).to(DEV, DTYPE)
ctx = torch.from_numpy(ctx_np.astype(np.float32)).to(DEV, DTYPE)

# Load weights
sd_r = torch.load('/mnt/d/AI/anima_phone/models/diffusion_weights_fp16.pt', weights_only=True, map_location='cpu')
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
h_a = F.silu(t_emb.float()) @ w1a.T
out_a = h_a @ w2a.T + lora.float()
shift_sa, scale_sa, gate_sa = torch.chunk(out_a, 3, dim=-1)
scale_sa = scale_sa + 1.0
bc = lambda t: t.to(DTYPE).repeat_interleave(S, dim=0)

# LN + AdaLN
y = F.layer_norm(x, (D,), weight=None, bias=None, eps=1e-6)
y = y * bc(scale_sa) + bc(shift_sa)

# QKV
qw = sd['blocks.0.self_attn.q_proj.weight'].to(DEV).to(DTYPE)
kw = sd['blocks.0.self_attn.k_proj.weight'].to(DEV).to(DTYPE)
vw = sd['blocks.0.self_attn.v_proj.weight'].to(DEV).to(DTYPE)
qnw = sd['blocks.0.self_attn.q_norm.weight'].to(DEV).to(DTYPE)
knw = sd['blocks.0.self_attn.k_norm.weight'].to(DEV).to(DTYPE)
ow = sd['blocks.0.self_attn.output_proj.weight'].to(DEV).to(DTYPE)

q = F.linear(y, qw); k = F.linear(y, kw); v = F.linear(y, vw)
ph = MS * NH
qn = F.rms_norm(q.view(ph, HD), (HD,), weight=qnw, eps=1e-6)
kn = F.rms_norm(k.view(ph, HD), (HD,), weight=knw, eps=1e-6)
vf = v.view(ph, HD)

# C++ style RoPE
from compare_realpipe import compute_rope_freqs_cpp, apply_rope_cpp
rope = compute_rope_freqs_cpp()
q_roped = apply_rope_cpp(qn, rope)
k_roped = apply_rope_cpp(kn, rope)

# Self-attention
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
# SA residual
sa = x + bc(gate_sa) * o_proj

# ── Compare each intermediate ──
comparisons = [
    ("b0_q_norm_cpp", qn.view(ph, HD), "Q after RMSNorm"),
    ("b0_k_norm_cpp", kn.view(ph, HD), "K after RMSNorm"),
    ("b0_v_raw_cpp", vf.view(ph, HD), "V raw"),
    ("b0_attn_o_cpp", attn_o.view(ph, HD), "Attn output"),
    ("b0_o_proj_cpp", o_proj.view(MS, D), "O_proj GEMM"),
    ("b0_sa_residual_cpp", sa.view(MS, D), "SA residual"),
]

print(f"{'Variable':<20s} {'max_err':<12s} {'mean_err':<12s} {'bit-exact?':<12s}")
print("-" * 56)
for name, pt_tensor, label in comparisons:
    cpp_arr = np.load(f'{RPDIR}/{name}.npy').astype(np.float32)
    pt_arr = pt_tensor.cpu().numpy().astype(np.float32)
    mi = min(cpp_arr.size, pt_arr.size)
    ok = np.isfinite(cpp_arr.flatten()[:mi]) & np.isfinite(pt_arr.flatten()[:mi])
    diff = np.abs(cpp_arr.flatten()[:mi][ok] - pt_arr.flatten()[:mi][ok])
    exact = "✓ BIT-EXACT" if diff.max() == 0 else f"✗ {diff.max():.6f}"
    print(f"{label:<20s} {diff.max():<12.6f} {diff.mean():<12.8f} {exact:<12s}")
    if diff.max() > 0:
        idx = np.argmax(diff)
        r = idx // min(cpp_arr.shape[-1] if cpp_arr.ndim > 1 else 1, 999)
        c = idx % min(cpp_arr.shape[-1] if cpp_arr.ndim > 1 else 999, 999)
        print(f"  worst at idx={idx}: phone={cpp_arr.flatten()[idx]:.8f} pt={pt_arr.flatten()[idx]:.8f}")

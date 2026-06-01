"""Compare phone Q_norm (FMA GEMM) vs PyTorch."""
import numpy as np, torch, torch.nn.functional as F, sys
sys.path.insert(0, '/mnt/d/AI/anima_phone/src')

DEV='cuda'; DTYPE=torch.float16; D=2048; HD=128; NH=16; M,S=2,256; MS=M*S; HP=16
RPDIR = '/mnt/d/AI/anima_phone/output/realpipe'

x_np = np.load(f'{RPDIR}/x_flat.npy').astype(np.float16)
sd_r = torch.load('/mnt/d/AI/anima_phone/models/diffusion_weights_fp16.pt', weights_only=True, map_location='cpu')
sd = {}
for k,v in sd_r.items():
    ck = k
    while ck.startswith('net.'): ck = ck[4:]
    sd[ck] = v
del sd_r

x = torch.from_numpy(x_np.astype(np.float32)).to(DEV, DTYPE)

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

w1a = sd['blocks.0.adaln_modulation_self_attn.1.weight'].float().to(DEV)
w2a = sd['blocks.0.adaln_modulation_self_attn.2.weight'].float().to(DEV)
h = F.silu(t_emb.float()) @ w1a.T
out_a = h @ w2a.T + lora.float()
shift, scale, gate = torch.chunk(out_a, 3, dim=-1)
scale = scale + 1.0
bc = lambda t: t.to(DTYPE).repeat_interleave(S, dim=0)

y = F.layer_norm(x, (D,), weight=None, bias=None, eps=1e-6)
y = y * bc(scale) + bc(shift)

qw = sd['blocks.0.self_attn.q_proj.weight'].to(DEV).to(DTYPE)
q = F.linear(y, qw)

qnw = sd['blocks.0.self_attn.q_norm.weight'].to(DEV).to(DTYPE)
ph = MS * NH
qn = F.rms_norm(q.view(ph, HD), (HD,), weight=qnw, eps=1e-6)

ph_q = np.load(f'{RPDIR}/q_norm_fma.npy').astype(np.float32)
pt_q = qn.cpu().numpy().astype(np.float32).flatten()
diff = np.abs(ph_q.flatten() - pt_q)
print(f'Q_norm FMA vs PT: max_err={diff.max():.6f} mean_err={diff.mean():.8f}')
if diff.max() > 0:
    idx = np.argmax(diff)
    print(f'  worst idx={idx}: phone={ph_q.flatten()[idx]:.8f} pt={pt_q.flatten()[idx]:.8f}')
    n_diff = (diff > 0).sum()
    print(f'  {n_diff}/{len(diff)} elements differ ({100*n_diff/len(diff):.2f}%)')

    # Compare Q_raw (before RMSNorm) to see if GEMM worked
    q_pt = q.cpu().numpy().astype(np.float32).flatten()
    # We don't have phone Q_raw, but we can check RMSNorm alone
    # RMSNorm: for each row, mean, rms, normalize
    q_reshaped = q.view(ph, HD)
    qn_computed = F.rms_norm(q_reshaped, (HD,), weight=qnw, eps=1e-6)
    qn_pt2 = qn_computed.cpu().numpy().astype(np.float32).flatten()
    diff2 = np.abs(qn_pt2 - pt_q)
    print(f'  RMSNorm self-consistency: max_err={diff2.max():.10f}')
else:
    print('BIT-EXACT!')

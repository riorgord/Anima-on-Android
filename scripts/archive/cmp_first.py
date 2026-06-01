"""Find first op with bit error in real pipeline."""
import numpy as np, torch, torch.nn.functional as F, sys
sys.path.insert(0, '/mnt/d/AI/anima_phone/src')
RPDIR='/mnt/d/AI/anima_phone/output/realpipe'
DEV='cuda'; DTYPE=torch.float16; D=2048; HD=128; NH=16; M,S=2,256; MS=M*S

x_np = np.load(f'{RPDIR}/x_flat.npy').astype(np.float16)
sd_r = torch.load('/mnt/d/AI/anima_phone/models/diffusion_weights_fp16.pt', weights_only=True, map_location='cpu')
sd = {}
for k,v in sd_r.items():
    ck = k
    while ck.startswith('net.'): ck = ck[4:]
    sd[ck] = v
del sd_r
x = torch.from_numpy(x_np.astype(np.float32)).to(DEV, DTYPE)

# t_emb, lora
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

# AdaLN
w1a = sd['blocks.0.adaln_modulation_self_attn.1.weight'].float().to(DEV)
w2a = sd['blocks.0.adaln_modulation_self_attn.2.weight'].float().to(DEV)
h_a = F.silu(t_emb.float()) @ w1a.T
out_a = h_a @ w2a.T + lora.float()
shift, scale, gate = torch.chunk(out_a, 3, dim=-1)
scale = scale + 1.0
bc = lambda t: t.to(DTYPE).repeat_interleave(S, dim=0)

# ----- 1. LN output (phone has it as part of modulated, but LN alone isn't captured) -----
# Compare LN alone
ln_pt = F.layer_norm(x, (D,), weight=None, bias=None, eps=1e-6)

# ----- 2. Modulated = LN * scale + shift -----
mod_pt = ln_pt * bc(scale) + bc(shift)
ph_mod = np.load(f'{RPDIR}/b0_modulated_cpp.npy').astype(np.float32).reshape(MS, D)
diff = np.abs(ph_mod.flatten() - mod_pt.cpu().numpy().astype(np.float32).flatten())
print(f'1. Modulated (LN*scale+shift): max_err={diff.max():.6f} bit_exact={(diff.max()==0)}')

# ----- 3. Q_raw = GEMM(mod, Q_weight) -----
qw = sd['blocks.0.self_attn.q_proj.weight'].to(DEV).to(DTYPE)
q_pt = F.linear(mod_pt, qw)
ph_qr = np.load(f'{RPDIR}/b0_q_raw_cpp.npy').astype(np.float32).reshape(MS, D)
diff2 = np.abs(ph_qr.flatten() - q_pt.cpu().numpy().astype(np.float32).flatten())
print(f'2. Q_raw (GEMM): max_err={diff2.max():.6f} bit_exact={(diff2.max()==0)}')

# ----- 4. Q_norm = RMSNorm(Q_raw) -----
# Use PHONE Q_raw as input to PC RMSNorm — tests if RMSNorm shader matches PyTorch
qnw = sd['blocks.0.self_attn.q_norm.weight'].to(DEV).to(DTYPE)
ph_q_raw_t = torch.from_numpy(ph_qr).to(DEV, DTYPE).view(MS*NH, HD)
qn_from_ph_raw = F.rms_norm(ph_q_raw_t, (HD,), weight=qnw, eps=1e-6)
ph_qn = np.load(f'{RPDIR}/b0_q_norm_cpp.npy').astype(np.float32)
diff3 = np.abs(ph_qn.flatten() - qn_from_ph_raw.cpu().numpy().astype(np.float32).flatten())
print(f'3. RMSNorm(phone Q_raw): max_err={diff3.max():.6f} bit_exact={(diff3.max()==0)}')

# ----- Summary -----
print()
if diff.max() == 0:
    print('✓ LN+modulate is BIT-EXACT')
else:
    print(f'✗ LN+modulate error = {diff.max():.6f} — FIRST OP WITH ERROR')
if diff2.max() == 0 and diff.max() == 0:
    print('✓ GEMM is BIT-EXACT')
if diff3.max() == 0:
    print('✓ RMSNorm is BIT-EXACT')
else:
    print(f'✗ RMSNorm error (from same Q_raw) = {diff3.max():.6f}')

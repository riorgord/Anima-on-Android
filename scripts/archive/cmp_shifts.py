"""Compare AdaLN shift/scale/gate between phone C++ and PC."""
import numpy as np, torch, torch.nn.functional as F
RPDIR='/mnt/d/AI/anima_phone/output/realpipe'; D=2048; M=2

sd_r = torch.load('/mnt/d/AI/anima_phone/models/diffusion_weights_fp16.pt', weights_only=True, map_location='cpu')
sd = {}
for k,v in sd_r.items():
    ck = k
    while ck.startswith('net.'): ck = ck[4:]
    sd[ck] = v
del sd_r

# C++ style t_emb, lora
w1 = sd['t_embedder.1.linear_1.weight'].float()
w2 = sd['t_embedder.1.linear_2.weight'].float()
w_ln = sd['t_embedding_norm.weight'].float()
halfD = D//2; sigma=1.0
j = np.arange(halfD, dtype=np.float32)
freqs = sigma * np.exp(-np.log(10000.0) * j / halfD)
sin_emb = np.zeros((M, D), dtype=np.float32)
sin_emb[:, :halfD] = np.cos(freqs)
sin_emb[:, halfD:] = np.sin(freqs)
rms = np.sqrt((sin_emb*sin_emb).mean(-1, keepdims=True) + 1e-6)
t_emb_np = (sin_emb * w_ln.numpy() / rms).astype(np.float16)
lora_np = (F.silu(torch.from_numpy(sin_emb) @ w1.T) @ w2.T).numpy().astype(np.float16)

# AdaLN
w1a = sd['blocks.0.adaln_modulation_self_attn.1.weight'].float().numpy()
w2a = sd['blocks.0.adaln_modulation_self_attn.2.weight'].float().numpy()
h = F.silu(torch.from_numpy(sin_emb) @ torch.from_numpy(w1a).T).numpy()
out = (torch.from_numpy(h) @ torch.from_numpy(w2a).T).numpy() + lora_np.astype(np.float32)
shift_pt, scale_pt, gate_pt = np.split(out, 3, axis=-1)
scale_pt = scale_pt + 1.0

# Phone shifts
ph = np.load(f'{RPDIR}/b0_shifts_cpp.npy').astype(np.float32)
ph_shift = ph[:M*D].reshape(M, D)
ph_scale = ph[M*D:2*M*D].reshape(M, D)
ph_gate  = ph[2*M*D:3*M*D].reshape(M, D)

for name, ph_v, pt_v in [("shift", ph_shift, shift_pt), ("scale", ph_scale, scale_pt), ("gate", ph_gate, gate_pt)]:
    diff = np.abs(ph_v.flatten() - pt_v.flatten())
    exact = "BIT-EXACT" if diff.max() == 0 else f"max_err={diff.max():.6f}"
    n_diff = (diff > 0).sum()
    print(f"AdaLN {name}: {exact}  n_diff={n_diff}/{ph_v.size}  "
          f"ph=[{ph_v.min():.3f},{ph_v.max():.3f}] pt=[{pt_v.min():.3f},{pt_v.max():.3f}]")

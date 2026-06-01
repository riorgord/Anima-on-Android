"""Compare AdaLN shift/scale/gate first elements only."""
import numpy as np, torch, torch.nn.functional as F
RPDIR='/mnt/d/AI/anima_phone/output/realpipe'; D=2048; M=2; ADALN=256

sd_r = torch.load('/mnt/d/AI/anima_phone/models/diffusion_weights_fp16.pt', weights_only=True, map_location='cpu')
sd = {}
for k,v in sd_r.items():
    ck = k
    while ck.startswith('net.'): ck = ck[4:]
    sd[ck] = v
del sd_r

# C++ style t_emb
w_ln = sd['t_embedding_norm.weight'].float().numpy()
halfD = D//2; sigma=1.0
j = np.arange(halfD, dtype=np.float32)
freqs = sigma * np.exp(-np.log(10000.0) * j / halfD)
sin_emb = np.zeros((M, D), dtype=np.float32)
sin_emb[:, :halfD] = np.cos(freqs)
sin_emb[:, halfD:] = np.sin(freqs)
rms = np.sqrt((sin_emb*sin_emb).mean(-1, keepdims=True) + 1e-6)
t_emb_np = (sin_emb * w_ln / rms).astype(np.float16)
print(f"t_emb[0,0..3] = {t_emb_np[0,:3]}")

# SiLU — uses t_emb (RMSNorm output), NOT sin_emb!
t_f32 = torch.from_numpy(t_emb_np.astype(np.float32))
si = F.silu(t_f32).numpy()
si_h = si.astype(np.float16)
print(f"SiLU(t_emb)[0,0..3] = {si_h[0,:3]}")
print(f"SiLU range = [{si_h.min():.4f}, {si_h.max():.4f}]")

# AdaLN: SiLU(t_emb) @ w1
w1 = sd['blocks.0.adaln_modulation_self_attn.1.weight'].float().numpy()
h = (torch.from_numpy(si.astype(np.float32)) @ torch.from_numpy(w1).T).numpy().astype(np.float16)
print(f"h=SiLU@w1[0,0..3] = {h[0,:3]}")
print(f"h range = [{h.min():.4f}, {h.max():.4f}]")

# AdaLN: h @ w2 + lora
w1_t = sd['t_embedder.1.linear_1.weight'].float().numpy()
w2_t = sd['t_embedder.1.linear_2.weight'].float().numpy()
lora_np = (F.silu(torch.from_numpy(sin_emb) @ torch.from_numpy(w1_t).T) @ torch.from_numpy(w2_t).T).numpy().astype(np.float16)
w2a = sd['blocks.0.adaln_modulation_self_attn.2.weight'].float().numpy()
out = (torch.from_numpy(h.astype(np.float32)) @ torch.from_numpy(w2a).T).numpy().astype(np.float16)
out = out + lora_np
shift_pt, scale_pt, gate_pt = np.split(out, 3, axis=-1)
scale_pt = scale_pt + 1.0
print(f"shift[0,0..3] = {shift_pt[0,:3]}")
print(f"scale[0,0..3] = {scale_pt[0,:3]}")
print(f"gate[0,0..3]  = {gate_pt[0,:3]}")

# Phone
ph = np.load(f'{RPDIR}/b0_shifts_cpp.npy').astype(np.float16)
ph_shift = ph[:M*D].reshape(M, D)
ph_scale = ph[M*D:2*M*D].reshape(M, D)
ph_gate  = ph[2*M*D:3*M*D].reshape(M, D)
print(f"\nPhone shift[0,0..3] = {ph_shift[0,:3]}")
print(f"Phone scale[0,0..3] = {ph_scale[0,:3]}")
print(f"Phone gate[0,0..3]  = {ph_gate[0,:3]}")

print(f"\nDiff shift[0,:3]: {np.abs(ph_shift[0,:3].astype(np.float32) - shift_pt[0,:3].astype(np.float32))}")
print(f"Diff scale[0,:3]: {np.abs(ph_scale[0,:3].astype(np.float32) - scale_pt[0,:3].astype(np.float32))}")
print(f"Diff gate[0,:3]:  {np.abs(ph_gate[0,:3].astype(np.float32) - gate_pt[0,:3].astype(np.float32))}")

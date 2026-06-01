"""WSL: Pre-compute FULL AdaLN (internal + external lora) for all 28 blocks × 3 steps.
Saves: adaln_full_step{N}.bin — [28, 9, MS, D] fp16 per step (504MB each)"""
import sys; sys.path.insert(0,"/mnt/d/AI/anima_phone/src")
import numpy as np, torch, os
import torch.nn.functional as F
from safetensors import safe_open

MS,D,M=512,2048,2; S=MS//M; NH=16; HD=128
OUT="/mnt/d/AI/anima_phone/output"
os.makedirs(OUT,exist_ok=True)

# Load weights
print("Loading weights...")
sd={}
with safe_open("/mnt/d/AI/手坤的anima/models/diffusion_model/anima-base-v1.0.safetensors",
               framework="pt",device="cpu") as f:
    for key in f.keys():
        sd[key[4:] if key.startswith("net.") else key]=f.get_tensor(key).float()

t_w1=sd["t_embedder.1.linear_1.weight"]  # [2048, 2048]
t_w2=sd["t_embedder.1.linear_2.weight"]  # [6144, 2048]
t_ln_w=sd["t_embedding_norm.weight"]     # [2048]

# Compute external adaln_lora for a given sigma
def compute_external_lora(sigma_val):
    sigma=torch.tensor([sigma_val,sigma_val]).unsqueeze(1)
    half_dim=D//2
    exponent=-np.log(10000)*np.arange(half_dim,dtype=np.float32)/half_dim
    emb_val=sigma.float().numpy()*np.exp(exponent)
    emb_sincos=np.concatenate([np.cos(emb_val),np.sin(emb_val)],-1)
    t_input=torch.from_numpy(emb_sincos).float()  # [2,1,2048]
    # TimestepEmbedding forward (use_adaln_lora=True)
    h=F.linear(t_input,t_w1)  # [2,1,2048]
    h=F.silu(h)
    ext_lora=F.linear(h,t_w2)  # [2,1,6144] — this is adaln_lora_B_T_3D
    # t_emb
    t_emb=F.rms_norm(t_input.squeeze(1),(D,),weight=t_ln_w,eps=1e-6)  # [2,2048]
    return t_emb, ext_lora.squeeze(1)  # [2,2048], [2,6144]

# For each sigma, compute all 28 blocks' full AdaLN
for step, sigma_val in enumerate([1.0, 0.667, 0.333]):
    print(f"Step {step+1}/3 sigma={sigma_val:.3f}")
    t_emb, ext_lora = compute_external_lora(sigma_val)

    adaln_data = np.zeros(28 * 9 * MS * D, dtype=np.uint16)

    for b in range(28):
        pfx = f"blocks.{b}."
        for ch_name, base_comp in [
            ("self_attn", 0), ("cross_attn", 3), ("mlp", 6)
        ]:
            w1 = sd[f"{pfx}adaln_modulation_{ch_name}.1.weight"]  # [256, 2048]
            w2 = sd[f"{pfx}adaln_modulation_{ch_name}.2.weight"]  # [6144, 256]

            # Internal AdaLN: SiLU(t_emb) → LoRA down → SiLU → LoRA up
            h = F.silu(t_emb)                    # [2, 2048]
            h = F.linear(h, w1)                   # [2, 256]
            h = F.silu(h)
            internal = F.linear(h, w2)            # [2, 6144]

            # Full AdaLN: internal + external
            full = internal + ext_lora             # [2, 6144]
            shift, scale, gate = torch.chunk(full, 3, dim=-1)  # each [2, 2048]
            scale_p1 = scale + 1.0

            n_elem = MS * D
            base = b * 9 * n_elem + base_comp * n_elem
            adaln_data[base+0*n_elem:base+1*n_elem] = scale_p1.repeat_interleave(S,0).numpy().astype(np.float16).ravel().view(np.uint16)
            adaln_data[base+1*n_elem:base+2*n_elem] = shift.repeat_interleave(S,0).numpy().astype(np.float16).ravel().view(np.uint16)
            adaln_data[base+2*n_elem:base+3*n_elem] = gate.repeat_interleave(S,0).numpy().astype(np.float16).ravel().view(np.uint16)

    adaln_data.tofile(f"{OUT}/adaln_full_step{step}.bin")
    print(f"  Saved: {adaln_data.nbytes/1e6:.0f}MB")

print("DONE")

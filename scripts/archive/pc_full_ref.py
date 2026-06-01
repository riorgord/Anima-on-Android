"""PC full reference pipeline — matching phone_pipeline.py 3-step denoising.

Loads the SAME diffusion_weights_fp16.pt as phone, uses same context/scheduler/seed.
Runs with num_blocks=28 in pure PyTorch (no C++ engine).
Dumps per-block intermediates from step 1 for C++ comparison.
Saves final 3-step image as reference.

Usage (WSL2):
    source /home/riorg/miniconda3/etc/profile.d/conda.sh
    conda activate /home/riorg/anima-work/.conda
    python /mnt/d/AI/anima_phone/scripts/pc_full_ref.py
"""
import sys, os, time, gc
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image

SRC = "/mnt/d/AI/anima_phone/src"
sys.path.insert(0, SRC)

import predict2
import wan_vae

DEV = "cuda"
DTYPE = torch.float16
SEED = 6666
STEPS = 3
CFG = 5.0
H_LAT = 32          # latent spatial dim → 256×256 image
HP = H_LAT // 2     # patch spatial dim (patch_spatial=2) → 16
D = 2048
N_HEADS = 16
HEAD_DIM = 128
MS = 2 * 1 * HP * HP  # total tokens after patch embedding = 512
HEAD_DIM = 128
NCTX = 512
CTXD = 1024

OUTDIR = "/mnt/d/AI/anima_phone/output"
os.makedirs(OUTDIR, exist_ok=True)

torch.manual_seed(SEED)
np.random.seed(SEED)

# ═══════════════════════════════════════════════════════════════
# 1. Load context (same as phone: pre-computed LLMAdapter output)
# ═══════════════════════════════════════════════════════════════
print("=" * 60)
print("Step 1: Load context")

ctx_path = "/mnt/d/AI/anima_phone/models/context_cond.pt"
if not os.path.exists(ctx_path):
    print("  Context not found, generating now...")
    import subprocess
    subprocess.run([sys.executable, "/mnt/d/AI/anima_phone/scripts/pc_context.py"], check=True)

ctx_cond = torch.load(ctx_path, weights_only=True).to(DEV).to(DTYPE)
ctx_uncond = torch.load("/mnt/d/AI/anima_phone/models/context_uncond.pt",
                        weights_only=True).to(DEV).to(DTYPE)
# Context shape: [1, 512, 1024] — repeat for CFG batch
ctx = torch.cat([ctx_cond, ctx_uncond], dim=0)  # [2, 512, 1024]
ctx_flat = ctx.reshape(2 * NCTX, CTXD)          # [1024, 1024] — for flat attention
print(f"  Context: {ctx.shape} (cond + uncond)")
del ctx_cond, ctx_uncond; gc.collect()

# ═══════════════════════════════════════════════════════════════
# 2. Load DiT (num_blocks=28, same weights as phone)
# ═══════════════════════════════════════════════════════════════
print("Step 2: Load DiT (28 blocks)")

config = dict(
    max_img_h=240, max_img_w=240, max_frames=128,
    in_channels=16, out_channels=16,
    patch_spatial=2, patch_temporal=1, concat_padding_mask=True,
    model_channels=D, num_blocks=28, num_heads=N_HEADS, mlp_ratio=4.0,
    crossattn_emb_channels=CTXD,
    pos_emb_cls="rope3d", pos_emb_learnable=True, pos_emb_interpolation="crop",
    min_fps=1, max_fps=30, use_adaln_lora=True, adaln_lora_dim=256,
    rope_h_extrapolation_ratio=4.0, rope_w_extrapolation_ratio=4.0,
    rope_t_extrapolation_ratio=1.0,
    extra_per_block_abs_pos_emb=False, rope_enable_fps_modulation=False,
)

t0 = time.time()
wt_path = "/mnt/d/AI/anima_phone/models/diffusion_weights_fp16.pt"
sd_raw = torch.load(wt_path, map_location="cpu", weights_only=True)
# Strip "net." prefix if present
sd = {}
for k, v in sd_raw.items():
    sd[k[4:] if k.startswith("net.") else k] = v
del sd_raw

dit = predict2.MiniTrainDIT(**config, device=DEV, dtype=DTYPE, operations=torch.nn)
dit.load_state_dict(sd, strict=False)
dit.eval()
del sd; gc.collect(); torch.cuda.empty_cache()
print(f"  Loaded in {time.time()-t0:.1f}s, {len(dit.blocks)} blocks")

# ═══════════════════════════════════════════════════════════════
# 3. Load VAE (WanVAE, same as phone)
# ═══════════════════════════════════════════════════════════════
print("Step 3: Load VAE")
vae_wt = torch.load("/mnt/d/AI/anima_phone/models/vae_weights_fp16.pt",
                    map_location="cpu", weights_only=True)
vae = wan_vae.WanVAE(dim=96, z_dim=16, dim_mult=[1,2,4,4], num_res_blocks=2, attn_scales=[],
                     temperal_downsample=[False,False,False])
vae = vae.to(DEV).to(DTYPE)
vae.load_state_dict({k: v.float() for k, v in vae_wt.items()}, strict=False)
vae.eval()
del vae_wt; gc.collect()
print(f"  VAE loaded")

# ═══════════════════════════════════════════════════════════════
# 4. Denoising loop (matching phone_pipeline.py)
# ═══════════════════════════════════════════════════════════════
print("=" * 60)
print(f"Step 4: Denoising {STEPS} steps, {H_LAT*8}x{H_LAT*8}, CFG={CFG}, seed={SEED}")

# Sigma schedule (same as phone)
def time_snr_shift(a, t):
    return a * t / (1.0 + (a - 1.0) * t)

linear = torch.linspace(1.0, 0.0, STEPS + 1)[:-1]
sigmas = (3.0 * linear / (1.0 + 2.0 * linear)).tolist() + [0.0]
print(f"  Sigmas: {[f'{s:.3f}' for s in sigmas]}")

# Initial latent (same seed as phone)
torch.manual_seed(SEED)
latent = torch.randn(1, 16, H_LAT, H_LAT, dtype=DTYPE, device=DEV)  # [1, 16, 32, 32]

# ── Patch and embed latent (matching phone prepare_embedded_sequence) ──
def prepare_input(lat, dt):
    """Replicate phone pipeline's x_embedder step.
    Returns flat [MS, D] and also the 5D format for block forward."""
    B = 2  # CFG batch
    x_5d = lat.unsqueeze(2).repeat(B, 1, 1, 1, 1).to(dt)  # [2, 16, 1, 32, 32]
    # Use model's prepare_embedded_sequence for correct patch embedding
    x_emb, rope, extra_pos = dit.prepare_embedded_sequence(x_5d)
    # rope and extra_pos are NOT used (C++ engine has no RoPE)
    return x_emb.reshape(B, 1, HP, HP, D), rope, extra_pos  # [2, 1, 16, 16, 2048]

# ── Block forward (flat format, no RoPE — matching C++ engine) ──
def run_blocks_no_rope(x_5d, t_emb_2d, ctx_3d, adaln_lora):
    """Run 28 blocks, using block.forward() with rope_emb=None (matching C++ no-RoPE)."""
    per_block = []
    with torch.no_grad():
        x = x_5d
        for i, block in enumerate(dit.blocks):
            x = block.forward(
                x, t_emb_2d, ctx_3d,
                rope_emb_L_1_1_D=None,
                adaln_lora_B_T_3D=adaln_lora,
            )
            per_block.append(x.clone())
    return x, per_block

# ══════════════════════════════════════════════════════════
# Main denoising loop
# ══════════════════════════════════════════════════════════
all_block_outputs = []  # [step][block] → numpy array

for step_idx in range(STEPS):
    sigma = sigmas[step_idx]
    next_sigma = sigmas[step_idx + 1]
    dt = next_sigma - sigma

    print(f"\n  Step {step_idx+1}/{STEPS}: sigma={sigma:.3f} → {next_sigma:.3f}")

    # Prepare x from latent
    x_5d, rope, extra_pos = prepare_input(latent, DTYPE)

    # Run 28 blocks (no RoPE)
    t_forward = time.time()
    ctx_3d = ctx  # [2, 512, 1024]

    # Compute t_emb for this step
    ts = torch.tensor([sigma, sigma], dtype=DTYPE, device=DEV).unsqueeze(1)  # [2, 1]
    t_emb_raw = dit.t_embedder[0](ts).to(DTYPE)
    t_emb_out, lora_out = dit.t_embedder[1](t_emb_raw)
    t_emb = dit.t_embedding_norm(t_emb_out)  # [2, 1, 2048]

    out_5d, per_block = run_blocks_no_rope(x_5d, t_emb, ctx_3d, lora_out)
    dt_forward = time.time() - t_forward

    # Flatten output
    noise_pred = out_5d.reshape(MS, D)  # [512, 2048]

    # CFG combination
    noise_cond = noise_pred[:MS//2]      # batch 0
    noise_uncond = noise_pred[MS//2:]    # batch 1
    noise_cfg = noise_uncond + CFG * (noise_cond - noise_uncond)

    # Apply final_layer + unpatch (matching phone)
    # final_layer: Linear(D, patch_out) → unpatch to [1, 16, 32, 32]
    noise_latent = dit.final_layer(noise_cfg.reshape(1, 1, HP, HP, D))

    # Scheduler step: x = x + dt * noise
    latent = latent + dt * noise_latent.squeeze(2).to(DTYPE)

    # Stats
    with torch.no_grad():
        f = noise_pred.float()
        print(f"    forward: {dt_forward:.1f}s  "
              f"noise range=[{f.min():.2f}, {f.max():.2f}]  "
              f"latent std={latent.float().std():.3f}")

    # Save per-block outputs from step 1 only (for C++ comparison)
    if step_idx == 0:
        for b_idx, block_out in enumerate(per_block):
            flat_out = block_out.reshape(MS, D).cpu().numpy().astype(np.float16)
            all_block_outputs.append(flat_out)
        print(f"    saved {len(all_block_outputs)} block outputs for C++ comparison")

# ═══════════════════════════════════════════════════════════
# 5. VAE decode
# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("Step 5: VAE decode")

with torch.no_grad():
    latent_fp32 = latent.float()
    # WanVAE expects [B, C, T, H, W] = [1, 16, 1, 32, 32]
    img = vae.decode(latent_fp32.unsqueeze(2))  # add T dim
    img_np = (img.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)

out_img = Image.fromarray(img_np)
img_path = f"{OUTDIR}/pc_ref_3step.png"
out_img.save(img_path)
print(f"  Image saved: {img_path}")
print(f"  Pixel range: [{img_np.min()}, {img_np.max()}], mean={img_np.mean():.1f}")

# ═══════════════════════════════════════════════════════════
# 6. Save per-block reference outputs for C++ comparison
# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("Step 6: Save per-block references")

cmp_dir = f"{OUTDIR}/validate"
os.makedirs(cmp_dir, exist_ok=True)

# Also save the step-1 inputs (x, t_emb, ctx) for C++ alignment
x_input = x_5d.reshape(MS, D).cpu().numpy().astype(np.float16)
t_input = t_emb.cpu().numpy().astype(np.float16)
c_input = ctx_flat.cpu().numpy().astype(np.float16)

np.save(f"{cmp_dir}/x_step1.npy", x_input)
np.save(f"{cmp_dir}/t_emb_step1.npy", t_input)
np.save(f"{cmp_dir}/ctx_step1.npy", c_input)
np.save(f"{cmp_dir}/sigma_step1.npy", np.array([sigmas[0]], dtype=np.float32))

for i, out in enumerate(all_block_outputs):
    np.save(f"{cmp_dir}/block_{i:02d}_pt.npy", out)

print(f"  Saved inputs + {len(all_block_outputs)} block outputs to {cmp_dir}/")
print(f"  x={x_input.shape} t_emb={t_input.shape} ctx={c_input.shape}")
print(f"  Final block output range: [{all_block_outputs[-1].min():.2f}, {all_block_outputs[-1].max():.2f}]")

print("\n" + "=" * 60)
print("DONE — PC reference pipeline complete")
print(f"  Reference image: {img_path}")
print(f"  Block outputs for C++ comparison: {cmp_dir}/block_*_pt.npy")
print("=" * 60)

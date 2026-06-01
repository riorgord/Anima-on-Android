"""PC reference pipeline — adapted from 2026-05-25 phone_pipeline.py (commit 9ac2df9).

Runs full DiT in pure PyTorch (no C++ engine), matching phone parameters:
  256×256, 3 steps, CFG=5, seed=6666, same FP16 weights & context.

Also dumps per-block intermediate outputs from step 1 for C++ engine comparison.

Usage (WSL2):
    source /home/riorg/miniconda3/etc/profile.d/conda.sh
    conda activate /home/riorg/anima-work/.conda
    python /mnt/d/AI/anima_phone/scripts/pc_ref_25.py
"""
import sys, time, gc, os
sys.path.insert(0, "/mnt/d/AI/anima_phone/src")
import torch, numpy as np
from PIL import Image
import predict2, llm_adapter
import wan_vae

DEV = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16
STEPS = 3
CFG = 5.0
SEED = 6666
H = 32  # 256×256

OUTDIR = "/mnt/d/AI/anima_phone/output"
MODELDIR = "/mnt/d/AI/anima_phone/models"
os.makedirs(OUTDIR, exist_ok=True)

# ── Load contexts ──
print("Loading contexts...")
ctx_cond = torch.load(f"{MODELDIR}/context_cond.pt", weights_only=True).to(DEV).to(DTYPE)
ctx_uncond = torch.load(f"{MODELDIR}/context_uncond.pt", weights_only=True).to(DEV).to(DTYPE)
print(f"  cond: {ctx_cond.shape}, uncond: {ctx_uncond.shape}")

# ── Load DiT ──
print("Loading DiT (28 blocks, FP16)...")
t0 = time.time()
config = dict(max_img_h=240, max_img_w=240, max_frames=128, in_channels=16, out_channels=16,
    patch_spatial=2, patch_temporal=1, concat_padding_mask=True, model_channels=2048,
    num_blocks=28, num_heads=16, mlp_ratio=4.0, crossattn_emb_channels=1024,
    pos_emb_cls="rope3d", pos_emb_learnable=True, pos_emb_interpolation="crop",
    min_fps=1, max_fps=30, use_adaln_lora=True, adaln_lora_dim=256,
    rope_h_extrapolation_ratio=4.0, rope_w_extrapolation_ratio=4.0, rope_t_extrapolation_ratio=1.0,
    extra_per_block_abs_pos_emb=False, rope_enable_fps_modulation=False)
dit_sd = torch.load(f"{MODELDIR}/diffusion_weights_fp16.pt", weights_only=True)
# Strip "net." prefix (may appear multiple times: safetensors has "net.X", export adds "net.net.X")
sd = {}
for k, v in dit_sd.items():
    while k.startswith("net."):
        k = k[4:]
    sd[k] = v
del dit_sd

dit = predict2.MiniTrainDIT(**config, device=DEV, dtype=DTYPE, operations=torch.nn)
dit.load_state_dict(sd, strict=False)  # llm_adapter keys are extra (not in predict2 base model)
dit.eval()
del sd; gc.collect()
print(f"  Loaded in {time.time()-t0:.0f}s, {len(dit.blocks)} blocks")

# ── Hook: capture per-block intermediate outputs ──
block_outputs = {}
def make_hook(idx):
    def hook(module, input, output):
        # output is 5D: [B, T, Hp, Wp, D]
        block_outputs[idx] = output.detach().cpu().numpy().astype(np.float16)
    return hook

hooks = []
for i, block in enumerate(dit.blocks):
    h = block.register_forward_hook(make_hook(i))
    hooks.append(h)

# ── Scheduler (same as phone) ──
def time_snr_shift(a, t): return a * t / (1.0 + (a - 1.0) * t)
linear = torch.linspace(1.0, 0.0, STEPS + 1)[:-1]
sigmas = (3.0 * linear / (1.0 + 2.0 * linear)).tolist() + [0.0]
print(f"Sigmas: {[f'{s:.3f}' for s in sigmas]}")

# ── Denoising ──
print(f"\nDenoising {STEPS} steps, {H*8}x{H*8}...")
gen = torch.Generator(device="cuda" if DEV == "cuda" else "cpu").manual_seed(SEED)
x = torch.randn(1, 16, H, H, generator=gen, dtype=DTYPE, device=DEV)
t_start = time.time()

for i in range(STEPS):
    sigma = sigmas[i]
    sigma_next = sigmas[i + 1]
    ts = torch.tensor([sigma], dtype=DTYPE, device=DEV)
    x_b = x.unsqueeze(2).repeat(2, 1, 1, 1, 1)  # [2, 16, 1, 32, 32]
    ctx_b = torch.cat([ctx_cond, ctx_uncond], dim=0)  # [2, 512, 1024]
    ts_b = ts.repeat(2)  # [2]

    # Clear previous block outputs
    block_outputs.clear()

    t0_step = time.time()
    with torch.no_grad():
        v_b = dit(x_b, ts_b, ctx_b)
    dit_time = time.time() - t0_step

    # Capture block outputs from step 1
    if i == 0:
        step1_outputs = dict(block_outputs)

    v_cond = v_b[0:1].float()
    v_uncond = v_b[1:2].float()
    v_cfg = v_uncond + CFG * (v_cond - v_uncond)
    x = (x.float() + v_cfg.squeeze(2) * (sigma_next - sigma)).to(DTYPE)

    print(f"  step {i+1}/{STEPS}: {dit_time:.1f}s  "
          f"latent std={x.float().std():.3f}  (total {time.time()-t_start:.0f}s)")

# Remove hooks
for h in hooks:
    h.remove()

# ── VAE ──
print("Loading VAE...")
vae_sd = torch.load(f"{MODELDIR}/vae_weights_fp16.pt", weights_only=True)
vae = wan_vae.WanVAE(dim=96, z_dim=16, dim_mult=[1,2,4,4], num_res_blocks=2, attn_scales=[],
    temperal_downsample=[False,True,True], image_channels=3, conv_out_channels=3, dropout=0.0)
vae = vae.to(DEV)
vae.load_state_dict({k: v.float() for k, v in vae_sd.items()}, strict=False)
vae.eval(); del vae_sd

print("Decoding...")
with torch.no_grad():
    image = vae.decode(x.float().unsqueeze(2))
img = image[0,:,0].clamp(-1,1)
img = ((img+1)/2*255).permute(1,2,0).cpu().numpy().astype(np.uint8)

out_path = f"{OUTDIR}/pc_ref_25.png"
Image.fromarray(img).save(out_path)
total_t = time.time() - t_start
print(f"Saved: {out_path}")
print(f"Pixel range: [{img.min()},{img.max()}], mean={img.mean():.1f}")
print(f"TOTAL: {STEPS} steps, {total_t:.0f}s ({total_t/STEPS:.0f}s/step), {H*8}x{H*8}")

# ── Save per-block outputs for C++ comparison ──
print(f"\nSaving {len(step1_outputs)} per-block outputs from step 1...")
cmp_dir = f"{OUTDIR}/validate"
os.makedirs(cmp_dir, exist_ok=True)
for b_idx in sorted(step1_outputs.keys()):
    out_5d = step1_outputs[b_idx]
    # out_5d is [B, T, Hp, Wp, D] = [2, 1, 16, 16, 2048]
    flat = out_5d.reshape(2 * 1 * 16 * 16, 2048)  # [512, 2048]
    np.save(f"{cmp_dir}/block_{b_idx:02d}_pt25.npy", flat)

# Also save step-1 inputs for C++ reproduction
# x input (before model forward): the latent reshaped
x_input = x_b.detach().cpu().numpy().astype(np.float16)  # [2, 16, 1, 32, 32]
# ts input: sigma values
ts_input = ts_b.detach().cpu().numpy().astype(np.float32)
# ctx input
ctx_input = ctx_b.detach().cpu().numpy().astype(np.float16)
np.save(f"{cmp_dir}/x_step1_25.npy", x_input)
np.save(f"{cmp_dir}/ts_step1_25.npy", ts_input)
np.save(f"{cmp_dir}/ctx_step1_25.npy", ctx_input)
np.save(f"{cmp_dir}/sigma_step1_25.npy", np.array([sigmas[0]], dtype=np.float32))

print(f"  Inputs + {len(step1_outputs)} block outputs saved to {cmp_dir}/")
print(f"  Final block output range: [{step1_outputs[27].min():.2f}, {step1_outputs[27].max():.2f}]")

print("\nDone!")

"""Generate real pipeline inputs (x_flat, t_emb, ctx_flat) and PyTorch block outputs.

Uses the SAME latent (seed=6666), context, and sigma=1.0 as phone_pipeline.py step 1.
Runs dit.forward() (WITH RoPE) and captures per-block outputs via hooks.

Usage (WSL2):
    source /home/riorg/miniconda3/etc/profile.d/conda.sh
    conda activate /home/riorg/anima-work/.conda
    python /mnt/d/AI/anima_phone/scripts/gen_real_inputs.py
"""
import sys, torch, numpy as np, time, os
sys.path.insert(0, '/mnt/d/AI/anima_phone/src')
import predict2

DEV = 'cuda'; DTYPE = torch.float16; SEED = 6666; D = 2048; M = 2

config = dict(
    max_img_h=240, max_img_w=240, max_frames=128, in_channels=16, out_channels=16,
    patch_spatial=2, patch_temporal=1, concat_padding_mask=True,
    model_channels=D, num_blocks=28, num_heads=16, mlp_ratio=4.0,
    crossattn_emb_channels=1024, pos_emb_cls='rope3d', pos_emb_learnable=True,
    pos_emb_interpolation='crop', min_fps=1, max_fps=30,
    use_adaln_lora=True, adaln_lora_dim=256,
    rope_h_extrapolation_ratio=4.0, rope_w_extrapolation_ratio=4.0,
    rope_t_extrapolation_ratio=1.0,
    extra_per_block_abs_pos_emb=False, rope_enable_fps_modulation=False,
)

# ── Load ──
print("Loading DiT...")
sd_pt = torch.load('/mnt/d/AI/anima_phone/models/diffusion_weights_fp16.pt',
                   weights_only=True, map_location='cpu')
sd = {}
for k, v in sd_pt.items():
    ck = k
    while ck.startswith('net.'):
        ck = ck[4:]
    sd[ck] = v
del sd_pt

dit = predict2.MiniTrainDIT(**config, device=DEV, dtype=DTYPE, operations=torch.nn)
dit.load_state_dict(sd, strict=False)
dit.eval()  # sets requires_grad=False
del sd

ctx_cond = torch.load('/mnt/d/AI/anima_phone/models/context_cond.pt', map_location=DEV).to(DTYPE)
ctx_uncond = torch.load('/mnt/d/AI/anima_phone/models/context_uncond.pt', map_location=DEV).to(DTYPE)
print(f"  Context: cond={ctx_cond.shape} uncond={ctx_uncond.shape}")

OUTDIR = '/mnt/d/AI/anima_phone/output/realpipe'
os.makedirs(OUTDIR, exist_ok=True)

# ── Generate latent (matching phone pipeline seed=6666, step 1) ──
gen = torch.Generator(device=DEV).manual_seed(SEED)
x = torch.randn(1, 16, 32, 32, generator=gen, dtype=DTYPE, device=DEV)
print(f"  Latent: shape={x.shape} std={x.float().std():.3f}")

x_b = x.unsqueeze(2).repeat(2, 1, 1, 1, 1)    # [2, 16, 1, 32, 32]
ctx_b = torch.cat([ctx_cond, ctx_uncond], dim=0)  # [2, 512, 1024]

# ── Register hooks for per-block outputs ──
block_outs = {}
def make_hook(idx):
    def hook(module, inp, out):
        flat = out.detach().cpu().reshape(M * 256, D).float().numpy().astype(np.float16)
        block_outs[idx] = flat
    return hook
hooks = [dit.blocks[i].register_forward_hook(make_hook(i)) for i in range(28)]

# ── Get x_emb (what goes into the blocks) ──
torch.set_grad_enabled(False)
x_emb, rope_emb, extra_pos = dit.prepare_embedded_sequence(x_b)
x_flat = x_emb.detach().cpu().reshape(512, D).float().numpy().astype(np.float16)
ctx_flat = ctx_b.detach().cpu().reshape(1024, 1024).float().numpy().astype(np.float16)
print(f"  x_emb: {x_emb.shape} -> x_flat: {x_flat.shape} [{x_flat.min():.4f}, {x_flat.max():.4f}]")

# ── C++ style t_emb (sinusoidal + RMSNorm, sigma=1.0) ──
w1 = dit.t_embedder[1].linear_1.weight.float().to(DEV)
w2 = dit.t_embedder[1].linear_2.weight.float().to(DEV)
w_ln = dit.t_embedding_norm.weight.float().to(DEV)
halfD = D // 2
sigma = 1.0
j = torch.arange(halfD, dtype=torch.float32, device=DEV)
freqs = sigma * torch.exp(-torch.log(torch.tensor(10000.0)) * j / halfD)
sin_emb = torch.zeros(M, D, dtype=torch.float32, device=DEV)
sin_emb[:, :halfD] = torch.cos(freqs).unsqueeze(0)
sin_emb[:, halfD:] = torch.sin(freqs).unsqueeze(0)
rms = torch.sqrt((sin_emb * sin_emb).mean(-1, keepdim=True) + 1e-6)
t_emb = (sin_emb * w_ln.unsqueeze(0) / rms).detach().cpu().float().numpy().astype(np.float16)
print(f"  t_emb: {t_emb.shape} [{t_emb.min():.4f}, {t_emb.max():.4f}]")

# ── Save inputs ──
np.save(f'{OUTDIR}/x_flat.npy', x_flat)
np.save(f'{OUTDIR}/t_emb.npy', t_emb)
np.save(f'{OUTDIR}/ctx_flat.npy', ctx_flat)
print(f"  Saved inputs to {OUTDIR}/")

# ── Run dit.forward() ──
ts = torch.tensor([sigma, sigma], dtype=DTYPE, device=DEV)
print("\nRunning dit.forward() (PyTorch native, WITH RoPE)...")
t0 = time.time()
with torch.no_grad():
    v_b = dit(x_b, ts, ctx_b)
dt = time.time() - t0
print(f"  Forward: {dt:.1f}s")

# ── Save per-block outputs ──
print("\nPer-block outputs:")
for i in range(28):
    if i in block_outs:
        out = block_outs[i]
        np.save(f'{OUTDIR}/block_{i:02d}_pt.npy', out)
        ok = np.isfinite(out)
        if ok.sum() > 0:
            print(f"  Block {i:2d}: [{out[ok].min():.1f}, {out[ok].max():.1f}] nan={np.sum(~ok)}")
        else:
            print(f"  Block {i:2d}: ALL NaN! nan={np.sum(~ok)}")
    else:
        print(f"  Block {i:2d}: MISSING")

# ── Cleanup ──
for h in hooks:
    h.remove()

v_flat = v_b.detach().cpu().float().reshape(512, D).numpy()
ok = np.isfinite(v_flat)
print(f"\nFinal dit output: [{v_flat[ok].min():.2f}, {v_flat[ok].max():.2f}] nan={np.sum(~ok)}")
print(f"DONE. Files in {OUTDIR}/")

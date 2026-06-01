"""Compare C++ engine block outputs against PyTorch reference (no RoPE).

Usage (WSL2):
    source /home/riorg/miniconda3/etc/profile.d/conda.sh
    conda activate /home/riorg/anima-work/.conda
    python /mnt/d/AI/anima_phone/scripts/compare_blocks.py
"""
import sys, os, time, gc
sys.path.insert(0, "/mnt/d/AI/anima_phone/src")
import torch
import torch.nn.functional as F
import numpy as np

DEV = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16
D = 2048; N_HEADS = 16; HEAD_DIM = 128; MLP_HIDDEN = 8192; NCTX = 512; CTXD = 1024

# Load C++ outputs
CMP = "/mnt/d/AI/anima_phone/output/cmp"
cpp_x = np.load(f"{CMP}/x_phone.npy")   # [512, 2048] fp16
cpp_ctx = np.load(f"{CMP}/ctx_phone.npy")  # [1024, 1024] fp16

print(f"C++ x: {cpp_x.shape} range=[{cpp_x.min():.4f}, {cpp_x.max():.4f}]")
print(f"C++ ctx: {cpp_ctx.shape} range=[{cpp_ctx.min():.4f}, {cpp_ctx.max():.4f}]")

cpp_blocks = []
for b in range(28):
    a = np.load(f"{CMP}/block_{b:02d}_cpp.npy")
    cpp_blocks.append(a)
    if b < 3 or b > 24:
        print(f"  C++ block {b:2d}: range=[{a.min():.2f}, {a.max():.2f}]  nan={np.isnan(a).sum()}")
print(f"  ... [28 blocks loaded]")

# ── PyTorch reference ──
print("\nLoading DiT...")
import predict2

config = dict(max_img_h=240, max_img_w=240, max_frames=128, in_channels=16, out_channels=16,
    patch_spatial=2, patch_temporal=1, concat_padding_mask=True, model_channels=D,
    num_blocks=28, num_heads=N_HEADS, mlp_ratio=4.0, crossattn_emb_channels=CTXD,
    pos_emb_cls="rope3d", pos_emb_learnable=True, pos_emb_interpolation="crop",
    min_fps=1, max_fps=30, use_adaln_lora=True, adaln_lora_dim=256,
    rope_h_extrapolation_ratio=4.0, rope_w_extrapolation_ratio=4.0, rope_t_extrapolation_ratio=1.0,
    extra_per_block_abs_pos_emb=False, rope_enable_fps_modulation=False)

t0 = time.time()
sd_raw = torch.load("/mnt/d/AI/anima_phone/models/diffusion_weights_fp16.pt", weights_only=True)
sd = {}
for k, v in sd_raw.items():
    while k.startswith("net."):
        k = k[4:]
    sd[k] = v
del sd_raw

dit = predict2.MiniTrainDIT(**config, device=DEV, dtype=DTYPE, operations=torch.nn)
dit.load_state_dict(sd, strict=False)
dit.eval()
del sd; gc.collect(); torch.cuda.empty_cache()
print(f"  Loaded in {time.time()-t0:.1f}s, {len(dit.blocks)} blocks")

# ── Compute t_emb matching phone's dit_compute_timestep(sigma=1.0) ──
sigma = 1.0
M = 2  # CFG batch
ts = torch.tensor([sigma, sigma], dtype=DTYPE, device=DEV).unsqueeze(1)  # [2, 1]
with torch.no_grad():
    t_emb_raw = dit.t_embedder[0](ts).to(DTYPE)
    t_emb_out, lora = dit.t_embedder[1](t_emb_raw)
    t_emb = dit.t_embedding_norm(t_emb_out)  # [2, 1, D]
print(f"  t_emb: {t_emb.shape} range=[{t_emb.min():.4f}, {t_emb.max():.4f}]")
print(f"  lora: {lora.shape} range=[{lora.min():.4f}, {lora.max():.4f}]")

# ── Run blocks with RoPE disabled (matching C++ engine) ──
MS = M * 256  # = 512
x_pt = torch.from_numpy(cpp_x.astype(np.float32)).to(DEV, DTYPE)  # [512, 2048]
ctx_pt = torch.from_numpy(cpp_ctx.astype(np.float32)).to(DEV, DTYPE)  # [1024, 1024]
ctx_3d = ctx_pt.reshape(M, NCTX, CTXD)  # [2, 512, 1024]

# Reshape x to 5D for Block.forward
HP = 16  # patch spatial = 32/2
x_5d = x_pt.reshape(M, 1, HP, HP, D)  # [2, 1, 16, 16, 2048]

print(f"\nRunning 28 blocks (RoPE=None)...")
t0 = time.time()
pt_blocks = []
with torch.no_grad():
    x = x_5d
    for i, block in enumerate(dit.blocks):
        x = block.forward(x, t_emb, ctx_3d, rope_emb_L_1_1_D=None, adaln_lora_B_T_3D=lora)
        flat = x.reshape(MS, D).cpu().numpy().astype(np.float16)
        pt_blocks.append(flat)

        # Print early and late blocks
        if i < 3 or i > 24:
            f = flat[np.isfinite(flat)]
            print(f"  PT  block {i:2d}: range=[{f.min():.2f}, {f.max():.2f}]"
                  if len(f) > 0 else f"  PT  block {i:2d}: ALL NaN")

print(f"  ... [28 blocks done in {time.time()-t0:.1f}s]")

# ── Compare ──
print(f"\n{'='*80}")
print(f"Block-by-block comparison: C++ vs PyTorch (no RoPE)")
print(f"{'Block':<6} {'C++ range':<30} {'PT range':<30} {'max_err':<12} {'mean_err':<12}")
print(f"{'-'*6} {'-'*30} {'-'*30} {'-'*12} {'-'*12}")

for b in range(28):
    cpp = cpp_blocks[b].astype(np.float32)
    pt = pt_blocks[b].astype(np.float32)

    # Compute error only on finite values
    cpp_ok = np.isfinite(cpp)
    pt_ok = np.isfinite(pt)
    both_ok = cpp_ok & pt_ok

    if both_ok.sum() > 0:
        diff = np.abs(cpp[both_ok] - pt[both_ok])
        max_err = diff.max()
        mean_err = diff.mean()
    else:
        max_err = float('inf')
        mean_err = float('inf')

    cpp_nan = (~cpp_ok).sum()
    pt_nan = (~pt_ok).sum()

    cpp_rng = f"[{cpp[cpp_ok].min():.1f}, {cpp[cpp_ok].max():.1f}]" if cpp_ok.sum() > 0 else "ALL NaN"
    pt_rng = f"[{pt[pt_ok].min():.1f}, {pt[pt_ok].max():.1f}]" if pt_ok.sum() > 0 else "ALL NaN"

    flag = ""
    if max_err > 100:
        flag = " ⚠️ LARGE"
    elif max_err > 1:
        flag = " ⚠️"

    print(f"  {b:2d}    {cpp_rng:<30} {pt_rng:<30} {max_err:<12.2f} {mean_err:<12.2f}{flag}")

    # Stop after first bad block for drill-down
    if max_err > 100 and b >= 1:
        print(f"\n  → Divergence starts at block {b} (max_err={max_err:.1f})")
        print(f"  → Block {b-1} was within tolerance — the bug is in this block's computation")
        break

print(f"\nDone. First diverging block identified.")

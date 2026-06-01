"""Test: C++ vs PyTorch with M=1 (no batch mixing in attention).
Runs on PC using same inputs, compares block outputs.
"""
import sys, os, time, gc
sys.path.insert(0, "/mnt/d/AI/anima_phone/src")
import torch, torch.nn.functional as F, numpy as np
import predict2

DEV = "cuda"; DTYPE = torch.float16
M, S, D = 1, 512, 2048  # M=1, but keep MS=512 by doubling S
MS = M * S  # = 512
N_HEADS, HEAD_DIM = 16, 128
NCTX, CTXD = 512, 1024

# Generate same inputs as phone but with M=1
SEED = 12345
rng = np.random.RandomState(SEED)
x_np = (rng.randn(MS, D).astype(np.float32) * 0.02).astype(np.float16)
ctx_np = (rng.randn(M * NCTX, CTXD).astype(np.float32) * 0.5).astype(np.float16)

# Load DiT
print("Loading DiT...")
config = dict(max_img_h=240, max_img_w=240, max_frames=128, in_channels=16, out_channels=16,
    patch_spatial=2, patch_temporal=1, concat_padding_mask=True, model_channels=D,
    num_blocks=28, num_heads=N_HEADS, mlp_ratio=4.0, crossattn_emb_channels=CTXD,
    pos_emb_cls="rope3d", pos_emb_learnable=True, pos_emb_interpolation="crop",
    min_fps=1, max_fps=30, use_adaln_lora=True, adaln_lora_dim=256,
    rope_h_extrapolation_ratio=4.0, rope_w_extrapolation_ratio=4.0, rope_t_extrapolation_ratio=1.0,
    extra_per_block_abs_pos_emb=False, rope_enable_fps_modulation=False)

sd_raw = torch.load("/mnt/d/AI/anima_phone/models/diffusion_weights_fp16.pt", weights_only=True)
sd = {}
for k, v in sd_raw.items():
    while k.startswith("net."): k = k[4:]
    sd[k] = v
del sd_raw

dit = predict2.MiniTrainDIT(**config, device=DEV, dtype=DTYPE, operations=torch.nn)
dit.load_state_dict(sd, strict=False); dit.eval()
del sd; gc.collect(); torch.cuda.empty_cache()

# ── PyTorch: per-batch attention (correct, Block.forward with rope_emb=None) ──
sigma = 1.0
ts = torch.tensor([sigma]*M, dtype=DTYPE, device=DEV).unsqueeze(1)
ctx_pt = torch.from_numpy(ctx_np.astype(np.float32)).to(DEV, DTYPE).reshape(M, NCTX, CTXD)
x_pt = torch.from_numpy(x_np.astype(np.float32)).to(DEV, DTYPE)
x_5d = x_pt.reshape(M, 1, S//(M), S//(M), D) if M == 1 else x_pt.reshape(M, 1, 16, 16, D)
# For M=1, S=512 → spatial = sqrt(512) = 22.6... not perfect square!
# Let me use a simpler approach: just run without reshape for M=1

print(f"M={M}, MS={MS}, spatial={int(np.sqrt(MS))}")

# Actually for M=1, MS=512 tokens → spatial ≈ 22.6, which doesn't work for 5D format.
# Block.forward expects 5D input. Let me use S_per_batch = MS//M = 512.
# But spatial = sqrt(512) ≈ 22.6 — not a square! This won't work for 5D reshaping.
#
# So for M=1 test, let me use S=256 (spatial=16), MS=256, M=1.
# This matches the original test but with single batch.

MS = 256; S = 256; M = 1
x_np = (rng.randn(MS, D).astype(np.float32) * 0.02).astype(np.float16)
ctx_np = (rng.randn(M * NCTX, CTXD).astype(np.float32) * 0.5).astype(np.float16)

x_pt = torch.from_numpy(x_np.astype(np.float32)).to(DEV, DTYPE)
ctx_pt = torch.from_numpy(ctx_np.astype(np.float32)).to(DEV, DTYPE)
ctx_3d = ctx_pt.reshape(M, NCTX, CTXD)

HP = int(np.sqrt(MS))  # = 16
x_5d = x_pt.reshape(M, 1, HP, HP, D)

# Compute t_emb
ts = torch.tensor([sigma]*M, dtype=DTYPE, device=DEV).unsqueeze(1)
with torch.no_grad():
    t_emb_raw = dit.t_embedder[0](ts).to(DTYPE)
    t_emb_out, lora = dit.t_embedder[1](t_emb_raw)
    t_emb = dit.t_embedding_norm(t_emb_out)

print(f"t_emb: {t_emb.shape}  lora: {lora.shape}")
print(f"x_5d: {x_5d.shape}  ctx_3d: {ctx_3d.shape}")

# Run blocks
print("\nRunning 28 blocks (PyTorch, M=1, RoPE=None)...")
pt_outs = []
with torch.no_grad():
    x = x_5d
    for i in range(28):
        x = dit.blocks[i].forward(x, t_emb, ctx_3d, rope_emb_L_1_1_D=None, adaln_lora_B_T_3D=lora)
        flat = x.reshape(MS, D).cpu().numpy().astype(np.float16)
        pt_outs.append(flat)
        if i < 3 or i > 24:
            f = flat[np.isfinite(flat)]
            print(f"  Block {i:2d}: range=[{f.min():.2f}, {f.max():.2f}]" if len(f) else f"  Block {i:2d}: ALL NaN")

# Compare with C++ outputs (pulled from phone)
# C++ outputs are M=2, MS=512. For M=1 comparison, we'd need to re-run C++ with M=1.
# But we haven't done that yet.
#
# Let me just check if PyTorch M=1 produces cleaner values than M=2.
# If so, batch mixing was the issue.
print(f"\nPyTorch M=1 final block range: [{pt_outs[-1].min():.2f}, {pt_outs[-1].max():.2f}]")
print(f"(For reference, C++ M=2 final block range at Block 0 was [-8.20, 16.94])")
print(f"(PyTorch M=2 final block 0 range was [-6.48, 16.16], block 1 was [-368, 801])")

# The key insight: with M=1, PyTorch should have much cleaner values.
# If so, the C++ engine needs per-batch attention.

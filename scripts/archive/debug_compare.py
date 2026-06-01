"""PC-side comparison: PyTorch vs C++ engine for same inputs."""
import sys, os
sys.path.insert(0, "/mnt/d/AI/anima_phone/src")
import torch
import numpy as np
import predict2

DTYPE = torch.float16
DEV = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEV}")

# Load debug data
data_dir = "/mnt/d/AI/anima_phone/output"
x_np = np.load(f"{data_dir}/debug_x.npy")  # [512, 2048] fp16
ctx_np = np.load(f"{data_dir}/debug_ctx.npy")  # [1024, 2048] fp16
out_cpp = np.load(f"{data_dir}/debug_out_cpp.npy")  # [512, 2048] fp16
sigma = float(np.load(f"{data_dir}/debug_sigma.npy")[0])

print(f"x: shape={x_np.shape} range=[{x_np.min():.3f}, {x_np.max():.3f}]")
print(f"ctx: shape={ctx_np.shape} range=[{ctx_np.min():.3f}, {ctx_np.max():.3f}]")
print(f"out_cpp: range=[{out_cpp.min():.3f}, {out_cpp.max():.3f}]")
print(f"sigma: {sigma}")

# Load full DiT model (28 blocks)
print("\nLoading full DiT (28 blocks)...")
config = dict(
    max_img_h=240, max_img_w=240, max_frames=128,
    in_channels=16, out_channels=16,
    patch_spatial=2, patch_temporal=1, concat_padding_mask=True,
    model_channels=2048, num_blocks=28, num_heads=16, mlp_ratio=4.0,
    crossattn_emb_channels=1024,
    pos_emb_cls="rope3d", pos_emb_learnable=True, pos_emb_interpolation="crop",
    min_fps=1, max_fps=30, use_adaln_lora=True, adaln_lora_dim=256,
    rope_h_extrapolation_ratio=4.0, rope_w_extrapolation_ratio=4.0,
    rope_t_extrapolation_ratio=1.0,
    extra_per_block_abs_pos_emb=False, rope_enable_fps_modulation=False,
)

from vk_ops import HybridOps
dit = predict2.MiniTrainDIT(**config, device=DEV, dtype=DTYPE, operations=HybridOps)
sd = torch.load("/mnt/d/AI/anima_phone/models/diffusion_weights_fp16.pt",
                map_location=DEV, weights_only=True)

# Rename keys: phone .pt uses "net." prefix, PyTorch model has no prefix
new_sd = {}
for k, v in sd.items():
    if k.startswith("net."):
        new_sd[k[4:]] = v
    else:
        new_sd[k] = v
sd = new_sD

dit.load_state_dict(sd, strict=True)
dit.eval()
print(f"DiT loaded: {len(dit.blocks)} blocks, {sum(v.numel() for v in sd.values())/1e6:.0f}M params")

# Prepare inputs matching C++ dit_forward_step flow
M, MS, D = 2, 512, 2048
Nctx, CtxD = 512, 1024

x_pt = torch.from_numpy(x_np).to(DEV).to(DTYPE)  # [512, 2048]
ctx_pt = torch.from_numpy(ctx_np).to(DEV).to(DTYPE)  # [1024, 2048]

# Need t_emb and lora. C++ uses dit_compute_timestep(sigma=1.0).
# Replicate: PyTorch t_embedder
ts = torch.tensor([sigma, sigma], dtype=DTYPE, device=DEV).unsqueeze(1)  # [2, 1]
t_raw = dit.t_embedder[0](ts).to(DTYPE)
t_emb_pt, lora_pt = dit.t_embedder[1](t_raw)
t_emb_pt = dit.t_embedding_norm(t_emb_pt)
print(f"t_emb: range=[{t_emb_pt.min():.3f}, {t_emb_pt.max():.3f}]")
print(f"lora: shape={lora_pt.shape} range=[{lora_pt.min():.3f}, {lora_pt.max():.3f}]")

# Prepare x_emb: need spatial dims. With 512 tokens and B=2, M=512:
# x_emb should be [B, T, H, W, D] = [2, 1, 16, 16, 2048] for 256x256
# But we got flat x_np [512, 2048], need to reshape
x_emb_pt = x_pt.reshape(2, 1, 16, 16, D)  # [2, 1, 16, 16, 2048]
ctx_pt_2 = ctx_pt.reshape(2, 512, CtxD)  # [2, 512, 1024]

# Note: C++ engine does NOT use RoPE or extra_pos_emb. The blocks compute
# self/cross attention without positional embeddings (RoPE not integrated).
# So we should skip RoPE in PyTorch comparison too.
# Actually, looking at C++ record_segment_self_attn, there's NO RoPE dispatch.
# So the comparison should match — no RoPE on either side.

# Run PyTorch block-level forward
# The C++ engine computes blocks sequentially on flat [MS, D]
# We'll iterate blocks manually and compare with C++ output

print("\n=== Block-by-block comparison ===")
x_flat_pt = x_pt.clone()  # [512, 2048]
ctx_flat_pt = ctx_pt.clone()  # [1024, 2048]

# Replicate C++ forward step: for each block, process through self/cross/MLP
# But PyTorch's block.forward includes RoPE inside. We need to bypass RoPE.
# The simplest: use PyTorch's block but set rope_emb to None/zero
# Actually, blocks in predict2 use RoPE internally. We should match the C++ behavior
# where RoPE is NOT computed.

# Let's just compute the full forward and compare final output,
# then if there's a big difference, drill down into individual blocks.

with torch.no_grad():
    # Full PyTorch forward (includes RoPE, which C++ doesn't have!)
    # This won't match exactly, but let's see the magnitude of difference
    x_3d = x_emb_pt  # [2, 1, 16, 16, 2048]

    # Prepare RoPE embeddings (PyTorch needs them, C++ doesn't use them)
    B, T, H_img, W_img = x_3d.shape[:4]
    # Manually call prepare_embedded_sequence? No, that changes x_3d.

    # Actually, the forward method signature from predict2:
    # def forward(self, x, t_emb, context, rope_emb, adaln_lora_B_T_3D=None, extra_pos_emb=None)

    # We need rope_emb. Let's use zero rope_emb to match C++ (no RoPE)
    # rope_emb is per-block position info
    # Actually, let's just call with rope_emb=None to see what happens
    # But the model might crash without rope_emb

    # Simplest: use the full prepare_embedded_sequence + forward
    # This includes RoPE in PyTorch but NOT in C++, so expect difference

    # Actually let's just check if the per-block output can be extracted
    # by hooking into the blocks

# Let's try a different approach: iterate blocks manually
# Each block has: self_attn, cross_attn, mlp, adaln_modulation
# The block.forward signature: (x, t_emb, context, rope_emb, adaln_lora)

# To match C++ (no RoPE), we pass rope_emb=None
# C++ block computation: for each block b in 0..27:
#   AdaLN from t_emb + lora → bcBuf
#   Self-attn: LN→AdaLN→QKV→norms→attn→O→gate
#   Cross-attn: LN→AdaLN→QKV→norms→attn→O→gate
#   MLP: LN→AdaLN→fc1→SiLU→fc2→gate

# PyTorch implementation in predict2 block:
# class Block:
#   def forward(x, t_emb, context, rope_emb, adaln_lora=None):
#     # self-attn
#     m = self.adaln_modulation_self_attn(t_emb, adaln_lora)
#     x_norm = self.self_attn.norm1(x)
#     x_attn = x_norm * (1+m.scale) + m.shift  # AdaLN apply
#     qkv = self.self_attn.qkv(x_attn)
#     q,k,v = split(qkv)
#     q,k = apply_rope(q, k, rope_emb)  # <-- RoPE here!
#     attn_out = sdpa(q, k, v)
#     x = x + m.gate * self.self_attn.proj(attn_out)

# The C++ equivalent skips RoPE (apply_rope step).

# For comparison, we should use a PyTorch baseline that also skips RoPE.
# Options:
# 1. Monkey-patch blocks to skip RoPE
# 2. Pass rope_emb=zeros
# 3. Accept the RoPE difference and look for large errors

# Let's try option 2: zero rope_emb
# But rope_emb might be cos/sin frequencies, zero gives wrong rotation
# Actually, with freqs=0: cos(0)=1, sin(0)=0. RoPE with zero freqs = identity.
# So rope_emb with all-zero cos/sin should make RoPE a no-op.
# But rope_emb shape depends on the block's RoPE implementation.

# Let me just load the full model, manually call each block with rope_emb=None,
# and see if predict2 handles None rope_emb.

# Actually, looking at predict2.py's block code, rope_emb is used in self_attn and cross_attn.
# If we don't pass it through, the attention computation might crash.
# The simplest approach: compute the rope_emb from the model but check if
# the per-block forward matches C++ with zero freqs.

# Let me just compute per-block outputs manually.
# For each block, we need: x_flat [512, 2048], ctx_flat [1024, 2048], t_emb, lora.

# Since we can't easily replicate C++'s no-RoPE behavior in PyTorch blocks,
# let me compare the output value ranges between C++ and PyTorch (with RoPE).
# If C++ output is extreme while PyTorch is reasonable, RoPE is not the issue.

print("Running PyTorch forward (with RoPE, for range comparison)...")
with torch.no_grad():
    # Full PyTorch pipeline
    B = 2
    x_3d = x_np.reshape(B, 1, 16, 16, D)  # [2, 1, 16, 16, 2048]
    x_3d_pt = torch.from_numpy(x_3d).to(DEV).to(DTYPE)
    ctx_3d_pt = torch.from_numpy(ctx_np.reshape(B, Nctx, CtxD)).to(DEV).to(DTYPE)

    # Get rope_emb
    rope_emb = dit.rope_embedder(x_3d_pt)

    # Manual block iteration matching C++ flat layout
    x_flat = x_pt.clone()
    for b_idx, block in enumerate(dit.blocks):
        # Get block output (includes RoPE)
        # block.forward expects [B, T, H, W, D] format, not flat

        # Actually this is getting messy. Let me just run the full forward
        # with prepare_embedded_sequence to handle RoPE properly,
        # then compare final output stats.
        pass

# Let me just run the full standard forward for comparison
print("Running full PyTorch DiT forward...")
with torch.no_grad():
    # Standard forward with all processing
    B = 2
    x_in = torch.from_numpy(x_np.reshape(B, 1, 16, 16, D)).to(DEV).to(DTYPE)
    ctx_in = torch.from_numpy(ctx_np.reshape(B, Nctx, CtxD)).to(DEV).to(DTYPE)

    # This matches phone_pipeline.py's pre-processing
    x_emb, rope_emb_data, extra_pos = dit.prepare_embedded_sequence(x_in)

    t_raw = dit.t_embedder[0](ts).to(DTYPE)
    t_emb_pt2, lora_pt2 = dit.t_embedder[1](t_raw)
    t_emb_pt2 = dit.t_embedding_norm(t_emb_pt2)

    # Forward through all blocks
    out_pt = dit.forward(x_emb, t_emb_pt2, ctx_in, rope_emb_data,
                         adaln_lora_B_T_3D=lora_pt2, extra_pos_emb=extra_pos)

    # Flatten for comparison
    out_pt_flat = out_pt.reshape(-1, D)
    print(f"PyTorch out: shape={out_pt_flat.shape} range=[{out_pt_flat.min():.3f}, {out_pt_flat.max():.3f}]")

# Compare
diff = np.abs(out_cpp.astype(np.float32) - out_pt_flat.cpu().numpy().astype(np.float32))
print(f"\nMax diff: {diff.max():.3f}")
print(f"Mean diff: {diff.mean():.3f}")
print(f"C++ range: [{out_cpp.min():.3f}, {out_cpp.max():.3f}]")
print(f"PT range: [{out_pt_flat.min():.3f}, {out_pt_flat.max():.3f}]")

# If diff is huge (>100), drill into individual blocks

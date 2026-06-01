"""Step 0: PC-side block-by-block PyTorch reference vs C++ engine alignment.

Replicates C++ engine's exact computation path:
- No RoPE (C++ hasn't integrated it)
- Flat attention (all positions attend to all positions, matching C++'s flat layout)
- Same compute order: AdaLN→self_attn→cross_attn→MLP per block
- FP16 compute for direct C++ comparison, FP32 as ground truth

Usage (in WSL2 conda env):
    source /home/riorg/miniconda3/etc/profile.d/conda.sh
    conda activate /home/riorg/anima-work/.conda
    python /mnt/d/AI/anima_phone/scripts/validate_blocks.py [--dump]

Output:
    output/validate/block_XX_fp16.npy — PyTorch FP16 block output
    output/validate/block_XX_fp32.npy — PyTorch FP32 ground truth
    output/validate/block_XX_inputs.npz  — input tensors for C++ comparison
"""
import sys, os, argparse, time
import torch
import torch.nn.functional as F
import numpy as np

SRC = "/mnt/d/AI/anima_phone/src"
sys.path.insert(0, SRC)

DTYPE_FP16 = torch.float16
DTYPE_FP32 = torch.float32
DEV = "cuda" if torch.cuda.is_available() else "cpu"

OUT = "/mnt/d/AI/anima_phone/output/validate"
os.makedirs(OUT, exist_ok=True)

# ── Config: match C++ engine constants ──
M = 1               # single batch for clean alignment (C++ engine doesn't separate batches in attention)
S = 256             # spatial tokens
MS = M * S          # 256
D = 2048
Nctx = 512          # context tokens (padded LLMAdapter output)
CtxD = 1024          # context hidden dim
N_HEADS = 16
HEAD_DIM = 128
MLP_HIDDEN = 8192
ADALN_LORA_DIM = 256
EPS = 1e-6


def main(dump_inputs=True):
    print(f"Device: {DEV}")
    print(f"Dtype: FP16={DTYPE_FP16}, FP32={DTYPE_FP32}")
    print(f"M={M} S={S} MS={MS} D={D} Nctx={Nctx} CtxD={CtxD}")

    # ── Load model ──
    print("\nLoading DiT model...")
    t0 = time.time()
    # Import predict2 AFTER setting device (it checks os.environ)
    import predict2

    config = dict(
        max_img_h=240, max_img_w=240, max_frames=128,
        in_channels=16, out_channels=16,
        patch_spatial=2, patch_temporal=1, concat_padding_mask=True,
        model_channels=D, num_blocks=28, num_heads=N_HEADS, mlp_ratio=4.0,
        crossattn_emb_channels=CtxD,
        pos_emb_cls="rope3d", pos_emb_learnable=True, pos_emb_interpolation="crop",
        min_fps=1, max_fps=30, use_adaln_lora=True, adaln_lora_dim=ADALN_LORA_DIM,
        rope_h_extrapolation_ratio=4.0, rope_w_extrapolation_ratio=4.0,
        rope_t_extrapolation_ratio=1.0,
        extra_per_block_abs_pos_emb=False, rope_enable_fps_modulation=False,
    )

    wt_path = "/mnt/d/AI/anima_phone/models/diffusion_weights_fp16.pt"
    sd_raw = torch.load(wt_path, map_location="cpu", weights_only=True)
    # Strip "net." prefix (phone .pt uses it)
    sd = {}
    for k, v in sd_raw.items():
        if k.startswith("net."):
            sd[k[4:]] = v
        else:
            sd[k] = v
    del sd_raw

    print(f"  Loaded weights in {time.time()-t0:.1f}s, {len(sd)} keys")

    # ── Generate inputs matching C++ engine ──
    # Fixed seed for reproducibility
    rng = np.random.RandomState(12345)

    # x: typical pipeline input (small values around zero, matching latent stats)
    x_np = (rng.randn(MS, D).astype(np.float32) * 0.02)

    # ctx: random context (mimics LLMAdapter output range)
    ctx_np = (rng.randn(M * Nctx, CtxD).astype(np.float32) * 0.5)

    # sigma: first step
    sigma = 1.0

    print(f"\nInput stats:")
    print(f"  x:    shape={x_np.shape}  range=[{x_np.min():.4f}, {x_np.max():.4f}]")
    print(f"  ctx:  shape={ctx_np.shape}  range=[{ctx_np.min():.4f}, {ctx_np.max():.4f}]")
    print(f"  sigma: {sigma}")

    # Save inputs for C++ comparison
    if dump_inputs:
        np.save(f"{OUT}/x_input.npy", x_np.astype(np.float16))
        np.save(f"{OUT}/ctx_input.npy", ctx_np.astype(np.float16))
        np.save(f"{OUT}/sigma.npy", np.array([sigma], dtype=np.float32))
        print(f"  Inputs saved to {OUT}/x_input.npy, ctx_input.npy, sigma.npy")

    # ── Convert to tensors ──
    x_fp32 = torch.from_numpy(x_np).to(DEV, DTYPE_FP32)  # [MS, D]
    ctx_fp32 = torch.from_numpy(ctx_np).to(DEV, DTYPE_FP32)  # [M*Nctx, CtxD]
    x_fp16 = x_fp32.to(DTYPE_FP16)
    ctx_fp16 = ctx_fp32.to(DTYPE_FP16)

    # ── Compute t_emb + lora (matching C++ dit_compute_timestep) ──
    # C++ t_embedder: no sigma→timestep mapping, uses sigma directly as float
    # We need PyTorch's t_embedder to get t_emb and lora
    # Load t_embedder weights
    tw = {k: v for k, v in sd.items() if k.startswith("t_embedder") or k == "t_embedding_norm.weight"}

    # t_embedder is nn.Sequential:
    #   [0] Linear(D, D, bias=False) — from t_embedder.0.weight
    #   [1] TimestepEmbedder style: generates t_emb + lora
    # Actually from predict2.py: t_embedder = nn.Sequential(t_embedder_0, t_embedder_1)
    # t_embedder_1 is from Cosmos predict2, outputs (t_emb, adaln_lora_B_T_3D)
    # We need to replicate C++ t_emb computation directly using the weights.

    # Let's just load the key weights and compute manually:
    # C++ computes t_emb via: SiLU(ts_linear) -> t_emb_linear -> t_embedding_norm
    # No, C++ calls dit_compute_timestep which is a completely separate C++ path!
    #
    # Looking at the C++ code, dit_compute_timestep computes t_emb and lora
    # using CPU-side computation (not Vulkan). The t_emb is stored in g_tEmbBuf.
    #
    # For our validation, we need to REPLICATE this t_emb computation exactly.
    # Let me check what dit_compute_timestep does...

    # From the C++ engine: dit_compute_timestep is defined later.
    # It likely does: sin/cos embedding of sigma → linear → embedding_norm
    # But we have the FULL PyTorch model, so let's just run the PyTorch t_embedder
    # on the same sigma and verify C++ produces the same t_emb.

    # For now, let me create a minimal t_embedder from the loaded weights
    # t_embedder structure:
    #   [0]: Linear(D, D, bias=False) with weight t_embedder.0.weight
    #   [1]: TimestepEmbedder (custom, generates t_emb + lora)
    # t_embedding_norm: nn.LayerNorm(D, eps=1e-6)

    from collections import OrderedDict

    # Build t_embedder from raw weights
    # Looking at predict2.py more carefully:
    # self.t_embedder = CosmosTEmbedder or similar
    # The exact structure depends on predict2 internals

    # Actually, let me just instantiate the full model and use its forward
    # But we need to be careful about memory

    print("\nInstantiating DiT (FP16 on GPU)...")
    t0 = time.time()
    dit_fp16 = predict2.MiniTrainDIT(**config, device=DEV, dtype=DTYPE_FP16, operations=torch.nn)
    dit_fp16.load_state_dict(sd, strict=False)
    dit_fp16.eval()
    print(f"  Init in {time.time()-t0:.1f}s, {len(dit_fp16.blocks)} blocks")
    del sd  # free CPU memory

    # ── Compute reference t_emb ──
    # t_embedder expects timestep tensor, not sigma
    # The C++ dit_compute_timestep(sigma) computes:
    #   ts = flow timestep from sigma
    #   t_emb via t_embedder
    #   lora via t_embedder (second output)
    #
    # For our test, we can use a simple approach:
    #   Use PyTorch's t_embedder with a fixed timestep
    #
    # But to match C++ exactly, we should call dit_compute_timestep.
    # Since we can't call C++ from PC, let's compute t_emb manually.
    #
    # Actually, looking at the C++ side more carefully in phone_pipeline.py:
    # _lib_vk.dit_compute_timestep(ct.c_float(sigma))
    # This computes t_emb from sigma using the same formula as PyTorch.
    #
    # But the t_embedder's actual computation uses the flow timestep,
    # not sigma. Let me trace what PyTorch does...

    # Actually, for the block comparison, the exact t_emb doesn't matter as long as
    # BOTH PyTorch and C++ use the SAME t_emb. We can export t_emb from C++ later.
    # For now, let's use PyTorch's t_embedder with sigma=1 to get a representative t_emb.

    # FlowMatchScheduler: ts = time_snr_shift(alpha, sigma)
    # For sigma=1.0: ts = 3.0 * 1.0 / (1.0 + 2.0 * 1.0) = 1.0
    # But actually, the flow model uses a different mapping...

    # Let me use a simpler approach: just pick a representative sigma and
    # compute the PyTorch t_emb. Then for C++ comparison, we'll make sure
    # C++ uses the same t_emb (we can pass t_emb directly instead of sigma).

    # Use sigma directly in the timestep (some flow models do this)
    ts = torch.tensor([[sigma]], dtype=DTYPE_FP16, device=DEV)  # [1, 1]
    with torch.no_grad():
        t_emb_raw = dit_fp16.t_embedder[0](ts).to(DTYPE_FP16)  # sine/cosine embedding → [1, D]
        t_emb_out, lora = dit_fp16.t_embedder[1](t_emb_raw)
        # t_emb_out: [1, D], lora: [1, 3*D]
        t_emb_fp16 = dit_fp16.t_embedding_norm(t_emb_out)  # [1, D]
        # Repeat for MS: t_emb is [M, D], one per batch
        # For M=1, t_emb_fp16 shape is [1, D] already

    # But wait, t_embedder might not handle [1,1] shape correctly.
    # Let me check what shape t_embedder expects...

    # Looking at predict2, the t_embedder is a nn.Sequential.
    # Let me check the first layer...

    # Squeeze temporal dim: t_emb [B, T, D] → [M, D], lora [B, T, 3D] → [M, 3*D]
    t_emb_fp16 = t_emb_fp16.squeeze(1)  # [M, D]
    lora = lora.squeeze(1)              # [M, 3*D]

    print(f"  t_emb shape: {t_emb_fp16.shape} range=[{t_emb_fp16.min():.4f}, {t_emb_fp16.max():.4f}]")
    print(f"  lora shape: {lora.shape} range=[{lora.min():.4f}, {lora.max():.4f}]")

    # ── Block-by-block forward (FP16, matching C++ engine) ──
    print("\n=== Block-by-block FP16 forward (C++ engine logic) ===")

    x_fp16_block = x_fp16.clone()  # [MS, D] — flat input
    torch.set_grad_enabled(False)  # ensure no grad accumulation with weight params
    block_outputs_fp16 = []  # store output after each block

    for b in range(28):
        pfx = f"blocks.{b}."

        # Get block weights
        # AdaLN modulation weights
        w0_s = dit_fp16.blocks[b].adaln_modulation_self_attn[1].weight  # [256, D]
        w2_s = dit_fp16.blocks[b].adaln_modulation_self_attn[2].weight  # [3*D, 256]
        w0_c = dit_fp16.blocks[b].adaln_modulation_cross_attn[1].weight
        w2_c = dit_fp16.blocks[b].adaln_modulation_cross_attn[2].weight
        w0_m = dit_fp16.blocks[b].adaln_modulation_mlp[1].weight
        w2_m = dit_fp16.blocks[b].adaln_modulation_mlp[2].weight

        # Attention weights
        q_w = dit_fp16.blocks[b].self_attn.q_proj.weight      # [D, D]
        k_w = dit_fp16.blocks[b].self_attn.k_proj.weight      # [D, D]
        v_w = dit_fp16.blocks[b].self_attn.v_proj.weight      # [D, D]
        o_w = dit_fp16.blocks[b].self_attn.output_proj.weight # [D, D]
        qn_w = dit_fp16.blocks[b].self_attn.q_norm.weight     # [head_dim] = [128]
        kn_w = dit_fp16.blocks[b].self_attn.k_norm.weight

        cx_q_w = dit_fp16.blocks[b].cross_attn.q_proj.weight  # [D, D]
        cx_k_w = dit_fp16.blocks[b].cross_attn.k_proj.weight  # [CtxD, D] = [1024, 2048]
        cx_v_w = dit_fp16.blocks[b].cross_attn.v_proj.weight  # [CtxD, D]
        cx_o_w = dit_fp16.blocks[b].cross_attn.output_proj.weight  # [D, D]
        cx_qn_w = dit_fp16.blocks[b].cross_attn.q_norm.weight  # [head_dim]
        cx_kn_w = dit_fp16.blocks[b].cross_attn.k_norm.weight

        l1_w = dit_fp16.blocks[b].mlp.layer1.weight  # [MLP_HIDDEN, D] = [8192, 2048]
        l2_w = dit_fp16.blocks[b].mlp.layer2.weight  # [D, MLP_HIDDEN] = [2048, 8192]

        # ── Helper: AdaLN compute (matches C++ adaln_gpu) ──
        def compute_adaln(w1, w2, lora_part):
            """Compute shift, scale, gate for one channel.
            w1: [256, D]  w2: [3*D, 256]  lora_part: [M, D] slice of full lora
            Returns: shift[M,D], scale[M,D], gate[M,D] each float16
            """
            h = F.silu(t_emb_fp16.float())                       # [M, D]
            h = F.linear(h, w1.float())                          # [M, 256]
            h = F.linear(h, w2.float())                          # [M, 3*D]
            h = h + lora_part.float()                            # add external lora
            shift, scale, gate = torch.chunk(h, 3, dim=-1)       # each [M, D]
            scale = scale + 1.0                                  # scale+1
            return shift.to(DTYPE_FP16), scale.to(DTYPE_FP16), gate.to(DTYPE_FP16)

        # C++ engine adds the SAME full lora [M, 3*D] to each AdaLN channel output.
        # PyTorch's Block.forward also uses the same adaln_lora_B_T_3D for all channels.
        # So pass full lora to all 3 compute_adaln calls.

        # Compute AdaLN for all 3 channels (each gets the same full lora)
        shift_s, scale_s, gate_s = compute_adaln(w0_s, w2_s, lora)
        shift_c, scale_c, gate_c = compute_adaln(w0_c, w2_c, lora)
        shift_m, scale_m, gate_m = compute_adaln(w0_m, w2_m, lora)

        # Broadcast [M, D] → [MS, D] (repeat_interleave S times along dim 0)
        # S = MS // M
        def broadcast(t):
            return t.repeat_interleave(S, dim=0)  # [MS, D]

        bc_scale_s = broadcast(scale_s)
        bc_shift_s = broadcast(shift_s)
        bc_gate_s  = broadcast(gate_s)
        bc_scale_c = broadcast(scale_c)
        bc_shift_c = broadcast(shift_c)
        bc_gate_c  = broadcast(gate_c)
        bc_scale_m = broadcast(scale_m)
        bc_shift_m = broadcast(shift_m)
        bc_gate_m  = broadcast(gate_m)

        x_in = x_fp16_block  # block input, [MS, D]

        # ── Self-attention (matching C++ Segment B) ──
        # LN
        y = F.layer_norm(x_in, (D,), weight=None, bias=None, eps=EPS)
        # AdaLN modulate: y * scale + shift
        y = y * bc_scale_s + bc_shift_s
        # QKV
        q = F.linear(y, q_w)    # [MS, D]
        k = F.linear(y, k_w)    # [MS, D]
        v = F.linear(y, v_w)    # [MS, D]
        # RMSNorm per-head: reshape to [MS*H, head_dim]
        q = F.rms_norm(q.reshape(MS * N_HEADS, HEAD_DIM), (HEAD_DIM,), weight=qn_w, eps=EPS)
        k = F.rms_norm(k.reshape(MS * N_HEADS, HEAD_DIM), (HEAD_DIM,), weight=kn_w, eps=EPS)
        # v_norm is Identity in PyTorch → no-op
        v_flat = v.reshape(MS * N_HEADS, HEAD_DIM)  # [MS*H, head_dim]

        # Flat attention: QK^T → softmax → AV (matching C++ attn_qkt + attn_softmax + attn_out)
        scale_attn = 1.0 / np.sqrt(HEAD_DIM)
        # Q: [MS*H, head_dim]  K: [MS*H, head_dim]
        # scores: [MS*H, MS] — each (position, head) attends to all positions
        # s[i, j] = dot(Q[i], K[j]) * scale where i = m_q*H+h, j = m_kv*H+h for same h only
        # In flat layout: Q[i] and K[j] have the same head only if i%H == j%H
        # The C++ shader only computes for same-head pairs.
        #
        # To replicate efficiently: reshape to [H, MS, head_dim] and batch matmul
        q_h = q.reshape(MS, N_HEADS, HEAD_DIM).permute(1, 0, 2)  # [H, MS, head_dim]
        k_h = k.reshape(MS, N_HEADS, HEAD_DIM).permute(1, 0, 2)  # [H, MS, head_dim]
        v_h = v_flat.reshape(MS, N_HEADS, HEAD_DIM).permute(1, 0, 2)  # [H, MS, head_dim]

        # QK^T per-head: [H, MS, MS]
        scores = torch.bmm(q_h, k_h.transpose(1, 2)) * scale_attn  # [H, MS, MS]

        # Softmax per-(query, head): softmax over key dimension (MS)
        attn_w = F.softmax(scores, dim=-1)  # [H, MS, MS]

        # AV: [H, MS, MS] @ [H, MS, head_dim] → [H, MS, head_dim]
        attn_o = torch.bmm(attn_w, v_h)  # [H, MS, head_dim]
        attn_o = attn_o.permute(1, 0, 2).reshape(MS, D)  # [MS, D]

        # Output projection
        sa_out = F.linear(attn_o, o_w)  # [MS, D]
        # Gated residual
        x_sa = x_in + bc_gate_s * sa_out  # [MS, D]

        # ── Cross-attention (matching C++ Segment C1 + C2) ──
        # LN
        y = F.layer_norm(x_sa, (D,), weight=None, bias=None, eps=EPS)
        # AdaLN modulate
        y = y * bc_scale_c + bc_shift_c
        # Q from x, K/V from ctx
        q_cx = F.linear(y, cx_q_w)        # [MS, D]
        k_cx = F.linear(ctx_fp16, cx_k_w)  # [M*Nctx, D]
        v_cx = F.linear(ctx_fp16, cx_v_w)  # [M*Nctx, D]
        # RMSNorm
        q_cx = F.rms_norm(q_cx.reshape(MS * N_HEADS, HEAD_DIM), (HEAD_DIM,), weight=cx_qn_w, eps=EPS)
        k_cx = F.rms_norm(k_cx.reshape(M * Nctx * N_HEADS, HEAD_DIM), (HEAD_DIM,), weight=cx_kn_w, eps=EPS)

        M_kv = M * Nctx  # 512
        q_cx_h = q_cx.reshape(MS, N_HEADS, HEAD_DIM).permute(1, 0, 2)       # [H, MS, head_dim]
        k_cx_h = k_cx.reshape(M_kv, N_HEADS, HEAD_DIM).permute(1, 0, 2)     # [H, M_kv, head_dim]
        v_cx_h = v_cx.reshape(M_kv, N_HEADS, HEAD_DIM).permute(1, 0, 2)     # [H, M_kv, head_dim]

        # QK^T: [H, MS, M_kv]
        cx_scores = torch.bmm(q_cx_h, k_cx_h.transpose(1, 2)) * scale_attn
        cx_attn_w = F.softmax(cx_scores, dim=-1)  # [H, MS, M_kv]
        # AV: [H, MS, M_kv] @ [H, M_kv, head_dim] → [H, MS, head_dim]
        cx_attn_o = torch.bmm(cx_attn_w, v_cx_h)
        cx_attn_o = cx_attn_o.permute(1, 0, 2).reshape(MS, D)  # [MS, D]

        # Output projection
        cx_out = F.linear(cx_attn_o, cx_o_w)  # [MS, D]
        # Gated residual
        x_cx = x_sa + bc_gate_c * cx_out  # [MS, D]

        # ── MLP (matching C++ Segment D) ──
        y = F.layer_norm(x_cx, (D,), weight=None, bias=None, eps=EPS)
        y = y * bc_scale_m + bc_shift_m
        y = F.linear(y, l1_w)           # [MS, MLP_HIDDEN]
        y = F.silu(y)
        y = F.linear(y, l2_w)           # [MS, D]
        # Gated residual
        x_out = x_cx + bc_gate_m * y    # outBuf

        x_fp16_block = x_out
        block_outputs_fp16.append(x_out.cpu().numpy().astype(np.float16))

        if b == 0 or (b + 1) % 7 == 0:
            f = x_out.float()
            print(f"  Block {b:2d}: out range=[{f.min():.3f}, {f.max():.3f}]  "
                  f"abs_mean={f.abs().mean():.3f}  nan={torch.isnan(f).sum().item()}")

    print(f"\nFinal output range: [{x_fp16_block.min():.3f}, {x_fp16_block.max():.3f}]")

    # ── Save per-block outputs ──
    for i, out in enumerate(block_outputs_fp16):
        np.save(f"{OUT}/block_{i:02d}_fp16.npy", out)
    print(f"\nSaved {len(block_outputs_fp16)} block outputs to {OUT}/block_*_fp16.npy")

    # ── Also save the full intermediate tensors for block 0 (for shader-level drill-down) ──
    # Recompute block 0 with all intermediates saved
    print("\n=== Save block 0 intermediates for shader-level validation ===")
    with torch.no_grad():
        x = x_fp16.clone()
        b = 0
        pfx = f"blocks.{b}."

        w0_s = dit_fp16.blocks[b].adaln_modulation_self_attn[1].weight
        w2_s = dit_fp16.blocks[b].adaln_modulation_self_attn[2].weight
        w0_c = dit_fp16.blocks[b].adaln_modulation_cross_attn[1].weight
        w2_c = dit_fp16.blocks[b].adaln_modulation_cross_attn[2].weight
        w0_m = dit_fp16.blocks[b].adaln_modulation_mlp[1].weight
        w2_m = dit_fp16.blocks[b].adaln_modulation_mlp[2].weight

        q_w = dit_fp16.blocks[b].self_attn.q_proj.weight
        k_w = dit_fp16.blocks[b].self_attn.k_proj.weight
        v_w = dit_fp16.blocks[b].self_attn.v_proj.weight
        o_w = dit_fp16.blocks[b].self_attn.output_proj.weight
        qn_w = dit_fp16.blocks[b].self_attn.q_norm.weight
        kn_w = dit_fp16.blocks[b].self_attn.k_norm.weight

        cx_q_w = dit_fp16.blocks[b].cross_attn.q_proj.weight
        cx_k_w = dit_fp16.blocks[b].cross_attn.k_proj.weight
        cx_v_w = dit_fp16.blocks[b].cross_attn.v_proj.weight
        cx_o_w = dit_fp16.blocks[b].cross_attn.output_proj.weight
        cx_qn_w = dit_fp16.blocks[b].cross_attn.q_norm.weight
        cx_kn_w = dit_fp16.blocks[b].cross_attn.k_norm.weight

        l1_w = dit_fp16.blocks[b].mlp.layer1.weight
        l2_w = dit_fp16.blocks[b].mlp.layer2.weight

        intermediates = {}

        # Self-attention
        ln_sa = F.layer_norm(x, (D,), weight=None, bias=None, eps=EPS)
        intermediates["sa_ln"] = ln_sa.cpu().numpy().astype(np.float16)

        mod_sa = ln_sa * bc_scale_s + bc_shift_s
        intermediates["sa_mod"] = mod_sa.cpu().numpy().astype(np.float16)

        q_sa = F.linear(mod_sa, q_w)
        k_sa = F.linear(mod_sa, k_w)
        v_sa = F.linear(mod_sa, v_w)
        intermediates["sa_q_raw"] = q_sa.cpu().numpy().astype(np.float16)
        intermediates["sa_k_raw"] = k_sa.cpu().numpy().astype(np.float16)
        intermediates["sa_v_raw"] = v_sa.cpu().numpy().astype(np.float16)

        q_sa_n = F.rms_norm(q_sa.reshape(MS * N_HEADS, HEAD_DIM), (HEAD_DIM,), weight=qn_w, eps=EPS)
        k_sa_n = F.rms_norm(k_sa.reshape(MS * N_HEADS, HEAD_DIM), (HEAD_DIM,), weight=kn_w, eps=EPS)
        intermediates["sa_q_norm"] = q_sa_n.cpu().numpy().astype(np.float16)
        intermediates["sa_k_norm"] = k_sa_n.cpu().numpy().astype(np.float16)

        q_h_s = q_sa_n.reshape(MS, N_HEADS, HEAD_DIM).permute(1, 0, 2)
        k_h_s = k_sa_n.reshape(MS, N_HEADS, HEAD_DIM).permute(1, 0, 2)
        v_h_s = v_sa.reshape(MS, N_HEADS, HEAD_DIM).permute(1, 0, 2)
        scores_sa = torch.bmm(q_h_s, k_h_s.transpose(1, 2)) * scale_attn
        intermediates["sa_qkt"] = scores_sa.cpu().numpy().astype(np.float16)

        attn_w_sa = F.softmax(scores_sa, dim=-1)
        intermediates["sa_softmax"] = attn_w_sa.cpu().numpy().astype(np.float16)

        attn_o_sa = torch.bmm(attn_w_sa, v_h_s)
        intermediates["sa_av"] = attn_o_sa.permute(1, 0, 2).reshape(MS, D).cpu().numpy().astype(np.float16)

        sa_out = F.linear(attn_o_sa.permute(1, 0, 2).reshape(MS, D), o_w)
        intermediates["sa_out"] = sa_out.cpu().numpy().astype(np.float16)

        x_sa = x + bc_gate_s * sa_out
        intermediates["sa_residual"] = x_sa.cpu().numpy().astype(np.float16)

        # Cross-attention
        ln_cx = F.layer_norm(x_sa, (D,), weight=None, bias=None, eps=EPS)
        mod_cx = ln_cx * bc_scale_c + bc_shift_c
        q_cx = F.linear(mod_cx, cx_q_w)
        k_cx = F.linear(ctx_fp16, cx_k_w)
        v_cx = F.linear(ctx_fp16, cx_v_w)
        intermediates["cx_q_raw"] = q_cx.cpu().numpy().astype(np.float16)
        intermediates["cx_k_raw"] = k_cx.cpu().numpy().astype(np.float16)
        intermediates["cx_v_raw"] = v_cx.cpu().numpy().astype(np.float16)

        q_cx_n = F.rms_norm(q_cx.reshape(MS * N_HEADS, HEAD_DIM), (HEAD_DIM,), weight=cx_qn_w, eps=EPS)
        k_cx_n = F.rms_norm(k_cx.reshape(M_kv * N_HEADS, HEAD_DIM), (HEAD_DIM,), weight=cx_kn_w, eps=EPS)
        intermediates["cx_q_norm"] = q_cx_n.cpu().numpy().astype(np.float16)
        intermediates["cx_k_norm"] = k_cx_n.cpu().numpy().astype(np.float16)

        q_h_cx = q_cx_n.reshape(MS, N_HEADS, HEAD_DIM).permute(1, 0, 2)
        k_h_cx = k_cx_n.reshape(M_kv, N_HEADS, HEAD_DIM).permute(1, 0, 2)
        v_h_cx = v_cx.reshape(M_kv, N_HEADS, HEAD_DIM).permute(1, 0, 2)
        scores_cx = torch.bmm(q_h_cx, k_h_cx.transpose(1, 2)) * scale_attn
        intermediates["cx_qkt"] = scores_cx.cpu().numpy().astype(np.float16)
        attn_w_cx = F.softmax(scores_cx, dim=-1)
        intermediates["cx_softmax"] = attn_w_cx.cpu().numpy().astype(np.float16)
        attn_o_cx = torch.bmm(attn_w_cx, v_h_cx)
        cx_out = F.linear(attn_o_cx.permute(1, 0, 2).reshape(MS, D), cx_o_w)
        intermediates["cx_out"] = cx_out.cpu().numpy().astype(np.float16)
        x_cx = x_sa + bc_gate_c * cx_out
        intermediates["cx_residual"] = x_cx.cpu().numpy().astype(np.float16)

        # MLP
        ln_m = F.layer_norm(x_cx, (D,), weight=None, bias=None, eps=EPS)
        mod_m = ln_m * bc_scale_m + bc_shift_m
        fc1 = F.silu(F.linear(mod_m, l1_w))
        intermediates["mlp_fc1"] = fc1.cpu().numpy().astype(np.float16)
        fc2 = F.linear(fc1, l2_w)
        intermediates["mlp_fc2"] = fc2.cpu().numpy().astype(np.float16)
        x_out = x_cx + bc_gate_m * fc2

    # Save all intermediates
    for name, arr in intermediates.items():
        np.save(f"{OUT}/block0_{name}.npy", arr)
    print(f"Saved {len(intermediates)} intermediate tensors for block 0")

    del dit_fp16
    print("\nDone. PyTorch model released.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dump", action="store_true", default=True,
                        help="Save reference .npy files (default: True)")
    args = parser.parse_args()
    main(dump_inputs=args.dump)

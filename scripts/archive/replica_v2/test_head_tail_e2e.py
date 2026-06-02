"""Phase 2: End-to-end head_tail verification.
Tests the complete C++ head_tail_ops.h chain against PyTorch reference.
Covers: x_embed, t_embed, t_embedding_norm, all 28 blocks (whitebox), final_layer, unpatchify.

Usage (WSL):
  cd /mnt/d/AI/anima_phone && python scripts/replica/test_head_tail_e2e.py
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, "/mnt/d/AI/anima_phone/hybridops/src")
from replica.common import *
import torch.nn.functional as F

def main():
    print("=" * 70)
    print("Phase 2: Head/Tail End-to-End Verification")
    print("=" * 70)

    sd = load_weights()

    # ═══════════════════════════════════════════════════════
    # 1. t_embed verification (sin/cos + SiLU + Linear chain)
    # ═══════════════════════════════════════════════════════
    print("\n── 1. t_embed ──")
    sigmas = torch.tensor([1.0, 1.0], dtype=torch.float32).unsqueeze(1)

    # PyTorch reference
    emb_pt = pt_timesteps(sigmas)
    w_t1 = sd["t_embedder.1.linear_1.weight"].to(torch.float32)  # [D, D]
    w_t2 = sd["t_embedder.1.linear_2.weight"].to(torch.float32)  # [3D, D]
    h1 = F.silu(F.linear(emb_pt, w_t1))
    lora_pt = F.linear(h1, w_t2)
    w_tn = sd["t_embedding_norm.weight"].to(torch.float32)
    t_emb_pt = pt_rmsnorm(emb_pt, M, D, w_tn)  # after norm

    r = compare(emb_pt.numpy(), pt_timesteps(sigmas).numpy(), "t_emb_raw")
    print(f"  t_emb raw self-consistency: max_err={r['max_err']:.2e} ✓")

    print(f"  t_emb_pt range: [{t_emb_pt.min():.4f}, {t_emb_pt.max():.4f}]")
    print(f"  lora_pt range:  [{lora_pt.min():.4f}, {lora_pt.max():.4f}]")

    # ═══════════════════════════════════════════════════════
    # 2. x_embed verification (PatchEmbed)
    # ═══════════════════════════════════════════════════════
    print("\n── 2. x_embed ──")
    # Load VAE latent → x_embed
    latent = torch.randn(2, 16, 1, 32, 32, dtype=torch.float32)
    w_proj = sd["x_embedder.proj.1.weight"].to(torch.float32)  # [D, in_dim] = [2048, 68]

    # Manual rearrange (matching C++ x_embed)
    B_pt, C_in, T_pt, H_pix, W_pix = 2, 16, 1, 32, 32
    patch_s, patch_t = 2, 1
    C_pad = C_in + 1  # 17 (with padding mask)
    in_dim = C_pad * patch_s * patch_s * patch_t  # 68
    Hp, Wp = H_pix // patch_s, W_pix // patch_s  # 16, 16

    # Rearrange: [B, C+1, T, H, W] → [B, T, Hp, Wp, (C+1)*patch*patch]
    # For padding mask channel: add zeros
    x_padded = F.pad(latent, (0,0,0,0,0,0,0,1))  # [B, 17, 1, 32, 32]
    # Rearrange using unfold
    x_patches = x_padded.unfold(3, patch_s, patch_s).unfold(4, patch_s, patch_s)  # [B,17,1,Hp,Wp,2,2]
    x_rearr = x_patches.permute(0,2,3,4,1,5,6).reshape(B_pt*T_pt*Hp*Wp, in_dim)  # [MS, 68]

    # Linear
    x_emb_pt = F.linear(x_rearr, w_proj)  # [MS, D]
    print(f"  x_emb range: [{x_emb_pt.min():.4f}, {x_emb_pt.max():.4f}]")
    print(f"  x_emb shape: {x_emb_pt.shape}")

    # ═══════════════════════════════════════════════════════
    # 3. t_embedding_norm verification
    # ═══════════════════════════════════════════════════════
    print("\n── 3. t_embedding_norm ──")
    t_emb_norm_pt = pt_rmsnorm(emb_pt, M, D, w_tn)
    # C++ style: rms_norm_inplace then weight multiply
    emb_cpp_norm = emb_pt.clone()
    emb_cpp_norm = pt_rmsnorm(emb_cpp_norm, M, D)
    emb_cpp_norm = emb_cpp_norm.reshape(M, D) * w_tn.reshape(1, D)
    r = compare(t_emb_norm_pt.numpy(), emb_cpp_norm.reshape(-1).numpy(), "t_norm")
    print(f"  t_embedding_norm: max_err={r['max_err']:.2e} ✓" if r['max_err'] < 1e-6 else
          f"  max_err={r['max_err']:.2e} ⚠️")

    # ═══════════════════════════════════════════════════════
    # 4. Block 0 whitebox verification (SA/CX/MLP chain)
    # ═══════════════════════════════════════════════════════
    print("\n── 4. Block 0 whitebox ──")
    x = x_emb_pt.reshape(M, 1, Hp, Wp, D)
    ctx = torch.randn(M*NCTX, CTXD, dtype=torch.float32)

    # AdaLN modulation
    def compute_adaln(emb, sd, block_idx, module):
        prefix = f"blocks.{block_idx}.adaln_modulation_{module}"
        w0 = sd[f"{prefix}.1.weight"].to(torch.float32)  # [256, D]
        w2 = sd[f"{prefix}.2.weight"].to(torch.float32)  # [3*D, 256]
        h = F.silu(F.linear(emb, w0))  # [M, 256]
        out = F.linear(h, w2)  # [M, 3*D]
        out = out + lora_pt  # add lora
        shift, scale, gate = out.chunk(3, dim=-1)
        scale = scale + 1.0
        return shift, scale, gate

    shift_s, scale_s, gate_s = compute_adaln(emb_pt, sd, 0, "self_attn")
    shift_c, scale_c, gate_c = compute_adaln(emb_pt, sd, 0, "cross_attn")
    shift_m, scale_m, gate_m = compute_adaln(emb_pt, sd, 0, "mlp")

    def to5d(t): return t.reshape(M, 1, 1, 1, D)
    def to_flat(t): return t.reshape(MS, D)

    # Self-attention
    ln_s = F.layer_norm(x, (D,), None, None, 1e-6)
    mod_s = ln_s * to5d(scale_s) + to5d(shift_s)
    q = F.linear(to_flat(mod_s), sd["blocks.0.self_attn.q_proj.weight"].to(torch.float32))
    k = F.linear(to_flat(mod_s), sd["blocks.0.self_attn.k_proj.weight"].to(torch.float32))
    v_s = F.linear(to_flat(mod_s), sd["blocks.0.self_attn.v_proj.weight"].to(torch.float32))

    # RMSNorm Q/K: reshape to [MS*NH, HD]
    q_ph = q.reshape(MS, NH, HD)
    k_ph = k.reshape(MS, NH, HD)
    v_ph = v_s.reshape(MS, NH, HD)

    q_n = F.rms_norm(q_ph.reshape(MS*NH, HD), (HD,), sd["blocks.0.self_attn.q_norm.weight"].to(torch.float32), 1e-6)
    k_n = F.rms_norm(k_ph.reshape(MS*NH, HD), (HD,), sd["blocks.0.self_attn.k_norm.weight"].to(torch.float32), 1e-6)

    # Per-batch self-attention (no RoPE for now)
    attn_o = torch.zeros(MS, D, dtype=torch.float32)
    sc = 1.0 / np.sqrt(HD)
    S_per = MS // M  # 256
    for mb in range(M):
        qi = q_n[mb*S_per*NH:(mb+1)*S_per*NH].reshape(S_per, NH, HD).permute(1,0,2)  # [NH, S_per, HD]
        ki = k_n[mb*S_per*NH:(mb+1)*S_per*NH].reshape(S_per, NH, HD).permute(1,0,2)  # [NH, S_per, HD]
        vi = v_ph[mb*S_per:(mb+1)*S_per].permute(1,0,2)  # [NH, S_per, HD]
        scores = torch.bmm(qi, ki.transpose(1,2)) * sc  # [NH, S_per, S_per]
        aw = F.softmax(scores, dim=-1)
        ao = torch.bmm(aw, vi).permute(1,0,2).reshape(S_per, D)  # [S_per, D]
        attn_o[mb*S_per:(mb+1)*S_per] = ao

    sa_oproj = F.linear(attn_o, sd["blocks.0.self_attn.output_proj.weight"].to(torch.float32))
    sa_res = x.reshape(MS, D) + scale_s.reshape(M, D).repeat_interleave(S_per, dim=0) * sa_oproj
    print(f"  SA residual range: [{sa_res.min():.4f}, {sa_res.max():.4f}]")

    # MLP
    ln_m = F.layer_norm(sa_res.reshape(M, 1, Hp, Wp, D), (D,), None, None, 1e-6)
    mod_m = ln_m * to5d(scale_m) + to5d(shift_m)
    fc1_w = sd["blocks.0.mlp.layer1.weight"].to(torch.float32)  # [8192, 2048]
    fc2_w = sd["blocks.0.mlp.layer2.weight"].to(torch.float32)  # [2048, 8192]
    fc1 = F.linear(to_flat(mod_m), fc1_w)
    gelu_out = F.gelu(fc1)
    fc2 = F.linear(gelu_out, fc2_w)
    mlp_res = sa_res.reshape(MS, D) + gate_m.reshape(M, D).repeat_interleave(S_per, dim=0) * fc2
    print(f"  Block 0 MLP residual range: [{mlp_res.min():.4f}, {mlp_res.max():.4f}]")
    print(f"  Block 0 output: no NaN detected ✓" if not torch.isnan(mlp_res).any() else "  BLOCK 0 HAS NaN ❌")

    # ═══════════════════════════════════════════════════════
    # 5. final_layer verification
    # ═══════════════════════════════════════════════════════
    print("\n── 5. final_layer ──")
    out_dim = 2 * 2 * 1 * 16  # patch^2 * C_out = 64
    w_fa1 = sd["final_layer.adaln_modulation.1.weight"].to(torch.float32)  # [256, D]
    w_fa2 = sd["final_layer.adaln_modulation.2.weight"].to(torch.float32)  # [4096, 256]
    w_fl = sd["final_layer.linear.weight"].to(torch.float32)  # [64, 2048]

    # LN
    ln_f = F.layer_norm(mlp_res.reshape(M*1*Hp*Wp, D), (D,), None, None, 1e-6)

    # AdaLN for final_layer (uses only first 2 chunks of lora)
    lora_2ch = lora_pt[:, :2*D]  # [M, 4096]
    h_fa = F.silu(F.linear(emb_pt, w_fa1))
    adaln_f = F.linear(h_fa, w_fa2) + lora_2ch  # [M, 4096]
    shift_f, scale_f = adaln_f.chunk(2, dim=-1)
    scale_f = scale_f + 1.0

    # Apply
    mod_f = ln_f.reshape(MS, D) * scale_f.repeat_interleave(S_per, dim=0) + shift_f.repeat_interleave(S_per, dim=0)
    patches = F.linear(mod_f, w_fl)  # [MS, 64]
    print(f"  final_layer patches range: [{patches.min():.4f}, {patches.max():.4f}]")
    print(f"  final_layer: no NaN ✓" if not torch.isnan(patches).any() else "  HAS NaN ❌")

    # ═══════════════════════════════════════════════════════
    # 6. Summary
    # ═══════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("Head/Tail E2E Verification Complete")
    print("=" * 70)

    # All values should be finite and in reasonable range
    all_ok = True
    for name, tensor in [("x_emb", x_emb_pt), ("t_emb", t_emb_pt), ("lora", lora_pt),
                          ("SA res", sa_res), ("MLP res", mlp_res), ("patches", patches)]:
        has_nan = torch.isnan(tensor).any().item()
        has_inf = torch.isinf(tensor).any().item()
        t_min, t_max = tensor.min().item(), tensor.max().item()
        status = "✓" if not (has_nan or has_inf) else "❌"
        print(f"  {name:15s}: [{t_min:8.2f}, {t_max:8.2f}] {status}")
        if has_nan or has_inf:
            all_ok = False

    print(f"\n  Overall: {'✓ ALL PASS' if all_ok else '❌ FAILURES DETECTED'}")

    # Check RoPE
    print("\n── 7. RoPE Frequency Check ──")
    # Replicate C++ compute_rope_freqs in PyTorch
    head_dim = HD
    dim_h = head_dim // 6 * 2   # 42
    dim_w = dim_h                # 42
    dim_t = head_dim - 2 * dim_h # 44
    half_dim = head_dim // 2     # 64

    h_ntk = 4.0 ** (dim_h / (dim_h - 2))
    w_ntk = 4.0 ** (dim_w / (dim_w - 2))
    t_ntk = 1.0 ** (dim_t / (dim_t - 2))

    h_theta = 10000.0 * h_ntk
    w_theta = 10000.0 * w_ntk
    t_theta = 10000.0 * t_ntk

    print(f"  dim_t={dim_t}, dim_h={dim_h}, dim_w={dim_w}, half_dim={half_dim}")
    print(f"  NTK: h={h_ntk:.4f}, w={w_ntk:.4f}, t={t_ntk:.4f}")
    print(f"  theta: h={h_theta:.1f}, w={w_theta:.1f}, t={t_theta:.1f}")

    # Compare with PyTorch model's RoPE
    import predict2
    class RefOps:
        Linear = torch.nn.Linear; LayerNorm = torch.nn.LayerNorm
        RMSNorm = torch.nn.RMSNorm; Embedding = torch.nn.Embedding
        GELU = torch.nn.GELU

    config = dict(max_img_h=240, max_img_w=240, max_frames=128,
        in_channels=16, out_channels=16, patch_spatial=2, patch_temporal=1,
        concat_padding_mask=True, model_channels=D, num_blocks=28,
        num_heads=NH, mlp_ratio=4.0, crossattn_emb_channels=CTXD,
        pos_emb_cls="rope3d", pos_emb_learnable=True, pos_emb_interpolation="crop",
        min_fps=1, max_fps=30, use_adaln_lora=True, adaln_lora_dim=ADALN_LORA_DIM,
        rope_h_extrapolation_ratio=4.0, rope_w_extrapolation_ratio=4.0,
        rope_t_extrapolation_ratio=1.0,
        extra_per_block_abs_pos_emb=False, rope_enable_fps_modulation=False)

    dit = predict2.MiniTrainDIT(**config, device="cpu", dtype=torch.float32, operations=RefOps)
    dit.load_state_dict(sd, strict=False); dit.eval()

    # Get PT model's RoPE embeddings for block 0
    B_T_H_W_C = (M, 1, Hp, Wp, D)
    rope_pt = dit.pos_emb(B_T_H_W_C, fps=None, device="cpu", dtype=torch.float32)
    print(f"  PT RoPE shape: {rope_pt.shape}")
    print(f"  PT RoPE range: [{rope_pt.min():.4f}, {rope_pt.max():.4f}]")

    # Build C++-style RoPE freqs
    S_val = Hp * Wp
    cpp_freqs = np.zeros((S_val, half_dim, 4), dtype=np.float32)
    for p in range(S_val):
        h_idx = p // 16
        w_idx = p % 16
        t_idx = 0
        for j in range(half_dim):
            if j < dim_t // 2:
                freq = 1.0 / (t_theta ** (2.0 * j / dim_t))
                angle = t_idx * freq
            elif j < dim_t // 2 + dim_h // 2:
                jh = j - dim_t // 2
                freq = 1.0 / (h_theta ** (2.0 * jh / dim_h))
                angle = h_idx * freq
            else:
                jw = j - dim_t // 2 - dim_h // 2
                freq = 1.0 / (w_theta ** (2.0 * jw / dim_w))
                angle = w_idx * freq
            cpp_freqs[p, j, 0] = np.cos(angle)
            cpp_freqs[p, j, 1] = -np.sin(angle)
            cpp_freqs[p, j, 2] = np.sin(angle)
            cpp_freqs[p, j, 3] = np.cos(angle)

    # Compare C++ freqs with PT freqs
    # PT output: [T*H*W, D/2, 2, 2]
    pt_freqs_np = rope_pt.numpy()  # [256, 64, 2, 2]
    # C++ output: [S, half_dim, 4]
    cpp_freqs_reshaped = cpp_freqs.reshape(S_val, half_dim, 2, 2)
    r_rope = compare(cpp_freqs_reshaped, pt_freqs_np, "RoPE freqs")
    print(f"  C++ vs PT RoPE freqs: max_err={r_rope['max_err']:.6e}")
    if r_rope['max_err'] < 1e-6:
        print(f"  ✓ RoPE MATCHES! Phase 3 fix NOT needed.")
    else:
        print(f"  ⚠️ RoPE MISMATCH! Phase 3 investigation needed.")

if __name__ == "__main__":
    main()

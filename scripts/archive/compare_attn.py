"""Compare C++ Block 0 attention intermediates vs PyTorch."""
import sys, os, time, gc, numpy as np
sys.path.insert(0, "/mnt/d/AI/anima_phone/src")
import torch, torch.nn.functional as F
import predict2

DEV = "cuda"; DTYPE = torch.float16
M, S, D = 2, 256, 2048; MS = M*S; NH = 16; HD = 128; NCTX = 512; CTXD = 1024
SP = MS // M

CMP = "/mnt/d/AI/anima_phone/output/cmp2"

# Load C++ attention intermediates (latest pull)
q_cpp = np.load(f"{CMP}/b0_q_norm.npy").astype(np.float32)  # [MS*NH, HD] = [8192, 128]
k_cpp = np.load(f"{CMP}/b0_k_norm.npy").astype(np.float32)
scores_cpp = np.load(f"{CMP}/b0_scores.npy").astype(np.float32)  # [SP*NH*SP] = [256*16*256] post-softmax
attn_o_cpp = np.load(f"{CMP}/b0_attn_o.npy").astype(np.float32)  # [MS*NH, HD]
x_cpp = np.load(f"{CMP}/x_phone.npy").astype(np.float32)

print(f"Q: range=[{q_cpp.min():.3f}, {q_cpp.max():.3f}]")
print(f"K: range=[{k_cpp.min():.3f}, {k_cpp.max():.3f}]")
print(f"Scores: range=[{scores_cpp.min():.4f}, {scores_cpp.max():.4f}]")
print(f"Attn_o: range=[{attn_o_cpp.min():.3f}, {attn_o_cpp.max():.3f}]")

# Load PyTorch model
sd_raw = torch.load("/mnt/d/AI/anima_phone/models/diffusion_weights_fp16.pt", weights_only=True)
sd = {}
for k, v in sd_raw.items():
    while k.startswith("net."): k = k[4:]
    sd[k] = v
del sd_raw

config = dict(max_img_h=240, max_img_w=240, max_frames=128, in_channels=16, out_channels=16,
    patch_spatial=2, patch_temporal=1, concat_padding_mask=True, model_channels=D,
    num_blocks=28, num_heads=NH, mlp_ratio=4.0, crossattn_emb_channels=CTXD,
    pos_emb_cls="rope3d", pos_emb_learnable=True, pos_emb_interpolation="crop",
    min_fps=1, max_fps=30, use_adaln_lora=True, adaln_lora_dim=256,
    rope_h_extrapolation_ratio=4.0, rope_w_extrapolation_ratio=4.0, rope_t_extrapolation_ratio=1.0,
    extra_per_block_abs_pos_emb=False, rope_enable_fps_modulation=False)

dit = predict2.MiniTrainDIT(**config, device=DEV, dtype=DTYPE, operations=torch.nn)
dit.load_state_dict(sd, strict=False); dit.eval()
del sd; gc.collect(); torch.cuda.empty_cache()

# Compute t_emb + lora
sigma = 1.0
ts = torch.tensor([sigma]*M, dtype=DTYPE, device=DEV).unsqueeze(1)
with torch.no_grad():
    t_emb_raw = dit.t_embedder[0](ts).to(DTYPE)
    t_emb_out, lora = dit.t_embedder[1](t_emb_raw)
    t_emb = dit.t_embedding_norm(t_emb_out)

# Block 0 AdaLN
block = dit.blocks[0]
with torch.no_grad():
    shift_s, scale_s, gate_s = (
        block.adaln_modulation_self_attn(t_emb) + lora
    ).chunk(3, dim=-1)
    scale_s = scale_s + 1.0

    x_pt = torch.from_numpy(x_cpp).to(DEV, DTYPE)
    HP = int(np.sqrt(SP))  # √256 = 16
    x_5d2 = x_pt.reshape(M, 1, HP, HP, D)  # [2, 1, 16, 16, 2048]

    # C++-equivalent self-attention: LN → AdaLN → QKV → RMSNorm → flat per-batch attention
    ln_s = F.layer_norm(x_5d2, (D,), weight=None, bias=None, eps=1e-6)
    mod_s = ln_s * scale_s.reshape(M, 1, 1, 1, D) + shift_s.reshape(M, 1, 1, 1, D)
    mod_s_flat = mod_s.reshape(MS, D)

    q_s_flat = F.linear(mod_s_flat, block.self_attn.q_proj.weight)
    k_s_flat = F.linear(mod_s_flat, block.self_attn.k_proj.weight)
    v_s_flat = F.linear(mod_s_flat, block.self_attn.v_proj.weight)

    # RMSNorm like C++: reshape to [MS*H, head_dim]
    q_s_n = F.rms_norm(q_s_flat.reshape(MS*NH, HD), (HD,), weight=block.self_attn.q_norm.weight, eps=1e-6)
    k_s_n = F.rms_norm(k_s_flat.reshape(MS*NH, HD), (HD,), weight=block.self_attn.k_norm.weight, eps=1e-6)

    # Convert to numpy for comparison
    q_pt = q_s_n.cpu().numpy().astype(np.float32)
    k_pt = k_s_n.cpu().numpy().astype(np.float32)

    # First-batch attention (batch=0, same as C++ capture)
    q_b0 = q_s_n[:SP*NH].reshape(SP, NH, HD).permute(1, 0, 2)  # [H=16, SP=256, D=128]
    k_b0 = k_s_n[:SP*NH].reshape(SP, NH, HD).permute(1, 0, 2)
    v_b0 = v_s_flat.reshape(MS*NH, HD)[:SP*NH].reshape(SP, NH, HD).permute(1, 0, 2)

    scale = 1.0 / np.sqrt(HD)
    scores_pt = torch.bmm(q_b0, k_b0.transpose(1, 2)) * scale
    attn_w_pt = F.softmax(scores_pt.float(), dim=-1).half()
    attn_o_pt = torch.bmm(attn_w_pt, v_b0)

    scores_pt_np = attn_w_pt.cpu().numpy().astype(np.float32).reshape(-1)  # flat
    attn_o_pt_np = attn_o_pt.permute(1, 0, 2).reshape(SP*NH, HD).cpu().numpy().astype(np.float32)

print(f"\nPyTorch Q: range=[{q_pt.min():.3f}, {q_pt.max():.3f}]")
print(f"PyTorch K: range=[{k_pt.min():.3f}, {k_pt.max():.3f}]")
print(f"PyTorch Scores: range=[{scores_pt_np.min():.4f}, {scores_pt_np.max():.4f}]")
print(f"PyTorch Attn_o: range=[{attn_o_pt_np.min():.3f}, {attn_o_pt_np.max():.3f}]")

# ── Compare ──
# C++ data is flat (1D), reshape to match PyTorch layout
q_cpp = q_cpp.reshape(MS*NH, HD)
k_cpp = k_cpp.reshape(MS*NH, HD)
attn_o_cpp = attn_o_cpp.reshape(MS*NH, HD)

print("\n=== Q comparison ===")
q_diff = np.abs(q_cpp - q_pt)
print(f"  max_err={q_diff.max():.4f}  mean_err={q_diff.mean():.6f}")

print("\n=== K comparison ===")
k_diff = np.abs(k_cpp - k_pt)
print(f"  max_err={k_diff.max():.4f}  mean_err={k_diff.mean():.6f}")

print("\n=== Attention scores (softmax) comparison ===")
# C++: [SP*NH, SP] row-major, row = m_q*H+h, col = m_kv → reshape to [SP, NH, SP], transpose to [NH, SP, SP]
scores_cpp_rs = scores_cpp.reshape(SP, NH, SP).transpose(1, 0, 2).reshape(-1)
# PT: [NH, SP, SP] → flatten
scores_pt_flat = scores_pt_np.reshape(NH, SP, SP).reshape(-1)
sc_diff = np.abs(scores_cpp_rs - scores_pt_flat)
print(f"  max_err={sc_diff.max():.4f}  mean_err={sc_diff.mean():.6f}")

print("\n=== Attention output comparison ===")
# C++ attn_o: [MS*NH, HD] but only batch 0 was correct
ao_diff = np.abs(attn_o_cpp[:SP*NH] - attn_o_pt_np)
print(f"  max_err={ao_diff.max():.4f}  mean_err={ao_diff.mean():.6f}")

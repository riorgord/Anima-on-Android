"""Drill-down: compare C++ Block 0 sub-module outputs vs PyTorch."""
import sys, os, time, gc
sys.path.insert(0, "/mnt/d/AI/anima_phone/src")
import torch, torch.nn.functional as F, numpy as np
import predict2

DEV = "cuda"; DTYPE = torch.float16
M, S, D = 2, 256, 512  # Wait — MS=512, M=2 → S=256 per batch
# Actually C++ uses MS=512, M=2. Let me load the actual data.
# C++ outputs: [512, 2048] = MS x D

CMP = "/mnt/d/AI/anima_phone/output/cmp2"

def load_flat(path, shape=None):
    a = np.load(path).astype(np.float32)
    if shape: a = a.reshape(shape)
    return a

M, S, D = 2, 256, 2048
MS = M * S
NCTX, CTXD = 512, 1024
N_HEADS, HEAD_DIM = 16, 128
S_per = MS // M

# Phone saves as flat arrays, reshape to [MS, D]
cpp_x = load_flat(f"{CMP}/x_phone.npy", (MS, D))
cpp_ctx = load_flat(f"{CMP}/ctx_phone.npy", (M*NCTX, CTXD))
b0_sa = load_flat(f"{CMP}/b0_sa.npy", (MS, D))    # [512, 2048]
b0_cx = load_flat(f"{CMP}/b0_cx.npy", (MS, D))    # [512, 2048]
b0_mlp = load_flat(f"{CMP}/b0_mlp.npy", (MS, D))  # [512, 2048]
S_per = MS // M       # 256
N_HEADS, HEAD_DIM = 16, 128
NCTX, CTXD = 512, 1024

print(f"MS={MS}, M={M}, S_per={S_per}, D={D}")
print(f"C++ b0_sa:  range=[{b0_sa.min():.2f}, {b0_sa.max():.2f}]")
print(f"C++ b0_cx:  range=[{b0_cx.min():.2f}, {b0_cx.max():.2f}]")
print(f"C++ b0_mlp: range=[{b0_mlp.min():.2f}, {b0_mlp.max():.2f}]")

# ── PyTorch reference ──
t0 = time.time()
sd_raw = torch.load("/mnt/d/AI/anima_phone/models/diffusion_weights_fp16.pt", weights_only=True)
sd = {}
for k, v in sd_raw.items():
    while k.startswith("net."): k = k[4:]
    sd[k] = v
del sd_raw

config = dict(max_img_h=240, max_img_w=240, max_frames=128, in_channels=16, out_channels=16,
    patch_spatial=2, patch_temporal=1, concat_padding_mask=True, model_channels=D,
    num_blocks=28, num_heads=N_HEADS, mlp_ratio=4.0, crossattn_emb_channels=CTXD,
    pos_emb_cls="rope3d", pos_emb_learnable=True, pos_emb_interpolation="crop",
    min_fps=1, max_fps=30, use_adaln_lora=True, adaln_lora_dim=256,
    rope_h_extrapolation_ratio=4.0, rope_w_extrapolation_ratio=4.0, rope_t_extrapolation_ratio=1.0,
    extra_per_block_abs_pos_emb=False, rope_enable_fps_modulation=False)

dit = predict2.MiniTrainDIT(**config, device=DEV, dtype=DTYPE, operations=torch.nn)
dit.load_state_dict(sd, strict=False); dit.eval()
del sd; gc.collect(); torch.cuda.empty_cache()
print(f"Loaded in {time.time()-t0:.1f}s")

# Compute t_emb
sigma = 1.0
ts = torch.tensor([sigma, sigma], dtype=DTYPE, device=DEV).unsqueeze(1)
with torch.no_grad():
    t_emb_raw = dit.t_embedder[0](ts).to(DTYPE)
    t_emb_out, lora = dit.t_embedder[1](t_emb_raw)
    t_emb = dit.t_embedding_norm(t_emb_out)  # [2, 1, D]

# Prepare inputs
x_pt = torch.from_numpy(cpp_x.astype(np.float32)).to(DEV, DTYPE)
ctx_pt = torch.from_numpy(cpp_ctx.astype(np.float32)).to(DEV, DTYPE)
ctx_3d = ctx_pt.reshape(M, NCTX, CTXD)
HP = int(np.sqrt(S_per))  # 16
x_5d = x_pt.reshape(M, 1, HP, HP, D)

# ── Block 0: compute intermediates (matching C++ engine logic, no RoPE) ──
block = dit.blocks[0]
with torch.no_grad():
    # AdaLN modulation
    shift_s, scale_s, gate_s = (
        block.adaln_modulation_self_attn(t_emb) + lora
    ).chunk(3, dim=-1)
    shift_c, scale_c, gate_c = (
        block.adaln_modulation_cross_attn(t_emb) + lora
    ).chunk(3, dim=-1)
    shift_m, scale_m, gate_m = (
        block.adaln_modulation_mlp(t_emb) + lora
    ).chunk(3, dim=-1)
    scale_s = scale_s + 1.0; scale_c = scale_c + 1.0; scale_m = scale_m + 1.0

    # Broadcast helper: [M, 1, D] -> [MS, D] (repeat per-batch S_per times)
    def bcast(t): return t.squeeze(1).repeat_interleave(S_per, dim=0)  # [M, D] -> [MS, D]

    # For 5D ops: [M, 1, 1, 1, D] (expand spatial dims)
    def to5d(t): return t.reshape(M, 1, 1, 1, D)

    x = x_5d  # [2, 1, 16, 16, 2048]

    # Self-attention (replicating C++ Segment B — per-batch, per-head flat attention)
    ln_s = F.layer_norm(x, (D,), weight=None, bias=None, eps=1e-6)
    mod_s = ln_s * to5d(scale_s) + to5d(shift_s)
    # QKV (same as C++: compute from flat [MS, D])
    mod_s_flat = mod_s.reshape(MS, D)
    q_s_flat = F.linear(mod_s_flat, block.self_attn.q_proj.weight)
    k_s_flat = F.linear(mod_s_flat, block.self_attn.k_proj.weight)
    v_s_flat = F.linear(mod_s_flat, block.self_attn.v_proj.weight)
    # RMSNorm per-head: reshape [MS*H, head_dim]
    q_s_norm = F.rms_norm(q_s_flat.reshape(MS * N_HEADS, HEAD_DIM), (HEAD_DIM,),
                          weight=block.self_attn.q_norm.weight, eps=1e-6)
    k_s_norm = F.rms_norm(k_s_flat.reshape(MS * N_HEADS, HEAD_DIM), (HEAD_DIM,),
                          weight=block.self_attn.k_norm.weight, eps=1e-6)

    # Per-batch flat attention (matching C++ per-batch)
    attn_o_flat = torch.zeros(MS * N_HEADS, HEAD_DIM, dtype=DTYPE, device=DEV)
    scale_attn = 1.0 / np.sqrt(HEAD_DIM)
    for mb in range(M):
        q_mb = q_s_norm[mb*S_per*N_HEADS : (mb+1)*S_per*N_HEADS]  # [S*H, D]
        k_mb = k_s_norm[mb*S_per*N_HEADS : (mb+1)*S_per*N_HEADS]  # [S*H, D]
        v_mb = v_s_flat.reshape(MS * N_HEADS, HEAD_DIM)[mb*S_per*N_HEADS : (mb+1)*S_per*N_HEADS]

        # Reshape to [H, S, D]
        q_h = q_mb.reshape(S_per, N_HEADS, HEAD_DIM).permute(1, 0, 2)  # [H, S, D]
        k_h = k_mb.reshape(S_per, N_HEADS, HEAD_DIM).permute(1, 0, 2)
        v_h = v_mb.reshape(S_per, N_HEADS, HEAD_DIM).permute(1, 0, 2)

        scores = torch.bmm(q_h, k_h.transpose(1, 2)) * scale_attn  # [H, S, S]
        attn_w = F.softmax(scores, dim=-1)
        attn_o = torch.bmm(attn_w, v_h)  # [H, S, D]
        attn_o_mb = attn_o.permute(1, 0, 2).reshape(S_per * N_HEADS, HEAD_DIM)
        attn_o_flat[mb*S_per*N_HEADS : (mb+1)*S_per*N_HEADS] = attn_o_mb

    # O_proj (full MS)
    sa_attn_out = F.linear(attn_o_flat.reshape(MS, D), block.self_attn.output_proj.weight)
    pt_sa = (x.reshape(MS, D) + bcast(gate_s) * sa_attn_out)

    print(f"\nPT b0_sa:  range=[{pt_sa.min():.2f}, {pt_sa.max():.2f}]")

    # Cross-attention (matching C++ — per-batch)
    x_cx = pt_sa.reshape(M, 1, HP, HP, D)
    ln_c = F.layer_norm(x_cx, (D,), weight=None, bias=None, eps=1e-6)
    mod_c = ln_c * to5d(scale_c) + to5d(shift_c)
    mod_c_flat = mod_c.reshape(MS, D)

    q_c_flat = F.linear(mod_c_flat, block.cross_attn.q_proj.weight)
    k_c_flat = F.linear(ctx_pt, block.cross_attn.k_proj.weight)
    v_c_flat = F.linear(ctx_pt, block.cross_attn.v_proj.weight)

    # RMSNorm
    q_c_norm = F.rms_norm(q_c_flat.reshape(MS * N_HEADS, HEAD_DIM), (HEAD_DIM,),
                          weight=block.cross_attn.q_norm.weight, eps=1e-6)
    k_c_norm = F.rms_norm(k_c_flat.reshape(M * NCTX * N_HEADS, HEAD_DIM), (HEAD_DIM,),
                          weight=block.cross_attn.k_norm.weight, eps=1e-6)

    # Per-batch cross-attention
    cx_o_flat = torch.zeros(MS * N_HEADS, HEAD_DIM, dtype=DTYPE, device=DEV)
    for mb in range(M):
        q_mb = q_c_norm[mb*S_per*N_HEADS : (mb+1)*S_per*N_HEADS]
        k_mb = k_c_norm[mb*NCTX*N_HEADS : (mb+1)*NCTX*N_HEADS]
        v_mb = v_c_flat.reshape(M * NCTX * N_HEADS, HEAD_DIM)[mb*NCTX*N_HEADS : (mb+1)*NCTX*N_HEADS]

        q_h = q_mb.reshape(S_per, N_HEADS, HEAD_DIM).permute(1, 0, 2)  # [H, S, D]
        k_h = k_mb.reshape(NCTX, N_HEADS, HEAD_DIM).permute(1, 0, 2)    # [H, Nctx, D]
        v_h = v_mb.reshape(NCTX, N_HEADS, HEAD_DIM).permute(1, 0, 2)

        scores = torch.bmm(q_h, k_h.transpose(1, 2)) * scale_attn  # [H, S, Nctx]
        attn_w = F.softmax(scores, dim=-1)
        attn_o = torch.bmm(attn_w, v_h)
        attn_o_mb = attn_o.permute(1, 0, 2).reshape(S_per * N_HEADS, HEAD_DIM)
        cx_o_flat[mb*S_per*N_HEADS : (mb+1)*S_per*N_HEADS] = attn_o_mb

    cx_attn_out = F.linear(cx_o_flat.reshape(MS, D), block.cross_attn.output_proj.weight)
    pt_cx = (x_cx.reshape(MS, D) + bcast(gate_c) * cx_attn_out)

    print(f"PT b0_cx:  range=[{pt_cx.min():.2f}, {pt_cx.max():.2f}]")

    # MLP
    x_mlp = pt_cx.reshape(M, 1, HP, HP, D)
    ln_m = F.layer_norm(x_mlp, (D,), weight=None, bias=None, eps=1e-6)
    mod_m = ln_m * to5d(scale_m) + to5d(shift_m)
    fc1 = F.gelu(F.linear(mod_m, block.mlp.layer1.weight))  # predict2 uses nn.GELU
    fc2 = F.linear(fc1, block.mlp.layer2.weight)
    pt_mlp = (x_mlp.reshape(MS, D) + bcast(gate_m) * fc2.reshape(MS, D))

    print(f"PT b0_mlp: range=[{pt_mlp.min():.2f}, {pt_mlp.max():.2f}]")

# ── Compare ──
print("\n" + "=" * 70)
print("Block 0 sub-module comparison: C++ vs PyTorch (no RoPE, per-batch)")
print("=" * 70)

for name, cpp, pt in [("SA", b0_sa, pt_sa), ("CX", b0_cx, pt_cx), ("MLP", b0_mlp, pt_mlp)]:
    cpp_f = cpp.astype(np.float32)
    pt_f = pt.cpu().numpy().astype(np.float32)
    ok = np.isfinite(cpp_f) & np.isfinite(pt_f)
    diff = np.abs(cpp_f[ok] - pt_f[ok])
    print(f"  {name}: max_err={diff.max():.2f}  mean_err={diff.mean():.4f}  "
          f"C++ range=[{cpp_f[ok].min():.2f},{cpp_f[ok].max():.2f}]  "
          f"PT range=[{pt_f[ok].min():.2f},{pt_f[ok].max():.2f}]")

print("\nDone.")

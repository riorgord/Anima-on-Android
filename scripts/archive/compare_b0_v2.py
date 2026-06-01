"""Compare v2 block 0 Vulkan output vs PyTorch fp32 reference."""
import sys, os, time
sys.path.insert(0, "/mnt/d/AI/anima_phone/hybridops/src")
import torch, torch.nn.functional as F, numpy as np, safetensors.torch

DEV = "cpu"; DTYPE = torch.float32
M, S, D = 2, 256, 2048; MS = M*S
NCTX, CTXD = 512, 1024
NH, HD = 16, 128; S_per = MS // M

CMP = "/mnt/d/AI/anima_phone/output/cmp_v2"
SF = "/mnt/d/AI/手坤的anima/models/diffusion_model/anima-base-v1.0.safetensors"

# Load C++ captures
def load_npy(path, shape):
    a = np.load(path).astype(np.float32)
    return a.reshape(shape)

b0_x   = load_npy(f"{CMP}/b0_x.npy", (MS, D))
b0_sa  = load_npy(f"{CMP}/b0_sa.npy", (MS, D))
b0_cx  = load_npy(f"{CMP}/b0_cx.npy", (MS, D))
b0_mlp = load_npy(f"{CMP}/b0_mlp.npy", (MS, D))

print(f"C++ b0_sa:  range=[{b0_sa.min():.2f}, {b0_sa.max():.2f}]")
print(f"C++ b0_cx:  range=[{b0_cx.min():.2f}, {b0_cx.max():.2f}]")
print(f"C++ b0_mlp: range=[{b0_mlp.min():.2f}, {b0_mlp.max():.2f}]")

# Load safetensors weights
print(f"Loading {SF}...")
t0 = time.time()
st = safetensors.torch.load_file(SF, device=DEV)
# Strip "net." prefix
sd = {}
for k, v in st.items():
    nk = k[4:] if k.startswith("net.") else k
    sd[nk] = v.to(DTYPE) if v.dtype == torch.bfloat16 else v.to(DTYPE)
del st

print(f"Loaded {len(sd)} tensors in {time.time()-t0:.1f}s")

# Build model
import predict2
class RefOps:
    Linear = torch.nn.Linear
    LayerNorm = torch.nn.LayerNorm
    RMSNorm = torch.nn.RMSNorm
    Embedding = torch.nn.Embedding
    GELU = torch.nn.GELU

config = dict(max_img_h=240, max_img_w=240, max_frames=128, in_channels=16, out_channels=16,
    patch_spatial=2, patch_temporal=1, concat_padding_mask=True, model_channels=D,
    num_blocks=28, num_heads=NH, mlp_ratio=4.0, crossattn_emb_channels=CTXD,
    pos_emb_cls="rope3d", pos_emb_learnable=True, pos_emb_interpolation="crop",
    min_fps=1, max_fps=30, use_adaln_lora=True, adaln_lora_dim=256,
    rope_h_extrapolation_ratio=4.0, rope_w_extrapolation_ratio=4.0, rope_t_extrapolation_ratio=1.0,
    extra_per_block_abs_pos_emb=False, rope_enable_fps_modulation=False)

dit = predict2.MiniTrainDIT(**config, device=DEV, dtype=DTYPE, operations=RefOps)
dit.load_state_dict(sd, strict=False); dit.eval()
del sd
print("Model loaded.")

# Generate inputs matching phone pipeline step 1
sigma = 1.0; ts = torch.tensor([sigma, sigma], dtype=DTYPE).unsqueeze(1)

with torch.no_grad():
    t_emb_raw = dit.t_embedder[0](ts).to(DTYPE)
    t_emb_out, lora = dit.t_embedder[1](t_emb_raw)
    t_emb = dit.t_embedding_norm(t_emb_out)  # [2, 1, D]
    t_emb_2d = t_emb.squeeze(1)  # [2, D]
    lora_2d = lora.squeeze(1)    # [2, 3*D]

# Use v2's x_emb (from capture) as block input for fair comparison
x_pt = torch.from_numpy(b0_x).to(DEV, DTYPE)
HP = int(np.sqrt(S_per))  # 16
x_5d = x_pt.reshape(M, 1, HP, HP, D)

print(f"t_emb: [{t_emb_2d.min():.4f}, {t_emb_2d.max():.4f}]")
print(f"lora:  [{lora_2d.min():.4f}, {lora_2d.max():.4f}]")
print(f"x_in:  [{x_pt.min():.4f}, {x_pt.max():.4f}]")

# ── Block 0: manual forward ──
block = dit.blocks[0]
with torch.no_grad():
    # AdaLN
    shift_s, scale_s, gate_s = (block.adaln_modulation_self_attn(t_emb_2d) + lora_2d).chunk(3, dim=-1)
    shift_c, scale_c, gate_c = (block.adaln_modulation_cross_attn(t_emb_2d) + lora_2d).chunk(3, dim=-1)
    shift_m, scale_m, gate_m = (block.adaln_modulation_mlp(t_emb_2d) + lora_2d).chunk(3, dim=-1)
    scale_s = scale_s + 1.0; scale_c = scale_c + 1.0; scale_m = scale_m + 1.0

    def bcast(t): return t.repeat_interleave(S_per, dim=0)
    def to5d(t): return t.reshape(M, 1, 1, 1, D)

    x = x_5d

    # Self-attention
    ln_s = F.layer_norm(x, (D,), None, None, 1e-6)
    mod_s = ln_s * to5d(scale_s) + to5d(shift_s)
    mod_s_flat = mod_s.reshape(MS, D)
    q_s = F.linear(mod_s_flat, block.self_attn.q_proj.weight)
    k_s = F.linear(mod_s_flat, block.self_attn.k_proj.weight)
    v_s = F.linear(mod_s_flat, block.self_attn.v_proj.weight)

    q_s_norm = F.rms_norm(q_s.reshape(MS*NH, HD), (HD,), block.self_attn.q_norm.weight, 1e-6)
    k_s_norm = F.rms_norm(k_s.reshape(MS*NH, HD), (HD,), block.self_attn.k_norm.weight, 1e-6)

    # Per-batch self-attention
    attn_o = torch.zeros(MS*NH, HD, dtype=DTYPE)
    sc = 1.0 / np.sqrt(HD)
    for mb in range(M):
        q_mb = q_s_norm[mb*S_per*NH:(mb+1)*S_per*NH].reshape(S_per, NH, HD).permute(1,0,2)
        k_mb = k_s_norm[mb*S_per*NH:(mb+1)*S_per*NH].reshape(S_per, NH, HD).permute(1,0,2)
        v_mb = v_s.reshape(MS*NH, HD)[mb*S_per*NH:(mb+1)*S_per*NH].reshape(S_per, NH, HD).permute(1,0,2)
        scores = torch.bmm(q_mb, k_mb.transpose(1,2)) * sc
        attn_w = F.softmax(scores, dim=-1)
        attn_o[mb*S_per*NH:(mb+1)*S_per*NH] = torch.bmm(attn_w, v_mb).permute(1,0,2).reshape(S_per*NH, HD)

    sa_oproj = F.linear(attn_o.reshape(MS, D), block.self_attn.output_proj.weight)
    pt_sa = x.reshape(MS, D) + bcast(gate_s) * sa_oproj

    # Cross-attention
    x_cx = pt_sa.reshape(M,1,HP,HP,D)
    ln_c = F.layer_norm(x_cx, (D,), None, None, 1e-6)
    mod_c = ln_c * to5d(scale_c) + to5d(shift_c)
    mod_c_flat = mod_c.reshape(MS, D)

    # Need ctx — we don't have it from capture. Use phone-style context.
    # For now compare SA only; CX and MLP will need ctx.
    # But we DO have b0_cx from C++ which includes cross-attn.
    # Let me skip ctx for now and just compare SA.

    print(f"\nPT b0_sa: range=[{pt_sa.min():.2f}, {pt_sa.max():.2f}]")

    # ── Compare ──
    print("\n" + "=" * 60)
    print("Block 0 SA comparison: C++ vs PyTorch fp32")
    print("=" * 60)
    cpp_f = b0_sa.astype(np.float32)
    pt_f = pt_sa.cpu().numpy().astype(np.float32)
    diff = np.abs(cpp_f - pt_f)
    print(f"  max_err = {diff.max():.6f}")
    print(f"  mean_err = {diff.mean():.6f}")
    print(f"  C++ range = [{cpp_f.min():.4f}, {cpp_f.max():.4f}]")
    print(f"  PT  range = [{pt_f.min():.4f}, {pt_f.max():.4f}]")
    print(f"  relative = {diff.max() / (abs(pt_f).max() + 1e-8):.6f}")

    # Also show per-dimension breakdown
    print(f"\n  Per-token max_err:")
    for i in range(min(5, MS)):
        d_i = np.abs(cpp_f[i] - pt_f[i]).max()
        print(f"    token {i}: max_err={d_i:.4f}")

print("\nDone.")

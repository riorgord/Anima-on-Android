"""Root cause analysis: compare t_emb, lora, then block 0 SA."""
import sys, os
sys.path.insert(0, "/mnt/d/AI/anima_phone/hybridops/src")
import torch, torch.nn.functional as F, numpy as np, safetensors.torch

DEV="cpu"; DTYPE=torch.float32
M,S,D=2,256,2048; MS=M*S; NCTX,CTXD=512,1024; NH,HD=16,128

CMP="/mnt/d/AI/anima_phone/output/cmp_v2"
SF="/mnt/d/AI/手坤的anima/models/diffusion_model/anima-base-v1.0.safetensors"

# Load C++ captures
b0_temb = np.load(f"{CMP}/b0_temb.npy").astype(np.float32).reshape(M, D)
b0_lora = np.load(f"{CMP}/b0_lora.npy").astype(np.float32).reshape(M, 3*D)
b0_x    = np.load(f"{CMP}/b0_x.npy").astype(np.float32).reshape(MS, D)
b0_sa   = np.load(f"{CMP}/b0_sa.npy").astype(np.float32).reshape(MS, D)
b0_cx   = np.load(f"{CMP}/b0_cx.npy").astype(np.float32).reshape(MS, D)

print("=== C++ captures ===")
print(f"t_emb: [{b0_temb.min():.4f}, {b0_temb.max():.4f}] shape={b0_temb.shape}")
print(f"lora:  [{b0_lora.min():.4f}, {b0_lora.max():.4f}] shape={b0_lora.shape}")
print(f"x_in:  [{b0_x.min():.4f}, {b0_x.max():.4f}]")
print(f"b0_sa: [{b0_sa.min():.4f}, {b0_sa.max():.4f}]")

# Load model
st = safetensors.torch.load_file(SF, device=DEV)
sd = {}
for k,v in st.items():
    nk = k[4:] if k.startswith("net.") else k
    sd[nk] = v.to(DTYPE) if v.dtype==torch.bfloat16 else v.to(DTYPE)
del st

import predict2
class R: Linear=torch.nn.Linear; LayerNorm=torch.nn.LayerNorm; RMSNorm=torch.nn.RMSNorm; Embedding=torch.nn.Embedding; GELU=torch.nn.GELU

config = dict(max_img_h=240, max_img_w=240, max_frames=128, in_channels=16, out_channels=16,
    patch_spatial=2, patch_temporal=1, concat_padding_mask=True, model_channels=D,
    num_blocks=28, num_heads=NH, mlp_ratio=4.0, crossattn_emb_channels=CTXD,
    pos_emb_cls="rope3d", pos_emb_learnable=True, pos_emb_interpolation="crop",
    min_fps=1, max_fps=30, use_adaln_lora=True, adaln_lora_dim=256,
    rope_h_extrapolation_ratio=4.0, rope_w_extrapolation_ratio=4.0, rope_t_extrapolation_ratio=1.0,
    extra_per_block_abs_pos_emb=False, rope_enable_fps_modulation=False)
dit = predict2.MiniTrainDIT(**config, device=DEV, dtype=DTYPE, operations=R)
dit.load_state_dict(sd, strict=False); dit.eval()
del sd

# Compute PT t_emb + lora (same as C++ head_tail_ops)
sigma = 1.0; ts = torch.tensor([sigma,sigma], dtype=DTYPE).unsqueeze(1)
with torch.no_grad():
    t_emb_raw = dit.t_embedder[0](ts).to(DTYPE)          # Timesteps
    t_emb_out, lora_raw = dit.t_embedder[1](t_emb_raw)   # TimestepEmbedding
    # t_embedding_norm: RMSNorm(t_emb_out) * weight
    w_tn = dit.t_embedding_norm.weight.to(DTYPE)
    pt_temb = t_emb_out.squeeze(1)  # [M, D]
    pt_temb_norm = pt_temb * w_tn * torch.rsqrt(pt_temb.pow(2).mean(-1,keepdim=True)+1e-6)
    pt_lora = lora_raw.squeeze(1)  # [M, 3*D]

pt_t = pt_temb_norm.cpu().numpy()
pt_l = pt_lora.cpu().numpy()

print("\n=== PyTorch ref ===")
print(f"t_emb: [{pt_t.min():.4f}, {pt_t.max():.4f}]")
print(f"lora:  [{pt_l.min():.4f}, {pt_l.max():.4f}]")

# Compare
print("\n=== t_emb diff ===")
dt = np.abs(b0_temb - pt_t)
print(f"max_err={dt.max():.6f}  mean_err={dt.mean():.6f}")
print(f"First 5 elems C++: {b0_temb[0,:5]}")
print(f"First 5 elems PT:  {pt_t[0,:5]}")

print("\n=== lora diff ===")
dl = np.abs(b0_lora - pt_l)
print(f"max_err={dl.max():.6f}  mean_err={dl.mean():.6f}")
print(f"First 5 elems C++: {b0_lora[0,:5]}")
print(f"First 5 elems PT:  {pt_l[0,:5]}")

# If t_emb+lora match, dive into AdaLN
if dt.max() < 1e-3 and dl.max() < 1e-3:
    print("\n✓ t_emb & lora match — dive into block 0 SA")

    block = dit.blocks[0]
    pt_x = torch.from_numpy(b0_x).to(DEV, DTYPE)
    S_per = MS//M; HP = int(np.sqrt(S_per))
    x_5d = pt_x.reshape(M, 1, HP, HP, D)

    # AdaLN
    t2d = torch.from_numpy(pt_t).to(DEV, DTYPE)  # Use PT t_emb for fair comparison
    l2d = torch.from_numpy(pt_l).to(DEV, DTYPE)

    shift_s, scale_s, gate_s = (block.adaln_modulation_self_attn(t2d) + l2d).chunk(3, dim=-1)
    scale_s = scale_s + 1.0
    def bcast(t): return t.repeat_interleave(S_per, dim=0)
    def to5d(t): return t.reshape(M, 1, 1, 1, D)

    # LN + modulate
    ln_s = F.layer_norm(x_5d, (D,), None, None, 1e-6)
    mod_s = ln_s * to5d(scale_s) + to5d(shift_s)
    mod_s_flat = mod_s.reshape(MS, D)

    # QKV
    q = F.linear(mod_s_flat, block.self_attn.q_proj.weight)
    k = F.linear(mod_s_flat, block.self_attn.k_proj.weight)
    v = F.linear(mod_s_flat, block.self_attn.v_proj.weight)

    print(f"\nq: [{q.min():.4f}, {q.max():.4f}]")
    print(f"k: [{k.min():.4f}, {k.max():.4f}]")

    # RMSNorm
    qn = F.rms_norm(q.reshape(MS*NH, HD), (HD,), block.self_attn.q_norm.weight, 1e-6)
    kn = F.rms_norm(k.reshape(MS*NH, HD), (HD,), block.self_attn.k_norm.weight, 1e-6)

    # Per-batch attention
    attn_o = torch.zeros(MS*NH, HD, dtype=DTYPE)
    sc = 1.0/np.sqrt(HD)
    for mb in range(M):
        qh = qn[mb*S_per*NH:(mb+1)*S_per*NH].reshape(S_per, NH, HD).permute(1,0,2)
        kh = kn[mb*S_per*NH:(mb+1)*S_per*NH].reshape(S_per, NH, HD).permute(1,0,2)
        vh = v.reshape(MS*NH, HD)[mb*S_per*NH:(mb+1)*S_per*NH].reshape(S_per, NH, HD).permute(1,0,2)
        scores = torch.bmm(qh, kh.transpose(1,2)) * sc
        aw = F.softmax(scores, dim=-1)
        attn_o[mb*S_per*NH:(mb+1)*S_per*NH] = torch.bmm(aw, vh).permute(1,0,2).reshape(S_per*NH, HD)

    oproj = F.linear(attn_o.reshape(MS, D), block.self_attn.output_proj.weight)
    pt_sa = x_5d.reshape(MS, D) + bcast(gate_s) * oproj

    print(f"\nSA oproj: [{oproj.min():.4f}, {oproj.max():.4f}]")
    print(f"PT b0_sa: [{pt_sa.min():.4f}, {pt_sa.max():.4f}]")

    cpp = b0_sa.astype(np.float32)
    ptn = pt_sa.cpu().numpy().astype(np.float32)
    diff = np.abs(cpp - ptn)
    print(f"\n=== SA diff (using PT t_emb+lora) ===")
    print(f"max_err={diff.max():.4f} mean_err={diff.mean():.4f}")
else:
    print("\n✗ t_emb or lora mismatch — fix head_tail_ops first")

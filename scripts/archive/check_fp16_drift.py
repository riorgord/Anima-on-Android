"""Check: is the C++ error just FP16 drift, or a real bug?
Compares PyTorch FP16 self-consistency vs C++ engine."""
import numpy as np, torch, torch.nn.functional as F, sys, gc
sys.path.insert(0, '/mnt/d/AI/anima_phone/src'); import predict2

CMP = '/mnt/d/AI/anima_phone/output/cmp2'
M, S, D = 2, 256, 2048; MS = M*S; NH = 16; HD = 128; NCTX = 512; CTXD = 1024; HP = 16

cpp_x = np.load(f'{CMP}/x_phone.npy').astype(np.float32).reshape(MS, D)
c_cpp = np.load(f'{CMP}/ctx_phone.npy').astype(np.float32).reshape(M*NCTX, CTXD)

sd_raw = torch.load('/mnt/d/AI/anima_phone/models/diffusion_weights_fp16.pt', weights_only=True)
sd = {}
for k, v in sd_raw.items():
    while k.startswith('net.'): k = k[4:]
    sd[k] = v
del sd_raw

config = dict(max_img_h=240, max_img_w=240, max_frames=128, in_channels=16, out_channels=16,
    patch_spatial=2, patch_temporal=1, concat_padding_mask=True, model_channels=D,
    num_blocks=28, num_heads=NH, mlp_ratio=4.0, crossattn_emb_channels=CTXD,
    pos_emb_cls='rope3d', pos_emb_learnable=True, pos_emb_interpolation='crop',
    min_fps=1, max_fps=30, use_adaln_lora=True, adaln_lora_dim=256,
    rope_h_extrapolation_ratio=4.0, rope_w_extrapolation_ratio=4.0, rope_t_extrapolation_ratio=1.0,
    extra_per_block_abs_pos_emb=False, rope_enable_fps_modulation=False)

dit = predict2.MiniTrainDIT(**config, device='cuda', dtype=torch.float16, operations=torch.nn)
dit.load_state_dict(sd, strict=False); dit.eval(); del sd; gc.collect(); torch.cuda.empty_cache()

sigma = 1.0; ts = torch.tensor([sigma]*M, dtype=torch.float16, device='cuda').unsqueeze(1)
with torch.no_grad():
    t_emb_raw = dit.t_embedder[0](ts).to(torch.float16)
    t_emb_out, lora = dit.t_embedder[1](t_emb_raw)
    t_emb_pt = dit.t_embedding_norm(t_emb_out)

x_pt = torch.from_numpy(cpp_x).cuda().half()
ctx_pt = torch.from_numpy(c_cpp).cuda().half().reshape(M, NCTX, CTXD)
x_5d = x_pt.reshape(M, 1, HP, HP, D)

# Run PT twice
pt1, pt2 = [], []
for run in range(2):
    with torch.no_grad():
        x = x_5d.clone()
        for i in range(3):
            x = dit.blocks[i].forward(x, t_emb_pt, ctx_pt, adaln_lora_B_T_3D=lora)
            (pt1 if run == 0 else pt2).append(x.reshape(MS, D).cpu().numpy().astype(np.float32))
    torch.cuda.empty_cache()

print("PyTorch FP16 run1 vs run2:")
for b in range(3):
    d = np.abs(pt1[b] - pt2[b])
    ok = np.isfinite(pt1[b]) & np.isfinite(pt2[b])
    print(f"  Block {b}: max_err={d[ok].max():.4f}  mean_err={d[ok].mean():.6f}")

print()
print("PyTorch FP16 run1 vs C++:")
cpp_blk = [np.load(f'{CMP}/block_{b:02d}_cpp.npy').astype(np.float32).reshape(MS, D) for b in range(3)]
for b in range(3):
    d = np.abs(pt1[b] - cpp_blk[b])
    ok = np.isfinite(pt1[b]) & np.isfinite(cpp_blk[b])
    print(f"  Block {b}: max_err={d[ok].max():.1f}  mean_err={d[ok].mean():.4f}")
    if d[ok].max() > 10:
        # Find the element with max error
        idx = np.unravel_index(np.argmax(d), d.shape)
        print(f"    max_err at {idx}: PT={pt1[b][idx]:.4f}  C++={cpp_blk[b][idx]:.4f}")

"""Compare C++ RoPE engine vs PyTorch WITH RoPE (full Block.forward)."""
import numpy as np, sys, time, gc
sys.path.insert(0,"/mnt/d/AI/anima_phone/src")
import torch, torch.nn.functional as F, predict2

CMP = "/mnt/d/AI/anima_phone/output/cmp2"
M,S,D = 2,256,2048; MS=M*S; NH=16; HD=128; NCTX=512; CTXD=1024; SP=MS//M
HP = int(np.sqrt(SP))  # 16

cpp_x = np.load(f"{CMP}/x_phone.npy").astype(np.float32).reshape(MS,D)
c_cpp = np.load(f"{CMP}/ctx_phone.npy").astype(np.float32).reshape(M*NCTX,CTXD)
cpp_blk = [np.load(f"{CMP}/block_{b:02d}_cpp.npy").astype(np.float32).reshape(MS,D) for b in range(28)]

sd_raw = torch.load("/mnt/d/AI/anima_phone/models/diffusion_weights_fp16.pt", weights_only=True)
sd = {}
for k,v in sd_raw.items():
    while k.startswith("net."): k=k[4:]
    sd[k]=v
del sd_raw

config = dict(max_img_h=240, max_img_w=240, max_frames=128, in_channels=16, out_channels=16,
    patch_spatial=2, patch_temporal=1, concat_padding_mask=True, model_channels=D,
    num_blocks=28, num_heads=NH, mlp_ratio=4.0, crossattn_emb_channels=CTXD,
    pos_emb_cls="rope3d", pos_emb_learnable=True, pos_emb_interpolation="crop",
    min_fps=1, max_fps=30, use_adaln_lora=True, adaln_lora_dim=256,
    rope_h_extrapolation_ratio=4.0, rope_w_extrapolation_ratio=4.0, rope_t_extrapolation_ratio=1.0,
    extra_per_block_abs_pos_emb=False, rope_enable_fps_modulation=False)

dit = predict2.MiniTrainDIT(**config, device="cuda", dtype=torch.float16, operations=torch.nn)
dit.load_state_dict(sd, strict=False); dit.eval(); del sd; gc.collect(); torch.cuda.empty_cache()

sigma=1.0; ts=torch.tensor([sigma]*M, dtype=torch.float16, device="cuda").unsqueeze(1)
with torch.no_grad():
    t_emb_raw = dit.t_embedder[0](ts).to(torch.float16)
    t_emb_out, lora = dit.t_embedder[1](t_emb_raw)
    t_emb_pt = dit.t_embedding_norm(t_emb_out)

x_pt = torch.from_numpy(cpp_x).cuda().half()
ctx_pt = torch.from_numpy(c_cpp).cuda().half().reshape(M, NCTX, CTXD)
x_5d = x_pt.reshape(M, 1, HP, HP, D)

# Run blocks WITH RoPE (full predict2 Block.forward)
pt_blk = []
with torch.no_grad():
    x = x_5d
    for i in range(28):
        x = dit.blocks[i].forward(x, t_emb_pt, ctx_pt, adaln_lora_B_T_3D=lora)
        pt_blk.append(x.reshape(MS, D).cpu().numpy().astype(np.float32))

print("Block  C++_range                  PT_range                    max_err    mean_err")
for b in range(28):
    cpp = cpp_blk[b]; pt = pt_blk[b]
    ok = np.isfinite(cpp) & np.isfinite(pt)
    diff = np.abs(cpp[ok] - pt[ok])
    flag = " ⚠️" if diff.max() > 100 else ""
    if b < 5 or b > 23 or diff.max() > 100:
        print(f"  {b:2d}   [{cpp[ok].min():7.1f},{cpp[ok].max():7.1f}]  [{pt[ok].min():7.1f},{pt[ok].max():7.1f}]  {diff.max():8.1f}  {diff.mean():8.4f}{flag}")
    if b == 4:
        print(f"  ... (middle blocks omitted)")

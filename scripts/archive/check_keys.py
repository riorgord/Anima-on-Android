"""Quick check: compare model keys vs state dict keys."""
import sys, torch
sys.path.insert(0, '/mnt/d/AI/anima_phone/src')
import predict2

config = dict(max_img_h=240, max_img_w=240, max_frames=128, in_channels=16, out_channels=16,
    patch_spatial=2, patch_temporal=1, concat_padding_mask=True, model_channels=2048,
    num_blocks=28, num_heads=16, mlp_ratio=4.0, crossattn_emb_channels=1024,
    pos_emb_cls="rope3d", pos_emb_learnable=True, pos_emb_interpolation="crop",
    min_fps=1, max_fps=30, use_adaln_lora=True, adaln_lora_dim=256,
    rope_h_extrapolation_ratio=4.0, rope_w_extrapolation_ratio=4.0, rope_t_extrapolation_ratio=1.0,
    extra_per_block_abs_pos_emb=False, rope_enable_fps_modulation=False)

dit = predict2.MiniTrainDIT(**config, device="cpu", dtype=torch.float16, operations=torch.nn)
model_keys = sorted(dit.state_dict().keys())

pt_keys = sorted(torch.load("/mnt/d/AI/anima_phone/models/diffusion_weights_fp16.pt", weights_only=True).keys())

stripped = sorted(k[4:] if k.startswith("net.") else k for k in pt_keys)

m_set = set(model_keys)
s_set = set(stripped)
missing = sorted(m_set - s_set)
extra = sorted(s_set - m_set)

print(f"Model keys: {len(model_keys)}")
for k in model_keys[:5]: print(f"  {k}")
print(f"\nPT keys: {len(pt_keys)}")
for k in pt_keys[:5]: print(f"  {k}")
print(f"\nStripped: {len(stripped)}")
for k in stripped[:5]: print(f"  {k}")
print(f"\nMissing from PT ({len(missing)}):")
for k in missing[:10]: print(f"  {k}")
print(f"\nExtra in PT ({len(extra)}):")
for k in extra[:10]: print(f"  {k}")

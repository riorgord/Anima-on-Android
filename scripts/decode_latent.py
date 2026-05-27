"""Process 2: Load latent from C++ engine, run final_layer + VAE decode, save PNG.
Runs AFTER C++ engine exits (dit_destroy). No OOM risk."""
import sys, torch, numpy as np, time, os
sys.path.insert(0,"/sdcard/anima_on_android/src")
import predict2, wan_vae, torch.nn.functional as F

DEV="cpu"; DTYPE=torch.float16
OUT="/sdcard/anima_on_android/output"
MS,D,M=512,2048,2

# Load DiT (for final_layer + t_embedder only)
print("Loading DiT for final_layer...")
config=dict(max_img_h=240,max_img_w=240,max_frames=128,in_channels=16,out_channels=16,
    patch_spatial=2,patch_temporal=1,concat_padding_mask=True,model_channels=2048,
    num_blocks=28,num_heads=16,mlp_ratio=4.0,crossattn_emb_channels=1024,
    pos_emb_cls="rope3d",pos_emb_learnable=True,pos_emb_interpolation="crop",
    min_fps=1,max_fps=30,use_adaln_lora=True,adaln_lora_dim=256,
    rope_h_extrapolation_ratio=4.0,rope_w_extrapolation_ratio=4.0,rope_t_extrapolation_ratio=1.0,
    extra_per_block_abs_pos_emb=False,rope_enable_fps_modulation=False)
sd=torch.load("/sdcard/anima_on_android/models/diffusion_weights_fp16.pt",weights_only=True,map_location="cpu")
dit=predict2.MiniTrainDIT(**config,device=DEV,dtype=DTYPE,operations=torch.nn)
dit.load_state_dict(sd,strict=False); dit.eval()
print("DiT loaded")

# Load C++ latent
latent=np.fromfile(f"{OUT}/latent_cpp_final.bin",dtype=np.float16).astype(np.float32)
x_t=torch.from_numpy(latent).reshape(2,1,16,16,2048)
print(f"C++ latent: mean={latent.mean():.4f} std={latent.std():.4f}")

# final_layer (sigma=0.333 for last step)
sigma=torch.tensor([0.333,0.333]).unsqueeze(1).to(DTYPE)
ts_emb=dit.t_embedder[0](sigma)
t_emb_fl,adaln_fl=dit.t_embedder[1](ts_emb.to(DTYPE))
t_emb_fl=dit.t_embedding_norm(t_emb_fl)
x_out=dit.final_layer(x_t.to(DTYPE),t_emb_fl,adaln_lora_B_T_3D=adaln_fl)
x_cond=x_out[0:1]
x_latent=dit.unpatchify(x_cond)
print(f"After final_layer: {x_latent.shape}")

# Free DiT
del dit,sd

# VAE decode
print("Loading VAE...")
vae_sd=torch.load("/sdcard/anima_on_android/models/vae_weights_fp16.pt",weights_only=True,map_location="cpu")
vae=wan_vae.WanVAE(dim=96, z_dim=16)
vae.load_state_dict(vae_sd,strict=False)
# Set latent_mean/std (not in checkpoint, must match phone VAE init format [1,C,1,1,1])
lm=torch.tensor([-0.7571,-0.7089,-0.9113,0.1075,-0.1745,0.9653,-0.1517,1.5508,0.4134,-0.0715,0.5517,-0.3632,-0.1922,-0.9497,0.2503,-0.2921])
ls=torch.tensor([0.7008,0.6666,0.8077,0.6013,0.6533,0.5949,0.6072,0.6538,0.5962,0.5265,0.6419,0.5935,0.5950,0.7603,0.6728,0.7236])
vae.latent_mean=lm.view(1,-1,1,1,1)
vae.latent_std=ls.view(1,-1,1,1,1)
vae.eval()

with torch.no_grad():
    t0=time.time()
    decoded=vae.decode(x_latent.float())
    print(f"VAE: {time.time()-t0:.1f}s")

img_np=decoded[0,:,0].permute(1,2,0).clamp(0,1).mul(255).byte().cpu().numpy()
from PIL import Image
out_path=f"{OUT}/round1_image.png"
Image.fromarray(img_np).save(out_path)
size=os.path.getsize(out_path)
print(f"Image saved: {size} bytes ({size/1024:.1f}KB)")

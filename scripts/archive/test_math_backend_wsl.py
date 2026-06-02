"""Test: if we use PT math backend SDPA (not flash), what image size do we get?
This tells us the target for our C math backend implementation.

Run on WSL:
  source /home/riorg/miniconda3/etc/profile.d/conda.sh
  conda activate /home/riorg/anima-work/.conda
  python /mnt/d/AI/anima_phone/anima_rt/scripts/test_math_backend_wsl.py
"""
import sys, json, struct, mmap, os, time, math
import torch, torch.nn as nn
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import predict2
from predict2 import MiniTrainDIT
import wan_vae

class Ops:
    Linear=nn.Linear; RMSNorm=nn.RMSNorm; LayerNorm=nn.LayerNorm; Embedding=nn.Embedding

SAFETENSORS = "/mnt/d/AI/手坤的anima/models/diffusion_model/anima-base-v1.0.safetensors"
VAE_WEIGHTS = "/mnt/d/AI/手坤的anima/models/vae/qwen_image_vae.safetensors"
STEPS = 3; SEED = 6666; H = 32

# ═══ Load DiT weights ═══
print("Loading DiT...")
with open(SAFETENSORS, 'rb') as f:
    header_len = struct.unpack('<Q', f.read(8))[0]
    header = json.loads(f.read(header_len))
data_start = 8 + header_len
sd = {}
with open(SAFETENSORS, 'rb') as f:
    mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
    for k, v in header.items():
        if k == '__metadata__': continue
        clean = k[4:] if k.startswith('net.') else k
        off = data_start + v['data_offsets'][0]
        end = data_start + v['data_offsets'][1]
        np_dtype = np.uint16 if v['dtype'] in ('BF16','F16') else np.float32
        data = torch.from_numpy(np.frombuffer(mm[off:end], dtype=np_dtype).copy()).reshape(v['shape'])
        if v['dtype'] == 'BF16': data = data.view(torch.bfloat16).to(torch.float32)
        elif v['dtype'] == 'F16': data = data.view(torch.float16).to(torch.float32)
        else: data = data.to(torch.float32)
        sd[clean] = data
    mm.close()
print(f"  {len(sd)} tensors")

config = dict(max_img_h=240, max_img_w=240, max_frames=128, in_channels=16, out_channels=16,
    patch_spatial=2, patch_temporal=1, concat_padding_mask=True,
    model_channels=2048, num_blocks=28, num_heads=16, mlp_ratio=4.0,
    crossattn_emb_channels=1024, pos_emb_cls="rope3d", pos_emb_learnable=True,
    pos_emb_interpolation="crop", min_fps=1, max_fps=30,
    use_adaln_lora=True, adaln_lora_dim=256,
    rope_h_extrapolation_ratio=4.0, rope_w_extrapolation_ratio=4.0,
    rope_t_extrapolation_ratio=1.0, extra_per_block_abs_pos_emb=False,
    rope_enable_fps_modulation=False)

dit = MiniTrainDIT(**config, device="cpu", dtype=torch.float32, operations=Ops)
dit.load_state_dict(sd, strict=False); dit.eval(); del sd

# ── Replace F.sdpa with manual math backend ──
import torch.nn.functional as F
_orig_sdpa = F.scaled_dot_product_attention

def math_sdpa(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None):
    """PT math backend: matmul → softmax → matmul. Same as our C code."""
    if scale is None:
        scale = 1.0 / math.sqrt(q.size(-1))
    q = q * scale; k = k * scale
    attn = q @ k.transpose(-2, -1)
    if is_causal:
        mask = torch.triu(torch.ones(attn.shape[-2:], device=q.device), diagonal=1).bool()
        attn = attn.masked_fill(mask, float('-inf'))
    attn = torch.softmax(attn, dim=-1)
    return attn @ v

F.scaled_dot_product_attention = math_sdpa
print("Patched SDPA → PT math backend")

# ═══ Denoising ═══
print(f"Denoising {STEPS} steps...")
gen = torch.Generator().manual_seed(SEED)
x = torch.randn(1, 16, H, H, generator=gen).float()
ctx = torch.randn(1, 512, 1024).float()  # random context for now

linear = torch.linspace(1.0, 0.0, STEPS + 1)[:-1]
sigmas = (3.0 * linear / (1.0 + 2.0 * linear)).tolist() + [0.0]

t0 = time.time()
for i in range(STEPS):
    sigma = sigmas[i]; sigma_next = sigmas[i+1]
    ts = torch.tensor([sigma])

    # CFG: cond + uncond
    x_b = x.unsqueeze(2).repeat(2, 1, 1, 1, 1)  # [2, 16, 1, H, W]
    ts_b = ts.repeat(2)
    ctx_b = ctx.repeat(2, 1, 1)

    with torch.no_grad():
        v_b = dit(x_b, ts_b, ctx_b)
    v_cond, v_uncond = v_b[0:1], v_b[1:2]
    v_cfg = v_uncond + 5.0 * (v_cond - v_uncond)
    x = (x + v_cfg.squeeze(2) * (sigma_next - sigma)).float()

    dt = time.time() - t0
    print(f"  step {i+1}: {dt:.0f}s v=[{v_cond.min():.2f},{v_cond.max():.2f}] nan={torch.isnan(v_cond).any()}")

# ═══ VAE Decode ═══
print("Loading VAE...")
vae_sd = {}
with open(VAE_WEIGHTS, 'rb') as f:
    header_len = struct.unpack('<Q', f.read(8))[0]
    header = json.loads(f.read(header_len))
vae_data_start = 8 + header_len
with open(VAE_WEIGHTS, 'rb') as f:
    mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
    for k, v in header.items():
        if k == '__metadata__': continue
        off = vae_data_start + v['data_offsets'][0]
        end = vae_data_start + v['data_offsets'][1]
        np_dtype = np.uint16 if v['dtype'] in ('BF16','F16') else np.float32
        data = torch.from_numpy(np.frombuffer(mm[off:end], dtype=np_dtype).copy()).reshape(v['shape'])
        if v['dtype'] == 'BF16': data = data.view(torch.bfloat16).to(torch.float32)
        vae_sd[k] = data
    mm.close()

vae = wan_vae.WanVAE(dim=96, z_dim=16, dim_mult=[1,2,4,4], num_res_blocks=2,
    attn_scales=[], temperal_downsample=[False,True,True],
    image_channels=3, conv_out_channels=3, dropout=0.0)
vae.load_state_dict(vae_sd, strict=False); vae.eval(); del vae_sd

print("Decoding...")
with torch.no_grad():
    image = vae.decode(x.unsqueeze(2).float())
img = image[0,:,0].clamp(-1,1)
img = ((img+1)/2*255).permute(1,2,0).cpu().numpy().astype(np.uint8)
out_path = "/mnt/d/AI/anima_phone/anima_rt/output/math_backend_ref.png"
Image.fromarray(img).save(out_path)
fsize = os.stat(out_path).st_size
print(f"Saved: {out_path} ({fsize} bytes)")
print(f"TOTAL: {time.time()-t0:.0f}s")

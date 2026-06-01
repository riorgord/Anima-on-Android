"""Phone pipeline: C++ engine for 28 blocks, lightweight PyTorch for rest"""
import sys, time, gc, ctypes, numpy as np, math, torch
from PIL import Image
import torch.nn.functional as F
sys.path.insert(0,"/sdcard/anima_on_android/src")
import wan_vae
from einops import rearrange

# === Init C++ engine ===
print("Init C++ engine...")
_lib = ctypes.CDLL("/data/local/tmp/libdit_vk.so")
_lib.dit_init.argtypes = [ctypes.c_char_p, ctypes.c_char_p]; _lib.dit_init.restype = ctypes.c_bool
_lib.dit_init_all_blocks.argtypes = []; _lib.dit_init_all_blocks.restype = ctypes.c_bool
_lib.dit_forward_28blocks.argtypes = [ctypes.c_void_p]*5 + [ctypes.c_int]*5
_lib.dit_forward_28blocks.restype = ctypes.c_bool

t0 = time.time()
ok = _lib.dit_init(b"/data/local/tmp/diffusion_weights.bin", b"/data/local/tmp")
print(f"  init = {ok} ({time.time()-t0:.1f}s)")
if not ok: sys.exit(1)
ok = _lib.dit_init_all_blocks()

# === Load lightweight PyTorch modules ===
print("Loading light PyTorch weights...")
DEV = "cpu"; DTYPE = torch.float16
light = torch.load("/sdcard/anima_on_android/models/dit_light.pt", weights_only=True)

w_x_proj   = light["x_embedder.proj.1.weight"].to(DTYPE)         # [2048, 68]
w_t1       = light["t_embedder.1.linear_1.weight"].to(DTYPE)     # [2048, 2048]
w_t2       = light["t_embedder.1.linear_2.weight"].to(DTYPE)     # [6144, 2048]
w_t_norm   = light["t_embedding_norm.weight"].float()             # [2048]
w_fa1      = light["final_layer.adaln_modulation.1.weight"].to(DTYPE)  # [256, 2048]
w_fa2      = light["final_layer.adaln_modulation.2.weight"].to(DTYPE)  # [4096, 256]
w_f_linear = light["final_layer.linear.weight"].to(DTYPE)         # [64, 2048]
# Missing: final_layer.layer_norm.weight [2048] — not in light file!
# Double-check: the original model has elementwise_affine=False for final_layer.layer_norm
# So it has NO weight (uses Identity). Good.

del light; gc.collect()
print(f"  Light weights loaded ({w_x_proj.nbytes + w_t1.nbytes + w_t2.nbytes + w_t_norm.nbytes + w_fa1.nbytes + w_fa2.nbytes + w_f_linear.nbytes} bytes)")

# Manual implementations
def run_x_embedder(x):
    """PatchEmbed: add padding mask → Rearrange → Linear(68, 2048)"""
    B = x.shape[0]
    mask = torch.zeros(B, 1, x.shape[2], x.shape[3], x.shape[4], dtype=x.dtype, device=x.device)
    x = torch.cat([x, mask], dim=1)  # [B, 17, T, H, W]
    x = rearrange(x, "b c (t r) (h m) (w n) -> b t h w (c r m n)", r=1, m=2, n=2)
    return F.linear(x.float(), w_x_proj.float())

def run_t_embedder(ts_b):
    """Timesteps + TimestepEmbedding. ts_b: [M] float. Returns t_emb[M,D], adaln_lora[M,3*D]"""
    M_val = ts_b.shape[0]
    ts = ts_b.flatten().float()
    half = 1024
    freq = 1.0 / (10000.0 ** (torch.arange(half, dtype=torch.float32) / (half - 1)))
    emb = ts[:, None] * freq[None, :]
    emb = torch.cat([torch.cos(emb), torch.sin(emb)], dim=-1)
    emb = rearrange(emb, "(b t) d -> b t d", b=M_val, t=1)
    h = F.linear(emb.to(DTYPE), w_t1); h = F.silu(h)
    h = F.linear(h, w_t2)  # [M, 3*D]
    return emb.squeeze(1).to(DTYPE), h.squeeze(1).to(DTYPE)  # t_emb [M,D], adaln [M,3*D]

def run_final_layer(x_bt, t_emb, adaln_lora_full):
    """FinalLayer: LN → AdaLN → Linear → patches"""
    D = 2048
    x_norm = F.layer_norm(x_bt.float(), (D,), weight=None, bias=None, eps=1e-6)
    h = F.silu(t_emb.float()[:,None,:])  # [M, 1, D]
    h = F.linear(h, w_fa1.float()); h = F.linear(h, w_fa2.float())  # [M, 1, 2*D]
    # final_layer uses only first 2 chunks of adaln_lora (n_adaln_chunks=2)
    adaln = adaln_lora_full.float()[:,:2*D]  # [M, 2*D]
    shift_scale = h + adaln[:,None,:]
    shift, scale = torch.chunk(shift_scale, 2, dim=-1)
    scale = scale + 1.0
    shift = shift[:,:,None,None,:]  # [M,T,D] → [M,T,1,1,D]
    scale = scale[:,:,None,None,:]
    x_mod = x_norm * (1 + scale) + shift
    return F.linear(x_mod.float(), w_f_linear.float())

# === Context ===
ctx_cond = torch.load("/sdcard/anima_on_android/models/context_cond.pt", weights_only=True).to(DEV).to(DTYPE)
ctx_uncond = torch.load("/sdcard/anima_on_android/models/context_uncond.pt", weights_only=True).to(DEV).to(DTYPE)

# === Parameters ===
STEPS = 3; CFG = 5.0; SEED = 6666; H = 32
MS, D, M = 512, 2048, 2

linear_t = torch.linspace(1.0, 0.0, STEPS + 1)[:-1]
sigmas = (3.0 * linear_t / (1.0 + 2.0 * linear_t)).tolist() + [0.0]

gen = torch.Generator(device=DEV).manual_seed(SEED)
x = torch.randn(1, 16, H, H, generator=gen, dtype=DTYPE)

# Compute RoPE freqs once (same for all steps)
from position_embedding import VideoRopePosition3DEmb
rope_emb = VideoRopePosition3DEmb(
    model_channels=2048, len_h=16, len_w=16, len_t=1,
    head_dim=128, is_learnable=False, interpolation="crop",
    h_extrapolation_ratio=4.0, w_extrapolation_ratio=4.0, t_extrapolation_ratio=1.0,
    enable_fps_modulation=False)
rope_freqs_pt = rope_emb.generate_embeddings([2, 1, 16, 16, 2048])  # [S, 64, 2, 2]
# Expand to per-head: [B*S*n_heads, 64, 4] = [8192, 64, 4]
rope_freqs_pt = rope_freqs_pt.reshape(256, 64, 4).repeat(2 * 16, 1, 1).numpy().astype(np.float16)
print(f"RoPE freqs: {rope_freqs_pt.shape} ({rope_freqs_pt.nbytes} bytes)")

print(f"Denoising {STEPS} steps, H={H}, CFG={CFG}...")
t_start = time.time()

for step_i in range(STEPS):
    sigma = sigmas[step_i]; sigma_next = sigmas[step_i + 1]
    ts = torch.tensor([sigma], dtype=DTYPE)  # [1]

    # CFG batch
    x_b = x.unsqueeze(2).repeat(2, 1, 1, 1, 1)  # [2, 16, 1, 32, 32]
    ctx_b = torch.cat([ctx_cond, ctx_uncond], dim=0)  # [2, 512, 1024]

    t0 = time.time()
    with torch.no_grad():
        # x_embedder (lightweight)
        x_flat = run_x_embedder(x_b).reshape(MS, D)  # [512, 2048]

        # t_embedder (lightweight)
        ts_b = ts.repeat(2)  # [2] for CFG
        t_emb, adaln_lora = run_t_embedder(ts_b)  # [2, 2048], [2, 3*D]

        # C++ engine: 28 blocks
        x_np = x_flat.float().numpy().astype(np.float16)
        t_np = t_emb.float().numpy().astype(np.float16)  # [2, 2048]
        c_np = ctx_b.numpy().astype(np.float16)
        out_np = np.zeros((MS, D), dtype=np.float16)

        ok = _lib.dit_forward_28blocks(
            x_np.ctypes.data_as(ctypes.c_void_p),
            t_np.ctypes.data_as(ctypes.c_void_p),
            c_np.ctypes.data_as(ctypes.c_void_p),
            rope_freqs_pt.ctypes.data_as(ctypes.c_void_p),
            out_np.ctypes.data_as(ctypes.c_void_p),
            MS, D, M, 512, 1024)

        # final_layer + unpatchify
        block_out = torch.from_numpy(out_np.astype(np.float32)).reshape(2, 1, 16, 16, D).to(DTYPE)
        # adaln_lora = h[:, :2*D] from the 3*D output (final_layer only needs 2 chunks)
        v_patches = run_final_layer(block_out, t_emb, adaln_lora)
        v_b = rearrange(v_patches, "B T H W (p1 p2 t C) -> B C (T t) (H p1) (W p2)",
                       p1=2, p2=2, t=1)[:, :, :1, :32, :32]

    cpp_time = time.time() - t0

    # CFG
    v_cond = v_b[0:1].float(); v_uncond = v_b[1:2].float()
    v_cfg = v_uncond + CFG * (v_cond - v_uncond)
    x = (x.float() + v_cfg.squeeze(2) * (sigma_next - sigma)).to(DTYPE)

    print(f"  step {step_i+1}/{STEPS}: {cpp_time:.1f}s  ok={ok}")

# === VAE ===
print("Loading VAE...")
vae_sd = torch.load("/sdcard/anima_on_android/models/vae_weights_fp16.pt", weights_only=True)
vae = wan_vae.WanVAE(dim=96, z_dim=16, dim_mult=[1,2,4,4], num_res_blocks=2, attn_scales=[],
    temperal_downsample=[False,True,True], image_channels=3, conv_out_channels=3, dropout=0.0)
vae.load_state_dict({k: v.float() for k, v in vae_sd.items()}, strict=False)
vae.eval(); del vae_sd

print("Decoding...")
with torch.no_grad():
    image = vae.decode(x.float().unsqueeze(2))
img = image[0,:,0].clamp(-1,1)
img = ((img+1)/2*255).permute(1,2,0).cpu().numpy().astype(np.uint8)

out_path = "/sdcard/anima_on_android/output/phone_cpp.png"
Image.fromarray(img).save(out_path)
total_t = time.time() - t_start
print(f"Saved: {out_path}")
print(f"TOTAL: {STEPS} steps, {total_t:.0f}s ({total_t/STEPS:.0f}s/step)")
_lib.dit_destroy()

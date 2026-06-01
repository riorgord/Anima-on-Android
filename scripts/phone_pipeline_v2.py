"""Anima phone pipeline v2 — C++ Vulkan engine for DiT, torch only for VAE."""
import sys, time, gc, os, ctypes, numpy as np
sys.path.insert(0, "/sdcard/anima_on_android/src")
import torch
from PIL import Image
import wan_vae

DEV = "cpu"
DTYPE = torch.float16
STEPS = 3; CFG = 5.0; SEED = 6666; H = 32  # 256×256
M, D, Nctx, CtxD, S, D3 = 2, 2048, 512, 1024, 256, 6144

# ── Load C++ engine ──
_lib = ctypes.CDLL("/data/local/tmp/libdit_vk_v2.so")

_lib.dit_load_safetensors.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
_lib.dit_load_safetensors.restype = ctypes.c_bool

_lib.dit_compute_timestep.argtypes = [ctypes.c_void_p, ctypes.c_int,
    ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p,
    ctypes.c_void_p, ctypes.c_void_p]
_lib.dit_compute_timestep.restype = ctypes.c_bool

_lib.dit_head_x_embed.argtypes = [ctypes.c_void_p, ctypes.c_char_p,
    ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ctypes.c_int, ctypes.c_int]
_lib.dit_head_x_embed.restype = ctypes.c_bool

_lib.dit_forward.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
    ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]
_lib.dit_forward.restype = ctypes.c_bool

_lib.dit_tail_final_layer.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
    ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p,
    ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
_lib.dit_tail_final_layer.restype = ctypes.c_bool

_lib.dit_tail_unpatchify.argtypes = [ctypes.c_void_p, ctypes.c_void_p,
    ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]
_lib.dit_tail_unpatchify.restype = None

_lib.dit_dump_captures.argtypes = [ctypes.c_char_p]; _lib.dit_dump_captures.restype = None
_lib.dit_reset_pool.argtypes = []; _lib.dit_reset_pool.restype = None
_lib.dit_destroy.argtypes = []; _lib.dit_destroy.restype = None

# ── Load contexts ──
print("Loading contexts...")
ctx_cond  = torch.load("/sdcard/anima_on_android/models/context_cond.pt",
                        weights_only=True).to(DEV).to(DTYPE)
ctx_uncond = torch.load("/sdcard/anima_on_android/models/context_uncond.pt",
                         weights_only=True).to(DEV).to(DTYPE)

# ── Init engine ──
print("Init C++ engine...")
t0 = time.time()
ok = _lib.dit_load_safetensors(
    b"/sdcard/anima_on_android/models/diffusion.safetensors",
    b"/data/local/tmp")
if not ok: print("FATAL: engine init failed"); sys.exit(1)
print(f"  OK ({time.time()-t0:.0f}s)")

# ── Scheduler ──
linear = torch.linspace(1.0, 0.0, STEPS + 1)[:-1]
sigmas = (3.0 * linear / (1.0 + 2.0 * linear)).tolist() + [0.0]

# ── Denoising ──
print(f"Denoising {STEPS} steps, H={H}...")
gen = torch.Generator(device=DEV).manual_seed(SEED)
x = torch.randn(1, 16, H, H, generator=gen, dtype=DTYPE)  # [1,16,32,32]
t_start = time.time()

# Prepare context buffer once (fp32 = ctx data upcasted)
ctx_cond_f32 = ctx_cond.float().numpy()
ctx_uncond_f32 = ctx_uncond.float().numpy()
ctx_buf = np.zeros((M, Nctx, CtxD), dtype=np.float32)
ctx_buf[0] = ctx_cond_f32.reshape(M//2, Nctx, CtxD)[0]
ctx_buf[1] = ctx_uncond_f32.reshape(M//2, Nctx, CtxD)[0]
ctx_flat = ctx_buf.reshape(-1).copy()

for i in range(STEPS):
    sigma = sigmas[i]; sigma_next = sigmas[i + 1]
    sigmas_c = (ctypes.c_float * M)(sigma, sigma)  # direct C array

    # Compute timestep embedding on CPU
    t_emb = np.zeros(M * D, dtype=np.float32)
    adaln_lora = np.zeros(M * D3, dtype=np.float32)
    _lib.dit_compute_timestep(
        sigmas_c, M,
        b"t_embedder.1.linear_1.weight",
        b"t_embedder.1.linear_2.weight",
        b"t_embedding_norm.weight",
        t_emb.ctypes.data, adaln_lora.ctypes.data)

    # x_embed: prepare input
    x_b = x.unsqueeze(2).repeat(2, 1, 1, 1, 1)  # [2, 16, 1, 32, 32]
    x_fp16 = x_b.contiguous().cpu().numpy().view(np.uint16).copy()
    x_emb = np.zeros(M * S * D, dtype=np.float32)
    _lib.dit_head_x_embed(
        x_fp16.ctypes.data, b"x_embedder.proj.1.weight",
        x_emb.ctypes.data, M, 16, 1, H, H)

    # Upload t_emb + x_emb to GPU? No — x_emb and t_emb/adaln_lora
    # are already in CPU memory from head ops.
    # Need to upload to Vulkan for block processing.
    # The dit_forward_step function reads from g_xBuf and g_ctxBuf
    # which should already have data.
    # For v2, we need to upload x_emb into g_xBuf before dit_forward_step.
    # But the current API doesn't expose buffer upload separately.
    # WORKAROUND: use dit_forward directly — it reads from the buffers
    # filled by dit_compute_timestep and uploaded externally.

    # ── 28-block forward ──
    # dit_forward internally uploads x_emb→g_xBuf, ctx→g_ctxBuf,
    # runs 28 blocks using pre-uploaded g_tEmbBuf (from dit_compute_timestep),
    # and downloads result to out_fp32
    t0_step = time.time()
    out_fp32 = np.zeros(M * S * D, dtype=np.float32)
    ok = _lib.dit_forward(
        x_emb.ctypes.data, ctx_flat.ctypes.data,
        out_fp32.ctypes.data,
        M * S, D, M, Nctx, CtxD)
    if not ok:
        print(f"  ERROR in step {i+1}: dit_forward failed")
        break
    dit_time = time.time() - t0_step

    # ── final_layer + unpatchify ──
    patches = np.zeros(M * S * 64, dtype=np.float32)
    _lib.dit_tail_final_layer(
        out_fp32.ctypes.data, t_emb.ctypes.data, adaln_lora.ctypes.data,
        b"final_layer.adaln_modulation.1.weight",
        b"final_layer.adaln_modulation.2.weight",
        b"final_layer.linear.weight",
        patches.ctypes.data, M * S, M)

    v_unpatch = np.zeros(M * 16 * 1 * H * H, dtype=np.float32)  # [2,16,1,32,32]
    _lib.dit_tail_unpatchify(patches.ctypes.data, v_unpatch.ctypes.data,
                              M, 1, H // 2, H // 2)

    # CFG mixing: v_b [2,16,1,32,32] → v_cond [1,16,1,32,32], v_uncond [1,16,1,32,32]
    v_b = torch.from_numpy(v_unpatch.reshape(M, 16, 1, H, H)).float()
    v_cond = v_b[0:1]; v_uncond = v_b[1:2]
    v_cfg = v_uncond + CFG * (v_cond - v_uncond)
    x_new = (x.float() + v_cfg.squeeze(2) * (sigma_next - sigma)).to(DTYPE)
    print(f"  step {i+1}/{STEPS}: dit={dit_time:.0f}s x=[{x_new.min():.4f},{x_new.max():.4f}]")
    if i == 0:
        os.makedirs("/sdcard/anima_on_android/output/cmp_v2", exist_ok=True)
        _lib.dit_dump_captures(b"/sdcard/anima_on_android/output/cmp_v2")
        print("  Block 0 captures dumped")
    x = x_new

print(f"TOTAL: {STEPS} steps, {time.time()-t_start:.0f}s")

# ── VAE ──
_lib.dit_destroy()
del _lib; gc.collect()

print("Loading VAE...")
vae_sd = torch.load("/sdcard/anima_on_android/models/vae_weights_fp16.pt", weights_only=True)
vae = wan_vae.WanVAE(dim=96, z_dim=16, dim_mult=[1,2,4,4], num_res_blocks=2,
    attn_scales=[], temperal_downsample=[False,True,True],
    image_channels=3, conv_out_channels=3, dropout=0.0)
vae.load_state_dict({k: v.float() for k, v in vae_sd.items()}, strict=False)
vae.eval(); del vae_sd

print("Decoding...")
with torch.no_grad():
    image = vae.decode(x.float().unsqueeze(2))
img = image[0,:,0].clamp(-1,1)
img = ((img+1)/2*255).permute(1,2,0).cpu().numpy().astype(np.uint8)
out = "/sdcard/anima_on_android/output/phone_v2.png"
Image.fromarray(img).save(out)
print(f"Saved: {out}")

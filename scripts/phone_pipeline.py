import sys, time, gc
sys.path.insert(0, "/sdcard/anima_on_android/src")
import torch, numpy as np
from PIL import Image
import predict2, llm_adapter
import wan_vae  # our fixed WanVAE (latent norm added)
sys.path.insert(0, "/sdcard/anima_on_android/scripts")
import vk_ops  # Vulkan-accelerated ops
import ctypes

DEV = "cpu"
DTYPE = torch.float16
STEPS = 3
CFG = 5.0
SEED = 6666
H = 32  # 256x256

# ── C++ Vulkan AdaLN engine (lightweight: only AdaLN weights ~340KB) ──
_lib_vk = ctypes.CDLL("/data/local/tmp/libdit_vk.so")
_lib_vk.dit_init_adaln_only.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
_lib_vk.dit_init_adaln_only.restype = ctypes.c_bool
_lib_vk.dit_write_lora.argtypes = [ctypes.c_void_p]
_lib_vk.dit_write_lora.restype = None
_lib_vk.dit_write_buf.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_size_t]
_lib_vk.dit_write_buf.restype = ctypes.c_bool
_lib_vk.dit_compute_timestep.argtypes = [ctypes.c_float]
_lib_vk.dit_compute_timestep.restype = ctypes.c_bool
_lib_vk.dit_reset_step_pool.argtypes = []
_lib_vk.dit_reset_step_pool.restype = ctypes.c_bool
_lib_vk.dit_adaln_one_block.argtypes = [ctypes.c_int, ctypes.c_void_p]
_lib_vk.dit_adaln_one_block.restype = ctypes.c_bool
_lib_vk.dit_forward_nblocks.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                                         ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                         ctypes.c_int]
_lib_vk.dit_forward_nblocks.restype = ctypes.c_bool
_lib_vk.dit_destroy.argtypes = []
_lib_vk.dit_destroy.restype = None

class PrecomputedAdaLN(torch.nn.Module):
    """Returns a precomputed [M, 3D] tensor, ignoring emb input."""
    def __init__(self, values_3D):
        super().__init__()
        self.register_buffer('v', values_3D.clone())
    def forward(self, emb):
        return self.v

# Load contexts (small, fast)
ctx_cond = torch.load("/sdcard/anima_on_android/models/context_cond.pt", weights_only=True).to(DEV).to(DTYPE)
ctx_uncond = torch.load("/sdcard/anima_on_android/models/context_uncond.pt", weights_only=True).to(DEV).to(DTYPE)

# Load DiT
print("Loading DiT...")
config = dict(max_img_h=240, max_img_w=240, max_frames=128, in_channels=16, out_channels=16,
    patch_spatial=2, patch_temporal=1, concat_padding_mask=True, model_channels=2048,
    num_blocks=0, num_heads=16, mlp_ratio=4.0, crossattn_emb_channels=1024,
    pos_emb_cls="rope3d", pos_emb_learnable=True, pos_emb_interpolation="crop",
    min_fps=1, max_fps=30, use_adaln_lora=True, adaln_lora_dim=256,
    rope_h_extrapolation_ratio=4.0, rope_w_extrapolation_ratio=4.0, rope_t_extrapolation_ratio=1.0,
    extra_per_block_abs_pos_emb=False, rope_enable_fps_modulation=False)
dit_sd = torch.load("/sdcard/anima_on_android/models/diffusion_weights_fp16.pt", weights_only=True)
dit = predict2.MiniTrainDIT(**config, device=DEV, dtype=DTYPE, operations=vk_ops.HybridOps)
dit.load_state_dict(dit_sd, strict=False)
dit.eval()
del dit_sd; gc.collect()
print("DiT loaded")

# ── Init C++ AdaLN engine ──
print("Init C++ AdaLN engine...")
t0 = time.time()
ok = _lib_vk.dit_init_adaln_only(b"/data/local/tmp/diffusion_weights.bin", b"/data/local/tmp")
print(f"  C++ AdaLN init: {ok} ({time.time()-t0:.0f}s)")
if not ok:
    print("FATAL: C++ AdaLN init failed")
    sys.exit(1)

M, D = 2, 2048  # batch=2 (cond+uncond), hidden=2048

# Replace MLP GELU + self-attention with Vulkan versions
for blk in dit.blocks:
    blk.mlp.activation = vk_ops.HybridGELU()
    _orig_attn = blk.self_attn
    _orig_compute = _orig_attn.compute_attention
    def _make_vk_attn(orig=_orig_attn, orig_comp=_orig_compute):
        scale = 1.0 / (orig.head_dim ** 0.5)
        def vk_compute_attention(q, k, v, transformer_options={}):
            B, S_q, H, D = q.shape
            _, S_kv, _, _ = k.shape
            M_q = B * S_q; M_kv = B * S_kv
            q_f16 = q.reshape(M_q*H, D).cpu().contiguous().to(torch.float16).numpy().view(np.uint16)
            k_f16 = k.reshape(M_kv*H, D).cpu().contiguous().to(torch.float16).numpy().view(np.uint16)
            v_f16 = v.reshape(M_kv*H, D).cpu().contiguous().to(torch.float16).numpy().view(np.uint16)
            o_flat = np.zeros(M_q * H * D, dtype=np.uint16)
            ok = vk_ops._lib_dit.dit_run_attention(
                q_f16.ctypes.data_as(ctypes.c_void_p),
                k_f16.ctypes.data_as(ctypes.c_void_p),
                v_f16.ctypes.data_as(ctypes.c_void_p),
                o_flat.ctypes.data_as(ctypes.c_void_p),
                M_q, M_kv, H, D, ctypes.c_float(scale))
            if not ok:
                return orig_comp(q, k, v, transformer_options)
            o_t = torch.tensor(o_flat.view(np.float16), device=q.device)
            o_t = o_t.reshape(B, S_q, H, D).reshape(B, S_q, H*D)
            return orig.output_dropout(orig.output_proj(o_t.to(q.dtype)))
        return vk_compute_attention
    blk.self_attn.compute_attention = _make_vk_attn()

    # Cross-attn: same 3-pass Vulkan, now batched (M_q split into ≤64 Q tokens,
    # ≤1024 WG per dispatch). Safe for Adreno 730 across steps.
    _orig_cross = blk.cross_attn
    _orig_cross_comp = _orig_cross.compute_attention
    def _make_vk_cross_attn(orig=_orig_cross, orig_comp=_orig_cross_comp):
        scale = 1.0 / (orig.head_dim ** 0.5)
        def vk_compute_attention(q, k, v, transformer_options={}):
            B, S_q, H, D = q.shape
            _, S_kv, _, _ = k.shape
            M_q = B * S_q; M_kv = B * S_kv
            q_f16 = q.reshape(M_q*H, D).cpu().contiguous().to(torch.float16).numpy().view(np.uint16)
            k_f16 = k.reshape(M_kv*H, D).cpu().contiguous().to(torch.float16).numpy().view(np.uint16)
            v_f16 = v.reshape(M_kv*H, D).cpu().contiguous().to(torch.float16).numpy().view(np.uint16)
            o_flat = np.zeros(M_q * H * D, dtype=np.uint16)
            ok = vk_ops._lib_dit.dit_run_attention(
                q_f16.ctypes.data_as(ctypes.c_void_p),
                k_f16.ctypes.data_as(ctypes.c_void_p),
                v_f16.ctypes.data_as(ctypes.c_void_p),
                o_flat.ctypes.data_as(ctypes.c_void_p),
                M_q, M_kv, H, D, ctypes.c_float(scale))
            if not ok:
                return orig_comp(q, k, v, transformer_options)
            o_t = torch.tensor(o_flat.view(np.float16), device=q.device)
            o_t = o_t.reshape(B, S_q, H, D).reshape(B, S_q, H*D)
            return orig.output_dropout(orig.output_proj(o_t.to(q.dtype)))
        return vk_compute_attention
    blk.cross_attn.compute_attention = _make_vk_cross_attn()

# Scheduler
def time_snr_shift(a, t): return a * t / (1.0 + (a - 1.0) * t)
linear = torch.linspace(1.0, 0.0, STEPS + 1)[:-1]
sigmas = (3.0 * linear / (1.0 + 2.0 * linear)).tolist() + [0.0]

# Denoising
print(f"Denoising {STEPS} steps, H={H}...")
gen = torch.Generator(device=DEV).manual_seed(SEED)
x = torch.randn(1, 16, H, H, generator=gen, dtype=DTYPE)
t_start = time.time()

for i in range(STEPS):
    sigma = sigmas[i]
    sigma_next = sigmas[i + 1]
    ts = torch.tensor([sigma], dtype=DTYPE)

    # C++ CPU: compute t_emb + lora from sigma directly into GPU buffers
    _lib_vk.dit_compute_timestep(float(sigma))

    # GPU AdaLN for all 28 blocks
    t0_adaln = time.time()
    adaln_all = []
    out_buf = np.zeros(9 * M * D, dtype=np.uint16)
    for blk_idx in range(28):
        ok = _lib_vk.dit_adaln_one_block(blk_idx, out_buf.ctypes.data_as(ctypes.c_void_p))
        if not ok:
            print(f"  ERROR: dit_adaln_one_block({blk_idx}) failed")
            break
        adaln_all.append(out_buf.copy().view(np.float16).reshape(9, M, D))
    adaln_time = time.time() - t0_adaln

    # Inject precomputed AdaLN into blocks
    for blk_idx, blk in enumerate(dit.blocks):
        a = adaln_all[blk_idx]  # [9, M, D]
        v_self  = torch.from_numpy(np.concatenate([a[0], a[1], a[2]], axis=-1)).reshape(2, 1, 3*D).to(DEV).to(DTYPE)
        v_cross = torch.from_numpy(np.concatenate([a[3], a[4], a[5]], axis=-1)).reshape(2, 1, 3*D).to(DEV).to(DTYPE)
        v_mlp   = torch.from_numpy(np.concatenate([a[6], a[7], a[8]], axis=-1)).reshape(2, 1, 3*D).to(DEV).to(DTYPE)
        blk.adaln_modulation_self_attn = PrecomputedAdaLN(v_self)
        blk.adaln_modulation_cross_attn = PrecomputedAdaLN(v_cross)
        blk.adaln_modulation_mlp = PrecomputedAdaLN(v_mlp)
        blk.use_adaln_lora = False

    x_b = x.unsqueeze(2).repeat(2, 1, 1, 1, 1)
    ctx_b = torch.cat([ctx_cond, ctx_uncond], dim=0)
    ts_b2 = ts.repeat(2)
    t0 = time.time()
    with torch.no_grad():
        v_b = dit(x_b, ts_b2, ctx_b)
    dit_time = time.time() - t0

    # C++ diag: run 28 pre-recorded blocks in parallel (skip-attn, values discarded)
    MS_val = 2 * 256  # M * S = 512
    x_flat = np.zeros(MS_val * D, dtype=np.uint16)
    _lib_vk.dit_forward_nblocks(x_flat.ctypes.data_as(ctypes.c_void_p),
        None, x_flat.ctypes.data_as(ctypes.c_void_p),
        x_flat.ctypes.data_as(ctypes.c_void_p),
        MS_val, D, 2, 512, 1024, 28)

    # Restore blocks for next step
    for blk in dit.blocks:
        blk.use_adaln_lora = True

    # Reset descriptor pool between steps
    _lib_vk.dit_reset_step_pool()

    v_cond = v_b[0:1].float()
    v_uncond = v_b[1:2].float()
    v_cfg = v_uncond + CFG * (v_cond - v_uncond)
    x = (x.float() + v_cfg.squeeze(2) * (sigma_next - sigma)).to(DTYPE)
    print(f"  step {i+1}/{STEPS}: dit={dit_time:.0f}s adaln={adaln_time:.0f}s "
          f"VK={vk_ops._VK_TIME:.0f}s CPU={vk_ops._CPU_TIME:.0f}s (total {time.time()-t_start:.0f}s)")

# Diagnostic
print(f"VkGEMM: {vk_ops._VK_COUNT} Vulkan calls, {vk_ops._CPU_COUNT} CPU calls")
print(f"  VK wall: {vk_ops._VK_TIME:.0f}s  CPU: {vk_ops._CPU_TIME:.0f}s")

# Internal phase breakdown (from C++ clock_gettime)
import vk_linear
_pack = ctypes.c_double(); _cmd = ctypes.c_double(); _gpu = ctypes.c_double(); _read = ctypes.c_double(); _nc = ctypes.c_int()
vk_linear._lib.vk_gemm_get_timings_us(ctypes.byref(_pack), ctypes.byref(_cmd), ctypes.byref(_gpu), ctypes.byref(_read), ctypes.byref(_nc))
_n = _nc.value or 1
print(f"  Phases (avg/{_n} calls): pack={_pack.value/_n*1e-3:.1f}ms cmd={_cmd.value/_n*1e-3:.1f}ms gpu={_gpu.value/_n*1e-3:.1f}ms read={_read.value/_n*1e-3:.1f}ms")
_pack_s = _pack.value * 1e-6; _cmd_s = _cmd.value * 1e-6; _gpu_s = _gpu.value * 1e-6; _read_s = _read.value * 1e-6
print(f"  Breakdown: pack={_pack_s:.0f}s cmd={_cmd_s:.0f}s submit+wait={_gpu_s:.0f}s read={_read_s:.0f}s")

# Unload C++ engine before VAE (free GPU memory)
print("Unloading C++ engine...")
_lib_vk.dit_destroy()
gc.collect()

# VAE — our fixed WanVAE (latent normalization added)
print("Loading VAE...")
vae_sd = torch.load("/sdcard/anima_on_android/models/vae_weights_fp16.pt", weights_only=True)
vae = wan_vae.WanVAE(dim=96, z_dim=16, dim_mult=[1,2,4,4], num_res_blocks=2, attn_scales=[],
    temperal_downsample=[False,True,True], image_channels=3, conv_out_channels=3, dropout=0.0)
vae.load_state_dict({k: v.float() for k, v in vae_sd.items()}, strict=False)  # strict=False for new buffers
vae.eval(); del vae_sd

print("Decoding...")
with torch.no_grad():
    image = vae.decode(x.float().unsqueeze(2))
img = image[0,:,0].clamp(-1,1)
img = ((img+1)/2*255).permute(1,2,0).cpu().numpy().astype(np.uint8)
out = "/sdcard/anima_on_android/output/phone_first.png"
Image.fromarray(img).save(out)
total_t = time.time() - t_start
print(f"Saved: {out}")
print(f"TOTAL: {STEPS} steps, {total_t:.0f}s ({total_t/STEPS:.0f}s/step), {H*8}x{H*8}")
del dit, vae, x, img; gc.collect()

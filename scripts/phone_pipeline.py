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
_lib_vk.dit_adaln_one_block.argtypes = [ctypes.c_int, ctypes.c_void_p]
_lib_vk.dit_adaln_one_block.restype = ctypes.c_bool
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
    num_blocks=28, num_heads=16, mlp_ratio=4.0, crossattn_emb_channels=1024,
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

    # Compute t_emb + lora for C++ GPU AdaLN (batch=2: cond+uncond)
    ts_b2_emb = ts.repeat(2).unsqueeze(1)  # [2, 1]
    with torch.no_grad():
        sin_emb = dit.t_embedder[0](ts_b2_emb).to(DTYPE)            # [2,1,D] sinusoidal
        t_emb_raw, lora_raw = dit.t_embedder[1](sin_emb)            # t_emb=[2,1,D], lora=[2,1,3D]
        t_emb_norm = dit.t_embedding_norm(t_emb_raw)                # [2, 1, D]

    t_emb_np = t_emb_norm.squeeze(1).cpu().numpy().astype(np.float16)  # [M, D]
    lora_np = lora_raw.squeeze(1).cpu().numpy().astype(np.float16)      # [M, 3D]
    lora_3MD = lora_np.reshape(M, 3, D).transpose(1, 0, 2).copy()      # [3, M, D]

    # Upload t_emb + lora to C++ engine
    _lib_vk.dit_write_buf(1, t_emb_np.ctypes.data_as(ctypes.c_void_p), t_emb_np.nbytes)
    _lib_vk.dit_write_lora(lora_3MD.ctypes.data_as(ctypes.c_void_p))

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

    # Inject precomputed AdaLN into blocks (lora already baked in)
    for blk_idx, blk in enumerate(dit.blocks):
        a = adaln_all[blk_idx]  # [9, M, D]: shift/scale/gate × self/cross/mlp
        v_self  = torch.from_numpy(np.concatenate([a[0], a[1], a[2]], axis=-1)).reshape(2, 1, 3*D).to(DEV).to(DTYPE)
        v_cross = torch.from_numpy(np.concatenate([a[3], a[4], a[5]], axis=-1)).reshape(2, 1, 3*D).to(DEV).to(DTYPE)
        v_mlp   = torch.from_numpy(np.concatenate([a[6], a[7], a[8]], axis=-1)).reshape(2, 1, 3*D).to(DEV).to(DTYPE)
        blk.adaln_modulation_self_attn = PrecomputedAdaLN(v_self)
        blk.adaln_modulation_cross_attn = PrecomputedAdaLN(v_cross)
        blk.adaln_modulation_mlp = PrecomputedAdaLN(v_mlp)
        blk.use_adaln_lora = False

    # DiT forward (blocks use precomputed AdaLN; final_layer uses PyTorch)
    x_b = x.unsqueeze(2).repeat(2, 1, 1, 1, 1)
    ctx_b = torch.cat([ctx_cond, ctx_uncond], dim=0)
    ts_b2 = ts.repeat(2)
    t0 = time.time()
    with torch.no_grad():
        v_b = dit(x_b, ts_b2, ctx_b)
    dit_time = time.time() - t0

    # Restore blocks for next step
    for blk in dit.blocks:
        blk.use_adaln_lora = True

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

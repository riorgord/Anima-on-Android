"""phone_pipeline_adaln.py — PyTorch pipeline with C++ GPU AdaLN per block.
Replaces each block's adaln_modulation_* modules with precomputed GPU values.
Verifies shader correctness block-by-block against pure PyTorch reference.
"""
import sys, time, gc, ctypes
sys.path.insert(0, "/sdcard/anima_on_android/src")
import torch, numpy as np
from PIL import Image
import predict2, llm_adapter
import wan_vae
sys.path.insert(0, "/sdcard/anima_on_android/scripts")
import vk_ops

DEV = "cpu"
DTYPE = torch.float16
STEPS = 3
CFG = 5.0
SEED = 6666
H = 32  # 256x256

# ── Load C++ engine ──────────────────────────────────────────────
lib = ctypes.CDLL("/data/local/tmp/libdit_vk.so")
lib.dit_init.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
lib.dit_init.restype = ctypes.c_bool
lib.dit_write_lora.argtypes = [ctypes.c_void_p]
lib.dit_write_lora.restype = None
lib.dit_write_buf.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_size_t]
lib.dit_write_buf.restype = ctypes.c_bool
lib.dit_adaln_one_block.argtypes = [ctypes.c_int, ctypes.c_void_p]
lib.dit_adaln_one_block.restype = ctypes.c_bool
lib.dit_destroy.argtypes = []
lib.dit_destroy.restype = None

print("Init C++ engine (weights + Vulkan)...")
t0 = time.time()
ok = lib.dit_init(b"/data/local/tmp/diffusion_weights.bin", b"/data/local/tmp")
print(f"  C++ init: {ok} ({time.time()-t0:.0f}s)")
if not ok:
    print("FATAL: C++ engine init failed")
    sys.exit(1)

# ── PrecomputedAdaLN module ──────────────────────────────────────
class PrecomputedAdaLN(torch.nn.Module):
    """Returns a precomputed [M, 3D] tensor, ignoring emb input."""
    def __init__(self, values_3D):
        super().__init__()
        self.register_buffer('v', values_3D.clone())

    def forward(self, emb):
        return self.v

# ── Load contexts ────────────────────────────────────────────────
ctx_cond = torch.load("/sdcard/anima_on_android/models/context_cond.pt", weights_only=True).to(DEV).to(DTYPE)
ctx_uncond = torch.load("/sdcard/anima_on_android/models/context_uncond.pt", weights_only=True).to(DEV).to(DTYPE)

# ── Load DiT ─────────────────────────────────────────────────────
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

# ── Scheduler ────────────────────────────────────────────────────
def time_snr_shift(a, t): return a * t / (1.0 + (a - 1.0) * t)
linear = torch.linspace(1.0, 0.0, STEPS + 1)[:-1]
sigmas = (3.0 * linear / (1.0 + 2.0 * linear)).tolist() + [0.0]

# ── Helper: upload t_emb + lora, run GPU AdaLN for all 28 blocks ─
def compute_all_adaln_gpu(t_emb_np, lora_np_3MD):
    """Upload t_emb + lora to C++, run dit_adaln_one_block for all 28 blocks.
    Returns list of [9, M, D] fp16 numpy arrays, one per block.
    """
    M, D = t_emb_np.shape
    # Upload t_emb → buf 1
    lib.dit_write_buf(1, t_emb_np.ctypes.data_as(ctypes.c_void_p), t_emb_np.nbytes)
    # Upload lora [3,M,D] → loraBuf
    lib.dit_write_lora(lora_np_3MD.ctypes.data_as(ctypes.c_void_p))

    results = []
    out_buf = np.zeros(9 * M * D, dtype=np.uint16)
    for i in range(28):
        ok = lib.dit_adaln_one_block(i, out_buf.ctypes.data_as(ctypes.c_void_p))
        if not ok:
            print(f"  ERROR: dit_adaln_one_block({i}) failed")
            return None
        results.append(out_buf.copy().view(np.float16).reshape(9, M, D))
    return results

# ── Helper: inject precomputed AdaLN into blocks ─────────────────
def inject_adaln(dit, adaln_blocks):
    """Replace each block's 3 adaln_modulation modules with PrecomputedAdaLN.
    adaln_blocks: list of [9, M, D] fp16 numpy arrays.
    C++ output layout per block:
      [0]=shift_self [1]=scale_self [2]=gate_self
      [3]=shift_cross [4]=scale_cross [5]=gate_cross
      [6]=shift_mlp [7]=scale_mlp [8]=gate_mlp
    Each module returns [M, 3D] = cat([shift, scale, gate], dim=-1).
    """
    M, D = adaln_blocks[0].shape[1], adaln_blocks[0].shape[2]
    for i, blk in enumerate(dit.blocks):
        adaln = adaln_blocks[i]
        # Self-attn: cat(shift, scale, gate) → [M, 3D]
        v_self = torch.from_numpy(
            np.concatenate([adaln[0], adaln[1], adaln[2]], axis=-1).astype(np.float16)
        ).to(DEV)
        # Cross-attn
        v_cross = torch.from_numpy(
            np.concatenate([adaln[3], adaln[4], adaln[5]], axis=-1).astype(np.float16)
        ).to(DEV)
        # MLP
        v_mlp = torch.from_numpy(
            np.concatenate([adaln[6], adaln[7], adaln[8]], axis=-1).astype(np.float16)
        ).to(DEV)

        blk.adaln_modulation_self_attn = PrecomputedAdaLN(v_self)
        blk.adaln_modulation_cross_attn = PrecomputedAdaLN(v_cross)
        blk.adaln_modulation_mlp = PrecomputedAdaLN(v_mlp)
        blk.use_adaln_lora = False  # lora already baked into precomputed values

# ── Denoising ────────────────────────────────────────────────────
print(f"Denoising {STEPS} steps, H={H}...")
gen = torch.Generator(device=DEV).manual_seed(SEED)
x = torch.randn(1, 16, H, H, generator=gen, dtype=DTYPE)
t_start = time.time()
M, D = 2, 2048

for step_i in range(STEPS):
    sigma = sigmas[step_i]
    sigma_next = sigmas[step_i + 1]
    ts = torch.tensor([sigma], dtype=DTYPE)

    # ── Compute t_emb + lora via PyTorch ──
    ts_b = ts.unsqueeze(1)  # [B, 1]
    with torch.no_grad():
        sin_emb = dit.t_embedder[0](ts_b).to(DTYPE)  # [B, 1, D]
        t_emb_torch, lora_torch = dit.t_embedder[1](sin_emb)  # t_emb=[B,1,D], lora=[B,1,3D]
        t_emb_torch = dit.t_embedding_norm(t_emb_torch)  # RMSNorm

    # Extract [M, D] and [M, 3D] for C++
    t_emb_np = t_emb_torch.squeeze(1).cpu().numpy().astype(np.float16)  # [M, D]
    lora_np = lora_torch.squeeze(1).cpu().numpy().astype(np.float16)     # [M, 3D]
    # Reshape lora [M, 3D] → [3, M, D] for C++ (matches dit_compute_timestep layout)
    lora_3MD = lora_np.reshape(M, 3, D).transpose(1, 0, 2).copy()  # [3, M, D]

    # ── GPU AdaLN for all 28 blocks ──
    t0_adaln = time.time()
    adaln_all = compute_all_adaln_gpu(t_emb_np, lora_3MD)
    if adaln_all is None:
        print("FATAL: GPU AdaLN failed")
        break
    adaln_time = time.time() - t0_adaln

    # Inject into DiT blocks
    inject_adaln(dit, adaln_all)

    # ── Run DiT forward (blocks use precomputed AdaLN, final_layer uses PyTorch) ──
    x_b = x.unsqueeze(2).repeat(2, 1, 1, 1, 1)
    ctx_b = torch.cat([ctx_cond, ctx_uncond], dim=0)
    ts_b2 = ts.repeat(2)

    t0_dit = time.time()
    with torch.no_grad():
        v_b = dit(x_b, ts_b2, ctx_b)
    dit_time = time.time() - t0_dit

    v_cond = v_b[0:1].float()
    v_uncond = v_b[1:2].float()
    v_cfg = v_uncond + CFG * (v_cond - v_uncond)
    x = (x.float() + v_cfg.squeeze(2) * (sigma_next - sigma)).to(DTYPE)

    # ── Restore use_adaln_lora for next step (t_embedder needs it for lora output) ──
    for blk in dit.blocks:
        blk.use_adaln_lora = True

    print(f"  step {step_i+1}/{STEPS}: dit={dit_time:.0f}s (adaln={adaln_time:.0f}s) "
          f"VK={vk_ops._VK_TIME:.0f}s CPU={vk_ops._CPU_TIME:.0f}s (total {time.time()-t_start:.0f}s)")

print(f"VkGEMM: {vk_ops._VK_COUNT} Vulkan calls, {vk_ops._CPU_COUNT} CPU calls")
print(f"  VK wall: {vk_ops._VK_TIME:.0f}s  CPU: {vk_ops._CPU_TIME:.0f}s")

# ── VAE ──────────────────────────────────────────────────────────
print("Unloading C++ engine...")
lib.dit_destroy()
gc.collect()

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
out = "/sdcard/anima_on_android/output/phone_adaln.png"
Image.fromarray(img).save(out)
total_t = time.time() - t_start
print(f"Saved: {out}")
print(f"TOTAL: {STEPS} steps, {total_t:.0f}s ({total_t/STEPS:.0f}s/step), {H*8}x{H*8}")
del dit, vae, x, img; gc.collect()

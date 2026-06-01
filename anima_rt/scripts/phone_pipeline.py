"""Anima HybridOps pipeline — loads safetensors directly, weights in Vulkan only."""
import sys, time, gc, json, struct, mmap, os, ctypes
sys.path.insert(0, "/sdcard/anima_on_android/src")
import torch, numpy as np
from PIL import Image
import predict2, llm_adapter
import wan_vae
sys.path.insert(0, "/sdcard/anima_on_android/scripts")
import vk_ops

# ── Memory tracking ──
def mem_rss_mb():
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) // 1024
    except:
        pass
    return -1

def print_mem(tag):
    rss = mem_rss_mb()
    print(f"  [MEM:{tag}] VmRSS={rss}MB" if rss > 0 else f"  [MEM:{tag}] unknown")

# Background memory logger
_mem_log = []
_mem_running = True
def _mem_logger(interval=0.5):
    import time as _time
    while _mem_running:
        rss = mem_rss_mb()
        _mem_log.append((_time.time(), rss))
        _time.sleep(interval)

import threading
_mem_thread = threading.Thread(target=_mem_logger, args=(0.5,), daemon=True)
_mem_thread.start()
print("Memory logger started (0.5s interval)")

DEV = "cpu"
DTYPE = torch.float16
STEPS = 3
CFG = 5.0
SEED = 6666
H = 32  # 256×256

# ═══════════════════════════════════════════════════════════════
# Safetensors reader (no safetensors package needed)
# ═══════════════════════════════════════════════════════════════
class SafetensorsReader:
    """Minimal safetensors reader using mmap."""
    def __init__(self, path):
        self._path = path
        with open(path, 'rb') as f:
            header_len = struct.unpack('<Q', f.read(8))[0]
            header = json.loads(f.read(header_len).decode('utf-8'))
        self.header = header
        self.data_start = 8 + header_len
        self._mmap = None
        self._file = None

    def keys(self):
        return list(self.header.keys())

    def dtype_code(self, key):
        """2=BF16, 1=F16, 0=F32"""
        d = self.header[key]['dtype']
        return {'BF16': 2, 'F16': 1, 'F32': 0}.get(d, 2)

    def numpy_dtype(self, key):
        d = self.header[key]['dtype']
        return np.uint16 if d in ('BF16', 'F16') else np.float32

    def get_tensor(self, key):
        if self._mmap is None:
            self._file = open(self._path, 'rb')
            self._mmap = mmap.mmap(self._file.fileno(), 0, access=mmap.ACCESS_READ)
        info = self.header[key]
        off = self.data_start + info['data_offsets'][0]
        end = self.data_start + info['data_offsets'][1]
        buf = self._mmap[off:end]
        arr = np.frombuffer(buf, dtype=self.numpy_dtype(key))
        return arr.reshape(info['shape'])

    def close(self):
        if self._mmap:
            self._mmap.close()
            self._file.close()
            self._mmap = None
            self._file = None


# ═══════════════════════════════════════════════════════════════
# Step 1: Init Vulkan engine
# ═══════════════════════════════════════════════════════════════
print("Init Vulkan engine...")
if not vk_ops._lib.vk_engine_init():
    print("FATAL: Vulkan engine init failed")
    sys.exit(1)
print_mem("vk_init")

# ═══════════════════════════════════════════════════════════════
# Step 2: Load model weights from safetensors
# ═══════════════════════════════════════════════════════════════
SAFETENSORS = "/sdcard/anima_on_android/models/diffusion.safetensors"
print(f"Loading safetensors: {SAFETENSORS}...")
st = SafetensorsReader(SAFETENSORS)
all_keys = st.keys()

# Detect prefix (e.g. "net.") — use first real tensor key, not __metadata__
PREFIX = ""
tensor_keys = [k for k in all_keys if k != "__metadata__"]
if tensor_keys and '.' in tensor_keys[0]:
    first_part = tensor_keys[0].split('.')[0]
    if first_part not in ('blocks', 'x_embedder', 't_embedder', 'final_layer',
                          't_embedding_norm', 'pos_embedder', 'llm_adapter'):
        PREFIX = first_part + '.'
        print(f"  Detected prefix: '{PREFIX}' — will strip from keys")

def strip_prefix(k):
    return k[len(PREFIX):] if PREFIX and k.startswith(PREFIX) else k

# Classify and load
shell_sd = {}
n_vk = 0
n_shell = 0

for key in all_keys:
    if key == "__metadata__":
        continue
    clean_key = strip_prefix(key)
    if vk_ops.is_block_gemm_key(clean_key):
        # → Vulkan: block GEMM weight
        data = st.get_tensor(key)
        dc = st.dtype_code(key)
        shape = list(data.shape)
        shape_arr = (ctypes.c_int * len(shape))(*shape)
        ret = vk_ops._lib.vk_weight_add(
            clean_key.encode(),
            data.ctypes.data,
            dc,
            shape_arr,
            len(shape))
        if ret < 0:
            print(f"  WARNING: vk_weight_add({clean_key}) failed: {ret}")
        del data
        n_vk += 1
    else:
        # → PyTorch: shell weight
        data = st.get_tensor(key)
        raw_dtype = st.header[key]['dtype']
        if raw_dtype == 'BF16':
            # BF16: uint16 bits → reinterpret as bfloat16 → convert to float16
            tensor = torch.from_numpy(data.copy()).view(torch.bfloat16).to(torch.float16)
        elif raw_dtype == 'F16':
            # FP16: uint16 bits → reinterpret as float16
            tensor = torch.from_numpy(data.copy()).view(torch.float16)
        elif raw_dtype == 'F32':
            tensor = torch.from_numpy(data.copy()).to(torch.float16)
        else:
            tensor = torch.from_numpy(data.copy()).to(torch.float16)
        shell_sd[clean_key] = tensor
        del data
        n_shell += 1

st.close()
gc.collect()
print(f"  {n_vk} → Vulkan, {n_shell} → PyTorch shell")

# Finalize Vulkan engine (creates GEMM pipeline + scratch buffers)
print("Finalizing Vulkan engine...")
if not vk_ops._lib.vk_engine_finalize():
    print("FATAL: vk_engine_finalize failed")
    sys.exit(1)

# ═══════════════════════════════════════════════════════════════
# Step 3: Create PyTorch model shell + load shell weights
# ═══════════════════════════════════════════════════════════════
print("Creating DiT shell (DummyOps — zero weight mem)...")
config = dict(max_img_h=240, max_img_w=240, max_frames=128, in_channels=16,
    out_channels=16, patch_spatial=2, patch_temporal=1, concat_padding_mask=True,
    model_channels=2048, num_blocks=28, num_heads=16, mlp_ratio=4.0,
    crossattn_emb_channels=1024, pos_emb_cls="rope3d", pos_emb_learnable=True,
    pos_emb_interpolation="crop", min_fps=1, max_fps=30,
    use_adaln_lora=True, adaln_lora_dim=256,
    rope_h_extrapolation_ratio=4.0, rope_w_extrapolation_ratio=4.0,
    rope_t_extrapolation_ratio=1.0, extra_per_block_abs_pos_emb=False,
    rope_enable_fps_modulation=False)

dit = predict2.MiniTrainDIT(**config, device=DEV, dtype=DTYPE,
                             operations=vk_ops.DummyOps)
print_mem("shell_created")
# Replace shell DummyLinear → nn.Linear so load_state_dict can fill them
vk_ops.patch_shell_linear(dit)
print_mem("shell_patched")
dit.load_state_dict(shell_sd, strict=False)
dit.eval()
del shell_sd; gc.collect()
print("Shell weights loaded")
print_mem("shell_loaded")

# ═══════════════════════════════════════════════════════════════
# Step 4: Patch block GEMM layers → VulkanGemmLinear
# ═══════════════════════════════════════════════════════════════
vk_ops.patch_block_layers(dit)

# ═══════════════════════════════════════════════════════════════
# Step 5: Load context + setup scheduler
# ═══════════════════════════════════════════════════════════════
ctx_cond = torch.load("/sdcard/anima_on_android/models/context_cond.pt",
                       weights_only=True).to(DEV).to(DTYPE)
ctx_uncond = torch.load("/sdcard/anima_on_android/models/context_uncond.pt",
                         weights_only=True).to(DEV).to(DTYPE)

def time_snr_shift(a, t): return a * t / (1.0 + (a - 1.0) * t)
linear = torch.linspace(1.0, 0.0, STEPS + 1)[:-1]
sigmas = (3.0 * linear / (1.0 + 2.0 * linear)).tolist() + [0.0]

# ═══════════════════════════════════════════════════════════════
# Step 6: Denoising
# ═══════════════════════════════════════════════════════════════
print(f"Denoising {STEPS} steps, H={H}...")
gen = torch.Generator(device=DEV).manual_seed(SEED)
x = torch.randn(1, 16, H, H, generator=gen, dtype=DTYPE)
t_start = time.time()

for i in range(STEPS):
    sigma = sigmas[i]
    sigma_next = sigmas[i + 1]
    ts = torch.tensor([sigma], dtype=DTYPE)

    x_b = x.unsqueeze(2).repeat(2, 1, 1, 1, 1)
    ctx_b = torch.cat([ctx_cond, ctx_uncond], dim=0)
    ts_b = ts.repeat(2)

    # Reset per-step descriptor pool
    vk_ops._lib.vk_reset_pool()

    t0 = time.time()
    try:
        with torch.no_grad():
            v_b = dit(x_b, ts_b, ctx_b)
    except Exception as e:
        print(f"  FATAL in step {i+1}: {e}")
        import traceback; traceback.print_exc()
        break
    dit_time = time.time() - t0

    v_cond = v_b[0:1].float()
    v_uncond = v_b[1:2].float()
    v_cfg = v_uncond + CFG * (v_cond - v_uncond)
    x_new = (x.float() + v_cfg.squeeze(2) * (sigma_next - sigma)).to(DTYPE)
    print(f"  step {i+1}/{STEPS}: dit={dit_time:.0f}s"
          f" v_cond=[{v_cond.min():.2f},{v_cond.max():.2f}]"
          f" nan={torch.isnan(v_cond).sum().item()}"
          f" x=[{x_new.min():.4f},{x_new.max():.4f}]"
          f" (total {time.time()-t_start:.0f}s)")
    x = x_new

print(f"VkGEMM: {vk_ops._VK_COUNT} Vulkan, {vk_ops._CPU_COUNT} CPU calls")
if vk_ops._VK_TIME > 0:
    print(f"  VK GEMM time: {vk_ops._VK_TIME:.0f}s  CPU: {vk_ops._CPU_TIME:.0f}s")

# ═══════════════════════════════════════════════════════════════
# Step 7: Cleanup Vulkan engine (free GPU memory for VAE)
# ═══════════════════════════════════════════════════════════════
print("Unloading Vulkan engine...")
vk_ops._lib.vk_engine_destroy()
del dit; gc.collect()

# ═══════════════════════════════════════════════════════════════
# Step 8: VAE decode
# ═══════════════════════════════════════════════════════════════
print("Loading VAE...")
vae_sd = torch.load("/sdcard/anima_on_android/models/vae_weights_fp16.pt",
                     weights_only=True)
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
out = "/sdcard/anima_on_android/output/phone_first.png"
Image.fromarray(img).save(out)
total_t = time.time() - t_start
print(f"Saved: {out}")
print(f"TOTAL: {STEPS} steps, {total_t:.0f}s ({total_t/STEPS:.0f}s/step), {H*8}x{H*8}")
del vae, x, img; gc.collect()

# ── Memory log summary ──
_mem_running = False
_mem_thread.join(timeout=1)
if _mem_log:
    rss_vals = [r for _, r in _mem_log if r > 0]
    if rss_vals:
        peak = max(rss_vals)
        avg = sum(rss_vals) // len(rss_vals)
        print(f"Memory trace ({len(rss_vals)} samples): peak={peak}MB avg={avg}MB")
        # Print top 5 peaks with timestamps
        peaks = sorted(_mem_log, key=lambda x: -x[1])[:5]
        for t, r in peaks:
            dt = t - _mem_log[0][0]
            print(f"  +{dt:.0f}s: {r}MB")

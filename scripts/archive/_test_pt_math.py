"""Test on phone: PT math backend SDPA (matmul+softmax+matmul using PT ops).
Same formula as our C code. If this produces ~74KB, our C SDPA has a bug.
If this also produces ~23KB, the math backend formula itself diverges from flash.
"""
import sys, time, gc, json, struct, mmap, ctypes, os
sys.path.insert(0, "/sdcard/anima_on_android/src")
import torch, numpy as np
from PIL import Image
import predict2, llm_adapter, wan_vae
sys.path.insert(0, "/sdcard/anima_on_android/scripts")
import vk_ops
import anima_rt_ops
import math

DEV = "cpu"; DTYPE = torch.float16; STEPS = 3; CFG = 5.0; SEED = 6666; H = 32

# ── PT math backend SDPA (same formula as our C code) ──
import torch.nn.functional as F_orig

def pt_math_sdpa(q, k, v):
    """PT math backend: scale→matmul→softmax→matmul. Same as our C SDPA."""
    scale = 1.0 / math.sqrt(q.size(-1))
    q_s = q.float() * scale
    k_s = k.float() * scale
    attn = q_s @ k_s.transpose(-2, -1)
    attn = torch.softmax(attn, dim=-1)
    out = (attn @ v.float()).to(v.dtype)
    return out

# Patch predict2's SDPA to use PT math backend instead of our C backend
import predict2 as _p2
_p2_orig = _p2._scaled_dot_product_attention
def _patched_pt_math(q, k, v, heads, skip_reshape=False, transformer_options=None):
    out = pt_math_sdpa(q, k, v)
    if skip_reshape:
        return out.transpose(1, 2).reshape(q.shape[0], -1, heads * q.shape[-1])
    return out.reshape(q.shape[0], q.shape[2], heads * q.shape[-1])
_p2._scaled_dot_product_attention = _patched_pt_math

# ── Same boilerplate as phone_pipeline ──
def mem_rss_mb():
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) // 1024
    except: pass
    return -1

print("Init Vulkan engine...")
if not vk_ops._lib.vk_engine_init(): print("FATAL"); sys.exit(1)

SAFETENSORS = "/sdcard/anima_on_android/models/diffusion.safetensors"
print(f"Loading: {SAFETENSORS}")

class SafetensorsReader:
    def __init__(self, path):
        with open(path, 'rb') as f:
            header_len = struct.unpack('<Q', f.read(8))[0]
            header = json.loads(f.read(header_len).decode('utf-8'))
        self.header = header; self.data_start = 8 + header_len
        self._mmap = None; self._file = None
    def keys(self): return list(self.header.keys())
    def dtype_code(self, key):
        d = self.header[key]['dtype']
        return {'BF16':2,'F16':1,'F32':0}.get(d,2)
    def numpy_dtype(self, key):
        d = self.header[key]['dtype']
        return np.uint16 if d in ('BF16','F16') else np.float32
    def get_tensor(self, key):
        if self._mmap is None:
            self._file = open(self._path, 'rb')
            self._mmap = mmap.mmap(self._file.fileno(), 0, access=mmap.ACCESS_READ)
        info = self.header[key]
        off = self.data_start + info['data_offsets'][0]
        end = self.data_start + info['data_offsets'][1]
        return np.frombuffer(self._mmap[off:end], dtype=self.numpy_dtype(key)).reshape(info['shape'])
    def close(self):
        if self._mmap: self._mmap.close(); self._file.close()

st = SafetensorsReader(SAFETENSORS)
all_keys = st.keys()
tensor_keys = [k for k in all_keys if k != "__metadata__"]
PREFIX = ""
if tensor_keys and '.' in tensor_keys[0]:
    first_part = tensor_keys[0].split('.')[0]
    if first_part not in ('blocks','x_embedder','t_embedder','final_layer',
                          't_embedding_norm','pos_embedder','llm_adapter'):
        PREFIX = first_part + '.'
def strip_prefix(k): return k[len(PREFIX):] if PREFIX and k.startswith(PREFIX) else k

shell_sd = {}
n_vk = n_shell = 0
for key in all_keys:
    if key == "__metadata__": continue
    clean_key = strip_prefix(key)
    if vk_ops.is_block_gemm_key(clean_key):
        data = st.get_tensor(key)
        shape = list(data.shape)
        shape_arr = (ctypes.c_int * len(shape))(*shape)
        ret = vk_ops._lib.vk_weight_add(clean_key.encode(), data.ctypes.data, st.dtype_code(key), shape_arr, len(shape))
        del data; n_vk += 1
    else:
        data = st.get_tensor(key)
        raw_dtype = st.header[key]['dtype']
        if raw_dtype == 'BF16':
            tensor = torch.from_numpy(data.copy()).view(torch.bfloat16).to(torch.float16)
        elif raw_dtype == 'F16':
            tensor = torch.from_numpy(data.copy()).view(torch.float16)
        elif raw_dtype == 'F32':
            tensor = torch.from_numpy(data.copy()).to(torch.float16)
        else:
            tensor = torch.from_numpy(data.copy()).to(torch.float16)
        shell_sd[clean_key] = tensor
        del data; n_shell += 1
st.close(); gc.collect()
print(f"  {n_vk} → Vulkan, {n_shell} → PyTorch shell")

vk_ops._lib.vk_engine_finalize()

config = dict(max_img_h=240, max_img_w=240, max_frames=128, in_channels=16,
    out_channels=16, patch_spatial=2, patch_temporal=1, concat_padding_mask=True,
    model_channels=2048, num_blocks=28, num_heads=16, mlp_ratio=4.0,
    crossattn_emb_channels=1024, pos_emb_cls="rope3d", pos_emb_learnable=True,
    pos_emb_interpolation="crop", min_fps=1, max_fps=30,
    use_adaln_lora=True, adaln_lora_dim=256,
    rope_h_extrapolation_ratio=4.0, rope_w_extrapolation_ratio=4.0,
    rope_t_extrapolation_ratio=1.0, extra_per_block_abs_pos_emb=False,
    rope_enable_fps_modulation=False)

dit = predict2.MiniTrainDIT(**config, device=DEV, dtype=DTYPE, operations=vk_ops.DummyOps)
vk_ops.patch_shell_linear(dit)
dit.load_state_dict(shell_sd, strict=False); dit.eval(); del shell_sd; gc.collect()
vk_ops.patch_block_layers(dit)
# KEEP SiLU/norm patches from anima_rt
import torch.nn as nn
for name, child in list(dit.named_children()):
    if isinstance(child, nn.LayerNorm):
        new = anima_rt_ops.AnimaRTLayerNorm(child.normalized_shape, eps=child.eps,
            elementwise_affine=child.elementwise_affine,
            dtype=child.weight.dtype if child.weight is not None else DTYPE,
            device=child.weight.device if child.weight is not None else DEV)
        if child.weight is not None: new.weight.data.copy_(child.weight.data)
        if child.bias is not None: new.bias.data.copy_(child.bias.data)
        setattr(dit, name, new)
    elif isinstance(child, nn.RMSNorm):
        new = anima_rt_ops.AnimaRTRMSNorm(child.normalized_shape, eps=child.eps,
            dtype=child.weight.dtype, device=child.weight.device)
        new.weight.data.copy_(child.weight.data)
        setattr(dit, name, new)
    elif isinstance(child, nn.GELU):
        setattr(dit, name, anima_rt_ops.AnimaRTGELU())
    elif isinstance(child, nn.SiLU):
        setattr(dit, name, anima_rt_ops.AnimaRTSiLU())
print("Patched: GEMM=Vulkan, norms/SiLU=anima_rt, SDPA=PT math backend")

ctx_cond = torch.load("/sdcard/anima_on_android/models/context_cond.pt", weights_only=True).to(DEV).to(DTYPE)
ctx_uncond = torch.load("/sdcard/anima_on_android/models/context_uncond.pt", weights_only=True).to(DEV).to(DTYPE)

linear = torch.linspace(1.0, 0.0, STEPS + 1)[:-1]
sigmas = (3.0 * linear / (1.0 + 2.0 * linear)).tolist() + [0.0]

print(f"Denoising {STEPS} steps...")
gen = torch.Generator(device=DEV).manual_seed(SEED)
x = torch.randn(1, 16, H, H, generator=gen, dtype=DTYPE)
t_start = time.time()

for i in range(STEPS):
    sigma = sigmas[i]; sigma_next = sigmas[i+1]
    ts = torch.tensor([sigma], dtype=DTYPE)
    x_b = x.unsqueeze(2).repeat(2, 1, 1, 1, 1)
    ctx_b = torch.cat([ctx_cond, ctx_uncond], dim=0)
    ts_b = ts.repeat(2)
    vk_ops._lib.vk_reset_pool()
    t0 = time.time()
    with torch.no_grad():
        v_b = dit(x_b, ts_b, ctx_b)
    dt = time.time() - t0
    v_cond = v_b[0:1].float(); v_uncond = v_b[1:2].float()
    v_cfg = v_uncond + CFG * (v_cond - v_uncond)
    x_new = (x.float() + v_cfg.squeeze(2) * (sigma_next - sigma)).to(DTYPE)
    print(f"  step {i+1}: dit={dt:.0f}s v=[{v_cond.min():.2f},{v_cond.max():.2f}] nan={torch.isnan(v_cond).any()}")
    x = x_new

print(f"Total denoising: {time.time()-t_start:.0f}s")
vk_ops._lib.vk_engine_destroy(); del dit; gc.collect()

# VAE decode
print("Loading VAE...")
vae_sd = torch.load("/sdcard/anima_on_android/models/vae_weights_fp16.pt", weights_only=True)
vae = wan_vae.WanVAE(dim=96, z_dim=16, dim_mult=[1,2,4,4], num_res_blocks=2,
    attn_scales=[], temperal_downsample=[False,True,True],
    image_channels=3, conv_out_channels=3, dropout=0.0)
vae.load_state_dict({k: v.float() for k, v in vae_sd.items()}, strict=False); vae.eval(); del vae_sd
with torch.no_grad():
    image = vae.decode(x.float().unsqueeze(2))
img = image[0,:,0].clamp(-1,1)
img = ((img+1)/2*255).permute(1,2,0).cpu().numpy().astype(np.uint8)
out = "/sdcard/anima_on_android/output/pt_math_sdpa.png"
Image.fromarray(img).save(out)
fsize = os.stat(out).st_size
print(f"Saved: {out} ({fsize} bytes)")
print(f"TOTAL: {time.time()-t_start:.0f}s")

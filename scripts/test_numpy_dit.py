"""Quick test: compare block 0 layer_norm + modulate vs PT."""
import sys, gc, math, types, struct, json, mmap, ctypes
sys.path.insert(0, "/sdcard/anima_on_android/src")
sys.path.insert(0, "/sdcard/anima_on_android/scripts")
import torch, numpy as np
import predict2, vk_ops, anima_rt_ops

# Init engines (minimal)
vk_ops._lib.vk_engine_init()
SAFETENSORS = "/sdcard/anima_on_android/models/diffusion.safetensors"
with open(SAFETENSORS, 'rb') as f:
    hl = struct.unpack('<Q', f.read(8))[0]
    hd = json.loads(f.read(hl).decode('utf-8'))
    ds = 8 + hl

def get_ten(key):
    info = hd[key]; off = ds + info['data_offsets'][0]; end = ds + info['data_offsets'][1]
    with open(SAFETENSORS, 'rb') as f:
        f.seek(off); buf = f.read(end-off)
    return np.frombuffer(buf, dtype=np.uint16 if info['dtype'] in ('BF16','F16') else np.float32).reshape(info['shape'])

PREFIX = "net."
def strip(k): return k[len(PREFIX):] if k.startswith(PREFIX) else k

norm_weights = {}; shell_sd = {}
for key in hd:
    if key == "__metadata__": continue
    ck = strip(key); sh = hd[key]['shape']
    if vk_ops.is_linear_weight(ck, shape=sh):
        data = get_ten(key); sl = list(sh)
        vk_ops._lib.vk_weight_add(ck.encode(), data.ctypes.data, {'BF16':2,'F16':1,'F32':0}.get(hd[key]['dtype'],2),
                                  (ctypes.c_int*len(sl))(*sl), len(sl))
    else:
        data = get_ten(key); raw = hd[key]['dtype']
        if raw == 'BF16':
            tensor = torch.from_numpy(data.copy()).view(torch.bfloat16).to(torch.float32)
        else:
            tensor = torch.from_numpy(data.copy()).to(torch.float32)
        shell_sd[ck] = tensor
        if raw == 'BF16':
            bits = data.copy().astype(np.uint32)<<16; norm_weights[ck] = bits.view(np.float32)
        else:
            norm_weights[ck] = data.copy().astype(np.float32)
vk_ops._lib.vk_engine_finalize()

# PT model
config = dict(max_img_h=240, max_img_w=240, max_frames=128, in_channels=16,
    out_channels=16, patch_spatial=2, patch_temporal=1, concat_padding_mask=True,
    model_channels=2048, num_blocks=28, num_heads=16, mlp_ratio=4.0,
    crossattn_emb_channels=1024, pos_emb_cls="rope3d", pos_emb_learnable=True,
    pos_emb_interpolation="crop", min_fps=1, max_fps=30,
    use_adaln_lora=True, adaln_lora_dim=256,
    rope_h_extrapolation_ratio=4.0, rope_w_extrapolation_ratio=4.0,
    rope_t_extrapolation_ratio=1.0, extra_per_block_abs_pos_emb=False,
    rope_enable_fps_modulation=False)
dit = predict2.MiniTrainDIT(**config, device="cpu", dtype=torch.float32, operations=vk_ops.DummyOps)
vk_ops.patch_shell_linear(dit)
dit.load_state_dict(shell_sd, strict=False); dit.eval()

# Patch SiLU → anima_rt (same as phone pipeline)
import torch.nn as nn
def _patch_seq(seq):
    for i, child in enumerate(seq):
        if isinstance(child, nn.SiLU): seq[i] = anima_rt_ops.AnimaRTSiLU()
        elif isinstance(child, nn.Sequential): _patch_seq(child)
def _patch_mod(model):
    for name, child in list(model.named_children()):
        if isinstance(child, nn.SiLU): setattr(model, name, anima_rt_ops.AnimaRTSiLU())
        elif isinstance(child, nn.Sequential): _patch_seq(child)
        else: _patch_mod(child)
_patch_mod(dit)

# ════════ Compare just the AdaLN modulation for block 0 self-attn ════════
# PT: extract the exact input/output of block 0's adaln_modulation_self_attn
np.random.seed(42)
emb_np = np.random.randn(1, 1, 2048).astype(np.float32)
lora_np = np.random.randn(1, 1, 6144).astype(np.float32)
emb_pt = torch.from_numpy(emb_np)  # stays FP32
lora_pt = torch.from_numpy(lora_np)  # stays FP32

# PT adaln_modulation_self_attn = Sequential(SiLU, Linear(2048→256), Linear(256→6144))
mod_pt = dit.blocks[0].adaln_modulation_self_attn(emb_pt)
mod_pt = (mod_pt + lora_pt).chunk(3, dim=-1)

# ND adaln: SiLU → Linear(2048→256) → SiLU → Linear(256→6144) → +lora → split
# NumpyRT wrapper
class NumpyRT:
    def __init__(self, nw): self._l=anima_rt_ops._lib; self._nw=nw
    def _kw(self, k):
        w=self._nw.get(k); return np.ascontiguousarray(w, dtype=np.float32) if w is not None else None
    def run_silu(self, x):
        out=np.zeros(len(x),dtype=np.float32)
        self._l.anima_rt_run_silu(x.ctypes.data,out.ctypes.data,len(x)); return out
    def run_layernorm(self, x_f32, D):
        M=x_f32.shape[0]; out=np.zeros((M,D),dtype=np.float32)
        self._l.anima_rt_run_layernorm(x_f32.ctypes.data,out.ctypes.data,M,D,ctypes.c_float(1e-6))
        return out
    def run_rmsnorm(self, x, key):
        *batch,D=x.shape; M=int(np.prod(batch)) if batch else 1
        x_f=np.ascontiguousarray(x.reshape(M,D).astype(np.float32),dtype=np.float32)
        w=self._kw(key+'.weight')
        out=np.zeros((M,D),dtype=np.float32)
        self._l.anima_rt_run_rmsnorm(x_f.ctypes.data,w.ctypes.data,out.ctypes.data,M,D,ctypes.c_float(1e-6))
        return out.reshape(*batch,D)

rt = NumpyRT(norm_weights)
class VK:
    def vk_run_gemm(self, name, x, out, M, N, K):
        if not vk_ops._lib.vk_run_gemm(name.encode(), x.ctypes.data, out.ctypes.data, M, N, K):
            raise RuntimeError(f"vk_run_gemm({name}) failed")
vk = VK()

# Reset descriptor pool before ND computation
vk_ops._lib.vk_reset_pool()

# ND adaln: exact same computation as NumpyBlock._adaln
B, T, Di = emb_np.shape; M = B*T
# SiLU
mod_nd = rt.run_silu(emb_np.reshape(M,Di).astype(np.float32).reshape(-1)).reshape(M, Di)
# Linear(2048→256)
m1 = np.zeros((M, 256), dtype=np.float32)
vk.vk_run_gemm('blocks.0.adaln_modulation_self_attn.1.weight', mod_nd, m1, M, 256, Di)
# PT Sequential(SiLU, Linear, Linear) — only ONE SiLU at start, no SiLU here
# Linear(256→3*2048)
m2 = np.zeros((M, 3*Di), dtype=np.float32)
vk.vk_run_gemm('blocks.0.adaln_modulation_self_attn.2.weight', m1, m2, M, 3*Di, 256)
m2 = m2 + lora_np.reshape(M, -1).astype(np.float32)
m2 = m2.reshape(B, T, 3*Di)
shift_nd, scale_nd, gate_nd = np.split(m2, 3, axis=-1)

shift_pt = mod_pt[0].float().cpu().numpy()
scale_pt = mod_pt[1].float().cpu().numpy()
gate_pt = mod_pt[2].float().cpu().numpy()

print(f"shift  PT [{shift_pt.min():.4f},{shift_pt.max():.4f}]  ND [{shift_nd.min():.4f},{shift_nd.max():.4f}]  err={np.abs(shift_nd-shift_pt).max():.6f}")
print(f"scale  PT [{scale_pt.min():.4f},{scale_pt.max():.4f}]  ND [{scale_nd.min():.4f},{scale_nd.max():.4f}]  err={np.abs(scale_nd-scale_pt).max():.6f}")
print(f"gate   PT [{gate_pt.min():.4f},{gate_pt.max():.4f}]   ND [{gate_nd.min():.4f},{gate_nd.max():.4f}]  err={np.abs(gate_nd-gate_pt).max():.6f}")

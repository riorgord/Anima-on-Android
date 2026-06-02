"""NumpyDiT pipeline — BF16 storage, FP32 compute, zero torch in DiT path."""
import sys, time, gc, math, struct, json, mmap, ctypes, os
import numpy as np
from PIL import Image
sys.path.insert(0, "/sdcard/anima_on_android/src")
sys.path.insert(0, "/sdcard/anima_on_android/scripts")
import torch  # ONLY for VAE decode at the end
import vk_ops, anima_rt_ops, numpy_dit

STEP=3; CFG=5.0; SEED=6666; H=32

# ════════ Load engines ════════
print("Init Vulkan...")
vk_ops._lib.vk_engine_init()
SAFETENSORS = "/sdcard/anima_on_android/models/diffusion.safetensors"
with open(SAFETENSORS, 'rb') as f:
    hl = struct.unpack('<Q', f.read(8))[0]
    hd = json.loads(f.read(hl).decode('utf-8'))
    ds = 8 + hl
PREFIX = "net."
def sk(k): return k[len(PREFIX):] if k.startswith(PREFIX) else k

norm_weights = {}
for key in hd:
    if key == "__metadata__": continue
    ck = sk(key); sh = hd[key]['shape']
    if vk_ops.is_linear_weight(ck, shape=sh):
        info = hd[key]; off = ds + info['data_offsets'][0]; end = ds + info['data_offsets'][1]
        with open(SAFETENSORS, 'rb') as f: f.seek(off); buf = f.read(end-off)
        data = np.frombuffer(buf, dtype=np.uint16 if info['dtype'] in ('BF16','F16') else np.float32).reshape(sh)
        sl = list(sh); vk_ops._lib.vk_weight_add(ck.encode(), data.ctypes.data,
            {'BF16':2,'F16':1,'F32':0}.get(info['dtype'],2), (ctypes.c_int*len(sl))(*sl), len(sl))
    else:
        info = hd[key]; off = ds + info['data_offsets'][0]; end = ds + info['data_offsets'][1]
        with open(SAFETENSORS, 'rb') as f: f.seek(off); buf = f.read(end-off)
        data = np.frombuffer(buf, dtype=np.uint16 if info['dtype'] in ('BF16','F16') else np.float32).reshape(sh)
        raw = info['dtype']
        if raw == 'BF16':
            bits = data.copy().astype(np.uint32) << 16
            norm_weights[ck] = bits.view(np.float32)
        else:
            norm_weights[ck] = data.copy().astype(np.float32)
vk_ops._lib.vk_engine_finalize()

# ════════ NumpyDiT ════════
class NumpyRT:
    def __init__(self,nw):self._l=anima_rt_ops._lib;self._nw=nw
    def _kw(self,k):
        w=self._nw.get(k);return np.ascontiguousarray(w,dtype=np.float32) if w is not None else None
    def run_layernorm(self,x,D):
        M=x.shape[0];o=np.zeros((M,D),dtype=np.float32)
        self._l.anima_rt_run_layernorm(x.ctypes.data,o.ctypes.data,M,D,ctypes.c_float(1e-6));return o
    def run_rmsnorm(self,x,key):
        *b,D=x.shape;M=int(np.prod(b)) if b else 1
        xf=np.ascontiguousarray(x.reshape(M,D).astype(np.float32),dtype=np.float32)
        w=self._kw(key+'.weight');o=np.zeros((M,D),dtype=np.float32)
        self._l.anima_rt_run_rmsnorm(xf.ctypes.data,w.ctypes.data,o.ctypes.data,M,D,ctypes.c_float(1e-6))
        return o.reshape(*b,D)
    def run_gelu(self,x):o=np.zeros(len(x),dtype=np.float32);self._l.anima_rt_run_gelu(x.ctypes.data,o.ctypes.data,len(x));return o
    def run_silu(self,x):o=np.zeros(len(x),dtype=np.float32);self._l.anima_rt_run_silu(x.ctypes.data,o.ctypes.data,len(x));return o
    def run_sdpa(self,q,k,v):
        B,H,Sq,D=q.shape;Skv=k.shape[2]
        qf=np.ascontiguousarray(q.reshape(B*H,Sq,D),dtype=np.float32)
        kf=np.ascontiguousarray(k.reshape(B*H,Skv,D),dtype=np.float32)
        vf=np.ascontiguousarray(v.reshape(B*H,Skv,D),dtype=np.float32)
        o=np.zeros((B*H,Sq,D),dtype=np.float32);s=1.0/np.sqrt(D)
        self._l.anima_rt_run_sdpa_flash(qf.ctypes.data,kf.ctypes.data,vf.ctypes.data,o.ctypes.data,B*H,Sq,Skv,D,ctypes.c_float(s),ctypes.c_bool(False))
        return o.reshape(B,H,Sq,D)
class NumpyVK:
    def vk_run_gemm(self,n,x,out,M,N,K):
        if not vk_ops._lib.vk_run_gemm(n.encode(),x.ctypes.data,out.ctypes.data,M,N,K):
            raise RuntimeError(f"vk_run_gemm({n}) failed")

rt = NumpyRT(norm_weights); vk = NumpyVK()
nd = numpy_dit.NumpyDiT(vk, rt)
print("NumpyDiT ready")

# ════════ Denoising ════════
ctx_cond = np.load("/sdcard/anima_on_android/models/context_cond.npy") if os.path.exists("/sdcard/anima_on_android/models/context_cond.npy") else None
ctx_uncond = np.load("/sdcard/anima_on_android/models/context_uncond.npy") if os.path.exists("/sdcard/anima_on_android/models/context_uncond.npy") else None

if ctx_cond is None:
    print("Loading context from .pt (convert to .npy for next time)")
    ctx_cond = torch.load("/sdcard/anima_on_android/models/context_cond.pt", weights_only=True).float().cpu().numpy()
    ctx_uncond = torch.load("/sdcard/anima_on_android/models/context_uncond.pt", weights_only=True).float().cpu().numpy()

# Scheduler (numpy, fork from flow_match)
lin = np.linspace(1.0, 0.0, STEP+1)[:-1]
sigmas_orig = (3.0*lin/(1.0+2.0*lin))
sigmas = np.append(sigmas_orig, 0.0).astype(np.float32)

# Latent init (no T dim, same as PT pipeline)
rng = np.random.default_rng(SEED)
x = rng.standard_normal((1, 16, H, H), dtype=np.float32).astype(np.float32)
t_start = time.time()

for i in range(STEP):
    sigma = float(sigmas[i]); sigma_next = float(sigmas[i+1])

    # Add T dim for DiT forward: [1, 16, 32, 32] → [1, 16, 1, 32, 32]
    # CFG batch: → [2, 16, 1, 32, 32]
    x_b = np.tile(np.expand_dims(x, 2), (2, 1, 1, 1, 1))
    ctx_b = np.concatenate([ctx_uncond, ctx_cond], axis=0)
    sigma_b = np.array([sigma, sigma], dtype=np.float32)

    vk_ops._lib.vk_reset_pool()
    t0 = time.time()
    v_b = nd.forward(x_b, sigma_b, ctx_b)
    dt = time.time()-t0

    v_cond = v_b[1:2]; v_uncond = v_b[0:1]
    v_cfg = v_uncond + CFG*(v_cond - v_uncond)
    x = x + v_cfg[:, :, 0, :, :]*(sigma_next - sigma)

    print(f"  step {i+1}/{STEP}: dit={dt:.0f}s"
          f" v=[{v_cond.min():.2f},{v_cond.max():.2f}]"
          f" x=[{x.min():.4f},{x.max():.4f}]"
          f" (total {time.time()-t_start:.0f}s)")

vk_ops._lib.vk_engine_destroy()
del nd; gc.collect()

# ════════ VAE decode (torch — only import here) ════════
import wan_vae
print("Loading VAE...")
vae_sd = torch.load("/sdcard/anima_on_android/models/vae_weights_fp16.pt", weights_only=True)
vae = wan_vae.WanVAE(dim=96, z_dim=16, dim_mult=[1,2,4,4], num_res_blocks=2,
    attn_scales=[], temperal_downsample=[False,True,True],
    image_channels=3, conv_out_channels=3, dropout=0.0)
vae.load_state_dict({k: v.float() for k, v in vae_sd.items()}, strict=False)
vae.eval(); del vae_sd
print("Decoding...")
x_vae = np.expand_dims(x, 2)  # [1,16,32,32] → [1,16,1,32,32]
with torch.no_grad():
    image = vae.decode(torch.from_numpy(x_vae.astype(np.float32)))
img = image[0,:,0].clamp(-1,1)
img = ((img+1)/2*255).permute(1,2,0).cpu().numpy().astype(np.uint8)
out = "/sdcard/anima_on_android/output/nd_phone.png"
Image.fromarray(img).save(out)
total_t = time.time()-t_start
fsize = os.stat(out).st_size if os.path.exists(out) else -1
print(f"Saved: {out} ({fsize} bytes)")
print(f"TOTAL: {STEP} steps, {total_t:.0f}s ({total_t/STEP:.0f}s/step), {H*8}x{H*8}")

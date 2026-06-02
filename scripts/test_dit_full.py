"""Full DiT comparison: NumpyDiT vs PT patched model."""
import sys, time, gc, math, types, struct, json, mmap, ctypes
sys.path.insert(0, "/sdcard/anima_on_android/src")
sys.path.insert(0, "/sdcard/anima_on_android/scripts")
import torch, numpy as np
import predict2, vk_ops, anima_rt_ops

# ════════ Load engines ════════
vk_ops._lib.vk_engine_init()
SAFETENSORS = "/sdcard/anima_on_android/models/diffusion.safetensors"
with open(SAFETENSORS, 'rb') as f:
    hl = struct.unpack('<Q', f.read(8))[0]
    hd = json.loads(f.read(hl).decode('utf-8'))
    ds = 8 + hl
PREFIX = "net."
def strip(k): return k[len(PREFIX):] if k.startswith(PREFIX) else k

def get_ten(key):
    info = hd[key]; off = ds + info['data_offsets'][0]; end = ds + info['data_offsets'][1]
    with open(SAFETENSORS, 'rb') as f: f.seek(off); buf = f.read(end-off)
    return np.frombuffer(buf, dtype=np.uint16 if info['dtype'] in ('BF16','F16') else np.float32).reshape(info['shape'])

norm_weights = {}; shell_sd = {}
for key in hd:
    if key == "__metadata__": continue
    ck = strip(key); sh = hd[key]['shape']
    if vk_ops.is_linear_weight(ck, shape=sh):
        data = get_ten(key); sl = list(sh)
        vk_ops._lib.vk_weight_add(ck.encode(), data.ctypes.data,
            {'BF16':2,'F16':1,'F32':0}.get(hd[key]['dtype'],2), (ctypes.c_int*len(sl))(*sl), len(sl))
    else:
        data = get_ten(key); raw = hd[key]['dtype']
        if raw == 'BF16':
            tensor = torch.from_numpy(data.copy()).view(torch.bfloat16).to(torch.float32)
        else:
            tensor = torch.from_numpy(data.copy()).to(torch.float32)
        shell_sd[ck] = tensor
        if raw == 'BF16':
            bits = data.copy().astype(np.uint32) << 16
            norm_weights[ck] = bits.view(np.float32)
        else:
            norm_weights[ck] = data.copy().astype(np.float32)
vk_ops._lib.vk_engine_finalize()

# ════════ PT model (same config as numpy_dit) ════════
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
dit.load_state_dict(shell_sd, strict=False); dit.eval(); del shell_sd; gc.collect()

# Apply all patches
import torch.nn as nn
def _pseq(seq):
    for i, child in enumerate(seq):
        if isinstance(child, nn.SiLU): seq[i] = anima_rt_ops.AnimaRTSiLU()
        elif isinstance(child, nn.Sequential): _pseq(child)
def _pmod(mod):
    for name, child in list(mod.named_children()):
        if isinstance(child, nn.LayerNorm):
            nw = anima_rt_ops.AnimaRTLayerNorm(child.normalized_shape, eps=child.eps,
                elementwise_affine=child.elementwise_affine, dtype=torch.float32)
            if child.weight is not None: nw.weight.data.copy_(child.weight.data)
            if child.bias is not None: nw.bias.data.copy_(child.bias.data)
            setattr(mod, name, nw)
        elif isinstance(child, nn.RMSNorm):
            nw = anima_rt_ops.AnimaRTRMSNorm(child.normalized_shape, eps=child.eps, dtype=torch.float32)
            nw.weight.data.copy_(child.weight.data); setattr(mod, name, nw)
        elif isinstance(child, nn.GELU): setattr(mod, name, anima_rt_ops.AnimaRTGELU())
        elif isinstance(child, nn.SiLU): setattr(mod, name, anima_rt_ops.AnimaRTSiLU())
        elif isinstance(child, nn.Sequential): _pseq(child)
        else: _pmod(child)
_pmod(dit)

import predict2 as _p2
def _psdpa(q,k,v,heads,skip_reshape=False,**kw):
    out = anima_rt_ops.anima_rt_sdpa(q,k,v)
    if skip_reshape: return out.transpose(1,2).reshape(q.shape[0],-1,heads*q.shape[-1])
    return out.reshape(q.shape[0],q.shape[2],heads*q.shape[-1])
_p2._scaled_dot_product_attention = _psdpa
def _prope(t, freqs):
    tn=t.float().cpu().numpy();fn=freqs.float().cpu().numpy()
    hd=tn.shape[-1]//2;ts=tn.shape
    t_=np.expand_dims(np.moveaxis(tn.reshape(*ts[:-1],2,hd),-2,-1),-2)
    return torch.from_numpy(np.moveaxis(fn[...,0]*t_[...,0]+fn[...,1]*t_[...,1],-1,-2).reshape(*ts)).to(device=t.device,dtype=t.dtype)
_p2.apply_rotary_pos_emb = _prope
def _pts(self,ts):
    tsn=ts.float().cpu().numpy();B=tsn.shape[0];T=1 if tsn.ndim<2 else tsn.shape[1]
    ts2=tsn.reshape(-1);half=self.num_channels//2
    exp=-math.log(10000)*np.arange(half,dtype=np.float32)/float(half)
    emb=np.exp(exp);emb=ts2[:,None]*emb[None,:]
    emb=np.concatenate([np.cos(emb),np.sin(emb)],axis=-1).reshape(B,T,-1)
    return torch.from_numpy(emb).to(device=ts.device,dtype=ts.dtype)
dit.t_embedder[0].forward = types.MethodType(_pts, dit.t_embedder[0])

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

import numpy_dit
rt = NumpyRT(norm_weights); vk = NumpyVK()
nd = numpy_dit.NumpyDiT(vk, rt)

# ════════ Compare ════════
np.random.seed(42)
x_np = np.random.randn(1,16,1,32,32).astype(np.float32)
sigma = 1.0
ctx_np = np.random.randn(1,512,1024).astype(np.float32)

x_pt = torch.from_numpy(x_np).to(torch.float32)
sigma_pt = torch.tensor([sigma], dtype=torch.float32)
ctx_pt = torch.from_numpy(ctx_np).to(torch.float32)

# Hook PT blocks to capture outputs
_pt_blk = {}
def _hb(mod, inp, out, idx):
    _pt_blk[idx] = out.float().cpu().numpy()
for i, blk in enumerate(dit.blocks):
    blk.register_forward_hook(lambda m,i,o,idx=i: _hb(m,i,o,idx))

vk_ops._lib.vk_reset_pool()
with torch.no_grad(): v_pt = dit(x_pt, sigma_pt, ctx_pt)
v_pt_np = v_pt.float().cpu().numpy()
print(f"PT v_cond: [{v_pt_np.min():.4f},{v_pt_np.max():.4f}]")
print(f"PT blk 0: [{_pt_blk[0].min():.2f},{_pt_blk[0].max():.2f}]  blk 27: [{_pt_blk[27].min():.2f},{_pt_blk[27].max():.2f}]")

vk_ops._lib.vk_reset_pool()
t0=time.time()
# Enable block trace in numpy_dit
numpy_dit.NumpyBlock.dbg.clear()
v_nd = nd.forward(x_np, sigma, ctx_np)
print(f"ND v_cond: [{v_nd.min():.4f},{v_nd.max():.4f}]  time={time.time()-t0:.0f}s")

# Block-by-block err
nd_dbg = numpy_dit.NumpyBlock.dbg
if 'b0_out' in nd_dbg:
    e0 = np.abs(nd_dbg['b0_out'][0] - _pt_blk[0][0]).max()
    e27 = np.abs(nd_dbg['b27_out'][0] - _pt_blk[27][0]).max() if 'b27_out' in nd_dbg else -1
    print(f"ND blk 0: [{nd_dbg['b0_out'][0].min():.2f},{nd_dbg['b0_out'][0].max():.2f}]  err_vs_PT={e0:.4f}")
    if 'b27_out' in nd_dbg:
        print(f"ND blk27: [{nd_dbg['b27_out'][0].min():.2f},{nd_dbg['b27_out'][0].max():.2f}]  err_vs_PT={e27:.4f}")

err = np.abs(v_nd.astype(np.float32) - v_pt_np)
print(f"\nmax_err: {err.max():.6f}  mean_err: {err.mean():.6f}")
print(f"nan: ND={np.isnan(v_nd).sum()}  PT={np.isnan(v_pt_np).sum()}")
print(f"RESULT: {'PASS (need <0.074)' if err.max() < 0.074 else 'FAIL'}")

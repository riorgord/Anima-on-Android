"""anima_rt ops — Python ctypes bindings for libanima_rt.so.
Drop-in replacement for hybridops vk_ops.py's HybridLayerNorm/HybridRMSNorm/HybridGELU.
GEMM stays in PyTorch (nn.Linear) or Vulkan (VulkanGemmLinear)."""
import ctypes as _ct
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_lib = _ct.CDLL("/data/local/tmp/libanima_rt.so")
_lib.anima_rt_init.restype = _ct.c_bool
_lib.anima_rt_run_gelu.argtypes = [_ct.c_void_p, _ct.c_void_p, _ct.c_int]
_lib.anima_rt_run_gelu.restype = _ct.c_bool
_lib.anima_rt_run_silu.argtypes = [_ct.c_void_p, _ct.c_void_p, _ct.c_int]
_lib.anima_rt_run_silu.restype = _ct.c_bool
_lib.anima_rt_run_layernorm.argtypes = [_ct.c_void_p, _ct.c_void_p, _ct.c_int, _ct.c_int, _ct.c_float]
_lib.anima_rt_run_layernorm.restype = _ct.c_bool
_lib.anima_rt_run_rmsnorm.argtypes = [_ct.c_void_p, _ct.c_void_p, _ct.c_void_p, _ct.c_int, _ct.c_int, _ct.c_float]
_lib.anima_rt_run_rmsnorm.restype = _ct.c_bool
_lib.anima_rt_run_softmax.argtypes = [_ct.c_void_p, _ct.c_void_p, _ct.c_int, _ct.c_int]
_lib.anima_rt_run_softmax.restype = _ct.c_bool

assert _lib.anima_rt_init(), "anima_rt_init failed"

# ── LayerNorm ──────────────────────────────────────────────────────
class AnimaRTLayerNorm(nn.LayerNorm):
    """nn.LayerNorm backed by libanima_rt.so CPU kernel (FP32 Welford).
    Works for elementwise_affine=False only (gamma=beta=nullptr)."""
    def forward(self, x):
        if self.elementwise_affine:
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        *batch, D = x.shape
        M = int(np.prod(batch)) if batch else 1
        x_f32 = x.reshape(M, D).cpu().contiguous().float().numpy()
        out_buf = np.zeros((M, D), dtype=np.float32)
        ok = _lib.anima_rt_run_layernorm(
            x_f32.ctypes.data, out_buf.ctypes.data, M, D, _ct.c_float(self.eps))
        if not ok:
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        result = torch.from_numpy(out_buf).to(device=x.device, dtype=x.dtype)
        return result.reshape(*batch, D) if batch else result.squeeze(0)


# ── RMSNorm ────────────────────────────────────────────────────────
class AnimaRTRMSNorm(nn.RMSNorm):
    """nn.RMSNorm backed by libanima_rt.so CPU kernel (FP32)."""
    def forward(self, x):
        *batch, D = x.shape
        M = int(np.prod(batch)) if batch else 1
        x_f32 = x.reshape(M, D).cpu().contiguous().float().numpy()
        w_f32 = self.weight.detach().cpu().float().numpy().copy()
        out_buf = np.zeros((M, D), dtype=np.float32)
        ok = _lib.anima_rt_run_rmsnorm(
            x_f32.ctypes.data, w_f32.ctypes.data,
            out_buf.ctypes.data, M, D, _ct.c_float(self.eps))
        if not ok:
            return F.rms_norm(x, self.normalized_shape, self.weight, self.eps)
        result = torch.from_numpy(out_buf).to(device=x.device, dtype=x.dtype)
        return result.reshape(*batch, D) if batch else result.squeeze(0)


# ── GELU ──────────────────────────────────────────────────────────
class AnimaRTGELU(nn.GELU):
    """nn.GELU backed by libanima_rt.so CPU kernel (FP32, exact erf)."""
    def forward(self, x):
        flat = x.reshape(-1).cpu().contiguous().float().numpy()
        out_buf = np.zeros(len(flat), dtype=np.float32)
        ok = _lib.anima_rt_run_gelu(flat.ctypes.data, out_buf.ctypes.data, len(flat))
        if not ok:
            return F.gelu(x)
        result = torch.from_numpy(out_buf).to(device=x.device, dtype=x.dtype)
        return result.reshape(x.shape)


# ── Operations class for predict2.py ──────────────────────────────
class AnimaRTOps:
    """Operations factory: GEMM stays nn.Linear, norms/activations → libanima_rt.so.
    Use this to replace hybridops DummyOps/ShellHybridOps."""
    Linear = nn.Linear
    RMSNorm = AnimaRTRMSNorm
    LayerNorm = AnimaRTLayerNorm
    Embedding = nn.Embedding
    GELU = AnimaRTGELU   # GPT2FeedForward uses nn.GELU directly, we intercept below

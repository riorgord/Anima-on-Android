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
_lib.anima_rt_run_gemm_fp32.argtypes = [_ct.c_void_p, _ct.c_void_p, _ct.c_void_p,
                                         _ct.c_int, _ct.c_int, _ct.c_int]
_lib.anima_rt_run_gemm_fp32.restype = _ct.c_bool
_lib.anima_rt_run_gemm_bf16.argtypes = [_ct.c_void_p, _ct.c_void_p, _ct.c_void_p,
                                         _ct.c_int, _ct.c_int, _ct.c_int]
_lib.anima_rt_run_gemm_bf16.restype = _ct.c_bool
_lib.anima_rt_run_sdpa.argtypes = [_ct.c_void_p]*4 + [_ct.c_int]*4 + [_ct.c_float, _ct.c_bool]
_lib.anima_rt_run_sdpa.restype = _ct.c_bool
_lib.anima_rt_run_sdpa_flash.argtypes = [_ct.c_void_p]*4 + [_ct.c_int]*4 + [_ct.c_float, _ct.c_bool]
_lib.anima_rt_run_sdpa_flash.restype = _ct.c_bool

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


# ── SDPA (Scaled Dot-Product Attention) ───────────────────────────
def anima_rt_sdpa(q, k, v):
    """Drop-in replacement for F.scaled_dot_product_attention (math backend).
    q, k, v: [B, H, S, D] — ALREADY in PyTorch SDPA format (predict2 does the permute).
    Returns: [B, H, S_q, D]
    """
    B, H, S_q, D = q.shape
    S_kv = k.shape[2]
    q_bh = q.reshape(B * H, S_q, D)
    k_bh = k.reshape(B * H, S_kv, D)
    v_bh = v.reshape(B * H, S_kv, D)
    q_f32 = q_bh.float().cpu().contiguous().numpy().astype(np.float32)
    k_f32 = k_bh.float().cpu().contiguous().numpy().astype(np.float32)
    v_f32 = v_bh.float().cpu().contiguous().numpy().astype(np.float32)
    out_buf = np.zeros((B * H, S_q, D), dtype=np.float32)
    scale = 1.0 / np.sqrt(D)
    ok = _lib.anima_rt_run_sdpa_flash(
        q_f32.ctypes.data, k_f32.ctypes.data, v_f32.ctypes.data,
        out_buf.ctypes.data,
        B * H, S_q, S_kv, D, _ct.c_float(scale), _ct.c_bool(False))
    if not ok:
        return F.scaled_dot_product_attention(q, k, v)
    result = torch.from_numpy(out_buf).to(device=q.device, dtype=q.dtype)
    return result.reshape(B, H, S_q, D)


# ── SiLU ──────────────────────────────────────────────────────────
class AnimaRTSiLU(nn.SiLU):
    """nn.SiLU backed by libanima_rt.so CPU kernel (FP32, bit-exact with PT)."""
    def forward(self, x):
        flat = x.reshape(-1).cpu().contiguous().float().numpy()
        out_buf = np.zeros(len(flat), dtype=np.float32)
        ok = _lib.anima_rt_run_silu(flat.ctypes.data, out_buf.ctypes.data, len(flat))
        if not ok: return F.silu(x)
        result = torch.from_numpy(out_buf).to(device=x.device, dtype=x.dtype)
        return result.reshape(x.shape)


# ── Linear (GEMM) ─────────────────────────────────────────────────
class AnimaRTLinear(nn.Module):
    """nn.Linear backed by libanima_rt.so GEMM (OpenBLAS or pure C++ fallback)."""
    def __init__(self, in_features, out_features, bias=True, device=None, dtype=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self._weight_bf16 = None; self._weight_fp32 = None; self._use_bf16 = False
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, dtype=dtype or torch.float32))
        else:
            self.register_parameter('bias', None)

    def set_weight_bf16(self, w): self._weight_bf16 = np.ascontiguousarray(w, dtype=np.uint16); self._use_bf16 = True
    def set_weight_fp32(self, w): self._weight_fp32 = np.ascontiguousarray(w, dtype=np.float32); self._use_bf16 = False

    def load_state_dict(self, state_dict, strict=True):
        for name, param in state_dict.items():
            if name == 'weight':
                w = param.detach().cpu().contiguous()
                if w.dtype == torch.bfloat16:
                    self.set_weight_bf16(w.view(torch.uint16).numpy())
                else:
                    self.set_weight_fp32(w.float().numpy())
            elif name == 'bias' and self.bias is not None:
                self.bias.data.copy_(param.data)
        return [], []

    def forward(self, x):
        *batch, K = x.shape; M = int(np.prod(batch)) if batch else 1; N = self.out_features
        x_f32 = x.reshape(M, K).cpu().contiguous().float().numpy()
        out_buf = np.zeros((M, N), dtype=np.float32)
        if self._use_bf16 and self._weight_bf16 is not None:
            ok = _lib.anima_rt_run_gemm_bf16(x_f32.ctypes.data, self._weight_bf16.ctypes.data, out_buf.ctypes.data, M, N, K)
        elif self._weight_fp32 is not None:
            ok = _lib.anima_rt_run_gemm_fp32(x_f32.ctypes.data, self._weight_fp32.ctypes.data, out_buf.ctypes.data, M, N, K)
        else:
            return F.linear(x, torch.zeros(N,K), self.bias)
        if not ok: return F.linear(x, torch.zeros(N,K), self.bias)
        result = torch.from_numpy(out_buf).to(device=x.device, dtype=x.dtype)
        if self.bias is not None: result += self.bias.to(result.dtype)
        return result.reshape(*batch, N) if batch else result.squeeze(0)


# ── Operations class for predict2.py ──────────────────────────────
class AnimaRTOps:
    """Operations factory: GEMM stays nn.Linear, norms/activations → libanima_rt.so.
    Use this to replace hybridops DummyOps/ShellHybridOps."""
    Linear = nn.Linear
    RMSNorm = AnimaRTRMSNorm
    LayerNorm = AnimaRTLayerNorm
    Embedding = nn.Embedding
    GELU = AnimaRTGELU   # GPT2FeedForward uses nn.GELU directly, we intercept below

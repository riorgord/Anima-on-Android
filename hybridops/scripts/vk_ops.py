"""Hybrid ops: Vulkan for large GEMM, CPU for small."""
import time, struct
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import ctypes as _ct

try:
    import sys
    sys.path.insert(0, "/sdcard/anima_on_android/scripts")
    import vk_linear as _vk
    _VK_AVAILABLE = True  # Vulkan enabled — dispatch swap bug fixed 2026-05-26
except ImportError:
    _VK_AVAILABLE = False

# ── libvk_hybrid.so (LayerNorm / SiLU / GELU shaders) ──
try:
    _lib_hy = _ct.CDLL("/data/local/tmp/libvk_hybrid.so")
    _lib_hy.vk_hybrid_init.argtypes = []
    _lib_hy.vk_hybrid_init.restype = _ct.c_bool
    _lib_hy.vk_hybrid_load.argtypes = [_ct.c_char_p, _ct.c_int, _ct.c_int]
    _lib_hy.vk_hybrid_load.restype = _ct.c_int
    _lib_hy.vk_hybrid_upload.argtypes = [_ct.c_int, _ct.c_int, _ct.c_void_p, _ct.c_size_t]
    _lib_hy.vk_hybrid_upload.restype = _ct.c_bool
    _lib_hy.vk_hybrid_download.argtypes = [_ct.c_int, _ct.c_int, _ct.c_void_p, _ct.c_size_t]
    _lib_hy.vk_hybrid_download.restype = _ct.c_bool
    _lib_hy.vk_hybrid_run.argtypes = [_ct.c_int, _ct.c_uint32, _ct.c_uint32, _ct.c_uint32, _ct.c_void_p]
    _lib_hy.vk_hybrid_run.restype = _ct.c_bool
    _VK_HYBRID_AVAILABLE = True
except Exception:
    _VK_HYBRID_AVAILABLE = False

# Threshold: Vulkan only when BOTH output dim >= 2048 AND batch (M) >= 64
# Avoids Adreno 7xx driver bug at small workgroup sizes
_VK_N_THRESHOLD = 2048


_VK_COUNT = 0
_CPU_COUNT = 0
_VK_TIME = 0.0
_CPU_TIME = 0.0
_VK_GPU_TIME = 0.0  # pure GPU compute time (from Vulkan timestamps)

# Register fp16-direct entry point (no f2h/h2f in .so)
if _VK_AVAILABLE:
    _vk._lib.vk_gemm_run_fp16.argtypes = [_ct.c_void_p, _ct.c_void_p, _ct.c_void_p,
                                          _ct.c_int, _ct.c_int, _ct.c_int]
    _vk._lib.vk_gemm_run_fp16.restype = _ct.c_bool
    _vk._lib.vk_gemm_get_timings_us.argtypes = [_ct.c_void_p, _ct.c_void_p, _ct.c_void_p, _ct.c_void_p, _ct.c_void_p]
    _vk._lib.vk_gemm_get_timings_us.restype = None

class HybridLinear(nn.Linear):
    """nn.Linear with Vulkan acceleration for large weight matrices."""

    def forward(self, x):
        global _VK_COUNT, _CPU_COUNT, _VK_TIME, _CPU_TIME, _VK_GPU_TIME
        import numpy as np, ctypes
        *batch, in_f = x.shape
        M = int(np.prod(batch)) if batch else 1
        out_f = self.weight.shape[0]
        if _VK_AVAILABLE and out_f >= _VK_N_THRESHOLD and M >= 16:
            if not _vk._INITIALIZED:
                _vk._lib.vk_gemm_init(1024, 8192, 8192, 16)
                _vk._INITIALIZED = True
            # fp16 direct — nn.Linear may store weight as fp32, force fp16
            x_u16 = x.reshape(M, in_f).cpu().contiguous().to(torch.float16).numpy().view(np.uint16).copy()
            w_u16 = self.weight.detach().cpu().to(torch.float16).numpy().view(np.uint16).copy()
            out = np.zeros((M, out_f), dtype=np.uint16)
            t0 = time.perf_counter()
            ok = _vk._lib.vk_gemm_run_fp16(
                out.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),
                x_u16.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),
                w_u16.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),
                M, out_f, in_f)
            _VK_TIME += time.perf_counter() - t0
            if not ok:
                return F.linear(x, self.weight, self.bias)
            # Diagnostic: compare with CPU for first 5 and every 50th call
            if _VK_COUNT < 5 or _VK_COUNT % 50 == 0:
                cpu_out = F.linear(x.float(), self.weight.float(),
                    self.bias.float() if self.bias is not None else None)
                vk_tensor = torch.tensor(out.view(np.float16), device=x.device, dtype=torch.float16)
                vk_tensor = vk_tensor.reshape(*batch, out_f) if batch else vk_tensor.squeeze(0)
                err = (cpu_out.float() - vk_tensor.float()).abs().max().item()
                print(f"  Vk#{_VK_COUNT} M={M} N={out_f} K={in_f} max_err={err:.6f}")
            result = torch.tensor(out.view(np.float16), device=x.device)
            result = result.reshape(*batch, out_f) if batch else result.squeeze(0)
            if self.bias is not None:
                result += self.bias.to(result.device, result.dtype)
            _VK_COUNT += 1
            return result
        _CPU_COUNT += 1
        t0 = time.perf_counter()
        out = F.linear(x, self.weight, self.bias)
        _CPU_TIME += time.perf_counter() - t0
        return out


# ── libdit_vk.so LayerNorm wrapper ──
try:
    _lib_dit = _ct.CDLL("/data/local/tmp/libdit_vk.so")
    _lib_dit.dit_run_layernorm.argtypes = [_ct.c_void_p, _ct.c_void_p, _ct.c_int, _ct.c_int, _ct.c_float]
    _lib_dit.dit_run_layernorm.restype = _ct.c_bool
    _lib_dit.dit_run_rmsnorm.argtypes = [_ct.c_void_p, _ct.c_void_p, _ct.c_int, _ct.c_void_p, _ct.c_int, _ct.c_int, _ct.c_float]
    _lib_dit.dit_run_rmsnorm.restype = _ct.c_bool
    _lib_dit.dit_run_gelu.argtypes = [_ct.c_void_p, _ct.c_void_p, _ct.c_int]
    _lib_dit.dit_run_gelu.restype = _ct.c_bool
    _lib_dit.dit_run_attention.argtypes = [_ct.c_void_p, _ct.c_void_p, _ct.c_void_p, _ct.c_void_p,
        _ct.c_int, _ct.c_int, _ct.c_int, _ct.c_int, _ct.c_float]
    _lib_dit.dit_run_attention.restype = _ct.c_bool
    _VK_LN_AVAILABLE = True
except Exception:
    _VK_LN_AVAILABLE = False

class HybridLayerNorm(nn.LayerNorm):
    """nn.LayerNorm with Vulkan acceleration via libdit_vk.so (FP32 I/O)."""
    _count = 0

    def forward(self, x):
        # Fallback: no Vulkan, or has affine params (not supported by shader)
        if not _VK_LN_AVAILABLE or self.elementwise_affine:
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)

        *batch, D = x.shape
        M = int(np.prod(batch)) if batch else 1

        x_f32 = x.reshape(M, D).cpu().contiguous().float().numpy()
        out_buf = np.zeros((M, D), dtype=np.float32)

        # Call libdit_vk.so LayerNorm (same Vulkan instance as AdaLN)
        ok = _lib_dit.dit_run_layernorm(
            x_f32.ctypes.data_as(_ct.c_void_p),
            out_buf.ctypes.data_as(_ct.c_void_p),
            M, D, _ct.c_float(self.eps))
        if not ok:
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)

        result = torch.from_numpy(out_buf).to(device=x.device, dtype=x.dtype)
        result = result.reshape(*batch, D) if batch else result.squeeze(0)

        cls = type(self)
        if cls._count < 5:
            ref = F.layer_norm(x.float(), self.normalized_shape, None, None, self.eps)
            err = (result.float() - ref.float()).abs().max().item()
            print(f"  VkLN#{cls._count} M={M} D={D} max_err={err:.6f}")
            cls._count += 1

        return result


class HybridRMSNorm(nn.RMSNorm):
    """nn.RMSNorm with Vulkan acceleration via libdit_vk.so (FP16 I/O)."""
    _count = 0

    def forward(self, x):
        if not _VK_LN_AVAILABLE:
            return F.rms_norm(x, self.normalized_shape, self.weight, self.eps)

        *batch, D = x.shape
        M = int(np.prod(batch)) if batch else 1

        x_f16 = x.reshape(M, D).cpu().contiguous().to(torch.float16).numpy().view(np.uint16)
        w_f16 = self.weight.detach().cpu().to(torch.float16).numpy().view(np.uint16)
        out_buf = np.zeros(M * D, dtype=np.uint16)

        ok = _lib_dit.dit_run_rmsnorm(
            x_f16.ctypes.data_as(_ct.c_void_p),
            w_f16.ctypes.data_as(_ct.c_void_p), int(w_f16.size),
            out_buf.ctypes.data_as(_ct.c_void_p),
            M, D, _ct.c_float(self.eps))
        if not ok:
            return F.rms_norm(x, self.normalized_shape, self.weight, self.eps)

        result = torch.tensor(out_buf.view(np.float16), device=x.device, dtype=x.dtype)
        result = result.reshape(*batch, D) if batch else result.squeeze(0)

        cls = type(self)
        if cls._count < 5:
            ref = F.rms_norm(x.float(), self.normalized_shape, self.weight.float(), self.eps)
            err = (result.float() - ref.float()).abs().max().item()
            print(f"  VkRMS#{cls._count} M={M} D={D} max_err={err:.6f}")
            cls._count += 1

        return result


class HybridGELU(nn.GELU):
    """nn.GELU with Vulkan acceleration via libdit_vk.so (FP16 I/O)."""
    _count = 0

    def forward(self, x):
        if not _VK_LN_AVAILABLE:
            return F.gelu(x)

        *batch, D = x.shape
        N = int(np.prod(batch)) * D if batch else D
        x_f16 = x.reshape(-1).cpu().contiguous().to(torch.float16).numpy().view(np.uint16)
        out_buf = np.zeros(N, dtype=np.uint16)

        ok = _lib_dit.dit_run_gelu(
            x_f16.ctypes.data_as(_ct.c_void_p),
            out_buf.ctypes.data_as(_ct.c_void_p),
            N)
        if not ok:
            return F.gelu(x)

        result = torch.tensor(out_buf.view(np.float16), device=x.device, dtype=x.dtype)
        result = result.reshape(*batch, D) if batch else result.view(D)

        cls = type(self)
        if cls._count < 3:
            ref = F.gelu(x.float())
            err = (result.float() - ref.float()).abs().max().item()
            print(f"  VkGELU#{cls._count} N={N} max_err={err:.6f}")
            cls._count += 1

        return result


class HybridOps:
    """Operations class with Vulkan acceleration for large Linear layers."""
    Linear = HybridLinear
    RMSNorm = HybridRMSNorm
    LayerNorm = HybridLayerNorm
    Embedding = nn.Embedding


def demo():
    """Quick test of HybridOps vs regular nn.Linear."""
    import time
    x = torch.randn(1024, 2048)
    w = torch.randn(2048, 2048)
    for _ in range(3):
        F.linear(x, w)
    t0 = time.time()
    for _ in range(5):
        F.linear(x, w)
    print(f"CPU: {(time.time()-t0)/5*1000:.0f}ms")
    if _VK_AVAILABLE:
        hl = HybridLinear(2048, 2048)
        hl.weight.data.copy_(w)
        for _ in range(3):
            hl(x)
        t0 = time.time()
        for _ in range(5):
            hl(x)
        print(f"VK:  {(time.time()-t0)/5*1000:.0f}ms")

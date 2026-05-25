"""Hybrid ops: Vulkan for large GEMM, CPU for small."""
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import sys
    sys.path.insert(0, "/sdcard/anima_on_android/scripts")
    import vk_linear as _vk
    _VK_AVAILABLE = False  # CPU-only: Vulkan deferred due to Adreno driver bug
except ImportError:
    _VK_AVAILABLE = False

# Threshold: Vulkan only when BOTH output dim >= 2048 AND batch (M) >= 64
# Avoids Adreno 7xx driver bug at small workgroup sizes
_VK_N_THRESHOLD = 2048


_VK_COUNT = 0
_CPU_COUNT = 0
_VK_COUNT = 0
_CPU_COUNT = 0

class HybridLinear(nn.Linear):
    """nn.Linear with Vulkan acceleration for large weight matrices."""

    def forward(self, x):
        global _VK_COUNT, _CPU_COUNT
        import numpy as np, ctypes
        *batch, in_f = x.shape
        M = int(np.prod(batch)) if batch else 1
        out_f = self.weight.shape[0]
        if _VK_AVAILABLE and out_f >= _VK_N_THRESHOLD and M >= 16:
            if not _vk._INITIALIZED:
                _vk._lib.vk_gemm_init(1024, 8192, 8192, 16)
                _vk._INITIALIZED = True
            x_f32 = x.reshape(M, in_f).float().cpu().numpy()
            w_f32 = self.weight.float().cpu().numpy()
            out = np.zeros((M, out_f), dtype=np.float32)
            ok = _vk._lib.vk_gemm_run(
                out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                x_f32.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                w_f32.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                M, out_f, in_f)
            if not ok:
                return F.linear(x, self.weight, self.bias)
            # Diagnostic: compare with CPU for first 5 and every 50th call
            if _VK_COUNT < 5 or _VK_COUNT % 50 == 0:
                cpu_out = F.linear(x.float(), self.weight.float(),
                    self.bias.float() if self.bias is not None else None)
                vk_tensor = torch.from_numpy(out).to(x.device, dtype=torch.float32)
                vk_tensor = vk_tensor.reshape(*batch, out_f) if batch else vk_tensor.squeeze(0)
                err = (cpu_out - vk_tensor).abs().max().item()
                print(f"  Vk#{_VK_COUNT} M={M} N={out_f} K={in_f} max_err={err:.6f}")
            result = torch.from_numpy(out).to(x.device).to(x.dtype)
            result = result.reshape(*batch, out_f) if batch else result.squeeze(0)
            if self.bias is not None:
                result += self.bias.to(result.device, result.dtype)
            _VK_COUNT += 1
            return result
        _CPU_COUNT += 1
        return F.linear(x, self.weight, self.bias)


class HybridOps:
    """Operations class with Vulkan acceleration for large Linear layers."""
    Linear = HybridLinear
    RMSNorm = nn.RMSNorm
    LayerNorm = nn.LayerNorm
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

"""Hybrid ops: Vulkan for large GEMM, CPU for small.

DIAGNOSTIC BUILD (2026-05-26): verify every .so call against CPU, raise on first divergence.

切换方法: 把 phone_pipeline.py 里的 `import vk_ops` 改成 `import vk_ops_diag as vk_ops`，
push 到手机后跑一步即可。本文件与 vk_ops.py 的差异:
- _VK_AVAILABLE 强制 True (即使常态版被设为 False 也强制走 Vulkan 路径)
- 每次 GEMM 都跑 CPU 对照, 异常 (nan/inf/max_err>0.5) 立即 raise
- 正常调用每 20 次打印一次心跳
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import sys
    sys.path.insert(0, "/sdcard/anima_on_android/scripts")
    import vk_linear as _vk
    _VK_AVAILABLE = True  # DIAGNOSTIC: forced on to exercise .so path
except ImportError:
    _VK_AVAILABLE = False

# Threshold: Vulkan only when BOTH output dim >= 2048 AND batch (M) >= 64
# Avoids Adreno 7xx driver bug at small workgroup sizes
_VK_N_THRESHOLD = 2048

_VK_COUNT = 0
_CPU_COUNT = 0

# Diagnostic thresholds
_ERR_THRESHOLD = 0.5      # fp16 GEMM normal error ~0.05; >0.5 = clear divergence
_HEARTBEAT_EVERY = 20     # print "OK" every N successful calls so we know it's moving


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
                print(f"  Vk#{_VK_COUNT} M={M} N={out_f} K={in_f} run returned False")
                raise RuntimeError("vk_gemm_run returned False")

            # === DIAGNOSTIC: verify EVERY call against CPU ===
            n_nan = int(np.isnan(out).sum())
            n_inf = int(np.isinf(out).sum())
            cpu_out = F.linear(x.float(), self.weight.float(), None).cpu().numpy()
            cpu_out = cpu_out.reshape(M, out_f)
            err = np.abs(cpu_out - out)
            max_err = float(err.max())
            mean_err = float(err.mean())

            bad = (n_nan > 0) or (n_inf > 0) or (max_err > _ERR_THRESHOLD)
            if bad:
                # Print samples to help diagnose
                print(f"!!! Vk#{_VK_COUNT} M={M} N={out_f} K={in_f} DIVERGED")
                print(f"    nan={n_nan} inf={n_inf} max_err={max_err:.4f} mean_err={mean_err:.4f}")
                print(f"    vk_out[0,:8] = {out[0,:8]}")
                print(f"    cpu_out[0,:8] = {cpu_out[0,:8]}")
                print(f"    x.abs().max() = {float(np.abs(x_f32).max()):.4f}")
                print(f"    w.abs().max() = {float(np.abs(w_f32).max()):.4f}")
                raise RuntimeError(f"Vulkan GEMM diverged at call #{_VK_COUNT}")

            if _VK_COUNT % _HEARTBEAT_EVERY == 0:
                print(f"  Vk#{_VK_COUNT} M={M} N={out_f} K={in_f} max_err={max_err:.4f} OK")

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

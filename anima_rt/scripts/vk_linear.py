"""VulkanLinear — float path with bit-exact FP16 conversion."""
import ctypes
import numpy as np
import torch

_lib = ctypes.CDLL("/data/local/tmp/libvk_gemm.so")
_lib.vk_gemm_init.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]
_lib.vk_gemm_init.restype = ctypes.c_bool
_lib.vk_gemm_run.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                              ctypes.c_int, ctypes.c_int, ctypes.c_int]
_lib.vk_gemm_run.restype = ctypes.c_bool
_lib.vk_gemm_destroy.argtypes = []
_lib.vk_gemm_destroy.restype = None

_MAX_M = _MAX_N = _MAX_K = 0
_INITIALIZED = False


def ensure_init(M, N, K):
    global _MAX_M, _MAX_N, _MAX_K, _INITIALIZED
    if M <= _MAX_M and N <= _MAX_N and K <= _MAX_K and _INITIALIZED:
        return
    if _INITIALIZED:
        _lib.vk_gemm_destroy()
    ok = _lib.vk_gemm_init(max(M, _MAX_M), max(N, _MAX_N), max(K, _MAX_K), 16)
    if not ok:
        raise RuntimeError("Vulkan init failed")
    _MAX_M = max(M, _MAX_M)
    _MAX_N = max(N, _MAX_N)
    _MAX_K = max(K, _MAX_K)
    _INITIALIZED = True


def vk_linear(x, weight, bias=None):
    *batch, in_f = x.shape
    out_f = weight.shape[0]
    total_M = int(np.prod(batch)) if batch else 1
    ensure_init(total_M, out_f, in_f)

    x_np = x.reshape(total_M, in_f).float().cpu().numpy()
    w_np = weight.float().cpu().numpy()
    out = np.zeros((total_M, out_f), dtype=np.float32)

    ok = _lib.vk_gemm_run(
        out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        x_np.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        w_np.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        total_M, out_f, in_f)
    if not ok:
        raise RuntimeError("Vulkan GEMM failed")

    result = torch.from_numpy(out).to(x.device).to(x.dtype)
    result = result.reshape(*batch, out_f) if batch else result.squeeze(0)
    if bias is not None:
        result += bias.to(result.device, result.dtype)
    return result


if __name__ == "__main__":
    import time
    x = torch.randn(1024, 2048, dtype=torch.float16)
    w = torch.randn(2048, 2048, dtype=torch.float16)
    ref = torch.nn.functional.linear(x.float(), w.float()).half()

    for _ in range(3):
        vk_linear(x, w)
    t0 = time.time()
    for _ in range(10):
        r = vk_linear(x, w)
    vk_t = (time.time() - t0) / 10
    diff = (r.float() - ref.float()).abs()
    print(f"Vulkan: {vk_t*1000:.0f}ms  max_err={float(diff.max()):.4f} mean_err={float(diff.mean()):.4f}")

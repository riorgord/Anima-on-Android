"""Per-op Vulkan acceleration via libvk_hybrid.so with fine-grained GPU timing.
Each op returns: (ok, result_tensor, {cpu_pack_us, gpu_us, cpu_unpack_us})
"""
import ctypes, struct, time, os
import numpy as np
import torch

# ============================================================
# libvk_hybrid.so bindings
# ============================================================
_lib = ctypes.CDLL("/data/local/tmp/libvk_hybrid.so")
_lib.vk_hybrid_init.argtypes = []
_lib.vk_hybrid_init.restype = ctypes.c_bool
_lib.vk_hybrid_load.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_int]
_lib.vk_hybrid_load.restype = ctypes.c_int
_lib.vk_hybrid_upload.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_size_t]
_lib.vk_hybrid_upload.restype = ctypes.c_bool
_lib.vk_hybrid_run.argtypes = [ctypes.c_int, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_void_p]
_lib.vk_hybrid_run.restype = ctypes.c_bool
_lib.vk_hybrid_download.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_size_t]
_lib.vk_hybrid_download.restype = ctypes.c_bool
_lib.vk_hybrid_last_gpu_us.argtypes = []
_lib.vk_hybrid_last_gpu_us.restype = ctypes.c_double
_lib.vk_hybrid_destroy.argtypes = []
_lib.vk_hybrid_destroy.restype = None

_init_done = False
_pipes = {}  # name → handle

# Timing accumulators (per op type)
_TIMERS = {}  # name → {'cpu_pack': 0, 'gpu': 0, 'cpu_unpack': 0, 'count': 0}


def _ensure_init():
    global _init_done
    if _init_done:
        return
    if not _lib.vk_hybrid_init():
        raise RuntimeError("vk_hybrid_init failed")
    _init_done = True


def _get_pipe(name, spv_file, n_bufs, push_sz):
    """Load SPIR-V pipeline, cached by name."""
    if name not in _pipes:
        h = _lib.vk_hybrid_load(spv_file.encode(), n_bufs, push_sz)
        if h < 0:
            raise RuntimeError(f"vk_hybrid_load failed for {name}")
        _pipes[name] = h
    return _pipes[name]


def _reset_timers():
    _TIMERS.clear()


def get_timers():
    """Return accumulated timers for all ops: {name: {cpu_pack, gpu, cpu_unpack, count}}"""
    return dict(_TIMERS)


def _to_u16(t: torch.Tensor) -> np.ndarray:
    """Convert torch tensor to contiguous fp16 numpy uint16 array."""
    return t.detach().cpu().contiguous().to(torch.float16).numpy().view(np.uint16).copy()


def _from_u16(arr: np.ndarray, shape, device, dtype) -> torch.Tensor:
    """Convert fp16 uint16 numpy array back to torch tensor."""
    return torch.from_numpy(arr.view(np.float16)).reshape(shape).to(device, dtype)


# ============================================================
# SiLU
# ============================================================
def vk_silu(x: torch.Tensor):
    """SiLU activation: x * sigmoid(x). In-place compatible (reads x, writes new result)."""
    _ensure_init()
    h = _get_pipe("silu", "/data/local/tmp/silu_fp16.spv", 2, 4)

    t0 = time.perf_counter()
    x_u16 = _to_u16(x.reshape(-1))
    out = np.zeros_like(x_u16)
    n = len(x_u16)

    _lib.vk_hybrid_upload(h, 0, x_u16.ctypes.data, x_u16.nbytes)
    _lib.vk_hybrid_upload(h, 1, out.ctypes.data, out.nbytes)
    t_pack = (time.perf_counter() - t0) * 1e6

    push = struct.pack("I", n)
    _lib.vk_hybrid_run(h, (n + 255) // 256, 1, 1, push)
    gpu_us = _lib.vk_hybrid_last_gpu_us()

    t1 = time.perf_counter()
    _lib.vk_hybrid_download(h, 1, out.ctypes.data, out.nbytes)
    result = _from_u16(out, x.shape, x.device, x.dtype)
    t_unpack = (time.perf_counter() - t1) * 1e6

    # Accumulate timers
    t = _TIMERS.setdefault("silu", {"cpu_pack": 0.0, "gpu": 0.0, "cpu_unpack": 0.0, "count": 0})
    t["cpu_pack"] += t_pack
    t["gpu"] += gpu_us if gpu_us > 0 else 0
    t["cpu_unpack"] += t_unpack
    t["count"] += 1

    return True, result, {"cpu_pack_us": t_pack, "gpu_us": gpu_us, "cpu_unpack_us": t_unpack}


# ============================================================
# GELU
# ============================================================
def vk_gelu(x: torch.Tensor):
    """GELU activation (tanh approx): 0.5 * x * (1 + tanh(c*(x+k*x^3)))."""
    _ensure_init()
    h = _get_pipe("gelu", "/data/local/tmp/gelu_fp16.spv", 2, 4)

    shape = x.shape
    t0 = time.perf_counter()
    x_u16 = _to_u16(x.reshape(-1))
    out = np.zeros_like(x_u16)
    n = len(x_u16)

    _lib.vk_hybrid_upload(h, 0, x_u16.ctypes.data, x_u16.nbytes)
    _lib.vk_hybrid_upload(h, 1, out.ctypes.data, out.nbytes)
    t_pack = (time.perf_counter() - t0) * 1e6

    push = struct.pack("I", n)
    _lib.vk_hybrid_run(h, (n + 255) // 256, 1, 1, push)
    gpu_us = _lib.vk_hybrid_last_gpu_us()

    t1 = time.perf_counter()
    _lib.vk_hybrid_download(h, 1, out.ctypes.data, out.nbytes)
    result = _from_u16(out, shape, x.device, x.dtype)
    t_unpack = (time.perf_counter() - t1) * 1e6

    t = _TIMERS.setdefault("gelu", {"cpu_pack": 0.0, "gpu": 0.0, "cpu_unpack": 0.0, "count": 0})
    t["cpu_pack"] += t_pack
    t["gpu"] += gpu_us if gpu_us > 0 else 0
    t["cpu_unpack"] += t_unpack
    t["count"] += 1

    return True, result, {"cpu_pack_us": t_pack, "gpu_us": gpu_us, "cpu_unpack_us": t_unpack}


# ============================================================
# LayerNorm (no affine — just normalize)
# ============================================================
def vk_layernorm(x: torch.Tensor, eps: float = 1e-6):
    """LayerNorm: (x - mean) / sqrt(var + eps). One workgroup per row. No affine weights."""
    _ensure_init()
    h = _get_pipe("layernorm", "/data/local/tmp/layernorm_fp16.spv", 2, 12)

    shape = x.shape
    x_flat = x.reshape(-1, shape[-1])
    n_rows, n_elems = x_flat.shape

    t0 = time.perf_counter()
    x_u16 = _to_u16(x_flat)
    out = np.zeros_like(x_u16)

    _lib.vk_hybrid_upload(h, 0, x_u16.ctypes.data, x_u16.nbytes)
    _lib.vk_hybrid_upload(h, 1, out.ctypes.data, out.nbytes)
    t_pack = (time.perf_counter() - t0) * 1e6

    push = struct.pack("IIf", n_rows, n_elems, eps)
    _lib.vk_hybrid_run(h, n_rows, 1, 1, push)  # one workgroup per row
    gpu_us = _lib.vk_hybrid_last_gpu_us()

    t1 = time.perf_counter()
    _lib.vk_hybrid_download(h, 1, out.ctypes.data, out.nbytes)
    result = _from_u16(out, x_flat.shape, x.device, x.dtype).reshape(shape)
    t_unpack = (time.perf_counter() - t1) * 1e6

    t = _TIMERS.setdefault("layernorm", {"cpu_pack": 0.0, "gpu": 0.0, "cpu_unpack": 0.0, "count": 0})
    t["cpu_pack"] += t_pack
    t["gpu"] += gpu_us if gpu_us > 0 else 0
    t["cpu_unpack"] += t_unpack
    t["count"] += 1

    return True, result, {"cpu_pack_us": t_pack, "gpu_us": gpu_us, "cpu_unpack_us": t_unpack}


# ============================================================
# GEMM (via libvk_hybrid — for comparison with old libvk_gemm)
# ============================================================
def vk_gemm(x: torch.Tensor, weight: torch.Tensor):
    """GEMM via hybrid wrapper. For comparison with old libvk_gemm.so."""
    _ensure_init()
    h = _get_pipe("gemm", "/data/local/tmp/gemm_fp16.spv", 3, 20)

    *batch, in_f = x.shape
    M = int(np.prod(batch)) if batch else 1
    N = weight.shape[0]
    K = in_f

    t0 = time.perf_counter()
    x_u16 = _to_u16(x.reshape(M, K))
    w_u16 = _to_u16(weight)
    out = np.zeros((M, N), dtype=np.uint16)

    _lib.vk_hybrid_upload(h, 0, x_u16.ctypes.data, x_u16.nbytes)
    _lib.vk_hybrid_upload(h, 1, w_u16.ctypes.data, w_u16.nbytes)
    _lib.vk_hybrid_upload(h, 2, None, M * N * 2)  # allocate output
    t_pack = (time.perf_counter() - t0) * 1e6

    push = struct.pack("IIIIf", M, N, K, 1, 1.0)
    _lib.vk_hybrid_run(h, (N + 7) // 8, (M + 7) // 8, 1, push)
    gpu_us = _lib.vk_hybrid_last_gpu_us()

    t1 = time.perf_counter()
    _lib.vk_hybrid_download(h, 2, out.ctypes.data, out.nbytes)
    result = _from_u16(out, (M, N), x.device, x.dtype)
    if batch:
        result = result.reshape(*batch, N)
    t_unpack = (time.perf_counter() - t1) * 1e6

    t = _TIMERS.setdefault("gemm", {"cpu_pack": 0.0, "gpu": 0.0, "cpu_unpack": 0.0, "count": 0})
    t["cpu_pack"] += t_pack
    t["gpu"] += gpu_us if gpu_us > 0 else 0
    t["cpu_unpack"] += t_unpack
    t["count"] += 1

    return True, result, {"cpu_pack_us": t_pack, "gpu_us": gpu_us, "cpu_unpack_us": t_unpack}


if __name__ == "__main__":
    print("=== SiLU test ===")
    x = torch.randn(1024, 2048, dtype=torch.float16)
    t0 = time.perf_counter()
    ok, y, ts = vk_silu(x)
    dt = (time.perf_counter() - t0) * 1000
    ref = x.float() * torch.sigmoid(x.float())
    err = (y.float() - ref).abs().max().item()
    print(f"  wall={dt:.1f}ms pack={ts['cpu_pack_us']:.0f}us gpu={ts['gpu_us']:.0f}us unpack={ts['cpu_unpack_us']:.0f}us")
    print(f"  max_err={err:.6f}")

    print("\n=== LayerNorm test ===")
    x = torch.randn(512, 2048, dtype=torch.float16)
    t0 = time.perf_counter()
    ok, y, ts = vk_layernorm(x, 1e-6)
    dt = (time.perf_counter() - t0) * 1000
    ref = torch.nn.functional.layer_norm(x.float(), [x.shape[-1]], eps=1e-6).half()
    err = (y.float() - ref.float()).abs().max().item()
    print(f"  wall={dt:.1f}ms pack={ts['cpu_pack_us']:.0f}us gpu={ts['gpu_us']:.0f}us unpack={ts['cpu_unpack_us']:.0f}us")
    print(f"  max_err={err:.6f}")

"""Monkey-patch torch F.silu, F.gelu, F.layer_norm with Vulkan for large tensors.
Only dispatches to GPU when profitable; small tensors stay on CPU.
"""
import torch
import torch.nn.functional as _F
import vk_hybrid_ops as _vk

_orig_silu = _F.silu
_orig_gelu = _F.gelu
_orig_layer_norm = _F.layer_norm

# Thresholds
_SILU_MIN_ELEMS = 100000
_GELU_MIN_ELEMS = 100000
_LN_MIN_ROWS = 64

_enabled = False


def enable():
    global _enabled
    if _enabled: return
    _F.silu = _vk_silu
    _F.gelu = _vk_gelu
    _F.layer_norm = _vk_layernorm
    _enabled = True


def disable():
    global _enabled
    if not _enabled: return
    _F.silu = _orig_silu
    _F.gelu = _orig_gelu
    _F.layer_norm = _orig_layer_norm
    _enabled = False


def is_enabled():
    return _enabled


def get_timers():
    return _vk.get_timers()


def reset_timers():
    _vk._reset_timers()


def _vk_silu(x, inplace=False):
    if inplace or not _enabled or x.numel() < _SILU_MIN_ELEMS:
        return _orig_silu(x, inplace=inplace)
    ok, result, _ = _vk.vk_silu(x)
    return result if ok else _orig_silu(x, inplace=inplace)


def _vk_gelu(x, approximate='none'):
    if not _enabled or x.numel() < _GELU_MIN_ELEMS:
        return _orig_gelu(x, approximate=approximate)
    ok, result, _ = _vk.vk_gelu(x)
    return result if ok else _orig_gelu(x, approximate=approximate)


def _vk_layernorm(input, normalized_shape, weight=None, bias=None, eps=1e-05):
    # Only handle no-affine case; otherwise fall through to CPU
    if (not _enabled or weight is not None or bias is not None
            or input.numel() == 0):
        return _orig_layer_norm(input, normalized_shape, weight, bias, eps)

    # Flatten to 2D: [B, ..., D] → [B*... , D]
    D = input.shape[-1]
    x_2d = input.reshape(-1, D)
    if x_2d.shape[0] < _LN_MIN_ROWS:
        return _orig_layer_norm(input, normalized_shape, weight, bias, eps)

    ok, result_2d, _ = _vk.vk_layernorm(x_2d, eps)
    if not ok:
        return _orig_layer_norm(input, normalized_shape, weight, bias, eps)
    return result_2d.reshape(input.shape)

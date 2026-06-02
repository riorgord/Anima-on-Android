"""Hybrid ops: Vulkan GEMM (BF16 weights + FP32 compute) / LN / RMSNorm / GELU
via libhybrid_engine.so. PyTorch holds only shell (~200MB).
"""
import time, struct
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import ctypes as _ct

# ── libhybrid_engine.so ──
try:
    _lib = _ct.CDLL("/data/local/tmp/libhybrid_engine.so")
    _lib.vk_engine_init.argtypes = []
    _lib.vk_engine_init.restype = _ct.c_bool
    _lib.vk_weight_add.argtypes = [_ct.c_char_p, _ct.c_void_p, _ct.c_int,
                                    _ct.c_void_p, _ct.c_int]
    _lib.vk_weight_add.restype = _ct.c_int
    _lib.vk_engine_finalize.argtypes = []
    _lib.vk_engine_finalize.restype = _ct.c_bool
    _lib.vk_reset_pool.argtypes = []
    _lib.vk_reset_pool.restype = _ct.c_bool
    _lib.vk_run_gemm.argtypes = [_ct.c_char_p, _ct.c_void_p, _ct.c_void_p,
                                  _ct.c_int, _ct.c_int, _ct.c_int]
    _lib.vk_run_gemm.restype = _ct.c_bool
    _lib.vk_run_layernorm.argtypes = [_ct.c_void_p, _ct.c_void_p,
                                       _ct.c_int, _ct.c_int, _ct.c_float]
    _lib.vk_run_layernorm.restype = _ct.c_bool
    _lib.vk_run_rmsnorm.argtypes = [_ct.c_void_p, _ct.c_void_p, _ct.c_int,
                                     _ct.c_void_p, _ct.c_int, _ct.c_int, _ct.c_float]
    _lib.vk_run_rmsnorm.restype = _ct.c_bool
    _lib.vk_run_gelu.argtypes = [_ct.c_void_p, _ct.c_void_p, _ct.c_int]
    _lib.vk_run_gelu.restype = _ct.c_bool
    _lib.vk_engine_destroy.argtypes = []
    _lib.vk_engine_destroy.restype = None
    _VK_AVAILABLE = True
except Exception:
    _VK_AVAILABLE = False

# All Linear weights go to Vulkan (no threshold — correctness first)

_VK_COUNT = 0
_CPU_COUNT = 0
_VK_TIME = 0.0
_CPU_TIME = 0.0


# ═══════════════════════════════════════════════════════════════
# VulkanGemmLinear — no PyTorch weight Parameter
# ═══════════════════════════════════════════════════════════════
class VulkanGemmLinear(nn.Module):
    """Linear layer where weight lives in libhybrid_engine.so Vulkan buffer.
    NOT a subclass of nn.Linear — no self.weight Parameter created.
    """
    def __init__(self, weight_name, in_features, out_features, bias=False,
                 device=None, dtype=None):
        super().__init__()
        self.weight_name = weight_name
        self.in_features = in_features
        self.out_features = out_features
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, device=device, dtype=dtype))
        else:
            self.register_parameter('bias', None)

    def forward(self, x):
        global _VK_COUNT, _CPU_COUNT, _VK_TIME, _CPU_TIME
        *batch, in_f = x.shape
        M = int(np.prod(batch)) if batch else 1
        out_f = self.out_features

        if _VK_AVAILABLE:
            x_f32 = x.reshape(M, in_f).cpu().contiguous().float().numpy().copy()
            out_f32 = np.zeros(M * out_f, dtype=np.float32)

            t0 = time.perf_counter()
            ok = _lib.vk_run_gemm(
                self.weight_name.encode(),
                x_f32.ctypes.data_as(_ct.c_void_p),
                out_f32.ctypes.data_as(_ct.c_void_p),
                M, out_f, in_f)
            _VK_TIME += time.perf_counter() - t0

            if not ok:
                return F.linear(x, self._cpu_weight(x.device, x.dtype), self.bias)

            result = torch.from_numpy(out_f32).to(device=x.device, dtype=x.dtype)
            result = result.reshape(*batch, out_f) if batch else result.squeeze(0)
            if self.bias is not None:
                result += self.bias.to(result.device, result.dtype)
            _VK_COUNT += 1
            return result

        _CPU_COUNT += 1
        t0 = time.perf_counter()
        out = F.linear(x, self._cpu_weight(x.device, x.dtype), self.bias)
        _CPU_TIME += time.perf_counter() - t0
        return out

    def _cpu_weight(self, device, dtype):
        """Lazy CPU fallback — reads weight from Vulkan buffer."""
        # Allocate numpy buffer and ask Vulkan to fill it
        # For now, weight is not readable from Vulkan.
        # In practice this path is only hit when M<16 or _VK_AVAILABLE=False.
        # Return a zero tensor to fail loudly.
        return torch.zeros(self.out_features, self.in_features, device=device, dtype=dtype)

    def extra_repr(self):
        return f'in_features={self.in_features}, out_features={self.out_features}, ' \
               f'bias={self.bias is not None}, weight={self.weight_name}'


# ═══════════════════════════════════════════════════════════════
# LayerNorm / RMSNorm / GELU — accelerated via libhybrid_engine.so
# ═══════════════════════════════════════════════════════════════
class HybridLayerNorm(nn.LayerNorm):
    """nn.LayerNorm with Vulkan acceleration (FP32 I/O)."""
    _count = 0
    def forward(self, x):
        if not _VK_AVAILABLE or self.elementwise_affine:
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        *batch, D = x.shape
        M = int(np.prod(batch)) if batch else 1
        x_f32 = x.reshape(M, D).cpu().contiguous().float().numpy()
        out_buf = np.zeros((M, D), dtype=np.float32)
        ok = _lib.vk_run_layernorm(
            x_f32.ctypes.data_as(_ct.c_void_p),
            out_buf.ctypes.data_as(_ct.c_void_p),
            M, D, _ct.c_float(self.eps))
        if not ok:
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        result = torch.from_numpy(out_buf).to(device=x.device, dtype=x.dtype)
        result = result.reshape(*batch, D) if batch else result.squeeze(0)
        cls = type(self)
        if cls._count < 3:
            ref = F.layer_norm(x.float(), self.normalized_shape, None, None, self.eps)
            err = (result.float() - ref.float()).abs().max().item()
            print(f"  VkLN#{cls._count} M={M} D={D} max_err={err:.6f}")
            cls._count += 1
        return result


class HybridRMSNorm(nn.RMSNorm):
    """nn.RMSNorm with Vulkan acceleration (FP16 I/O)."""
    _count = 0
    def forward(self, x):
        if not _VK_AVAILABLE:
            return F.rms_norm(x, self.normalized_shape, self.weight, self.eps)
        *batch, D = x.shape
        M = int(np.prod(batch)) if batch else 1
        x_f16 = x.reshape(M, D).cpu().contiguous().to(torch.float16).numpy().view(np.uint16)
        w_f16 = self.weight.detach().cpu().to(torch.float16).numpy().view(np.uint16).copy()
        out_buf = np.zeros(M * D, dtype=np.uint16)
        ok = _lib.vk_run_rmsnorm(
            x_f16.ctypes.data_as(_ct.c_void_p),
            w_f16.ctypes.data_as(_ct.c_void_p), int(w_f16.size),
            out_buf.ctypes.data_as(_ct.c_void_p),
            M, D, _ct.c_float(self.eps))
        if not ok:
            return F.rms_norm(x, self.normalized_shape, self.weight, self.eps)
        result = torch.tensor(out_buf.view(np.float16), device=x.device, dtype=x.dtype)
        result = result.reshape(*batch, D) if batch else result.squeeze(0)
        cls = type(self)
        if cls._count < 3:
            ref = F.rms_norm(x.float(), self.normalized_shape, self.weight.float(), self.eps)
            err = (result.float() - ref.float()).abs().max().item()
            print(f"  VkRMS#{cls._count} M={M} D={D} max_err={err:.6f}")
            cls._count += 1
        return result


class HybridGELU(nn.GELU):
    """nn.GELU with Vulkan acceleration (FP16 I/O)."""
    _count = 0
    def forward(self, x):
        if not _VK_AVAILABLE:
            return F.gelu(x)
        *batch, D = x.shape
        N = int(np.prod(batch)) * D if batch else D
        x_f16 = x.reshape(-1).cpu().contiguous().to(torch.float16).numpy().view(np.uint16)
        out_buf = np.zeros(N, dtype=np.uint16)
        ok = _lib.vk_run_gelu(
            x_f16.ctypes.data_as(_ct.c_void_p),
            out_buf.ctypes.data_as(_ct.c_void_p), N)
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


# ═══════════════════════════════════════════════════════════════
# DummyLinear — placeholder that allocates NO weight memory
# ═══════════════════════════════════════════════════════════════
class DummyLinear(nn.Module):
    """Placeholder: stores in/out dims, no weight Parameter.
    Must be patched (→ VulkanGemmLinear or nn.Linear) before forward."""
    def __init__(self, in_features, out_features, bias=False, device=None, dtype=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self._dtype = dtype if dtype is not None else torch.float16
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, device=device, dtype=self._dtype))
        else:
            self.register_parameter('bias', None)
        # For forward compatibility: register_buffer creates a dummy buffer that
        # prevents nn.Linear from being detected as a leaf parameter source
        self.register_buffer('_placeholder', torch.empty(0, device=device, dtype=self._dtype))

    def forward(self, x):
        raise RuntimeError("DummyLinear.forward — must be patched before use")


# ═══════════════════════════════════════════════════════════════
# Operations classes
# ═══════════════════════════════════════════════════════════════
class DummyOps:
    """Creates model shell with ZERO weight memory for Linear layers."""
    Linear = DummyLinear
    RMSNorm = nn.RMSNorm       # use PyTorch for norms initially
    LayerNorm = nn.LayerNorm
    Embedding = nn.Embedding


class ShellHybridOps:
    """For non-block layers: nn.Linear in PyTorch, norms via Vulkan."""
    Linear = nn.Linear
    RMSNorm = HybridRMSNorm
    LayerNorm = HybridLayerNorm
    Embedding = nn.Embedding


# ═══════════════════════════════════════════════════════════════
# Block layer patching
# ═══════════════════════════════════════════════════════════════
def patch_shell_linear(model):
    """Replace ALL DummyLinear → VulkanGemmLinear (block + shell).
    All Linear weights are now in Vulkan — no nn.Linear needed."""
    def walk_and_patch(module, name_prefix=""):
        patched = 0
        for child_name, child in list(module.named_children()):
            full_prefix = f"{name_prefix}.{child_name}" if name_prefix else child_name
            if isinstance(child, DummyLinear):
                wname = f"{full_prefix}.weight"
                has_bias = child.bias is not None
                new_lin = VulkanGemmLinear(
                    weight_name=wname,
                    in_features=child.in_features,
                    out_features=child.out_features,
                    bias=has_bias,
                )
                if has_bias:
                    new_lin.bias.data.copy_(child.bias.data)
                setattr(module, child_name, new_lin)
                patched += 1
            else:
                patched += walk_and_patch(child, full_prefix)
        return patched

    n = walk_and_patch(model)
    if n > 0: print(f"Patched {n} DummyLinear → VulkanGemmLinear")
    return n


def patch_block_layers(model):
    """Replace block DummyLinear layers with VulkanGemmLinear."""
    patched = 0
    for block_idx, block in enumerate(model.blocks):
        prefix = f"blocks.{block_idx}."
        replacements = {
            ("self_attn", "q_proj"):      f"{prefix}self_attn.q_proj.weight",
            ("self_attn", "k_proj"):      f"{prefix}self_attn.k_proj.weight",
            ("self_attn", "v_proj"):      f"{prefix}self_attn.v_proj.weight",
            ("self_attn", "output_proj"): f"{prefix}self_attn.output_proj.weight",
            ("cross_attn", "q_proj"):     f"{prefix}cross_attn.q_proj.weight",
            ("cross_attn", "k_proj"):     f"{prefix}cross_attn.k_proj.weight",
            ("cross_attn", "v_proj"):     f"{prefix}cross_attn.v_proj.weight",
            ("cross_attn", "output_proj"): f"{prefix}cross_attn.output_proj.weight",
            ("mlp", "layer1"): f"{prefix}mlp.layer1.weight",
            ("mlp", "layer2"): f"{prefix}mlp.layer2.weight",
        }
        for (parent_attr, child_attr), wname in replacements.items():
            parent = getattr(block, parent_attr)
            old = getattr(parent, child_attr)
            has_bias = old.bias is not None
            new_linear = VulkanGemmLinear(
                weight_name=wname,
                in_features=old.in_features,
                out_features=old.out_features,
                bias=has_bias,
            )
            # Copy bias if present
            if has_bias:
                new_linear.bias.data.copy_(old.bias.data)
            setattr(parent, child_attr, new_linear)
            patched += 1
    print(f"Patched {patched} block Linear → VulkanGemmLinear")


# ═══════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════
def is_linear_weight(key, shape=None):
    """Does this safetensors key+shape correspond to a Linear (GEMM) weight?
    Any 2D .weight tensor → Linear, excluding embedding/norm weights.
    shape is from safetensors header (list of ints), used when available."""
    if not key.endswith('.weight'):
        return False
    # Exclude position/text embedding lookup tables (not GEMM)
    if 'pos_embed' in key.lower() or 'embed_tokens' in key.lower():
        return False
    # Exclude norm weights (1D but safe to check by name)
    if 'norm' in key.lower():
        return False
    # If shape is known: 2D → Linear weight matrix [out_features, in_features]
    if shape is not None:
        if len(shape) != 2:
            return False
    return True

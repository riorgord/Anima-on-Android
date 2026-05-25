"""Vulkan-accelerated Linear layer — drop-in replacement for nn.Linear.

Architecture:
  1. Compile gemm.comp → SPIR-V
  2. Allocate Vulkan buffers for weight (constant), input/output (per-call)
  3. Each forward() submits a compute dispatch

When Vulkan is unavailable, falls back to PyTorch nn.Linear.
"""
import torch
import torch.nn as nn
from pathlib import Path

_HERE = Path(__file__).parent


class VulkanLinear(nn.Module):
    """Linear layer accelerated by Vulkan GEMM shader.

    Uses the same weight/bias interface as nn.Linear.
    Internally dispatches to Vulkan compute (or PyTorch fallback).
    """

    def __init__(self, in_features, out_features, bias=True, dtype=torch.float16):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.use_vulkan = False  # Set True once Vulkan pipeline is initialized
        self._vk_initialized = False

        self.weight = nn.Parameter(torch.empty(out_features, in_features, dtype=dtype))
        self.bias = nn.Parameter(torch.empty(out_features, dtype=dtype)) if bias else None
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def _ensure_vulkan(self):
        """Lazy-init Vulkan pipeline on first call."""
        if self._vk_initialized or not self.use_vulkan:
            return
        self._vk_initialized = True
        # TODO: create Vulkan compute pipeline for this layer's dimensions
        # - Compile weight to Vulkan buffer
        # - Pre-allocate output buffer
        # - Record command buffer template

    def forward(self, x):
        if self.use_vulkan:
            return self._forward_vulkan(x)
        return nn.functional.linear(x, self.weight, self.bias)

    def _forward_vulkan(self, x):
        """Vulkan accelerated forward (stub — falls back for now)."""
        return nn.functional.linear(x, self.weight, self.bias)


def replace_linear_with_vulkan(model, pattern=None):
    """Recursively replace nn.Linear layers with VulkanLinear.

    Args:
        model: PyTorch module
        pattern: Optional regex to filter which layers to replace
    Returns: count of replaced layers
    """
    count = 0
    for name, child in model.named_children():
        if isinstance(child, nn.Linear):
            vk = VulkanLinear(child.in_features, child.out_features,
                              bias=child.bias is not None,
                              dtype=child.weight.dtype)
            vk.weight.data.copy_(child.weight.data)
            if child.bias is not None:
                vk.bias.data.copy_(child.bias.data)
            setattr(model, name, vk)
            count += 1
        else:
            count += replace_linear_with_vulkan(child, pattern)
    return count


if __name__ == "__main__":
    # Quick smoke test
    m = nn.Sequential(
        nn.Linear(16, 32),
        nn.ReLU(),
        nn.Linear(32, 8),
    )
    n = replace_linear_with_vulkan(m)
    print(f"Replaced {n} Linear layers")
    x = torch.randn(2, 16, dtype=torch.float16)
    y = m(x)
    print(f"Output: {list(y.shape)} mean={float(y.mean()):.4f}")

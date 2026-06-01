"""Compare PyTorch LayerNorm with naive two-pass algorithm (what our shader does)."""
import torch, torch.nn.functional as F

x = torch.randn(512, 2048, dtype=torch.float16, device='cuda')

# PyTorch LN
pt_ln = F.layer_norm(x.float(), (2048,), weight=None, bias=None, eps=1e-6).half()

# Naive two-pass (same as our GLSL shader):
# Pass 1: mean
mean = x.float().mean(dim=-1, keepdim=True)
# Pass 2: var
var = ((x.float() - mean) ** 2).mean(dim=-1, keepdim=True)
# Normalize
naive_ln = ((x.float() - mean) / (var + 1e-6).sqrt()).half()

diff = (pt_ln.float() - naive_ln.float()).abs()
print(f"PyTorch LN vs naive 2-pass: max_err={diff.max():.6f} bit_exact={(diff.max()==0).item()}")

# Also compare with our shader's formula: (x-mean)/sqrt(var+eps)
# Which is the same naive 2-pass

# Check if PyTorch uses a different formula:
# Try: `torch.rsqrt(var+eps)` instead of `1/sqrt(var+eps)`
rsqrt_ln = ((x.float() - mean) * torch.rsqrt(var + 1e-6)).half()
diff2 = (pt_ln.float() - rsqrt_ln.float()).abs()
print(f"PyTorch LN vs rsqrt variant: max_err={diff2.max():.6f} bit_exact={(diff2.max()==0).item()}")

# Check if difference is in mean computation
# PyTorch might use a different reduction order
pt_mean = x.float().mean(dim=-1)
naive_mean = x.float().sum(dim=-1) / 2048.0
diff3 = (pt_mean - naive_mean).abs()
print(f"mean diff: max_err={diff3.max():.10f}")

# Check if difference is in 1/sqrt vs rsqrt
a = torch.tensor([2.0], device='cuda')
print(f"1/sqrt(2)={1.0/a.sqrt().item():.10f} rsqrt(2)={a.rsqrt().item():.10f} same={(1.0/a.sqrt()==a.rsqrt()).item()}")

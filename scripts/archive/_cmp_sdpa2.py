"""Test: does PT 2.11 F.sdpa scale by default?"""
import torch, math

torch.manual_seed(42); B,H,S,D = 1,1,8,16
q=torch.randn(B,H,S,D); k=torch.randn(B,H,S,D); v=torch.randn(B,H,S,D)

scale = 1.0/math.sqrt(D)
out_default = torch.nn.functional.scaled_dot_product_attention(q,k,v)
out_scaled  = torch.nn.functional.scaled_dot_product_attention(q,k,v, scale=scale)
out_noscale = torch.nn.functional.scaled_dot_product_attention(q*scale, k*scale, v)

# Manual math with scale
qs=q*scale; ks=k*scale
out_math = torch.softmax(qs@ks.transpose(-2,-1),dim=-1) @ v

print(f"Default:     range=[{out_default.min():.4f},{out_default.max():.4f}]")
print(f"Explicit scale: range=[{out_scaled.min():.4f},{out_scaled.max():.4f}]")
print(f"Pre-scaled:  range=[{out_noscale.min():.4f},{out_noscale.max():.4f}]")
print(f"Math scaled: range=[{out_math.min():.4f},{out_math.max():.4f}]")

# Check which matches which
import numpy as np
e1 = np.abs(out_default.numpy()-out_scaled.numpy()).max()
e2 = np.abs(out_default.numpy()-out_noscale.numpy()).max()
e3 = np.abs(out_default.numpy()-out_math.numpy()).max()
print(f"\nerr(default,explicit): {e1:.2e}")
print(f"err(default,pre-scaled): {e2:.2e}")
print(f"err(default,math): {e3:.2e}")
print(f"err(math,explicit): {np.abs(out_math.numpy()-out_scaled.numpy()).max():.2e}")

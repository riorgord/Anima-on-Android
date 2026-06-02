"""Compare PT default SDPA vs manual math backend on phone."""
import torch, math

torch.manual_seed(42)
B,H,S,D = 2,16,256,128
q = torch.randn(B,H,S,D).float()
k = torch.randn(B,H,S,D).float()
v = torch.randn(B,H,S,D).float()

# PT default
out_pt = torch.nn.functional.scaled_dot_product_attention(q,k,v)
print(f"PT default: range=[{out_pt.min():.4f},{out_pt.max():.4f}]")

# Manual math
scale = 1.0/math.sqrt(D)
qs = q*scale; ks = k*scale
attn = qs @ ks.transpose(-2,-1)
out_math = torch.softmax(attn,dim=-1) @ v
print(f"PT math:    range=[{out_math.min():.4f},{out_math.max():.4f}]")

err = (out_pt - out_math).abs().max().item()
print(f"max_err: {err:.2e}")

# Also test with FP16
q_h = q.half(); k_h = k.half(); v_h = v.half()
out_pt_h = torch.nn.functional.scaled_dot_product_attention(q_h,k_h,v_h)
qs_h = q_h.float()*scale; ks_h = k_h.float()*scale
attn_h = qs_h @ ks_h.transpose(-2,-1)
out_math_h = (torch.softmax(attn_h,dim=-1) @ v_h.float()).half()
err_h = (out_pt_h.float() - out_math_h.float()).abs().max().item()
print(f"FP16 PT default vs math: max_err={err_h:.2e}")
print(f"FP16 PT range=[{out_pt_h.float().min():.4f},{out_pt_h.float().max():.4f}]")
print(f"FP16 Math range=[{out_math_h.float().min():.4f},{out_math_h.float().max():.4f}]")

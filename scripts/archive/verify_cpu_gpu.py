"""Verify PyTorch determinism: CPU/GPU/FP16 bit-exact tests."""
import torch, torch.nn.functional as F

# FP32: GPU run twice
x = torch.randn(512, 2048, dtype=torch.float32, device='cuda')
ln1 = F.layer_norm(x, (2048,), weight=None, bias=None, eps=1e-6)
ln2 = F.layer_norm(x.clone(), (2048,), weight=None, bias=None, eps=1e-6)
d = (ln1 - ln2).abs()
print(f"LN fp32 GPU run1 vs run2: max_err={d.max():.10f} bit_exact={(d.max()==0).item()}")

# FP16: GPU run twice
xh = x.half()
ln1h = F.layer_norm(xh, (2048,), weight=None, bias=None, eps=1e-6)
ln2h = F.layer_norm(xh.clone(), (2048,), weight=None, bias=None, eps=1e-6)
d2 = (ln1h.float() - ln2h.float()).abs()
print(f"LN fp16 GPU run1 vs run2: max_err={d2.max():.10f} bit_exact={(d2.max()==0).item()}")

# FP16 GPU vs FP16 CPU (sequential)
xc = xh.cpu()
ln_cpu = F.layer_norm(xc, (2048,), weight=None, bias=None, eps=1e-6)
d3 = (ln1h.cpu().float() - ln_cpu.float()).abs()
print(f"LN fp16 GPU vs fp16 CPU: max_err={d3.max():.6f} bit_exact={(d3.max()==0).item()}")

# FP32 GPU vs FP32 CPU
xc_f32 = x.cpu()
ln_cpu_f32 = F.layer_norm(xc_f32, (2048,), weight=None, bias=None, eps=1e-6)
d4 = (ln1.cpu() - ln_cpu_f32).abs()
print(f"LN fp32 GPU vs fp32 CPU: max_err={d4.max():.10f} bit_exact={(d4.max()==0).item()}")

# RMSNorm fp16: run twice
xr = xh.reshape(-1, 128)
rn1 = F.rms_norm(xr, (128,), weight=None, eps=1e-6)
rn2 = F.rms_norm(xr.clone(), (128,), weight=None, eps=1e-6)
d5 = (rn1.float() - rn2.float()).abs()
print(f"RMSNorm fp16 run1 vs run2: max_err={d5.max():.10f} bit_exact={(d5.max()==0).item()}")


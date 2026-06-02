"""Verify numpy Timesteps vs PT — line-by-line translation."""
import sys, math, numpy as np, torch
sys.path.insert(0, "/sdcard/anima_on_android/src")
import predict2

# Create real Timesteps module
ts = predict2.Timesteps(num_channels=2048)

# Test input (pipeline shape: B=2 CFG batch, T=1)
timesteps_pt = torch.tensor([[1.0], [1.0]], dtype=torch.float32)

# PT reference
out_pt = ts.forward(timesteps_pt)

# Numpy: line-by-line translation
ts_np = timesteps_pt.float().cpu().numpy()
B, T = ts_np.shape[0], 1 if ts_np.ndim < 2 else ts_np.shape[1]
timesteps = ts_np.reshape(-1)
half_dim = ts.num_channels // 2  # 1024
# PT: exponent = -log(10000) * arange(half_dim) / (half_dim - 0.0)
exponent = -math.log(10000) * np.arange(half_dim, dtype=np.float32) / (half_dim - 0.0)
# PT: emb = exp(exponent)
emb = np.exp(exponent)
# PT: emb = timesteps[:, None].float() * emb[None, :]
emb = timesteps[:, None] * emb[None, :]
# PT: sin_emb = sin(emb); cos_emb = cos(emb)
sin_emb = np.sin(emb); cos_emb = np.cos(emb)
# PT: emb = cat([cos_emb, sin_emb], dim=-1)
emb = np.concatenate([cos_emb, sin_emb], axis=-1)
# PT: rearrange(emb, "(b t) d -> b t d", b=B, t=T)
emb = emb.reshape(B, T, -1)

err = np.abs(emb - out_pt.float().cpu().numpy())
print(f"out shape: {emb.shape}")
print(f"max_err: {err.max():.8f}")
print(f"mean_err: {err.mean():.8f}")
print(f"== 0.0: {(err == 0.0).mean()*100:.1f}%")
print(f"RESULT: {'PASS' if err.max() < 1e-7 else 'FAIL'}")

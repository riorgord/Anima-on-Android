"""Verify numpy RoPE vs PT — line-by-line translation, should be bit-exact."""
import sys, numpy as np, torch
sys.path.insert(0, "/sdcard/anima_on_android/src")
import predict2 as _p2, position_embedding

# Real pipeline shapes
pos_emb = position_embedding.VideoRopePosition3DEmb(
    model_channels=2048, len_h=16, len_w=16, len_t=1,
    max_fps=30, min_fps=1, is_learnable=True, interpolation="crop",
    head_dim=128, h_extrapolation_ratio=4.0, w_extrapolation_ratio=4.0,
    t_extrapolation_ratio=1.0, enable_fps_modulation=False,
)
freqs_pt = pos_emb.generate_embeddings(torch.Size([2, 1, 16, 16, 2048]))
freqs_pt = freqs_pt.unsqueeze(1).unsqueeze(0)  # pipeline unsqueeze
torch.manual_seed(42)
q_pt = torch.randn(2, 256, 16, 128, dtype=torch.float32)

# PT reference
out_pt = _p2.apply_rotary_pos_emb(q_pt, freqs_pt)

# Numpy: line-by-line translation of PT
t_np = q_pt.float().cpu().numpy()
f_np = freqs_pt.float().cpu().numpy()
half_D = t_np.shape[-1] // 2
t_shape = t_np.shape
t_ = t_np.reshape(*t_shape[:-1], 2, half_D)
t_ = np.moveaxis(t_, -2, -1)
t_ = np.expand_dims(t_, -2)
t_out = f_np[..., 0] * t_[..., 0] + f_np[..., 1] * t_[..., 1]
t_out = np.moveaxis(t_out, -1, -2)
t_out = t_out.reshape(*t_shape)

err = np.abs(t_out - out_pt.float().cpu().numpy())
print(f"Shapes: t={q_pt.shape}, freqs={freqs_pt.shape}")
print(f"max_err: {err.max():.8f}")
print(f"mean_err: {err.mean():.8f}")
print(f"== 0.0: {(err == 0.0).mean()*100:.1f}%")
print(f"RESULT: {'PASS' if err.max() < 1e-7 else 'FAIL'}")

"""Verify RoPE C++ vs PT vs pure numpy — three-way comparison at real scale."""
import sys, time
sys.path.insert(0, "/sdcard/anima_on_android/src")

import torch, numpy as np, ctypes as ct, predict2 as _p2
import position_embedding

lib = ct.CDLL("/data/local/tmp/libanima_rt.so")
lib.anima_rt_init.restype = ct.c_bool
lib.anima_rt_run_rope.argtypes = [ct.c_void_p]*3 + [ct.c_int]*3
lib.anima_rt_run_rope.restype = ct.c_bool
assert lib.anima_rt_init()

# Real model freqs
pos_emb = position_embedding.VideoRopePosition3DEmb(
    model_channels=2048, len_h=16, len_w=16, len_t=1,
    max_fps=30, min_fps=1, is_learnable=True, interpolation="crop",
    head_dim=128, h_extrapolation_ratio=4.0, w_extrapolation_ratio=4.0,
    t_extrapolation_ratio=1.0, enable_fps_modulation=False,
)
freqs_raw = pos_emb.generate_embeddings(torch.Size([2, 1, 16, 16, 2048]))
freqs_pt = freqs_raw.unsqueeze(1).unsqueeze(0)  # [256] → [1,256,1,64,2,2]
print(f"freqs raw: {freqs_raw.shape}, unsqueezed: {freqs_pt.shape}")

# Test Q (small scale: 1 head, 1 batch for debugging)
torch.manual_seed(42)
q_pt = torch.randn(1, 256, 1, 128, dtype=torch.float32)
B, S, H, D = 1, 256, 1, 128
print(f"q: [{B},{S},{H},{D}]")

# ---- 1. PT reference ----
out_pt = _p2.apply_rotary_pos_emb(q_pt, freqs_pt)

# ---- 2. Pure numpy (independent of C++) ----
q_np = q_pt.float().cpu().numpy()
f_np = freqs_pt.float().cpu().numpy()  # [1,256,1,64,2,2]

# numpy: t.reshape(B,S,H,2,D/2).movedim(-2,-1).unsqueeze(-2)
t_rs = q_np.reshape(B, S, H, 2, D//2)
t_mv = np.moveaxis(t_rs, -2, -1)  # movedim(-2, -1)
t_ = np.expand_dims(t_mv, -2)  # unsqueeze(-2), → [B,S,H,D/2,1,2]

# freqs[...,0] * t_[...,0] + freqs[...,1] * t_[...,1]
f0 = f_np[..., 0]  # [1,256,1,64,2]
f1 = f_np[..., 1]  # [1,256,1,64,2]
t0 = t_[..., 0]    # [B,S,H,64,1]
t1 = t_[..., 1]    # [B,S,H,64,1]

result = f0 * t0 + f1 * t1  # broadcast: [B,S,H,64,2]
out_np_flat = np.moveaxis(result, -1, -2).reshape(B,S,H,D)
print(f"numpy vs PT: max_err={np.abs(out_np_flat - out_pt.float().numpy()).max():.8f}")

# ---- 3. C++ kernel ----
t_flat = np.ascontiguousarray(q_np.reshape(B*H, S, D), dtype=np.float32)
f_flat = np.ascontiguousarray(f_np, dtype=np.float32)
out_cpp = np.zeros((B*H, S, D), dtype=np.float32)
t0_cpp = time.perf_counter()
ok = lib.anima_rt_run_rope(t_flat.ctypes.data, f_flat.ctypes.data, out_cpp.ctypes.data, B*H, S, D)
assert ok
dt_cpp = time.perf_counter() - t0_cpp
out_cpp = out_cpp.reshape(B, S, H, D)

# ---- Compare all three ----
err_cpp_vs_pt = np.abs(out_cpp - out_pt.float().cpu().numpy())
err_cpp_vs_np = np.abs(out_cpp - out_np_flat)

print(f"\nC++ vs PT:  max_err={err_cpp_vs_pt.max():.8f}  mean={err_cpp_vs_pt.mean():.8f}")
print(f"C++ vs numpy: max_err={err_cpp_vs_np.max():.8f}  mean={err_cpp_vs_np.mean():.8f}")
print(f"C++ time: {dt_cpp*1000:.1f}ms")

# Detail: worst mismatch location
if err_cpp_vs_np.max() > 1e-6:
    idx = np.unravel_index(err_cpp_vs_np.argmax(), err_cpp_vs_np.shape)
    print(f"\nWorst C++ vs numpy at {idx}:")
    print(f"  numpy: {out_np_flat[idx]:.10f}")
    print(f"  C++:   {out_cpp[idx]:.10f}")

THRESH = 5e-7
print(f"\nRESULT: {'PASS' if err_cpp_vs_np.max() < THRESH else 'FAIL — C++ != numpy'}")

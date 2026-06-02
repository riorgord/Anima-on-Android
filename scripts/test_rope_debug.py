"""Minimal RoPE debug: single position, check exact values."""
import sys, ctypes as ct, numpy as np, torch
sys.path.insert(0, "/sdcard/anima_on_android/src")
import predict2 as _p2

lib = ct.CDLL("/data/local/tmp/libanima_rt.so")
lib.anima_rt_init.restype = ct.c_bool
lib.anima_rt_run_rope.argtypes = [ct.c_void_p]*3 + [ct.c_int]*3
lib.anima_rt_run_rope.restype = ct.c_bool
assert lib.anima_rt_init()

# Simple test: D=4, pairs=2, half=2
# t: [1, 1, 1, 4], values [a, b, c, d] where [a,b] are even half, [c,d] are odd half
# freqs: [1, 2, 2, 2], values: [cos0, -sin0, sin0, cos0, cos1, -sin1, sin1, cos1]
# cos0=0.5, sin0=0.866 (60 deg), cos1=0.0, sin1=1.0 (90 deg)

D = 4; S = 1; H = 1; B = 1
t_pt = torch.tensor([[[[1.0, 2.0, 3.0, 4.0]]]], dtype=torch.float32)  # [1,1,1,4]
freqs_pt = torch.tensor([[[[0.5, -0.866], [0.866, 0.5]],
                           [[0.0, -1.0], [1.0, 0.0]]]], dtype=torch.float32)  # Not unsqueezed
freqs_pt = freqs_pt.unsqueeze(1).unsqueeze(0)  # → [1,1,1,2,2,2]
print(f"t: {t_pt.numpy()}")
print(f"freqs: {freqs_pt.numpy()}")

out_pt = _p2.apply_rotary_pos_emb(t_pt, freqs_pt)
print(f"PT: {out_pt.numpy()}")

# Manual computation (should match PT):
# t = [1, 2, 3, 4]: x_even[0]=1, x_odd[0]=3, x_even[1]=2, x_odd[1]=4
# p=0: cos=0.5, -sin=-0.866 → out[0] = 0.5*1 + (-0.866)*3 = -2.098
#                      out[2] = 0.866*1 + 0.5*3 = 2.366
# p=1: cos=0.0, -sin=-1.0 → out[1] = 0.0*2 + (-1.0)*4 = -4.0
#                     out[3] = 1.0*2 + 0.0*4 = 2.0
print(f"Expected: [[[[{-2.098:0.3f}, {-4.0:0.3f}, {2.366:0.3f}, {2.0:0.3f}]]]]")

# C++ kernel
t_np = t_pt.float().cpu().contiguous().numpy().reshape(B*H, S, D).astype(np.float32)
f_np = np.ascontiguousarray(freqs_pt.float().cpu().numpy(), dtype=np.float32)
out_buf = np.zeros((B*H, S, D), dtype=np.float32)
ok = lib.anima_rt_run_rope(t_np.ctypes.data, f_np.ctypes.data, out_buf.ctypes.data, B*H, S, D)
assert ok
print(f"CPP: {out_buf}")
print(f"MATCH: {np.allclose(out_buf, out_pt.float().numpy(), atol=1e-6)}")

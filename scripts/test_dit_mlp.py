"""Test MLP path: LN → fc1 → SiLU → fc2"""
import ctypes, numpy as np, torch, sys, time

_lib = ctypes.CDLL("/data/local/tmp/libdit_vk.so")
_lib.dit_init.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
_lib.dit_init.restype = ctypes.c_bool
_lib.dit_record_mlp.argtypes = [ctypes.c_int]
_lib.dit_record_mlp.restype = ctypes.c_bool
_lib.dit_forward.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
    ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]
_lib.dit_forward.restype = ctypes.c_bool

print("dit_init...")
t0 = time.time()
ok = _lib.dit_init(b"/data/local/tmp/diffusion_weights.bin", b"/data/local/tmp")
print(f"  init = {ok} ({time.time()-t0:.1f}s)")
if not ok: sys.exit(1)

print("Recording MLP block 0...")
ok = _lib.dit_record_mlp(0)
print(f"  record = {ok}")

Mv, Dv = 512, 2048
np.random.seed(42)
x = np.random.randn(Mv, Dv).astype(np.float16)
t_emb = np.zeros((2, Dv), dtype=np.float16)
ctx = np.zeros((2, 512, 1024), dtype=np.float16)
out = np.zeros((Mv, Dv), dtype=np.float16)

print("Running MLP...")
t0 = time.time()
ok = _lib.dit_forward(x.ctypes.data_as(ctypes.c_void_p), t_emb.ctypes.data_as(ctypes.c_void_p),
    ctx.ctypes.data_as(ctypes.c_void_p), out.ctypes.data_as(ctypes.c_void_p),
    Mv, Dv, 2, 512, 1024)
print(f"  forward = {ok} ({time.time()-t0:.3f}s)")

print("Computing reference...")
sd = torch.load("/sdcard/anima_on_android/models/diffusion_weights_fp16.pt",
    weights_only=True, map_location="cpu")
w1 = sd["blocks.0.mlp.layer1.weight"]
w2 = sd["blocks.0.mlp.layer2.weight"]
del sd
import torch.nn.functional as F
x_t = torch.from_numpy(x.astype(np.float32))
h = F.layer_norm(x_t, (2048,), weight=None, bias=None, eps=1e-6)
h = F.linear(h, w1.float())
h = F.silu(h)
ref = F.linear(h, w2.float()).half().numpy()

err = np.abs(out.astype(np.float32) - ref.astype(np.float32)).max()
print(f"max_err = {err:.6f}")
print(f"out mean={out.astype(np.float32).mean():.4f} std={out.astype(np.float32).std():.4f}")
print(f"ref mean={ref.astype(np.float32).mean():.4f} std={ref.astype(np.float32).std():.4f}")
print(f"non-zero: {(out != 0).sum()} / {out.size}")
print("PASS" if err < 0.1 else "FAIL")

_lib.dit_destroy()

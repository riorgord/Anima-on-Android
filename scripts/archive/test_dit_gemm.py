"""Test single GEMM dispatch with real DiT weights"""
import ctypes, numpy as np, torch, sys, time

_lib = ctypes.CDLL("/data/local/tmp/libdit_vk.so")
_lib.dit_init.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
_lib.dit_init.restype = ctypes.c_bool
_lib.dit_record_gemm_test.argtypes = [ctypes.c_char_p, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32]
_lib.dit_record_gemm_test.restype = ctypes.c_bool
_lib.dit_forward.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
    ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]
_lib.dit_forward.restype = ctypes.c_bool
_lib.dit_destroy.argtypes = []
_lib.dit_destroy.restype = None

# Load with real weights
print("dit_init with weights (3.9GB, takes ~30s to memcpy)...")
t0 = time.time()
ok = _lib.dit_init(b"/data/local/tmp/diffusion_weights.bin", b"/data/local/tmp")
print(f"  init = {ok} ({time.time()-t0:.1f}s)")
if not ok:
    # Try weightless
    print("Retrying weightless...")
    ok = _lib.dit_init(b"", b"/data/local/tmp")
    print(f"  init = {ok}")

if not ok:
    print("FAILED"); sys.exit(1)

# Test GEMM: q_proj weight [2048, 2048], input [512, 2048] → output [512, 2048]
Mv, Nv, Kv = 512, 2048, 2048
wname = b"blocks.0.self_attn.q_proj.weight"

print(f"Recording GEMM: {wname.decode()} M={Mv} N={Nv} K={Kv}")
ok = _lib.dit_record_gemm_test(wname, Mv, Nv, Kv)
print(f"  record = {ok}")

# Prepare test data
np.random.seed(42)
x = np.random.randn(Mv, Kv).astype(np.float16)
t_emb = np.zeros((2, 2048), dtype=np.float16)
ctx = np.zeros((2, 512, 1024), dtype=np.float16)
out = np.zeros((Mv, Nv), dtype=np.float16)

print("Running Vulkan GEMM...")
t0 = time.time()
ok = _lib.dit_forward(
    x.ctypes.data_as(ctypes.c_void_p),
    t_emb.ctypes.data_as(ctypes.c_void_p),
    ctx.ctypes.data_as(ctypes.c_void_p),
    out.ctypes.data_as(ctypes.c_void_p),
    Mv, Kv, 2, 512, 1024)
print(f"  forward = {ok} ({time.time()-t0:.3f}s)")

# CPU reference
print("Computing CPU reference...")
import torch.nn.functional as F
w_t = torch.load("/sdcard/anima_on_android/models/diffusion_weights_fp16.pt",
    weights_only=True, map_location="cpu")
w = w_t["blocks.0.self_attn.q_proj.weight"]
del w_t
x_t = torch.from_numpy(x.astype(np.float32))
ref = F.linear(x_t, w.float()).half().numpy()

err = np.abs(out.astype(np.float32) - ref.astype(np.float32)).max()
print(f"max_err = {err:.6f}")
print(f"out mean={out.astype(np.float32).mean():.4f} std={out.astype(np.float32).std():.4f}")
print(f"ref mean={ref.astype(np.float32).mean():.4f} std={ref.astype(np.float32).std():.4f}")
print(f"non-zero: {(out != 0).sum()} / {out.size}")

_lib.dit_destroy()
print("DONE")

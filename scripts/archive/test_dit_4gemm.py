"""Test 4 sequential GEMM dispatches (Q/K/V/O proj) with real weights"""
import ctypes, numpy as np, torch, sys, time

_lib = ctypes.CDLL("/data/local/tmp/libdit_vk.so")
_lib.dit_init.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
_lib.dit_init.restype = ctypes.c_bool
_lib.dit_record_self_attn.argtypes = [ctypes.c_int]
_lib.dit_record_self_attn.restype = ctypes.c_bool
_lib.dit_forward.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
    ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]
_lib.dit_forward.restype = ctypes.c_bool
_lib.dit_destroy.argtypes = []
_lib.dit_destroy.restype = None

print("dit_init with weights...")
t0 = time.time()
ok = _lib.dit_init(b"/data/local/tmp/diffusion_weights.bin", b"/data/local/tmp")
print(f"  init = {ok} ({time.time()-t0:.1f}s)")
if not ok: sys.exit(1)

# Record 4-GEMM sequence (Q/K/V/O proj for block 0)
print("Recording 4-GEMM block 0...")
ok = _lib.dit_record_self_attn(0)
print(f"  record = {ok}")
if not ok: sys.exit(1)

# Test data
Mv, Dv = 512, 2048
np.random.seed(42)
x = np.random.randn(Mv, Dv).astype(np.float16)
t_emb = np.zeros((2, Dv), dtype=np.float16)
ctx = np.zeros((2, 512, 1024), dtype=np.float16)
out = np.zeros((Mv, Dv), dtype=np.float16)

print("Running Vulkan 4-GEMM...")
t0 = time.time()
ok = _lib.dit_forward(x.ctypes.data_as(ctypes.c_void_p), t_emb.ctypes.data_as(ctypes.c_void_p),
    ctx.ctypes.data_as(ctypes.c_void_p), out.ctypes.data_as(ctypes.c_void_p),
    Mv, Dv, 2, 512, 1024)
print(f"  forward = {ok} ({time.time()-t0:.3f}s)")

# CPU reference: LN(x) → V_proj → O_proj
# (engine does LN → Q/K/V → Q/K norms → V → O)
print("Computing CPU reference...")
import torch.nn.functional as F
sd = torch.load("/sdcard/anima_on_android/models/diffusion_weights_fp16.pt",
    weights_only=True, map_location="cpu")
w_v = sd["blocks.0.self_attn.v_proj.weight"]
w_o = sd["blocks.0.self_attn.output_proj.weight"]
del sd
x_t = torch.from_numpy(x.astype(np.float32))
ln_x = F.layer_norm(x_t, (2048,), weight=None, bias=None, eps=1e-6)
v = F.linear(ln_x, w_v.float())
ref = F.linear(v, w_o.float()).half().numpy()

err = np.abs(out.astype(np.float32) - ref.astype(np.float32)).max()
print(f"max_err = {err:.6f}  (4-GEMM barrier chain, output=O_proj)")
print(f"out mean={out.astype(np.float32).mean():.4f} std={out.astype(np.float32).std():.4f}")
print(f"ref mean={ref.astype(np.float32).mean():.4f} std={ref.astype(np.float32).std():.4f}")
print(f"non-zero: {(out != 0).sum()} / {out.size}")
print("PASS" if err < 0.1 else "FAIL (barrier issue?)")

_lib.dit_destroy()
print("DONE")

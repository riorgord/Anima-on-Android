"""Phase 2: Load dumped data, run C++ engine, compare with reference."""
import ctypes, numpy as np, time, sys, os

OUT="/sdcard/anima_on_android/output/ref_dump"

print("Loading dumped data...")
x_in = np.load(f"{OUT}/x_in.npy")
t_emb = np.load(f"{OUT}/t_emb.npy")
ref_out = np.load(f"{OUT}/ref_out.npy")
adaln_all = np.fromfile(f"{OUT}/adaln_all.bin", dtype=np.uint16)
MS,D = x_in.shape; M = t_emb.shape[0]
print(f"  x: {x_in.shape}  t_emb: {t_emb.shape}  ref: {ref_out.shape}  adaln: {adaln_all.nbytes/1e6:.0f}MB")

print("Loading C++ engine...")
_lib=ctypes.CDLL("/data/local/tmp/libdit_vk.so")
_lib.dit_init.argtypes=[ctypes.c_char_p,ctypes.c_char_p]; _lib.dit_init.restype=ctypes.c_bool
_lib.dit_init_all_blocks.argtypes=[]; _lib.dit_init_all_blocks.restype=ctypes.c_bool
_lib.dit_forward_28blocks.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_void_p,
    ctypes.c_int,ctypes.c_int,ctypes.c_int]; _lib.dit_forward_28blocks.restype=ctypes.c_bool

t0=time.time()
ok=_lib.dit_init(b"/data/local/tmp/diffusion_weights.bin",b"/data/local/tmp")
print(f"  init={ok} ({time.time()-t0:.1f}s)")
if not ok: sys.exit(1)

ok=_lib.dit_init_all_blocks()
print(f"  record={ok}")

print("Running C++ engine...")
out_np = np.zeros((MS,D), dtype=np.float16)
t0=time.time()
ok=_lib.dit_forward_28blocks(x_in.ctypes.data_as(ctypes.c_void_p),
    adaln_all.ctypes.data_as(ctypes.c_void_p),
    out_np.ctypes.data_as(ctypes.c_void_p), MS, D, M)
dt=time.time()-t0
print(f"  forward={ok} ({dt:.3f}s)")

# Compare
err = np.abs(out_np.astype(np.float32) - ref_out.astype(np.float32)).max()
print(f"max_err = {err:.5f}")
print(f"C++ mean/std = {out_np.astype(np.float32).mean():.4f}/{out_np.astype(np.float32).std():.4f}")
print(f"REF mean/std = {ref_out.astype(np.float32).mean():.4f}/{ref_out.astype(np.float32).std():.4f}")
nz = (out_np != 0).sum()
print(f"non-zero: {nz}/{out_np.size} ({100*nz/out_np.size:.1f}%)")
print("PASS" if err < 10 else "FAIL")
_lib.dit_destroy()

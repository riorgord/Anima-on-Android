"""Bisect test: pre-record vs per-step (skip-attn vs full)."""
import ctypes as ct, numpy as np, time
lib = ct.CDLL("/data/local/tmp/libdit_vk.so")
lib.dit_init_adaln_only.argtypes = [ct.c_char_p, ct.c_char_p]
lib.dit_init_adaln_only.restype = ct.c_bool
lib.dit_forward_nblocks.argtypes = [ct.c_void_p]*4 + [ct.c_int]*6
lib.dit_forward_nblocks.restype = ct.c_bool
lib.dit_forward_step.argtypes = [ct.c_void_p]*4 + [ct.c_int]*6  # + skip_attn
lib.dit_forward_step.restype = ct.c_bool
lib.dit_destroy.argtypes = []
lib.dit_destroy.restype = None

MS, M, D, Nctx, CtxD = 512, 2, 2048, 512, 1024
rng = np.random.RandomState(42)
x = (rng.randn(MS*D).astype(np.float16)*0.1).view(np.uint16)
t = (rng.randn(M*D).astype(np.float16)*0.1).view(np.uint16)
c = (rng.randn(M*Nctx*CtxD).astype(np.float16)*0.1).view(np.uint16)
o = np.zeros(MS*D, dtype=np.uint16)

def data_ptr(arr): return arr.ctypes.data_as(ct.c_void_p)

# Test 1: per-step with skip_attn=1 (AdaLN + MLP only, no attention)
print("Test 1: per-step skip-attn...")
lib.dit_init_adaln_only(b"/data/local/tmp/diffusion_weights.bin", b"/data/local/tmp")
t0 = time.time()
ok = lib.dit_forward_step(data_ptr(x), data_ptr(t), data_ptr(c), data_ptr(o),
                           MS, D, M, Nctx, CtxD, 1)  # skip_attn=1
dt = time.time() - t0
o_f = o.view(np.float16)
nans = int(np.sum(np.isnan(o_f)))
ok_f = np.isfinite(o_f)
vmin = float(o_f[ok_f].min()) if ok_f.any() else 0
vmax = float(o_f[ok_f].max()) if ok_f.any() else 0
print(f"  skip-attn: {'OK' if ok else 'FAIL'} ({dt:.0f}s) min={vmin:+.1f} max={vmax:+.1f} nan={nans}")
lib.dit_destroy()

# Test 2: per-step with skip_attn=0 (full attention)
print("Test 2: per-step full attention...")
lib.dit_init_adaln_only(b"/data/local/tmp/diffusion_weights.bin", b"/data/local/tmp")
t0 = time.time()
ok = lib.dit_forward_step(data_ptr(x), data_ptr(t), data_ptr(c), data_ptr(o),
                           MS, D, M, Nctx, CtxD, 0)  # skip_attn=0
dt = time.time() - t0
o_f = o.view(np.float16)
nans = int(np.sum(np.isnan(o_f)))
ok_f = np.isfinite(o_f)
vmin = float(o_f[ok_f].min()) if ok_f.any() else 0
vmax = float(o_f[ok_f].max()) if ok_f.any() else 0
print(f"  full attn: {'OK' if ok else 'FAIL'} ({dt:.0f}s) min={vmin:+.1f} max={vmax:+.1f} nan={nans}")
lib.dit_destroy()

print("Done.")

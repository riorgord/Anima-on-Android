"""Test: self-only, cross-only, full attention."""
import ctypes as ct, numpy as np
lib = ct.CDLL("/data/local/tmp/libdit_vk.so")
lib.dit_init_adaln_only.argtypes = [ct.c_char_p, ct.c_char_p]
lib.dit_init_adaln_only.restype = ct.c_bool
lib.dit_forward_step.argtypes = [ct.c_void_p]*4 + [ct.c_int]*6
lib.dit_forward_step.restype = ct.c_bool
lib.dit_destroy.argtypes = []
lib.dit_destroy.restype = None

MS, M, D, Nctx, CtxD = 512, 2, 2048, 512, 1024
rng = np.random.RandomState(42)
x = (rng.randn(MS*D).astype(np.float16)*0.1).view(np.uint16)
t = (rng.randn(M*D).astype(np.float16)*0.1).view(np.uint16)
c = (rng.randn(M*Nctx*CtxD).astype(np.float16)*0.1).view(np.uint16)
o = np.zeros(MS*D, dtype=np.uint16)
dp = lambda a: a.ctypes.data_as(ct.c_void_p)

for mode, label in [(0, "full"), (2, "self only"), (3, "cross only")]:
    lib.dit_init_adaln_only(b"/data/local/tmp/diffusion_weights.bin", b"/data/local/tmp")
    ok = lib.dit_forward_step(dp(x), dp(t), dp(c), dp(o), MS, D, M, Nctx, CtxD, mode)
    of = o.view(np.float16)
    nans = int(np.sum(np.isnan(of)))
    ok_f = np.isfinite(of)
    vmin = float(of[ok_f].min()) if ok_f.any() else 0
    vmax = float(of[ok_f].max()) if ok_f.any() else 0
    print(f"  {label}: {'OK' if ok else 'FAIL'} nan={nans} min={vmin:+.1f} max={vmax:+.1f}")
    lib.dit_destroy()

print("Done.")

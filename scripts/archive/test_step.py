"""Test per-step recording: dit_forward_step (112 submit/step, TDR-safe at all freqs)."""
import ctypes, numpy as np, time

lib = ctypes.CDLL("/data/local/tmp/libdit_vk.so")

lib.dit_init_adaln_only.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
lib.dit_init_adaln_only.restype = ctypes.c_bool
lib.dit_forward_step.argtypes = [
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
    ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ctypes.c_int  # mode
]
lib.dit_forward_step.restype = ctypes.c_bool
lib.dit_destroy.argtypes = []
lib.dit_destroy.restype = None

M, D, S = 2, 2048, 256
MS = M * S
Nctx, CtxD = 512, 1024

def make_data():
    rng = np.random.RandomState(42)
    x = (rng.randn(MS * D).astype(np.float16) * 0.1).view(np.uint16)
    t = (rng.randn(M * D).astype(np.float16) * 0.1).view(np.uint16)
    c = (rng.randn(M * Nctx * CtxD).astype(np.float16) * 0.1).view(np.uint16)
    o = np.zeros(MS * D, dtype=np.uint16)
    return x, t, c, o

def scan(o):
    f = o.view(np.float16)
    ok = np.isfinite(f)
    return (float(f[ok].min()) if ok.any() else 0,
            float(f[ok].max()) if ok.any() else 0,
            int(np.sum(np.isnan(f))))

print("=== Per-Step Recording Test ===")
print()

print("Init engine...")
t0 = time.time()
ok = lib.dit_init_adaln_only(
    b"/data/local/tmp/diffusion_weights.bin",
    b"/data/local/tmp")
print(f"  Init: {'OK' if ok else 'FAIL'} ({time.time()-t0:.0f}s)")
assert ok

# ── 1 step (28 blocks, 112 submits) ──
print()
print("Running 1 step (28 blocks, 112 submits)...")
x, t, c, o = make_data()
t0 = time.time()
ok = lib.dit_forward_step(
    x.ctypes.data_as(ctypes.c_void_p),
    t.ctypes.data_as(ctypes.c_void_p),
    c.ctypes.data_as(ctypes.c_void_p),
    o.ctypes.data_as(ctypes.c_void_p),
    MS, D, M, Nctx, CtxD, 0)
dt = time.time() - t0
vmin, vmax, nans = scan(o)
print(f"  Step: {'OK' if ok else 'FAIL'} ({dt:.0f}s)  min={vmin:+.2f} max={vmax:+.2f} nan={nans}")

# ── 3 steps ──
if ok:
    print()
    print("Running 3 steps...")
    for step in range(3):
        x, t, c, o = make_data()
        t0 = time.time()
        ok = lib.dit_forward_step(
            x.ctypes.data_as(ctypes.c_void_p),
            t.ctypes.data_as(ctypes.c_void_p),
            c.ctypes.data_as(ctypes.c_void_p),
            o.ctypes.data_as(ctypes.c_void_p),
            MS, D, M, Nctx, CtxD, 0)
        dt = time.time() - t0
        vmin, vmax, nans = scan(o)
        print(f"  Step {step+1}/3: {'OK' if ok else 'FAIL'} ({dt:.0f}s)  "
              f"min={vmin:+.2f} max={vmax:+.2f} nan={nans}")
        if not ok: break

print()
print("=== Done ===")
lib.dit_destroy()

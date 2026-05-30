"""Test merged attention: fresh init per test to avoid descPool leak from re-recording.

Tests: 1, 2, 4, 8, 16, 28 blocks with merged real attention.
Each test: init fresh → record N blocks → forward 1 step → destroy.
"""

import ctypes, numpy as np, time

lib = ctypes.CDLL("/data/local/tmp/libdit_vk.so")

lib.dit_set_skip_attn_precord.argtypes = []
lib.dit_set_skip_attn_precord.restype = None
lib.dit_init_adaln_only.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
lib.dit_init_adaln_only.restype = ctypes.c_bool
lib.dit_record_blocks_with_attn.argtypes = [ctypes.c_int]
lib.dit_record_blocks_with_attn.restype = ctypes.c_bool
lib.dit_forward_merged.argtypes = [
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
    ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ctypes.c_int
]
lib.dit_forward_merged.restype = ctypes.c_bool
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

def scan_output(o):
    f = o.view(np.float16)
    ok = np.isfinite(f)
    return float(f[ok].min()) if ok.any() else 0, float(f[ok].max()) if ok.any() else 0, int(np.sum(np.isnan(f)))

def test_n(n):
    """Fresh init, record n blocks with real attn, forward 1 step."""
    lib.dit_set_skip_attn_precord()
    ok = lib.dit_init_adaln_only(
        b"/data/local/tmp/diffusion_weights.bin",
        b"/data/local/tmp")
    if not ok:
        return (0, 0, 0, 0), "init failed"

    ok = lib.dit_record_blocks_with_attn(n)
    if not ok:
        return (0, 0, 0, 0), "record failed"

    x, t, c, o = make_data()
    t0 = time.time()
    ok = lib.dit_forward_merged(
        x.ctypes.data_as(ctypes.c_void_p),
        t.ctypes.data_as(ctypes.c_void_p),
        c.ctypes.data_as(ctypes.c_void_p),
        o.ctypes.data_as(ctypes.c_void_p),
        MS, D, M, Nctx, CtxD, n)
    dt = time.time() - t0

    vmin, vmax, nans = scan_output(o)
    lib.dit_destroy()

    if not ok:
        return (dt, vmin, vmax, nans), "submit failed"
    return (dt, vmin, vmax, nans), None

# ── Run ──
print("=== Merged Attention: fresh init per test ===")
print(f"{'Blocks':>6s}  {'Status':>12s}  {'Time':>6s}  {'min':>8s}  {'max':>8s}  {'NaN'}")
print("-" * 60)

for n_blocks in [1, 2, 4, 8, 16, 28]:
    (dt, vmin, vmax, nans), err = test_n(n_blocks)
    if err:
        print(f"  {n_blocks:>4d}  {err:>12s}")
        if "submit failed" in err:
            # check logcat for VkResult
            break  # stop on first failure
    else:
        print(f"  {n_blocks:>4d}  {'OK':>12s}  {dt:>5.1f}s  {vmin:>+8.2f}  {vmax:>+8.2f}  {nans}")

print()
print("Done.")

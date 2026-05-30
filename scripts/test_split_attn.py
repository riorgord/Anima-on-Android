"""Test split architecture: cmd[i]=part1, cmd_attn[i]=part2.

Expected: 28 cmd[i] (50 dispatch) + 28 cmd_attn[i] (22 dispatch) per step.
Total 72 dispatch per block, split across 2 submits.
"""

import ctypes, numpy as np, time

lib = ctypes.CDLL("/data/local/tmp/libdit_vk.so")

lib.dit_init_adaln_only.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
lib.dit_init_adaln_only.restype = ctypes.c_bool
lib.dit_forward_nblocks.argtypes = [
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
    ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ctypes.c_int
]
lib.dit_forward_nblocks.restype = ctypes.c_bool
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

print("=== Split Architecture Test ===")
print()

# ── Init ──
print("Init engine (pre-record 28 cmd + 28 cmd_attn)...")
t0 = time.time()
ok = lib.dit_init_adaln_only(
    b"/data/local/tmp/diffusion_weights.bin",
    b"/data/local/tmp")
dt = time.time() - t0
print(f"  Init: {'OK' if ok else 'FAIL'} ({dt:.0f}s)")
assert ok, "init failed"

# ── Phase 1: 1 block ──
print()
print("Phase 1: Forward 1 block...")
x, t, c, o = make_data()
t0 = time.time()
ok = lib.dit_forward_nblocks(x.ctypes.data_as(ctypes.c_void_p),
    t.ctypes.data_as(ctypes.c_void_p),
    c.ctypes.data_as(ctypes.c_void_p),
    o.ctypes.data_as(ctypes.c_void_p),
    MS, D, M, Nctx, CtxD, 1)
dt = time.time() - t0
vmin, vmax, nans = scan(o)
print(f"  1 block: {'OK' if ok else 'FAIL'} ({dt:.1f}s)  min={vmin:+.2f} max={vmax:+.2f} nan={nans}")
assert ok, "1 block failed"
print("  Phase 1 PASSED")

# ── Phase 2: 2 blocks ──
print()
print("Phase 2: Forward 2 blocks...")
x, t, c, o = make_data()
t0 = time.time()
ok = lib.dit_forward_nblocks(x.ctypes.data_as(ctypes.c_void_p),
    t.ctypes.data_as(ctypes.c_void_p),
    c.ctypes.data_as(ctypes.c_void_p),
    o.ctypes.data_as(ctypes.c_void_p),
    MS, D, M, Nctx, CtxD, 2)
dt = time.time() - t0
vmin, vmax, nans = scan(o)
print(f"  2 blocks: {'OK' if ok else 'FAIL'} ({dt:.1f}s)  min={vmin:+.2f} max={vmax:+.2f} nan={nans}")
assert ok, "2 blocks failed"
print("  Phase 2 PASSED")

# ── Phase 3: 28 blocks × 3 steps ──
print()
print("Phase 3: Forward 28 blocks × 3 steps...")
for step in range(3):
    x, t, c, o = make_data()
    t0 = time.time()
    ok = lib.dit_forward_nblocks(x.ctypes.data_as(ctypes.c_void_p),
        t.ctypes.data_as(ctypes.c_void_p),
        c.ctypes.data_as(ctypes.c_void_p),
        o.ctypes.data_as(ctypes.c_void_p),
        MS, D, M, Nctx, CtxD, 28)
    dt = time.time() - t0
    vmin, vmax, nans = scan(o)
    print(f"  Step {step+1}/3: {'OK' if ok else 'FAIL'} ({dt:.1f}s)  "
          f"min={vmin:+.2f} max={vmax:+.2f} nan={nans}")
    assert ok, f"step {step+1} failed"

print()
print("=== All phases PASSED ===")
lib.dit_destroy()

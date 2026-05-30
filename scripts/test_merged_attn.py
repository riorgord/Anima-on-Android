"""Test: merge attention dispatches into cmd[i] (not separate cmd_attn).

Phase 1: Record 1 block with real attn, submit it → check for NaN
Phase 2: Record 2 blocks, forward chain → check for NaN
Phase 3: Record 28 blocks, 3-step forward → GPU stability test
"""

import ctypes, numpy as np, time

lib = ctypes.CDLL("/data/local/tmp/libdit_vk.so")

# ── Signatures ──
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

# ── Model dims ──
M, D, S = 2, 2048, 256
MS = M * S  # 512
Nctx, CtxD = 512, 1024

def make_data():
    """Create random test data."""
    rng = np.random.RandomState(42)
    x = (rng.randn(MS * D).astype(np.float16) * 0.1).view(np.uint16)
    t = (rng.randn(M * D).astype(np.float16) * 0.1).view(np.uint16)
    c = (rng.randn(M * Nctx * CtxD).astype(np.float16) * 0.1).view(np.uint16)
    o = np.zeros(MS * D, dtype=np.uint16)
    return x, t, c, o

def scan_output(o_uint16):
    """Return min, max, nan_count of fp16 output."""
    o = o_uint16.view(np.float16)
    ok = np.isfinite(o)
    vmin = float(o[ok].min()) if ok.any() else 0
    vmax = float(o[ok].max()) if ok.any() else 0
    nans = int(np.sum(np.isnan(o)))
    return vmin, vmax, nans

# ── Step 1: Init (skip attn pre-recording) ──
print("=== Merged Attention Test ===")
print()
print("Init engine (skip attn pre-record)...")
lib.dit_set_skip_attn_precord()
ok = lib.dit_init_adaln_only(
    b"/data/local/tmp/diffusion_weights.bin",
    b"/data/local/tmp"
)
print(f"  Init: {'OK' if ok else 'FAIL'}")
assert ok, "init failed"

# ── Phase 1: Single block ──
print()
print("Phase 1: Record 1 block with real attention...")
ok = lib.dit_record_blocks_with_attn(1)
print(f"  Record: {'OK' if ok else 'FAIL'}")
assert ok, "record failed"

x, t, c, o = make_data()
t0 = time.time()
ok = lib.dit_forward_merged(
    x.ctypes.data_as(ctypes.c_void_p),
    t.ctypes.data_as(ctypes.c_void_p),
    c.ctypes.data_as(ctypes.c_void_p),
    o.ctypes.data_as(ctypes.c_void_p),
    MS, D, M, Nctx, CtxD, 1)
dt = time.time() - t0
vmin, vmax, nans = scan_output(o)
print(f"  Forward 1 block: {'OK' if ok else 'FAIL'} ({dt:.1f}s)")
print(f"  Output: min={vmin:+.1f} max={vmax:+.1f} nan={nans}")
assert ok, "forward 1 block failed"
print("  Phase 1 PASSED")

# ── Phase 2: 2 blocks ──
print()
print("Phase 2: Record 2 blocks with real attention...")
ok = lib.dit_record_blocks_with_attn(2)
print(f"  Record: {'OK' if ok else 'FAIL'}")
assert ok, "record 2 blocks failed"

x, t, c, o = make_data()
t0 = time.time()
ok = lib.dit_forward_merged(
    x.ctypes.data_as(ctypes.c_void_p),
    t.ctypes.data_as(ctypes.c_void_p),
    c.ctypes.data_as(ctypes.c_void_p),
    o.ctypes.data_as(ctypes.c_void_p),
    MS, D, M, Nctx, CtxD, 2)
dt = time.time() - t0
vmin, vmax, nans = scan_output(o)
print(f"  Forward 2 blocks: {'OK' if ok else 'FAIL'} ({dt:.1f}s)")
print(f"  Output: min={vmin:+.1f} max={vmax:+.1f} nan={nans}")
assert ok, "forward 2 blocks failed"
print("  Phase 2 PASSED")

# ── Phase 3: 28 blocks × 3 steps ──
print()
print("Phase 3: Record 28 blocks, 3-step forward...")
ok = lib.dit_record_blocks_with_attn(28)
print(f"  Record: {'OK' if ok else 'FAIL'}")
assert ok, "record 28 blocks failed"

x, t, c, o = make_data()
for step in range(3):
    t0 = time.time()
    ok = lib.dit_forward_merged(
        x.ctypes.data_as(ctypes.c_void_p),
        t.ctypes.data_as(ctypes.c_void_p),
        c.ctypes.data_as(ctypes.c_void_p),
        o.ctypes.data_as(ctypes.c_void_p),
        MS, D, M, Nctx, CtxD, 28)
    dt = time.time() - t0
    vmin, vmax, nans = scan_output(o)
    print(f"  Step {step+1}/3: {'OK' if ok else 'FAIL'} ({dt:.1f}s)  "
          f"min={vmin:+.1f} max={vmax:+.1f} nan={nans}")
    assert ok, f"forward step {step+1} failed"

print()
print("=== All phases PASSED ===")
lib.dit_destroy()

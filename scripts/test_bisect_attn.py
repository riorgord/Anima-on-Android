"""Bisect the Adreno descriptor set threshold for pre-recorded attention blocks.

Strategy:
  Binary search over [1, 28] attention blocks to find the maximum N
  where pre-recording N cmd_attn (24 descriptor sets each) still allows
  cmd[0] submit to succeed.

  Init with skip_attn_precord flag → pre-record only block compute (cmd[0..27])
  Then test increasing N via dit_test_attn_precord(N) which:
    1. Resets descPool2
    2. Records N attention blocks into cmd_attn[0..N-1] (N*24 sets)
    3. Submits cmd[0] to check GPU state validity
    4. Returns true/false

Each test iteration reuses the same init (no re-init between bisect steps).
descPool2 is reset between iterations via vkResetDescriptorPool.
"""

import ctypes, sys

lib = ctypes.CDLL("/data/local/tmp/libdit_vk.so")

# ── Function signatures ──
lib.dit_set_skip_attn_precord.argtypes = []
lib.dit_set_skip_attn_precord.restype = None

lib.dit_init_adaln_only.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
lib.dit_init_adaln_only.restype = ctypes.c_bool

lib.dit_test_attn_precord.argtypes = [ctypes.c_int]
lib.dit_test_attn_precord.restype = ctypes.c_bool

lib.dit_destroy.argtypes = []
lib.dit_destroy.restype = None

# ── Init without attention pre-recording ──
print("=== Phase 0: Bisect Adreno descriptor set threshold ===")
print()
print("Step 1: Init engine (skip attention pre-record)...")
lib.dit_set_skip_attn_precord()
ok = lib.dit_init_adaln_only(
    b"/data/local/tmp/diffusion_weights.bin",
    b"/data/local/tmp"
)
if not ok:
    print("FATAL: init failed")
    sys.exit(1)
print("  Init OK (28 blocks pre-recorded, 0 attention blocks)")

# ── Binary search ──
print()
print("Step 2: Binary search over [1, 28] attention blocks...")
print(f"{'Test':>5s}  {'N':>5s}  {'Sets':>6s}  {'Result':>8s}  {'Note'}")
print("-" * 50)

lo, hi = 1, 28
best = 0  # highest N that works
tests = []

while lo <= hi:
    mid = (lo + hi) // 2
    sets = mid * 24  # 3 passes × 8 batches = 24 descriptor sets per block

    ok = lib.dit_test_attn_precord(mid)
    status = "OK" if ok else "FAIL"
    note = ""

    if ok:
        best = mid
        lo = mid + 1
    else:
        hi = mid - 1

    txt = f"BISECT" if lo <= hi else f"FINAL"
    print(f"{txt:>5s}  {mid:>5d}  {sets:>6d}  {status:>8s}  {note}")
    tests.append((mid, sets, ok))

# ── Summary ──
print()
print("=" * 50)
if best == 0:
    print("RESULT: Even 1 attention block (24 sets) fails!")
    print("  → Cannot pre-record ANY attention with current block count")
    print("  → Need per-step recording or reduce block descriptor sets")
elif best == 28:
    print("RESULT: All 28 attention blocks (672 sets) work!")
    print("  → The previous failure was a transient issue, retry full pipeline")
else:
    max_sets = best * 24
    total_sets = 1764 + max_sets  # 1764 = 28 blocks × 63 dispatches
    print(f"RESULT: Threshold = {best} attention blocks = {max_sets} sets")
    print(f"  Total descPool + descPool2 sets that work: {total_sets}")
    print(f"  Failed at: {best + 1} blocks = {(best + 1) * 24} sets")
    blocks_per_group = best
    groups_needed = (28 + blocks_per_group - 1) // blocks_per_group
    print(f"  → Pre-record attention in groups of {blocks_per_group}")
    print(f"  → Need {groups_needed} groups × {blocks_per_group} blocks")

# ── Cleanup ──
lib.dit_destroy()
print()
print("Done. lib destroyed.")

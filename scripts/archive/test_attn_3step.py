"""Test: pre-record all 28 cmd_attn at init, then run 3-step forward loop.

Verifies that 56 submit/step (28 cmd + 28 cmd_attn) works across all steps.
Uses random dummy data — correctness not expected, just GPU stability.
"""

import ctypes, numpy as np

lib = ctypes.CDLL("/data/local/tmp/libdit_vk.so")

# ── Function signatures ──
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

# ── Model dims ──
M = 2      # batch (cond+uncond)
D = 2048   # hidden dim
S = 256    # spatial tokens
MS = M * S  # 512
Nctx = 512  # cross-attn context tokens
CtxD = 1024
N_BLOCKS = 28

# ── Init: pre-records 28 cmd + 28 cmd_attn ──
print("=== 3-step pipeline smoke test (with attention pre-recording) ===")
print()
print("Step 0: Init engine (pre-record all 28 blocks + 28 cmd_attn)...")
ok = lib.dit_init_adaln_only(
    b"/data/local/tmp/diffusion_weights.bin",
    b"/data/local/tmp"
)
print(f"  Init: {'OK' if ok else 'FAIL'}")
if not ok:
    print("FATAL: init failed")
    import sys; sys.exit(1)

# ── Dummy test data ──
rng = np.random.RandomState(42)
x_data = rng.randn(MS * D).astype(np.float16) * 0.1
t_emb = rng.randn(M * D).astype(np.float16) * 0.1
ctx_data = rng.randn(M * Nctx * CtxD).astype(np.float16) * 0.1
out_data = np.zeros(MS * D, dtype=np.uint16)

x_ptr = x_data.view(np.uint16).ctypes.data_as(ctypes.c_void_p)
t_ptr = t_emb.view(np.uint16).ctypes.data_as(ctypes.c_void_p)
c_ptr = ctx_data.view(np.uint16).ctypes.data_as(ctypes.c_void_p)
o_ptr = out_data.ctypes.data_as(ctypes.c_void_p)

# ── Run 3 steps ──
print()
print("Step 1-3: Forward loop (56 submit/step)...")
import time

for step in range(3):
    t0 = time.time()
    ok = lib.dit_forward_nblocks(
        x_ptr, t_ptr, c_ptr, o_ptr,
        MS, D, M, Nctx, CtxD, N_BLOCKS
    )
    dt = time.time() - t0
    status = "OK" if ok else "FAIL"
    print(f"  Step {step+1}/3: {status}  ({dt:.1f}s)")

    if not ok:
        print(f"\nFATAL: Forward step {step+1} failed!")
        break

    # Quick NaN scan
    out = out_data.view(np.float16)
    nan_cnt = np.sum(np.isnan(out))
    inf_cnt = np.sum(np.isinf(out))
    vmin, vmax = out[np.isfinite(out)].min(), out[np.isfinite(out)].max()
    print(f"         output: min={float(vmin):+.2f} max={float(vmax):+.2f} nan={nan_cnt} inf={inf_cnt}")

print()
print("=== Done ===")
lib.dit_destroy()

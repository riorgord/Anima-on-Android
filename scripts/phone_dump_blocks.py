"""Phone-side: run C++ engine and dump per-block outputs for comparison.

MUST match PC reference inputs (same seed, same sigma, same MS/D).
Uses dit_forward_step with mode=0 (full self+cross attention).
Saves each block's output to separate .npy file.
"""
import ctypes as ct, numpy as np, os, time

lib = ct.CDLL("/data/local/tmp/libdit_vk.so")

# ── Bindings ──
lib.dit_set_skip_attn_precord.argtypes = []; lib.dit_set_skip_attn_precord.restype = None
lib.dit_init_adaln_only.argtypes = [ct.c_char_p, ct.c_char_p]; lib.dit_init_adaln_only.restype = ct.c_bool
lib.dit_compute_timestep.argtypes = [ct.c_float]; lib.dit_compute_timestep.restype = ct.c_bool
lib.dit_forward_step.argtypes = [ct.c_void_p]*4 + [ct.c_int]*6; lib.dit_forward_step.restype = ct.c_bool
lib.dit_get_block_output.argtypes = [ct.c_int, ct.c_void_p, ct.c_size_t]; lib.dit_get_block_output.restype = ct.c_bool
lib.dit_get_b0_intermediate.argtypes = [ct.c_int, ct.c_void_p, ct.c_size_t]; lib.dit_get_b0_intermediate.restype = ct.c_bool
lib.dit_get_timestep_output.argtypes = [ct.c_void_p, ct.c_size_t, ct.c_void_p, ct.c_size_t]
lib.dit_get_timestep_output.restype = ct.c_bool
lib.dit_destroy.argtypes = []; lib.dit_destroy.restype = None

# ── Constants (matching PC reference) ──
M, S, D = 2, 256, 2048
MS = M * S
Nctx, CtxD = 512, 1024
N_HEADS, HEAD_DIM = 16, 128
SEED = 12345

OUTDIR = "/sdcard/anima_on_android/output/cmp"
os.makedirs(OUTDIR, exist_ok=True)

# ── Generate inputs (same seed as PC reference) ──
rng = np.random.RandomState(SEED)
x_np = (rng.randn(MS, D).astype(np.float32) * 0.02).astype(np.float16)
ctx_np = (rng.randn(M * Nctx, CtxD).astype(np.float32) * 0.5).astype(np.float16)

print(f"x: {x_np.shape} range=[{x_np.min():.4f}, {x_np.max():.4f}]")
print(f"ctx: {ctx_np.shape} range=[{ctx_np.min():.4f}, {ctx_np.max():.4f}]")

# Save inputs for PC comparison
np.save(f"{OUTDIR}/x_phone.npy", x_np)
np.save(f"{OUTDIR}/ctx_phone.npy", ctx_np)

# ── Init C++ engine ──
print("Init engine...")
t0 = time.time()
lib.dit_set_skip_attn_precord()  # use per-step recording (not pre-recorded)
ok = lib.dit_init_adaln_only(b"/data/local/tmp/diffusion_weights.bin", b"/data/local/tmp")
print(f"  Init: {ok} ({time.time()-t0:.0f}s)")
if not ok:
    print("FATAL: init failed")
    lib.dit_destroy()
    exit(1)

# ── Compute t_emb (matching PC sigma=1.0) ──
sigma = 1.0
lib.dit_compute_timestep(ct.c_float(sigma))

# Dump t_emb and lora for PC comparison
t_emb_buf = np.zeros(M * D, dtype=np.uint16)
lora_buf = np.zeros(M * 3 * D, dtype=np.uint16)
lib.dit_get_timestep_output(
    t_emb_buf.ctypes.data_as(ct.c_void_p), M * D * 2,
    lora_buf.ctypes.data_as(ct.c_void_p), M * 3 * D * 2)
t_emb_arr = t_emb_buf.view(np.float16).reshape(M, D)
# lora is stored as [3, M, D] (component-major) → rearrange to [M, 3D] (batch-major)
lora_arr = lora_buf.view(np.float16).reshape(3, M, D).transpose(1, 0, 2).reshape(M, 3*D)
np.save(f"{OUTDIR}/t_emb_phone.npy", t_emb_arr)
np.save(f"{OUTDIR}/lora_phone.npy", lora_arr)
print(f"t_emb: {t_emb_arr.shape} range=[{t_emb_arr.min():.4f}, {t_emb_arr.max():.4f}]")
print(f"lora:  {lora_arr.shape} range=[{lora_arr.min():.4f}, {lora_arr.max():.4f}]")

# ── Run 28 blocks, capturing each output ──
dp = lambda a: a.ctypes.data_as(ct.c_void_p)
x = x_np.view(np.uint16).copy()
o = np.zeros(MS * D, dtype=np.uint16)

# We run dit_forward_step for blocks 0..N one at a time.
# The C++ engine chains blocks internally. To get per-block outputs,
# we run dit_forward_step with nblocks=1 for each block, feeding the
# output of block N as input to block N+1.
#
# BUT: dit_forward_step is designed for full 28-block runs.
# We need a simpler approach.
#
# Alternative: modify dit_forward_step to accept a start_block parameter.
# But we can't recompile C++ from here.
#
# SIMPLEST: run a FULL dit_forward_step (28 blocks), and
# look at the C++ engine's logcat output for per-block nan counts.
# The real per-block output will be from the phone_pipeline hooks.
#
# Actually, let me do it differently: use dit_run_layernorm-style per-call.
# But dit_forward_step is the only C++ API for block computation.
#
# OK let me just run the full 28-block forward and save the FINAL output,
# plus read intermediate outputs via logcat.
# For per-block comparison, I'll need to modify the C++ engine...

print("\nRunning dit_forward_step (28 blocks, mode=0 full attention)...")
t0 = time.time()
ok = lib.dit_forward_step(
    dp(x.reshape(-1)), ct.c_void_p(0), dp(ctx_np.view(np.uint16).reshape(-1)),
    dp(o), MS, D, M, Nctx, CtxD, 0)
dt = time.time() - t0

of = o.view(np.float16)
vmin = float(of[np.isfinite(of)].min()) if np.isfinite(of).any() else 0
vmax = float(of[np.isfinite(of)].max()) if np.isfinite(of).any() else 0
nans = int(np.sum(np.isnan(of)))
print(f"  Forward: {ok} ({dt:.0f}s)  range=[{vmin:.2f}, {vmax:.2f}]  nan={nans}")

# Save final output
np.save(f"{OUTDIR}/out_cpp.npy", o.view(np.float16).reshape(MS, D))

# ── Read per-block outputs via new C++ API ──
print(f"\nCapturing per-block outputs (MS={MS}, D={D})...")
block_sz = MS * D * 2  # bytes per block
block_arr = np.zeros(MS * D, dtype=np.uint16)
for b in range(28):
    ok = lib.dit_get_block_output(b, block_arr.ctypes.data_as(ct.c_void_p), block_sz)
    if ok:
        arr = block_arr.view(np.float16).reshape(MS, D).copy()
        np.save(f"{OUTDIR}/block_{b:02d}_cpp.npy", arr)
        f = arr[np.isfinite(arr)]
        print(f"  Block {b:2d}: range=[{f.min():.2f}, {f.max():.2f}]  nan={np.sum(np.isnan(arr))}" if len(f) > 0 else f"  Block {b:2d}: ALL NaN/Inf")
    else:
        print(f"  Block {b:2d}: READ FAILED")

print(f"\nSaved to {OUTDIR}/block_*_cpp.npy")

# ── Capture Block 0 intermediates ──
print(f"\nCapturing Block 0 intermediates...")
stages = [
    (0, "sa", MS*D*2),
    (1, "cx", MS*D*2),
    (2, "mlp", MS*D*2),
    (10, "q_norm", MS*N_HEADS*HEAD_DIM*2),
    (11, "k_norm", MS*N_HEADS*HEAD_DIM*2),
    (12, "v_raw", MS*N_HEADS*HEAD_DIM*2),
    (13, "scores", (MS//M)*N_HEADS*(MS//M)*2),
    (14, "attn_o", MS*N_HEADS*HEAD_DIM*2),
    (15, "q_roped", MS*N_HEADS*HEAD_DIM*2),
    (16, "k_roped", MS*N_HEADS*HEAD_DIM*2),
    (17, "o_proj", MS*D*2),
]
for stage_id, name, sz in stages:
    buf = np.zeros(sz // 2, dtype=np.uint16)  # size in bytes → uint16 count
    ok = lib.dit_get_b0_intermediate(stage_id, buf.ctypes.data_as(ct.c_void_p), sz)
    if ok:
        arr = buf.view(np.float16)
        np.save(f"{OUTDIR}/b0_{name}.npy", arr)
        f = arr[np.isfinite(arr)]
        print(f"  b0_{name}: shape={arr.shape} range=[{f.min():.2f}, {f.max():.2f}]  nan={np.sum(np.isnan(arr))}" if len(f) > 0 else f"  b0_{name}: ALL NaN")
    else:
        print(f"  b0_{name}: READ FAILED (stage={stage_id})")

lib.dit_destroy()
print("Done.")

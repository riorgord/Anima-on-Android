"""Phone: run C++ engine with real pipeline inputs from PC and dump per-block outputs."""
import ctypes as ct, numpy as np, os, time

lib = ct.CDLL("/data/local/tmp/libdit_vk.so")
lib.dit_init_adaln_only.argtypes = [ct.c_char_p, ct.c_char_p]; lib.dit_init_adaln_only.restype = ct.c_bool
lib.dit_write_buf.argtypes = [ct.c_int, ct.c_void_p, ct.c_size_t]; lib.dit_write_buf.restype = ct.c_bool
lib.dit_forward_step.argtypes = [ct.c_void_p]*4 + [ct.c_int]*6; lib.dit_forward_step.restype = ct.c_bool
lib.dit_get_block_output.argtypes = [ct.c_int, ct.c_void_p, ct.c_size_t]; lib.dit_get_block_output.restype = ct.c_bool
lib.dit_destroy.argtypes = []; lib.dit_destroy.restype = None

INDIR = "/sdcard/anima_on_android/output/realpipe"
OUTDIR = INDIR
os.makedirs(OUTDIR, exist_ok=True)

# Load PC inputs
x_np = np.load(f"{INDIR}/x_flat.npy").astype(np.float16)
t_np = np.load(f"{INDIR}/t_emb.npy").astype(np.float16)
c_np = np.load(f"{INDIR}/ctx_flat.npy").astype(np.float16)
print(f"x: {x_np.shape} [{x_np.min():.4f},{x_np.max():.4f}]")
print(f"t: {t_np.shape} [{t_np.min():.4f},{t_np.max():.4f}]")
print(f"c: {c_np.shape} [{c_np.min():.4f},{c_np.max():.4f}]")

# Init C++ engine
t0 = time.time()
ok = lib.dit_init_adaln_only(b"/data/local/tmp/diffusion_weights.bin", b"/data/local/tmp")
print(f"Init: {ok} ({time.time()-t0:.0f}s)")
if not ok:
    print("FATAL: init failed")
    exit(1)

# Compute t_emb + lora via C++ dit_compute_timestep (same as PC C++-style, max_err=0.00006)
# Then overwrite t_emb with PC's exactly-matching value for consistency
lib.dit_compute_timestep(ct.c_float(1.0))
lib.dit_write_buf(1, t_np.ctypes.data_as(ct.c_void_p), t_np.nbytes)

MS, D, M, Nctx, CtxD = 512, 2048, 2, 512, 1024; N_HEADS, HEAD_DIM = 16, 128
dp = lambda a: a.ctypes.data_as(ct.c_void_p)
o = np.zeros(MS * D, dtype=np.uint16)

# Run C++ forward
print(f"Running C++ forward with real inputs...")
t0 = time.time()
ok = lib.dit_forward_step(
    dp(x_np.view(np.uint16).reshape(-1)),
    ct.c_void_p(0),  # t_emb already in GPU
    dp(c_np.view(np.uint16).reshape(-1)),
    dp(o), MS, D, M, Nctx, CtxD, 0)
dt = time.time() - t0

of = o.view(np.float16)
nans = int(np.sum(np.isnan(of)))
vmin = float(of[np.isfinite(of)].min()) if np.isfinite(of).any() else 0
vmax = float(of[np.isfinite(of)].max()) if np.isfinite(of).any() else 0
print(f"Forward: {ok} ({dt:.0f}s) range=[{vmin:.2f},{vmax:.2f}] nan={nans}")

# Dump per-block
block_sz = MS * D * 2
block_arr = np.zeros(MS * D, dtype=np.uint16)
for b in range(28):
    ok = lib.dit_get_block_output(b, block_arr.ctypes.data_as(ct.c_void_p), block_sz)
    if ok:
        arr = block_arr.view(np.float16).reshape(MS, D).copy()
        np.save(f"{OUTDIR}/block_{b:02d}_cpp.npy", arr)
        ok_mask = np.isfinite(arr)
        if ok_mask.sum() > 0:
            print(f"  Block {b:2d}: [{arr[ok_mask].min():.1f},{arr[ok_mask].max():.1f}] nan={np.sum(~ok_mask)}")
        else:
            print(f"  Block {b:2d}: ALL NaN!")
    else:
        print(f"  Block {b:2d}: READ FAILED")

# Capture Block 0 intermediates
lib.dit_get_b0_intermediate.argtypes = [ct.c_int, ct.c_void_p, ct.c_size_t]
lib.dit_get_b0_intermediate.restype = ct.c_bool

qkv_sz = MS * N_HEADS * HEAD_DIM * 2
stages = [
    (10, "q_norm", qkv_sz),
    (11, "k_norm", qkv_sz),
    (12, "v_raw", qkv_sz),
    (14, "attn_o", qkv_sz),
    (17, "o_proj", MS*D*2),
    (18, "modulated", MS*D*2),
    (19, "q_raw", MS*D*2),
    (20, "shifts", M*3*D*2),
    (0, "sa_residual", MS*D*2),
]
for stage_id, name, sz in stages:
    buf = np.zeros(sz // 2, dtype=np.uint16)
    ok = lib.dit_get_b0_intermediate(stage_id, buf.ctypes.data_as(ct.c_void_p), sz)
    if ok:
        arr = buf.view(np.float16)
        np.save(f"{OUTDIR}/b0_{name}_cpp.npy", arr)
        f = arr[np.isfinite(arr)]
        rng_str = f"[{f.min():.4f}, {f.max():.4f}]" if len(f) > 0 else "ALL NaN"
        print(f"  b0_{name}: {rng_str}")
    else:
        print(f"  b0_{name}: READ FAILED")

lib.dit_destroy()
print(f"Done. Saved to {OUTDIR}/block_*_cpp.npy")

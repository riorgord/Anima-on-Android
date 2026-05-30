"""Export C++ forward inputs/outputs for PyTorch comparison."""
import ctypes as ct, numpy as np, os

lib = ct.CDLL("/data/local/tmp/libdit_vk.so")
lib.dit_set_skip_attn_precord.argtypes = []; lib.dit_set_skip_attn_precord.restype = None
lib.dit_init_adaln_only.argtypes = [ct.c_char_p, ct.c_char_p]; lib.dit_init_adaln_only.restype = ct.c_bool
lib.dit_compute_timestep.argtypes = [ct.c_float]; lib.dit_compute_timestep.restype = ct.c_bool
lib.dit_forward_step.argtypes = [ct.c_void_p]*4 + [ct.c_int]*6; lib.dit_forward_step.restype = ct.c_bool
lib.dit_destroy.argtypes = []; lib.dit_destroy.restype = None

outdir = "/sdcard/anima_on_android/output"
os.makedirs(outdir, exist_ok=True)

# Init
lib.dit_set_skip_attn_precord()
ok = lib.dit_init_adaln_only(b"/data/local/tmp/diffusion_weights.bin", b"/data/local/tmp")
print("init:", ok)

# Fixed seed for reproducibility
rng = np.random.RandomState(12345)
MS, M, D, Nctx, CtxD = 512, 2, 2048, 512, 1024

# Use sigma=1.0 (typical first step) for realistic t_emb
sigma = 1.0
lib.dit_compute_timestep(ct.c_float(sigma))

# Random input data in same range as real pipeline
x = (rng.randn(MS, D).astype(np.float16) * 0.02).view(np.uint16)
t = np.zeros(0, dtype=np.uint16)  # t_emb already in GPU via dit_compute_timestep
# ctx: typical context range
c = (rng.randn(M, Nctx, CtxD).astype(np.float16) * 0.5).reshape(M*Nctx, CtxD).view(np.uint16)
o = np.zeros(MS * D, dtype=np.uint16)

dp = lambda a: a.ctypes.data_as(ct.c_void_p)

print(f"x range: {x.view(np.float16).min():.3f} to {x.view(np.float16).max():.3f}")
print(f"ctx range: {c.view(np.float16).min():.3f} to {c.view(np.float16).max():.3f}")

print("Running forward...")
ok = lib.dit_forward_step(
    dp(x.reshape(-1)), ct.c_void_p(0), dp(c.reshape(-1)),
    dp(o), MS, D, M, Nctx, CtxD, 0)
print("forward:", ok)

of = o.view(np.float16)
print(f"out range: {of.min():.3f} to {of.max():.3f} nan={np.sum(np.isnan(of))}")

# Save inputs and output
np.save(f"{outdir}/debug_x.npy", x.view(np.float16).reshape(MS, D))
np.save(f"{outdir}/debug_ctx.npy", c.view(np.float16).reshape(M*Nctx, CtxD))
np.save(f"{outdir}/debug_out_cpp.npy", o.view(np.float16).reshape(MS, D))
# t_emb: we computed via dit_compute_timestep (sigma=1.0), save sigma for PC recalculation
np.save(f"{outdir}/debug_sigma.npy", np.array([sigma]))

print(f"Saved to {outdir}/debug_*.npy")
lib.dit_destroy()

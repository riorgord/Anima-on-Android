"""Compare C++ dit_compute_timestep vs PyTorch t_embedder directly on phone."""
import ctypes as ct, numpy as np, time, sys
sys.path.insert(0, "/sdcard/anima_on_android/src")
import torch, predict2

lib = ct.CDLL("/data/local/tmp/libdit_vk.so")
lib.dit_set_skip_attn_precord.argtypes = []; lib.dit_set_skip_attn_precord.restype = None
lib.dit_init_adaln_only.argtypes = [ct.c_char_p, ct.c_char_p]; lib.dit_init_adaln_only.restype = ct.c_bool
lib.dit_compute_timestep.argtypes = [ct.c_float]; lib.dit_compute_timestep.restype = ct.c_bool
lib.dit_read_buf.argtypes = [ct.c_int, ct.c_void_p, ct.c_size_t]; lib.dit_read_buf.restype = ct.c_bool
lib.dit_destroy.argtypes = []; lib.dit_destroy.restype = None

sigma = 1.0
M, D = 2, 2048

# ── Init C++ ──
lib.dit_set_skip_attn_precord()
ok = lib.dit_init_adaln_only(b"/data/local/tmp/diffusion_weights.bin", b"/data/local/tmp")
print(f"C++ init: {ok}")

# ── C++ compute t_emb ──
lib.dit_compute_timestep(ct.c_float(sigma))

# ── Read back C++ t_emb and lora ──
t_emb_cpp = np.zeros(M * D, dtype=np.uint16)
lora_cpp = np.zeros(3 * M * D, dtype=np.uint16)
lib.dit_read_buf(1, t_emb_cpp.ctypes.data_as(ct.c_void_p), t_emb_cpp.nbytes)  # buf_id=1: g_tEmbBuf
lib.dit_read_buf(8, lora_cpp.ctypes.data_as(ct.c_void_p), lora_cpp.nbytes)  # buf_id=8: g_loraBuf

t_emb_cpp_f = t_emb_cpp.view(np.float16).reshape(M, D)
lora_cpp_f = lora_cpp.view(np.float16)  # [3*M*D] flat

print(f"C++ t_emb: shape={t_emb_cpp_f.shape} range=[{t_emb_cpp_f.min():.4f}, {t_emb_cpp_f.max():.4f}]")
print(f"C++ lora: shape={lora_cpp_f.shape} range=[{lora_cpp_f.min():.4f}, {lora_cpp_f.max():.4f}]")

# ── PyTorch t_emb (same weights as C++) ──
print("\nLoading PyTorch t_embedder...")
small_sd = torch.load("/sdcard/anima_on_android/models/diffusion_weights_small.pt", weights_only=True)
sd = {}
for k, v in small_sd.items():
    while k.startswith("net."):
        k = k[4:]
    sd[k] = v
del small_sd

config = dict(max_img_h=240, max_img_w=240, max_frames=128, in_channels=16, out_channels=16,
    patch_spatial=2, patch_temporal=1, concat_padding_mask=True, model_channels=D,
    num_blocks=0, num_heads=16, mlp_ratio=4.0, crossattn_emb_channels=1024,
    pos_emb_cls="rope3d", pos_emb_learnable=True, pos_emb_interpolation="crop",
    min_fps=1, max_fps=30, use_adaln_lora=True, adaln_lora_dim=256,
    rope_h_extrapolation_ratio=4.0, rope_w_extrapolation_ratio=4.0, rope_t_extrapolation_ratio=1.0,
    extra_per_block_abs_pos_emb=False, rope_enable_fps_modulation=False)

dit = predict2.MiniTrainDIT(**config, device="cpu", dtype=torch.float16, operations=torch.nn)
dit.load_state_dict(sd, strict=False)
dit.eval()

ts = torch.tensor([sigma, sigma], dtype=torch.float16).unsqueeze(1)  # [2, 1]
with torch.no_grad():
    # Step 1: Timesteps — sinusoidal
    sin_emb = dit.t_embedder[0](ts).squeeze(1).float()  # [2, 2048]

    # Step 2: TimestepEmbedding.linear_1 + SiLU
    w1 = dit.t_embedder[1].linear_1.weight.float()  # [D, D]
    h = torch.nn.functional.silu(torch.nn.functional.linear(sin_emb, w1))  # [2, 2048]

    # Step 3: linear_2 → lora
    w2 = dit.t_embedder[1].linear_2.weight.float()  # [3*D, D]
    lora_pt = torch.nn.functional.linear(h, w2).half()  # [2, 6144]

    # Step 4: RMSNorm → t_emb
    w_norm = dit.t_embedding_norm.weight.float()  # [D]
    t_emb_pt = torch.nn.functional.rms_norm(sin_emb, (D,), weight=w_norm, eps=1e-6).half()

print(f"PT  t_emb: shape={t_emb_pt.shape} range=[{t_emb_pt.min():.4f}, {t_emb_pt.max():.4f}]")
print(f"PT  lora: shape={lora_pt.shape} range=[{lora_pt.min():.4f}, {lora_pt.max():.4f}]")

# ── Compare ──
t_emb_pt_f = t_emb_pt.numpy().astype(np.float16)  # [2, 2048]
lora_pt_f = lora_pt.numpy().astype(np.float16).reshape(-1)  # [2*3*D] = [12288]

print("\n=== C++ vs PyTorch t_emb comparison ===")
t_diff = np.abs(t_emb_cpp_f.astype(np.float32) - t_emb_pt_f.astype(np.float32))
print(f"  max_err={t_diff.max():.6f}  mean_err={t_diff.mean():.6f}")
print(f"  C++ first 5: {t_emb_cpp_f[0,:5]}")
print(f"  PT  first 5: {t_emb_pt_f[0,:5]}")

print("\n=== C++ vs PyTorch lora comparison ===")
# C++ lora layout: [3, M, D] = [3, 2, 2048] — component-major
# PT lora layout: [M, 3*D] = [2, 6144] — batch-major, then component-major
# Compare by reshaping C++ to match PT
lora_cpp_reshaped = lora_cpp_f.reshape(3, M, D)  # [3, 2, 2048]
lora_cpp_pt_format = np.zeros(M * 3 * D, dtype=np.float16)  # [2, 6144]
for b in range(M):
    lora_cpp_pt_format[b * 3 * D + 0 * D: b * 3 * D + 1 * D] = lora_cpp_reshaped[0, b, :]  # shift
    lora_cpp_pt_format[b * 3 * D + 1 * D: b * 3 * D + 2 * D] = lora_cpp_reshaped[1, b, :]  # scale
    lora_cpp_pt_format[b * 3 * D + 2 * D: b * 3 * D + 3 * D] = lora_cpp_reshaped[2, b, :]  # gate

l_diff = np.abs(lora_cpp_pt_format.astype(np.float32) - lora_pt_f.astype(np.float32))
print(f"  max_err={l_diff.max():.6f}  mean_err={l_diff.mean():.6f}")
print(f"  C++ first 5: {lora_cpp_pt_format[:5]}")
print(f"  PT  first 5: {lora_pt_f[:5]}")

# ── Save for PC comparison ──
OUT = "/sdcard/anima_on_android/output/cmp"
np.save(f"{OUT}/t_emb_cpp.npy", t_emb_cpp_f)
np.save(f"{OUT}/lora_cpp.npy", lora_cpp_pt_format.reshape(M, 3*D))
np.save(f"{OUT}/t_emb_pt.npy", t_emb_pt_f)
np.save(f"{OUT}/lora_pt.npy", lora_pt_f)

lib.dit_destroy()
print("\nDone.")

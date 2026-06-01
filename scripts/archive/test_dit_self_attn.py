"""Test full self-attention with pre-uploaded AdaLN: LN→AdaLN→QKV→norms→V→O→gate+res"""
import ctypes, numpy as np, torch, sys, time

_lib = ctypes.CDLL("/data/local/tmp/libdit_vk.so")
_lib.dit_init.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
_lib.dit_init.restype = ctypes.c_bool
_lib.dit_record_self_attn_full.argtypes = [ctypes.c_int]
_lib.dit_record_self_attn_full.restype = ctypes.c_bool
_lib.dit_write_buf.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_size_t]
_lib.dit_write_buf.restype = ctypes.c_bool
_lib.dit_forward.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
    ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]
_lib.dit_forward.restype = ctypes.c_bool

print("Init...")
t0 = time.time()
ok = _lib.dit_init(b"/data/local/tmp/diffusion_weights.bin", b"/data/local/tmp")
print(f"  init = {ok} ({time.time()-t0:.1f}s)")
if not ok: sys.exit(1)

# Test data
MS, D, M = 512, 2048, 2
np.random.seed(42)
x = torch.randn(MS, D, dtype=torch.float32)
t_emb = torch.randn(M, D, dtype=torch.float32)

print("Loading weights and computing AdaLN reference...")
sd = torch.load("/sdcard/anima_on_android/models/diffusion_weights_fp16.pt",
    weights_only=True, map_location="cpu")

# Extract block 0 weights (no "net." prefix)
pfx = "blocks.0."
w_q = sd[f"{pfx}self_attn.q_proj.weight"]
w_k = sd[f"{pfx}self_attn.k_proj.weight"]
w_v = sd[f"{pfx}self_attn.v_proj.weight"]
w_o = sd[f"{pfx}self_attn.output_proj.weight"]
w_qn = sd[f"{pfx}self_attn.q_norm.weight"].float()
w_kn = sd[f"{pfx}self_attn.k_norm.weight"].float()
# AdaLN LoRA weights
w_l0 = sd[f"{pfx}adaln_modulation_self_attn.1.weight"].float()
w_l2 = sd[f"{pfx}adaln_modulation_self_attn.2.weight"].float()
del sd

import torch.nn.functional as F

# Compute AdaLN: SiLU(t_emb) → Linear(256) → Linear(3*D)
h = F.silu(t_emb.float())
h = F.linear(h, w_l0)    # [M, 256]
h = F.linear(h, w_l2)    # [M, 3*D]
shift, scale, gate = torch.chunk(h, 3, dim=-1)  # each [M, D]

# Add 1 to scale
scale_p1 = scale + 1.0

# Broadcast from [M, D] to [MS, D] (S=256 per batch item)
scale_bcast = scale_p1.repeat_interleave(MS // M, dim=0)  # [MS, D]
shift_bcast = shift.repeat_interleave(MS // M, dim=0)
gate_bcast = gate.repeat_interleave(MS // M, dim=0)

# Upload AdaLN data to bcBuf (buf 4) at specific offsets:
#   [scale_bcast (MS*D) | shift_bcast (MS*D) | gate_bcast (MS*D)]
#   scale @ offset 0, shift @ MS*D*2 bytes, gate @ 2*MS*D*2 bytes
def upload_buf(buf_id, tensor, byte_offset=0):
    data = tensor.detach().to(torch.float16).numpy()
    # Stupid but works: upload entire concat buffer
    pass

# Build concatenated buffer [scale | shift | gate], each MS*D fp16 = MS*D*2 bytes
elem_size = 2  # fp16
n_elem = MS * D
concat = np.zeros(n_elem * 3, dtype=np.uint16)  # fp16 = uint16
concat[0*n_elem : 1*n_elem] = scale_bcast.numpy().astype(np.float16).ravel().view(np.uint16)
concat[1*n_elem : 2*n_elem] = shift_bcast.numpy().astype(np.float16).ravel().view(np.uint16)
concat[2*n_elem : 3*n_elem] = gate_bcast.numpy().astype(np.float16).ravel().view(np.uint16)

_lib.dit_write_buf(4, concat.ctypes.data_as(ctypes.c_void_p), concat.nbytes)

print("Recording self-attn full block 0...")
ok = _lib.dit_record_self_attn_full(0)
print(f"  record = {ok}")

# Upload input x
x_np = x.numpy().astype(np.float16)
_lib.dit_write_buf(0, x_np.ctypes.data_as(ctypes.c_void_p), x_np.nbytes)

# Upload dummy t_emb and ctx (not used in self-attn full)
t_np = np.zeros((M, D), dtype=np.float16)
c_np = np.zeros((M, 512, 1024), dtype=np.float16)
_lib.dit_write_buf(1, t_np.ctypes.data_as(ctypes.c_void_p), t_np.nbytes)
_lib.dit_write_buf(2, c_np.ctypes.data_as(ctypes.c_void_p), c_np.nbytes)

out_np = np.zeros((MS, D), dtype=np.float16)
_lib.dit_write_buf(3, out_np.ctypes.data_as(ctypes.c_void_p), out_np.nbytes)

print("Running Vulkan self-attn...")
t0 = time.time()
ok = _lib.dit_forward(x_np.ctypes.data_as(ctypes.c_void_p), t_np.ctypes.data_as(ctypes.c_void_p),
    c_np.ctypes.data_as(ctypes.c_void_p), out_np.ctypes.data_as(ctypes.c_void_p),
    MS, D, M, 512, 1024)
print(f"  forward = {ok} ({time.time()-t0:.3f}s)")

# CPU reference: replicate the same simplified path
# 1. LN(x)
ln_x = F.layer_norm(x, (D,), weight=None, bias=None, eps=1e-6)
# 2. AdaLN: ln_x * scale_p1_bcast + shift_bcast
modulated = ln_x * scale_bcast + shift_bcast
# 3. Q/K/V proj
q = F.linear(modulated, w_q.float())  # [MS, D]
k = F.linear(modulated, w_k.float())
v = F.linear(modulated, w_v.float())
# 4. Q/K norms (RMSNorm per head: reshape to [MS*16, 128])
q_rs = q.reshape(MS * 16, 128)
k_rs = k.reshape(MS * 16, 128)
q_hat = F.rms_norm(q_rs, (128,), weight=w_qn, eps=1e-6)
k_hat = F.rms_norm(k_rs, (128,), weight=w_kn, eps=1e-6)
q_hat = q_hat.reshape(MS, D)
k_hat = k_hat.reshape(MS, D)
# 5. Skip attention — use V as attention output
attn_out = v
# 6. O proj
o = F.linear(attn_out, w_o.float())
# 7. Gate + residual
ref = x + gate_bcast * o
ref_np = ref.half().numpy()

err = np.abs(out_np.astype(np.float32) - ref_np.astype(np.float32)).max()
print(f"max_err = {err:.6f}")
print(f"out mean={out_np.astype(np.float32).mean():.4f} std={out_np.astype(np.float32).std():.4f}")
print(f"ref mean={ref_np.astype(np.float32).mean():.4f} std={ref_np.astype(np.float32).std():.4f}")
nz = (out_np != 0).sum()
print(f"non-zero: {nz} / {out_np.size} ({100*nz/out_np.size:.1f}%)")
print("PASS" if err < 0.5 else "FAIL")

_lib.dit_destroy()
print("DONE")

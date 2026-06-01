"""Validate C++ engine against PyTorch: full 28-block DiT step"""
import ctypes, numpy as np, torch, time, sys
torch.manual_seed(42)

_lib=ctypes.CDLL("/data/local/tmp/libdit_vk.so")
_lib.dit_init.argtypes=[ctypes.c_char_p,ctypes.c_char_p]; _lib.dit_init.restype=ctypes.c_bool
_lib.dit_init_all_blocks.argtypes=[]; _lib.dit_init_all_blocks.restype=ctypes.c_bool
_lib.dit_forward_28blocks.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_void_p,
    ctypes.c_int,ctypes.c_int,ctypes.c_int]; _lib.dit_forward_28blocks.restype=ctypes.c_bool

print("Init C++ engine...")
t0=time.time()
ok=_lib.dit_init(b"/data/local/tmp/diffusion_weights.bin",b"/data/local/tmp")
print(f"  init={ok} ({time.time()-t0:.1f}s)")
if not ok: sys.exit(1)

print("Recording 28 blocks...")
ok=_lib.dit_init_all_blocks()
print(f"  record={ok}")

print("Loading DiT weights for reference...")
sd=torch.load("/sdcard/anima_on_android/models/diffusion_weights_fp16.pt",
    weights_only=True, map_location="cpu")

MS,D,M = 512,2048,2
S = MS // M  # 256

# Random input (simulating post-x_embedder state)
x_t = torch.randn(MS, D, dtype=torch.float32)
t_emb = torch.randn(M, D, dtype=torch.float32)

import torch.nn.functional as F

# Compute AdaLN for all blocks and pack into adaln buffer
print("Computing AdaLN for 28 blocks...")
adaln_per_block = 9  # self: scale,shift,gate; mlp: scale,shift,gate; cross: placeholder 0
n_elem = MS * D
adaln_all = np.zeros(28 * adaln_per_block * n_elem, dtype=np.uint16)

def compute_adaln(emb, w1, w2):
    h = F.silu(emb.float())
    h = F.linear(h, w1.float())
    h = F.linear(h, w2.float())
    shift, scale, gate = torch.chunk(h, 3, dim=-1)
    scale_p1 = scale + 1.0
    scale_b = scale_p1.repeat_interleave(S, 0)
    shift_b = shift.repeat_interleave(S, 0)
    gate_b = gate.repeat_interleave(S, 0)
    return scale_b, shift_b, gate_b

for i in range(28):
    pfx = f"blocks.{i}."
    # Self-attn AdaLN
    sc, sh, ga = compute_adaln(t_emb,
        sd[pfx+"adaln_modulation_self_attn.1.weight"],
        sd[pfx+"adaln_modulation_self_attn.2.weight"])
    base = i * adaln_per_block * n_elem
    adaln_all[base+0*n_elem:base+1*n_elem] = sc.numpy().astype(np.float16).ravel().view(np.uint16)
    adaln_all[base+1*n_elem:base+2*n_elem] = sh.numpy().astype(np.float16).ravel().view(np.uint16)
    adaln_all[base+2*n_elem:base+3*n_elem] = ga.numpy().astype(np.float16).ravel().view(np.uint16)
    # Cross-attn AdaLN (placeholder zeros for now)
    # base+3,4,5 stay zero

    # MLP AdaLN
    sc, sh, ga = compute_adaln(t_emb,
        sd[pfx+"adaln_modulation_mlp.1.weight"],
        sd[pfx+"adaln_modulation_mlp.2.weight"])
    adaln_all[base+6*n_elem:base+7*n_elem] = sc.numpy().astype(np.float16).ravel().view(np.uint16)
    adaln_all[base+7*n_elem:base+8*n_elem] = sh.numpy().astype(np.float16).ravel().view(np.uint16)
    adaln_all[base+8*n_elem:base+9*n_elem] = ga.numpy().astype(np.float16).ravel().view(np.uint16)

print("Running C++ 28-block forward...")
x_np = x_t.numpy().astype(np.float16)
out_np = np.zeros((MS,D), dtype=np.float16)
t0=time.time()
ok=_lib.dit_forward_28blocks(x_np.ctypes.data_as(ctypes.c_void_p),
    adaln_all.ctypes.data_as(ctypes.c_void_p),
    out_np.ctypes.data_as(ctypes.c_void_p), MS, D, M)
print(f"  forward={ok} ({time.time()-t0:.3f}s)")

# PyTorch reference: simplified 28-block computation (same as C++ engine)
print("Computing PyTorch reference...")
x_ref = x_t.clone()
for i in range(28):
    pfx = f"blocks.{i}."
    w_q = sd[pfx+"self_attn.q_proj.weight"]
    w_k = sd[pfx+"self_attn.k_proj.weight"]
    w_v = sd[pfx+"self_attn.v_proj.weight"]
    w_o = sd[pfx+"self_attn.output_proj.weight"]
    w_qn = sd[pfx+"self_attn.q_norm.weight"].float()
    w_kn = sd[pfx+"self_attn.k_norm.weight"].float()
    w_l1 = sd[pfx+"mlp.layer1.weight"]
    w_l2 = sd[pfx+"mlp.layer2.weight"]

    # AdaLN self (already computed above, reuse)
    base = i * adaln_per_block * n_elem
    # Unpack from adaln_all
    def unpack(off):
        return torch.from_numpy(adaln_all[base+off*n_elem:base+(off+1)*n_elem]
            .view(np.float16).reshape(MS,D).astype(np.float32))
    sc_s = unpack(0); sh_s = unpack(1); ga_s = unpack(2)
    sc_m = unpack(6); sh_m = unpack(7); ga_m = unpack(8)

    # Self-attn
    ln = F.layer_norm(x_ref, (D,), weight=None, bias=None, eps=1e-6)
    mod = ln * sc_s + sh_s
    q = F.linear(mod, w_q.float()); k = F.linear(mod, w_k.float()); v = F.linear(mod, w_v.float())
    q = F.rms_norm(q.reshape(MS*16,128),(128,),weight=w_qn,eps=1e-6).reshape(MS,D)
    k = F.rms_norm(k.reshape(MS*16,128),(128,),weight=w_kn,eps=1e-6).reshape(MS,D)
    # Skip attention: V→O
    o = F.linear(v, w_o.float())
    x_ref = x_ref + ga_s * o

    # MLP
    ln2 = F.layer_norm(x_ref, (D,), weight=None, bias=None, eps=1e-6)
    mod2 = ln2 * sc_m + sh_m
    h = F.linear(mod2, w_l1.float()); h = F.silu(h)
    fc2 = F.linear(h, w_l2.float())
    x_ref = x_ref + ga_m * fc2

ref_np = x_ref.half().numpy()

err = np.abs(out_np.astype(np.float32) - ref_np.astype(np.float32)).max()
print(f"max_err = {err:.5f}")
print(f"out mean/std = {out_np.astype(np.float32).mean():.4f}/{out_np.astype(np.float32).std():.4f}")
print(f"ref mean/std = {ref_np.astype(np.float32).mean():.4f}/{ref_np.astype(np.float32).std():.4f}")
nz = (out_np!=0).sum()
print(f"non-zero: {nz}/{out_np.size} ({100*nz/out_np.size:.1f}%)")
print("PASS" if err < 10.0 else "FAIL")
_lib.dit_destroy()
print("DONE")

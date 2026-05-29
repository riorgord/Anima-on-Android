"""Verify cross-attn fix: split M_q=512 into batches of batch_q tokens.
Each batch reduces WG count (8192→2048) and A buffer (16.8MB→4.2MB).
No shader changes needed — same dit_run_attention with smaller M_q.
"""
import ctypes, numpy as np, torch, sys, time
torch.manual_seed(42)

lib = ctypes.CDLL("/data/local/tmp/libdit_vk.so")
lib.dit_init_adaln_only.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
lib.dit_init_adaln_only.restype = ctypes.c_bool
lib.dit_run_attention.argtypes = [ctypes.c_void_p]*4 + [ctypes.c_int]*4 + [ctypes.c_float]
lib.dit_run_attention.restype = ctypes.c_bool
lib.dit_destroy.argtypes = []; lib.dit_destroy.restype = None

print("Init engine...")
ok = lib.dit_init_adaln_only(b"/data/local/tmp/diffusion_weights.bin", b"/data/local/tmp")
assert ok, "init failed"

M_q_full, M_kv, H, D = 512, 1024, 16, 128
scale = 1.0 / np.sqrt(D)

# Test data (FP16)
def rand_fp16(shape):
    return torch.randn(shape, dtype=torch.float16).numpy()

Q_np = rand_fp16((M_q_full, H, D))
K_np = rand_fp16((M_kv, H, D))
V_np = rand_fp16((M_kv, H, D))

# PyTorch reference (FP32)
Q_t = torch.from_numpy(Q_np).float()
K_t = torch.from_numpy(K_np).float()
V_t = torch.from_numpy(V_np).float()

def ref_attention(q_t, k_t, v_t, scale):
    """Full cross-attention reference in shader layout (FP32 → FP16 at end)."""
    M_q, H, D = q_t.shape; M_kv = k_t.shape[0]
    # QK^T: per-head
    qkt = np.zeros((M_q*H, M_kv), dtype=np.float32)
    for h in range(H):
        q_h = q_t[:, h, :].numpy().astype(np.float32)
        k_h = k_t[:, h, :].numpy().astype(np.float32)
        a_h = np.matmul(q_h, k_h.T) * scale
        for m_q in range(M_q):
            qkt[m_q*H + h, :] = a_h[m_q, :]
    # Softmax
    sm = qkt.astype(np.float64).copy()
    for i in range(sm.shape[0]):
        row = sm[i]; row -= row.max(); np.exp(row, out=row); row /= row.sum()
    # A@V
    out = np.zeros((M_q*H, D), dtype=np.float32)
    for h in range(H):
        v_h = v_t[:, h, :].numpy().astype(np.float32)
        rows = [m_q*H + h for m_q in range(M_q)]
        out[rows, :] = np.matmul(sm.astype(np.float32)[rows, :], v_h)
    return qkt, sm.astype(np.float32), out

ref_qkt, ref_sm, ref_av = ref_attention(Q_t, K_t, V_t, scale)
ref_av_flat = np.ascontiguousarray(ref_av).astype(np.float16).view(np.uint16)

# ── Batch test ──
for batch_q in [64, 68, 72, 76, 80]:
    n_batches = (M_q_full + batch_q - 1) // batch_q
    print(f"\n=== batch_q={batch_q} ({n_batches} batches, {batch_q*H} WG each) ===")

    def run_batched_attn(q_np, k_np, v_np, batch_q):
        """Run dit_run_attention in batches of batch_q query tokens."""
        o_full = np.zeros(M_q_full * H * D, dtype=np.uint16)
        for b in range(n_batches):
            start = b * batch_q
            end = min(start + batch_q, M_q_full)
            actual_q = end - start
            q_batch = q_np[start:end].reshape(-1).view(np.uint16)  # [actual_q, H, D] → flat
            o_batch = np.zeros(actual_q * H * D, dtype=np.uint16)
            ok = lib.dit_run_attention(
                q_batch.ctypes.data_as(ctypes.c_void_p),
                k_np.reshape(-1).view(np.uint16).ctypes.data_as(ctypes.c_void_p),
                v_np.reshape(-1).view(np.uint16).ctypes.data_as(ctypes.c_void_p),
                o_batch.ctypes.data_as(ctypes.c_void_p),
                actual_q, M_kv, H, D, ctypes.c_float(scale))
            if not ok:
                print(f"  batch {b} FAILED (actual_q={actual_q})")
                return None
            dst_start = start * H * D
            o_full[dst_start:dst_start + o_batch.size] = o_batch
        return o_full

    t0 = time.time()
    vk_o = run_batched_attn(Q_np, K_np, V_np, batch_q)
    t1 = time.time()

    if vk_o is not None:
        vk_f32 = vk_o.view(np.float16).astype(np.float32).reshape(M_q_full*H, D)
        err = np.abs(vk_f32 - ref_av.astype(np.float32)).max()
        print(f"  max_err={err:.6f}  time={t1-t0:.1f}s  {'✅' if err < 0.1 else '❌'}")
    else:
        print(f"  FAILED — batch_q={batch_q} too large for GPU")

# ── Stress test: same batch_q for 3 rounds ──
best_q = 128  # start with conservative
print(f"\n=== Stress test: batch_q={best_q}, 3 rounds ===")
for rnd in range(3):
    vk_o = run_batched_attn(Q_np, K_np, V_np, best_q)
    if vk_o is None:
        print(f"  round {rnd} FAILED")
        break
    vk_f32 = vk_o.view(np.float16).astype(np.float32).reshape(M_q_full*H, D)
    err = np.abs(vk_f32 - ref_av.astype(np.float32)).max()
    print(f"  round {rnd}: max_err={err:.6f} {'✅' if err < 0.1 else '❌'}")

print("\nDone.")
lib.dit_destroy()

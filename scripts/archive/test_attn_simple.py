"""Simple cross-attn test: M_q=512, batched internally by C++."""
import ctypes, numpy as np, torch, time
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

M_q, M_kv, H, D = 512, 1024, 16, 128
scale = 1.0 / np.sqrt(D)

def rand_fp16(shape):
    return torch.randn(shape, dtype=torch.float16).numpy()

Q_np = rand_fp16((M_q, H, D))
K_np = rand_fp16((M_kv, H, D))
V_np = rand_fp16((M_kv, H, D))

# PyTorch reference
Q_t = torch.from_numpy(Q_np).float()
K_t = torch.from_numpy(K_np).float()
V_t = torch.from_numpy(V_np).float()

def ref_attention(q_t, k_t, v_t, scale):
    M_q, H, D = q_t.shape; M_kv = k_t.shape[0]
    qkt = np.zeros((M_q*H, M_kv), dtype=np.float32)
    for h in range(H):
        q_h = q_t[:, h, :].numpy().astype(np.float32)
        k_h = k_t[:, h, :].numpy().astype(np.float32)
        a_h = np.matmul(q_h, k_h.T) * scale
        for m_q in range(M_q):
            qkt[m_q*H + h, :] = a_h[m_q, :]
    sm = qkt.astype(np.float64).copy()
    for i in range(sm.shape[0]):
        row = sm[i]; row -= row.max(); np.exp(row, out=row); row /= row.sum()
    out = np.zeros((M_q*H, D), dtype=np.float32)
    for h in range(H):
        v_h = v_t[:, h, :].numpy().astype(np.float32)
        rows = [m_q*H + h for m_q in range(M_q)]
        out[rows, :] = np.matmul(sm.astype(np.float32)[rows, :], v_h)
    return out

print("Computing reference...")
ref_av = ref_attention(Q_t, K_t, V_t, scale)

print("Running Vulkan (single call, C++ internal batching)...")
q_f16 = Q_np.reshape(-1).view(np.uint16)
k_f16 = K_np.reshape(-1).view(np.uint16)
v_f16 = V_np.reshape(-1).view(np.uint16)
o_buf = np.zeros(M_q * H * D, dtype=np.uint16)

t0 = time.time()
ok = lib.dit_run_attention(
    q_f16.ctypes.data_as(ctypes.c_void_p),
    k_f16.ctypes.data_as(ctypes.c_void_p),
    v_f16.ctypes.data_as(ctypes.c_void_p),
    o_buf.ctypes.data_as(ctypes.c_void_p),
    M_q, M_kv, H, D, ctypes.c_float(scale))
t1 = time.time()

if not ok:
    print(f"FAILED - dit_run_attention returned false")
else:
    vk_f32 = o_buf.view(np.float16).astype(np.float32).reshape(M_q*H, D)
    err = np.abs(vk_f32 - ref_av).max()
    print(f"max_err={err:.6f}  time={t1-t0:.1f}s  {'PASS' if err < 0.1 else 'FAIL'}")

# Stress: 3 rounds to check cross-step corruption
print("\nStress test: 3 rounds...")
for r in range(3):
    t0 = time.time()
    ok = lib.dit_run_attention(
        q_f16.ctypes.data_as(ctypes.c_void_p),
        k_f16.ctypes.data_as(ctypes.c_void_p),
        v_f16.ctypes.data_as(ctypes.c_void_p),
        o_buf.ctypes.data_as(ctypes.c_void_p),
        M_q, M_kv, H, D, ctypes.c_float(scale))
    t1 = time.time()
    if not ok:
        print(f"  round {r} FAILED")
    else:
        vk_f32 = o_buf.view(np.float16).astype(np.float32).reshape(M_q*H, D)
        err = np.abs(vk_f32 - ref_av).max()
        print(f"  round {r}: max_err={err:.6f}  time={t1-t0:.1f}s  {'PASS' if err < 0.1 else 'FAIL'}")

print("Done.")
lib.dit_destroy()

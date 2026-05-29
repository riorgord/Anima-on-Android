"""Isolate cross-attention VK_ERROR_DEVICE_LOST at M_kv=1024.
Tests each of 3 passes (QK^T, softmax, A@V) individually against PyTorch reference.
"""
import ctypes, numpy as np, torch, sys
torch.manual_seed(42)

lib = ctypes.CDLL("/data/local/tmp/libdit_vk.so")
lib.dit_init_adaln_only.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
lib.dit_init_adaln_only.restype = ctypes.c_bool
lib.dit_run_qkt.argtypes = [ctypes.c_void_p]*7 + [ctypes.c_float]
lib.dit_run_qkt.restype = ctypes.c_bool
lib.dit_run_softmax.argtypes = [ctypes.c_void_p] + [ctypes.c_int]*3
lib.dit_run_softmax.restype = ctypes.c_bool
lib.dit_run_av.argtypes = [ctypes.c_void_p]*4 + [ctypes.c_int]*4
lib.dit_run_av.restype = ctypes.c_bool
lib.dit_destroy.argtypes = []
lib.dit_destroy.restype = None

print("Init engine...")
ok = lib.dit_init_adaln_only(b"/data/local/tmp/diffusion_weights.bin", b"/data/local/tmp")
assert ok, "init failed"

# Cross-attention params
M_q, M_kv, H, D = 512, 1024, 16, 128
scale = 1.0 / np.sqrt(D)

# Random test data (FP16 range)
def rand_fp16(shape):
    return torch.randn(shape, dtype=torch.float16).numpy()

Q_np = rand_fp16((M_q, H, D))      # [512, 16, 128]
K_np = rand_fp16((M_kv, H, D))     # [1024, 16, 128]
V_np = rand_fp16((M_kv, H, D))

# Flatten for shader
Q_flat = Q_np.reshape(-1).view(np.uint16)
K_flat = K_np.reshape(-1).view(np.uint16)
V_flat = V_np.reshape(-1).view(np.uint16)

# PyTorch reference
Q_t = torch.from_numpy(Q_np).float()
K_t = torch.from_numpy(K_np).float()
V_t = torch.from_numpy(V_np).float()

# Shader layout: Q[M_q*H*D], K[M_kv*H*D], A[gid*M_kv + m_kv] where gid = m_q*H + h
# ↔ A_flat[(m_q*H+h)*M_kv + m_kv] = QK^T[m_q, h, m_kv]
# So we need ref in layout: [M_q*H, M_kv] → flatten row-major
def to_shader_qkt(q_f32, k_f32, scale):
    """QK^T per-head, shader layout: result[(m_q*H+h)*M_kv + m_kv] = sum_d Q[m_q,h,d]*K[m_kv,h,d]*scale"""
    M_q, H, D = q_f32.shape; M_kv = k_f32.shape[0]
    out = np.zeros((M_q*H, M_kv), dtype=np.float32)
    for h in range(H):
        q_h = q_f32[:, h, :].astype(np.float32)  # [M_q, D]
        k_h = k_f32[:, h, :].astype(np.float32)  # [M_kv, D]
        a_h = np.matmul(q_h, k_h.T) * scale      # [M_q, M_kv]
        for m_q in range(M_q):
            out[m_q*H + h, :] = a_h[m_q, :]
    return out

ref_qkt = to_shader_qkt(Q_t.numpy(), K_t.numpy(), scale)
ref_qkt_flat = np.ascontiguousarray(ref_qkt).astype(np.float16).view(np.uint16)

# softmax: same layout, per row (each row is gid)
ref_sm = ref_qkt.astype(np.float64).copy()  # [M_q*H, M_kv], fp64 for stable exp
for i in range(ref_sm.shape[0]):
    row = ref_sm[i]
    row -= row.max()
    np.exp(row, out=row)
    row /= row.sum()
ref_sm_flat = np.ascontiguousarray(ref_sm).astype(np.float16).view(np.uint16)

# A@V: O[(m_q*H+h)*D + d] = sum_{m_kv} A[(m_q*H+h)*M_kv + m_kv] * V[m_kv, h, d]
# Shader layout: A is [M_q*H, M_kv], V is [M_kv*H, D], O is [M_q*H, D]
# A@V per-head: O[(m_q*H+h)*D + d] = sum_{m_kv} A[(m_q*H+h)*M_kv+m_kv] * V[m_kv, h, d]
ref_av = np.zeros((M_q*H, D), dtype=np.float32)
for h in range(H):
    v_h = V_t.numpy()[:, h, :].astype(np.float32)  # [M_kv, D]
    a_rows = [m_q*H+h for m_q in range(M_q)]      # gid indices for this head
    ref_av[a_rows, :] = np.matmul(ref_sm.astype(np.float32)[a_rows, :], v_h)
ref_av_flat = np.ascontiguousarray(ref_av).astype(np.float16).view(np.uint16)

# ── Test 1: QK^T ──
print("\n--- Pass 1: QK^T ---")
vk_a = np.zeros(M_q * H * M_kv, dtype=np.uint16)
ok1 = lib.dit_run_qkt(Q_flat.ctypes, K_flat.ctypes, vk_a.ctypes, M_q, M_kv, H, D, ctypes.c_float(scale))
print(f"  submit: {'OK' if ok1 else 'FAILED'}")
if ok1:
    vk_f32 = vk_a.view(np.float16).astype(np.float32).reshape(M_q*H, M_kv)
    ref_f32 = ref_qkt.astype(np.float32)
    err = np.abs(vk_f32 - ref_f32).max()
    print(f"  max_err={err:.6f}")
    zeros = (vk_f32 == 0.0).sum()
    total = M_q*H*M_kv
    print(f"  zero elements: {zeros}/{total}")
    if err > 0.1:
        bad = np.unravel_index(np.abs(vk_f32-ref_f32).argmax(), ref_f32.shape)
        print(f"  worst at gid={bad[0]} m_kv={bad[1]}: vk={vk_f32[bad]:.4f} ref={ref_f32[bad]:.4f}")
        # Check if the shader output is all zeros
        if zeros == total:
            print("  ALL ZEROS — shader did not write A buffer")

# ── Test 2: softmax ──
print("\n--- Pass 2: softmax ---")
vk_sm = ref_qkt_flat.copy()  # start from correct QK^T
ok2 = lib.dit_run_softmax(vk_sm.ctypes, M_q, M_kv, H)
print(f"  submit: {'OK' if ok2 else 'FAILED'}")
if ok2:
    vk_f32_2 = vk_sm.view(np.float16).astype(np.float32).reshape(M_q*H, M_kv)
    err2 = np.abs(vk_f32_2 - ref_sm.astype(np.float32)).max()
    print(f"  max_err vs fp64-ref={err2:.6f}")
    # Check if any row has NaN
    nan_count = np.isnan(vk_f32_2).sum()
    print(f"  NaN count={nan_count}")
    # Check row 0 against fp16 reference
    fp16_ref_sm = ref_qkt.astype(np.float16)
    for ii in range(fp16_ref_sm.shape[0]):
        row = fp16_ref_sm[ii].astype(np.float32)
        row -= row.max(); row = np.exp(row); row /= row.sum()
        fp16_ref_sm[ii] = row.astype(np.float16)
    err_fp16 = np.abs(vk_f32_2 - fp16_ref_sm.astype(np.float32)).max()
    print(f"  max_err vs fp16-ref={err_fp16:.6f}")

# ── Test 3: A@V ──
print("\n--- Pass 3: A@V ---")
vk_o = np.zeros(M_q * H * D, dtype=np.uint16)
ok3 = lib.dit_run_av(ref_sm_flat.ctypes, V_flat.ctypes, vk_o.ctypes, M_q, M_kv, H, D)
print(f"  submit: {'OK' if ok3 else 'FAILED'}")
if ok3:
    vk_f32_3 = vk_o.view(np.float16).astype(np.float32).reshape(M_q*H, D)
    err3 = np.abs(vk_f32_3 - ref_av.astype(np.float32)).max()
    print(f"  max_err={err3:.6f}")

print("\nDone.")
lib.dit_destroy()

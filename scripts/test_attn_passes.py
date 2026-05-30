"""Test each attention pass separately: QK^T, softmax, A@V."""
import ctypes, numpy as np, torch
torch.manual_seed(42)

lib = ctypes.CDLL("/data/local/tmp/libdit_vk.so")
lib.dit_init_adaln_only.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
lib.dit_init_adaln_only.restype = ctypes.c_bool
lib.dit_run_qkt.argtypes = [ctypes.c_void_p]*3 + [ctypes.c_int]*4 + [ctypes.c_float]
lib.dit_run_qkt.restype = ctypes.c_bool
lib.dit_run_softmax.argtypes = [ctypes.c_void_p] + [ctypes.c_int]*3
lib.dit_run_softmax.restype = ctypes.c_bool
lib.dit_run_av.argtypes = [ctypes.c_void_p]*3 + [ctypes.c_int]*4
lib.dit_run_av.restype = ctypes.c_bool

print("Init...")
ok = lib.dit_init_adaln_only(b"/data/local/tmp/diffusion_weights.bin", b"/data/local/tmp")
assert ok

# Small test: M_q=8, H=1, D=4, M_kv=4, total_rows=8 → 1 WG with ROWS=8
M_q, M_kv, H, D = 8, 4, 1, 4
scale = 1.0 / np.sqrt(D)

Q = torch.randn(M_q, H, D, dtype=torch.float16).numpy()
K = torch.randn(M_kv, H, D, dtype=torch.float16).numpy()
V = torch.randn(M_kv, H, D, dtype=torch.float16).numpy()

# CPU reference
def ref_qkt(q, k, scale):
    Mq, H, D = q.shape
    Mk = k.shape[0]
    A = np.zeros((Mq*H, Mk), dtype=np.float32)
    for mq in range(Mq):
        for h in range(H):
            for mk in range(Mk):
                s = 0.0
                for d in range(D):
                    s += float(q[mq,h,d]) * float(k[mk,h,d])
                A[mq*H+h, mk] = s * scale
    return A

def ref_softmax(A):
    A32 = A.astype(np.float64).copy()
    for i in range(A32.shape[0]):
        row = A32[i]
        row -= row.max()
        np.exp(row, out=row)
        row /= row.sum()
    return A32.astype(np.float32)

def ref_av(A, v):
    MqH, Mk = A.shape
    Mq = MqH // 1  # H=1
    H = 1
    D = v.shape[-1]
    O = np.zeros((MqH, D), dtype=np.float32)
    for mq in range(Mq):
        for h in range(H):
            gid = mq*H + h
            for d in range(D):
                s = 0.0
                for mk in range(Mk):
                    s += A[gid, mk] * float(v[mk, h, d])
                O[gid, d] = s
    return O

ref_A = ref_qkt(Q, K, scale)
ref_S = ref_softmax(ref_A)
ref_O = ref_av(ref_S, V)
print(f"Ref: A max={ref_A.max():.3f}, S sum≈1={ref_S.sum(axis=1).mean():.4f}, O max={ref_O.max():.3f}")

# Test QK^T
print("\n--- QK^T ---")
A_vk = np.zeros(M_q * H * M_kv, dtype=np.uint16)
ok = lib.dit_run_qkt(
    Q.ravel().view(np.uint16).ctypes.data_as(ctypes.c_void_p),
    K.ravel().view(np.uint16).ctypes.data_as(ctypes.c_void_p),
    A_vk.ctypes.data_as(ctypes.c_void_p),
    M_q, M_kv, H, D, ctypes.c_float(scale))
A_vk32 = A_vk.view(np.float16).astype(np.float32).reshape(M_q*H, M_kv)
err_a = np.abs(A_vk32 - ref_A).max()
print(f"QK^T: ok={ok}, max_err={err_a:.6f}")

# Test softmax
print("\n--- Softmax ---")
S_vk = ref_A.astype(np.float16).view(np.uint16).copy()
ok = lib.dit_run_softmax(S_vk.ctypes.data_as(ctypes.c_void_p), M_q, M_kv, H)
S_vk32 = S_vk.view(np.float16).astype(np.float32).reshape(M_q*H, M_kv)
err_s = np.abs(S_vk32 - ref_S).max()
print(f"Softmax: ok={ok}, max_err={err_s:.6f}")

# Test A@V
print("\n--- A@V ---")
O_vk = np.zeros(M_q * H * D, dtype=np.uint16)
A_in = ref_S.astype(np.float16).view(np.uint16).copy()
ok = lib.dit_run_av(
    A_in.ctypes.data_as(ctypes.c_void_p),
    V.ravel().view(np.uint16).ctypes.data_as(ctypes.c_void_p),
    O_vk.ctypes.data_as(ctypes.c_void_p),
    M_q, M_kv, H, D)
O_vk32 = O_vk.view(np.float16).astype(np.float32).reshape(M_q*H, D)
err_o = np.abs(O_vk32 - ref_O).max()
print(f"A@V: ok={ok}, max_err={err_o:.6f}")

# Now test full pipeline
print("\n--- Full attention ---")
lib.dit_run_attention.argtypes = [ctypes.c_void_p]*4 + [ctypes.c_int]*4 + [ctypes.c_float]
lib.dit_run_attention.restype = ctypes.c_bool
O_full = np.zeros(M_q * H * D, dtype=np.uint16)
ok = lib.dit_run_attention(
    Q.ravel().view(np.uint16).ctypes.data_as(ctypes.c_void_p),
    K.ravel().view(np.uint16).ctypes.data_as(ctypes.c_void_p),
    V.ravel().view(np.uint16).ctypes.data_as(ctypes.c_void_p),
    O_full.ctypes.data_as(ctypes.c_void_p),
    M_q, M_kv, H, D, ctypes.c_float(scale))
err_full = np.abs(O_full.view(np.float16).astype(np.float32).reshape(M_q*H, D) - ref_O).max()
print(f"Full: ok={ok}, max_err={err_full:.6f}")

lib.dit_destroy()

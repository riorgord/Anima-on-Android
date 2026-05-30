"""Test attention with ROWS_PER_WG=1, M_q=64 (1024 WG, safe)."""
import ctypes, numpy as np, torch
torch.manual_seed(42)

lib = ctypes.CDLL("/data/local/tmp/libdit_vk.so")
lib.dit_init_adaln_only.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
lib.dit_init_adaln_only.restype = ctypes.c_bool
lib.dit_run_attention.argtypes = [ctypes.c_void_p]*4 + [ctypes.c_int]*4 + [ctypes.c_float]
lib.dit_run_attention.restype = ctypes.c_bool

print("Init...")
ok = lib.dit_init_adaln_only(b"/data/local/tmp/diffusion_weights.bin", b"/data/local/tmp")
assert ok

M_q, M_kv, H, D = 64, 1024, 16, 128  # small M_q = 64 → 64*16=1024 WG safe
scale = 1.0 / np.sqrt(D)

def rand_fp16(shape):
    return torch.randn(shape, dtype=torch.float16).numpy()

Q = rand_fp16((M_q, H, D))
K = rand_fp16((M_kv, H, D))
V = rand_fp16((M_kv, H, D))

# Reference
Qt = torch.from_numpy(Q).float(); Kt = torch.from_numpy(K).float(); Vt = torch.from_numpy(V).float()
qkt = np.zeros((M_q*H, M_kv), dtype=np.float32)
for h in range(H):
    a_h = np.matmul(Qt[:,h,:].numpy(), Kt[:,h,:].numpy().T) * scale
    for m_q in range(M_q):
        qkt[m_q*H+h, :] = a_h[m_q, :]
sm = qkt.astype(np.float64).copy()
for i in range(sm.shape[0]):
    row = sm[i]; row -= row.max(); np.exp(row, out=row); row /= row.sum()
ref = np.zeros((M_q*H, D), dtype=np.float32)
for h in range(H):
    v_h = Vt[:,h,:].numpy()
    rows = [m_q*H+h for m_q in range(M_q)]
    ref[rows, :] = np.matmul(sm.astype(np.float32)[rows, :], v_h)

O = np.zeros(M_q * H * D, dtype=np.uint16)
ok = lib.dit_run_attention(
    Q.reshape(-1).view(np.uint16).ctypes.data_as(ctypes.c_void_p),
    K.reshape(-1).view(np.uint16).ctypes.data_as(ctypes.c_void_p),
    V.reshape(-1).view(np.uint16).ctypes.data_as(ctypes.c_void_p),
    O.ctypes.data_as(ctypes.c_void_p),
    M_q, M_kv, H, D, ctypes.c_float(scale))
err = np.abs(O.view(np.float16).astype(np.float32).reshape(M_q*H, D) - ref).max()
print(f"max_err={err:.6f} {'OK' if err < 0.01 else 'FAIL'}")
lib.dit_destroy()

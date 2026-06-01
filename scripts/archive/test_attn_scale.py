"""Test ROWS=8 attention at different scales."""
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

def test(M_q, M_kv, H, D, label):
    scale = 1.0 / np.sqrt(D)
    Q = torch.randn(M_q, H, D, dtype=torch.float16).numpy()
    K = torch.randn(M_kv, H, D, dtype=torch.float16).numpy()
    V = torch.randn(M_kv, H, D, dtype=torch.float16).numpy()
    Qt = torch.from_numpy(Q).float()
    Kt = torch.from_numpy(K).float()
    Vt = torch.from_numpy(V).float()

    # reference
    qkt = np.zeros((M_q*H, M_kv), dtype=np.float32)
    for h in range(H):
        a_h = np.matmul(Qt[:,h,:].numpy(), Kt[:,h,:].numpy().T) * scale
        for mq in range(M_q): qkt[mq*H+h, :] = a_h[mq, :]
    sm = qkt.astype(np.float64).copy()
    for i in range(sm.shape[0]):
        row = sm[i]; row -= row.max(); np.exp(row, out=row); row /= row.sum()
    ref = np.zeros((M_q*H, D), dtype=np.float32)
    for h in range(H):
        v_h = Vt[:,h,:].numpy()
        rows = [mq*H+h for mq in range(M_q)]
        ref[rows, :] = np.matmul(sm.astype(np.float32)[rows, :], v_h)

    O = np.zeros(M_q * H * D, dtype=np.uint16)
    ok = lib.dit_run_attention(
        Q.ravel().view(np.uint16).ctypes.data_as(ctypes.c_void_p),
        K.ravel().view(np.uint16).ctypes.data_as(ctypes.c_void_p),
        V.ravel().view(np.uint16).ctypes.data_as(ctypes.c_void_p),
        O.ctypes.data_as(ctypes.c_void_p),
        M_q, M_kv, H, D, ctypes.c_float(scale))
    err = np.abs(O.view(np.float16).astype(np.float32).reshape(M_q*H, D) - ref).max()
    rows = M_q * H
    wgs = (rows + 7) // 8
    print(f"  {label}: M_q={M_q} H={H} rows={rows} WGs={wgs} max_err={err:.6f} {'OK' if err < 0.01 else 'FAIL'}")

# Different scales
test(8, 4, 1, 4, "tiny")       # 8 rows, 1 WG
test(64, 4, 1, 4, "64r-1h")    # 64 rows, 8 WGs
test(128, 4, 1, 4, "128r-1h")  # 128 rows, 16 WGs
test(256, 4, 1, 4, "256r-1h")  # 256 rows, 32 WGs
test(64, 512, 16, 128, "64q-sa")  # 1024 rows, 128 WGs
test(128, 512, 16, 128, "128q")   # 2048 rows, 256 WGs
test(256, 512, 16, 128, "256q")   # 4096 rows, 512 WGs
test(512, 512, 16, 128, "512q")   # 8192 rows, 1024 WGs

print("done")
lib.dit_destroy()

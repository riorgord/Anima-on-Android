"""Test QK^T and A@V individually with ROWS=8, M_kv=1024.
These have no barriers → streaming mode → should handle ROWS=8 fine.
Softmax has barriers → test separately."""

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

M_q, M_kv, H, D = 512, 1024, 16, 128
scale = 1.0 / np.sqrt(D)
Q = torch.randn(M_q, H, D, dtype=torch.float16).numpy()
K = torch.randn(M_kv, H, D, dtype=torch.float16).numpy()
V = torch.randn(M_kv, H, D, dtype=torch.float16).numpy()

# CPU reference QK^T
Qt=torch.from_numpy(Q).float(); Kt=torch.from_numpy(K).float()
ref_A=np.zeros((M_q*H, M_kv), dtype=np.float32)
for h in range(H):
    a_h=np.matmul(Qt[:,h,:].numpy(), Kt[:,h,:].numpy().T)*scale
    for mq in range(M_q): ref_A[mq*H+h,:]=a_h[mq,:]
print(f"Ref A: shape={ref_A.shape}, max={ref_A.max():.3f}")

# Test QK^T
A_vk=np.zeros(M_q*H*M_kv, dtype=np.uint16)
ok=lib.dit_run_qkt(Q.ravel().view(np.uint16).ctypes.data_as(ctypes.c_void_p),
    K.ravel().view(np.uint16).ctypes.data_as(ctypes.c_void_p),
    A_vk.ctypes.data_as(ctypes.c_void_p), M_q, M_kv, H, D, ctypes.c_float(scale))
err_a=np.abs(A_vk.view(np.float16).astype(np.float32).reshape(M_q*H,M_kv)-ref_A).max()
print(f"QK^T: ok={ok}, max_err={err_a:.6f} {'OK' if err_a<0.01 else 'FAIL'}")

# CPU reference softmax
S_ref=ref_A.astype(np.float64).copy()
for i in range(S_ref.shape[0]): row=S_ref[i];row-=row.max();np.exp(row,out=row);row/=row.sum()

# Test softmax
S_vk=ref_A.astype(np.float16).view(np.uint16).copy()
ok=lib.dit_run_softmax(S_vk.ctypes.data_as(ctypes.c_void_p), M_q, M_kv, H)
err_s=np.abs(S_vk.view(np.float16).astype(np.float32).reshape(M_q*H,M_kv)-S_ref.astype(np.float32)).max()
print(f"Softmax: ok={ok}, max_err={err_s:.6f} {'OK' if err_s<0.01 else 'FAIL'}")

# CPU reference A@V
S_f32=S_ref.astype(np.float32); Vt=torch.from_numpy(V).float()
ref_O=np.zeros((M_q*H, D), dtype=np.float32)
for h in range(H):
    v_h=Vt[:,h,:].numpy(); rows=[mq*H+h for mq in range(M_q)]
    ref_O[rows,:]=np.matmul(S_f32[rows,:], v_h)

# Test A@V
O_vk=np.zeros(M_q*H*D, dtype=np.uint16)
ok=lib.dit_run_av(S_f32.astype(np.float16).view(np.uint16).ctypes.data_as(ctypes.c_void_p),
    V.ravel().view(np.uint16).ctypes.data_as(ctypes.c_void_p),
    O_vk.ctypes.data_as(ctypes.c_void_p), M_q, M_kv, H, D)
err_o=np.abs(O_vk.view(np.float16).astype(np.float32).reshape(M_q*H,D)-ref_O).max()
print(f"A@V: ok={ok}, max_err={err_o:.6f} {'OK' if err_o<0.01 else 'FAIL'}")

print("done")
lib.dit_destroy()

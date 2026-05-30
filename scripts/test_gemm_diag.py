"""Diagnose dit_run_gemm: does it work?"""
import ctypes, numpy as np, torch, time
torch.manual_seed(42)

lib = ctypes.CDLL("/data/local/tmp/libdit_vk.so")
lib.dit_init_adaln_only.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
lib.dit_init_adaln_only.restype = ctypes.c_bool
lib.dit_run_gemm.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int]
lib.dit_run_gemm.restype = ctypes.c_bool

print("Init...")
ok = lib.dit_init_adaln_only(b"/data/local/tmp/diffusion_weights.bin", b"/data/local/tmp")
print(f"init: {ok}")

# Test various sizes
sizes = [
    (512, 2048, 2048, "Q_proj"),
    (512, 8192, 2048, "fc1"),
    (512, 2048, 8192, "fc2"),
    (1024, 2048, 1024, "cross_K_proj"),
    (512, 68, 2048, "x_embed"),
]
for M, N, K, label in sizes:
    A = torch.randn(M, K, dtype=torch.float16).numpy().view(np.uint16)
    B = torch.randn(N, K, dtype=torch.float16).numpy().view(np.uint16)
    C = np.zeros(M * N, dtype=np.uint16)
    C_ref = np.matmul(A.view(np.float16).reshape(M, K).astype(np.float32),
                      B.view(np.float16).reshape(N, K).T.astype(np.float32))

    ok = lib.dit_run_gemm(
        A.ctypes.data_as(ctypes.c_void_p),
        B.ctypes.data_as(ctypes.c_void_p), int(B.nbytes),
        C.ctypes.data_as(ctypes.c_void_p),
        M, N, K)
    if ok:
        err = np.abs(C.view(np.float16).astype(np.float32).reshape(M, N) - C_ref).max()
        print(f"{label} M={M} N={N} K={K}: max_err={err:.6f} {'OK' if err < 0.1 else 'FAIL'}")
    else:
        print(f"{label} M={M} N={N} K={K}: dit_run_gemm returned FALSE")

print("done")

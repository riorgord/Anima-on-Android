import ctypes, numpy as np, torch; torch.manual_seed(42)
lib = ctypes.CDLL("/data/local/tmp/libdit_vk.so")
lib.dit_init_adaln_only.argtypes=[ctypes.c_char_p,ctypes.c_char_p]; lib.dit_init_adaln_only.restype=ctypes.c_bool
lib.dit_run_gemm.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_int,ctypes.c_void_p,ctypes.c_int,ctypes.c_int,ctypes.c_int]
lib.dit_run_gemm.restype=ctypes.c_bool
ok=lib.dit_init_adaln_only(b"/data/local/tmp/diffusion_weights.bin",b"/data/local/tmp")
for M,N,K,label in [(512,2048,2048,'Q_proj'),(512,8192,2048,'fc1'),(512,2048,8192,'fc2'),(1024,2048,1024,'cross_K')]:
    A=torch.randn(M,K,dtype=torch.float16).numpy().view(np.uint16)
    B=torch.randn(N,K,dtype=torch.float16).numpy().view(np.uint16)
    C=np.zeros(M*N,dtype=np.uint16)
    ref=np.matmul(A.view(np.float16).reshape(M,K).astype(np.float32),B.view(np.float16).reshape(N,K).T.astype(np.float32))
    ok=lib.dit_run_gemm(A.ctypes.data_as(ctypes.c_void_p),B.ctypes.data_as(ctypes.c_void_p),int(B.nbytes),C.ctypes.data_as(ctypes.c_void_p),M,N,K)
    e=np.abs(C.view(np.float16).astype(np.float32).reshape(M,N)-ref).max()
    print(f"{label} M={M} N={N} K={K}: err={e:.6f} {'OK' if e<0.05 else 'FAIL'}")

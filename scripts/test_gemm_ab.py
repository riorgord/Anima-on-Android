"""Compare dit_run_gemm (new) vs vk_gemm_run_fp16 (old)."""
import ctypes, numpy as np, torch; torch.manual_seed(42)

lib_old = ctypes.CDLL("/data/local/tmp/libvk_gemm.so")
lib_old.vk_gemm_init.argtypes=[ctypes.c_int]*4; lib_old.vk_gemm_init.restype=ctypes.c_bool
lib_old.vk_gemm_run_fp16.argtypes=[ctypes.c_void_p]*3+[ctypes.c_int]*3; lib_old.vk_gemm_run_fp16.restype=ctypes.c_bool
lib_old.vk_gemm_init(1024,8192,8192,16)

lib_new = ctypes.CDLL("/data/local/tmp/libdit_vk.so")
lib_new.dit_init_adaln_only.argtypes=[ctypes.c_char_p,ctypes.c_char_p]; lib_new.dit_init_adaln_only.restype=ctypes.c_bool
lib_new.dit_run_gemm.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_int,ctypes.c_void_p,ctypes.c_int,ctypes.c_int,ctypes.c_int]; lib_new.dit_run_gemm.restype=ctypes.c_bool
lib_new.dit_init_adaln_only(b"/data/local/tmp/diffusion_weights.bin",b"/data/local/tmp")

for M,N,K,label in [(512,2048,2048,'Q_proj'),(512,8192,2048,'fc1')]:
    A=torch.randn(M,K,dtype=torch.float16).numpy().view(np.uint16)
    B=torch.randn(N,K,dtype=torch.float16).numpy().view(np.uint16)

    C_old=np.zeros(M*N,dtype=np.uint16)
    lib_old.vk_gemm_run_fp16(C_old.ctypes.data_as(ctypes.c_void_p),A.ctypes.data_as(ctypes.c_void_p),B.ctypes.data_as(ctypes.c_void_p),M,N,K)

    C_new=np.zeros(M*N,dtype=np.uint16)
    lib_new.dit_run_gemm(A.ctypes.data_as(ctypes.c_void_p),B.ctypes.data_as(ctypes.c_void_p),int(B.nbytes),C_new.ctypes.data_as(ctypes.c_void_p),M,N,K)

    diff=np.abs(C_old.view(np.float16).astype(np.float32)-C_new.view(np.float16).astype(np.float32)).max()
    print(f"{label}: old_vs_new_diff={diff:.6f} {'SAME' if diff<1e-6 else 'DIFFER!'}")

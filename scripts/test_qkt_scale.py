import ctypes, numpy as np, torch; torch.manual_seed(42)
lib = ctypes.CDLL("/data/local/tmp/libdit_vk.so")
lib.dit_init_adaln_only.argtypes=[ctypes.c_char_p,ctypes.c_char_p]; lib.dit_init_adaln_only.restype=ctypes.c_bool
lib.dit_run_qkt.argtypes=[ctypes.c_void_p]*3+[ctypes.c_int]*4+[ctypes.c_float]; lib.dit_run_qkt.restype=ctypes.c_bool
lib.dit_init_adaln_only(b"/data/local/tmp/diffusion_weights.bin",b"/data/local/tmp")
M_kv,H,D,s=1024,16,128,1.0/np.sqrt(128)
for M_q in [8,16,32,64,128,256,512]:
    Q=torch.randn(M_q,H,D,dtype=torch.float16).numpy(); K=torch.randn(M_kv,H,D,dtype=torch.float16).numpy()
    Qt,Kt=torch.from_numpy(Q).float(),torch.from_numpy(K).float()
    ref=np.zeros((M_q*H,M_kv),dtype=np.float32)
    for h in range(H):
        a=np.matmul(Qt[:,h,:].numpy(),Kt[:,h,:].numpy().T)*s
        for mq in range(M_q): ref[mq*H+h,:]=a[mq,:]
    A_vk=np.zeros(M_q*H*M_kv,dtype=np.uint16)
    ok=lib.dit_run_qkt(Q.ravel().view(np.uint16).ctypes.data_as(ctypes.c_void_p),K.ravel().view(np.uint16).ctypes.data_as(ctypes.c_void_p),A_vk.ctypes.data_as(ctypes.c_void_p),M_q,M_kv,H,D,ctypes.c_float(s))
    e=np.abs(A_vk.view(np.float16).astype(np.float32).reshape(M_q*H,M_kv)-ref).max()
    rows=M_q*H; wgs=(rows+7)//8
    print(f"M_q={M_q:3d} rows={rows:5d} WGs={wgs:4d} ok={ok} err={e:.6f} {'OK' if e<0.01 else 'FAIL'}")

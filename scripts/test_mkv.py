import ctypes, numpy as np, torch; torch.manual_seed(42)
lib = ctypes.CDLL("/data/local/tmp/libdit_vk.so")
lib.dit_init_adaln_only.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
lib.dit_init_adaln_only.restype = ctypes.c_bool
lib.dit_run_attention.argtypes = [ctypes.c_void_p]*4+[ctypes.c_int]*4+[ctypes.c_float]
lib.dit_run_attention.restype = ctypes.c_bool
lib.dit_init_adaln_only(b"/data/local/tmp/diffusion_weights.bin", b"/data/local/tmp")

for M_q in [64, 128, 256, 512]:
    M_kv, H, D, s = 1024, 16, 128, 1.0/np.sqrt(128)
    Q=torch.randn(M_q,H,D,dtype=torch.float16).numpy()
    K=torch.randn(M_kv,H,D,dtype=torch.float16).numpy()
    V=torch.randn(M_kv,H,D,dtype=torch.float16).numpy()
    Qt,Kt,Vt=torch.from_numpy(Q).float(),torch.from_numpy(K).float(),torch.from_numpy(V).float()
    qkt=np.zeros((M_q*H,M_kv),dtype=np.float32)
    for h in range(H):
        a=np.matmul(Qt[:,h,:].numpy(),Kt[:,h,:].numpy().T)*s
        for mq in range(M_q): qkt[mq*H+h,:]=a[mq,:]
    sm=qkt.astype(np.float64).copy()
    for i in range(sm.shape[0]): row=sm[i];row-=row.max();np.exp(row,out=row);row/=row.sum()
    ref=np.zeros((M_q*H,D),dtype=np.float32)
    for h in range(H):
        v_h=Vt[:,h,:].numpy();rows=[mq*H+h for mq in range(M_q)]
        ref[rows,:]=np.matmul(sm.astype(np.float32)[rows,:],v_h)
    O=np.zeros(M_q*H*D,dtype=np.uint16)
    lib.dit_run_attention(Q.ravel().view(np.uint16).ctypes.data_as(ctypes.c_void_p),K.ravel().view(np.uint16).ctypes.data_as(ctypes.c_void_p),V.ravel().view(np.uint16).ctypes.data_as(ctypes.c_void_p),O.ctypes.data_as(ctypes.c_void_p),M_q,M_kv,H,D,ctypes.c_float(s))
    e=np.abs(O.view(np.float16).astype(np.float32).reshape(M_q*H,D)-ref).max()
    rows=M_q*H;wgs=(rows+7)//8
    print(f"M_q={M_q:3d} rows={rows:5d} WGs={wgs:4d} max_err={e:.6f} {'OK' if e<0.01 else 'FAIL'}")

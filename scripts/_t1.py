"""Minimal test: init + single GEMM using dit_vk engine"""
import ctypes, torch, numpy as np, sys
sys.path.insert(0, "/sdcard/anima_on_android/scripts")

# Use libvk_gemm.so (the WORKING engine) to verify GEMM works
import vk_linear as _vk
if not _vk._INITIALIZED:
    _vk._lib.vk_gemm_init(1024,8192,8192,16)
    _vk._INITIALIZED = True

MS,D=512,2048
x=torch.randn(MS,D,dtype=torch.float16)
w=torch.randn(D,D,dtype=torch.float16)

out=np.zeros((MS,D),dtype=np.uint16)
ok=_vk._lib.vk_gemm_run_fp16(
    out.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),
    x.numpy().view(np.uint16).ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),
    w.numpy().view(np.uint16).ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),
    MS,D,D)
r=torch.from_numpy(out.view(np.float16))
ref=torch.nn.functional.linear(x.float(),w.float()).half()
err=(r.float()-ref.float()).abs().max()
print(f"libvk_gemm single GEMM: max_err={float(err):.5f} ok={ok}")
print(f"output mean={float(r.float().mean()):.4f} non-zero={(r!=0).sum().item()}")

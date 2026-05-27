"""Use verified libvk_gemm.so GEMM with dit_vk loaded weights — just check if weight buffers work"""
import ctypes, torch, numpy as np, sys, time
sys.path.insert(0, "/sdcard/anima_on_android/scripts")
import vk_linear as _vk
if not _vk._INITIALIZED:
    _vk._lib.vk_gemm_init(1024,8192,8192,16)
    _vk._INITIALIZED = True

# Do ONE GEMM with libvk_gemm — baseline verification
MS,D=512,2048
x=torch.randn(MS,D,dtype=torch.float16)
out=np.zeros((MS,D),dtype=np.uint16)
ok=_vk._lib.vk_gemm_run_fp16(
    out.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),
    x.numpy().view(np.uint16).ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),
    x.numpy().view(np.uint16).ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),
    MS,D,D)
r=torch.from_numpy(out.view(np.float16))
print(f"libvk_gemm: ok={ok} mean={float(r.float().mean()):.4f} non-zero={(r!=0).sum().item()}")

# Now can we load dit_vk weights from Python and pass to libvk_gemm?
# The weights are in Vulkan buffers inside libdit_vk.so — we can't access them from Python
# But we CAN verify that the dit_vk engine's init works (we already know it does)
print("Test complete")

"""Minimal test: compare bare Vulkan GEMM (PT vs ND) with same input."""
import sys, struct, json, ctypes
sys.path.insert(0, "/sdcard/anima_on_android/src")
sys.path.insert(0, "/sdcard/anima_on_android/scripts")
import torch, numpy as np, vk_ops

# Load just the first AdaLN weight into Vulkan
vk_ops._lib.vk_engine_init()
SAFETENSORS = "/sdcard/anima_on_android/models/diffusion.safetensors"
with open(SAFETENSORS, 'rb') as f:
    hl = struct.unpack('<Q', f.read(8))[0]
    hd = json.loads(f.read(hl).decode('utf-8'))
    ds = 8 + hl

PREFIX = "net."
for key in hd:
    if key == "__metadata__": continue
    ck = key[len(PREFIX):] if key.startswith(PREFIX) else key
    if ck == 'blocks.0.adaln_modulation_self_attn.1.weight':
        info = hd[key]; off = ds + info['data_offsets'][0]; end = ds + info['data_offsets'][1]
        with open(SAFETENSORS, 'rb') as f:
            f.seek(off); buf = f.read(end-off)
        data = np.frombuffer(buf, dtype=np.uint16).reshape(info['shape'])
        sl = list(info['shape'])
        dc = {'BF16':2,'F16':1,'F32':0}.get(info['dtype'],2)
        ret = vk_ops._lib.vk_weight_add(ck.encode(), data.ctypes.data, dc, (ctypes.c_int*len(sl))(*sl), len(sl))
        print(f"vk_weight_add({ck}): {ret}  shape={info['shape']}  dtype={info['dtype']}")
        break

vk_ops._lib.vk_engine_finalize()

# Same input for both
test_in = np.ones((1, 2048), dtype=np.float32) * 0.5
M, N, K = 1, 256, 2048

# PT: create a minimal PT model with just one VulkanGemmLinear
import torch.nn as nn
pt_lin = vk_ops.VulkanGemmLinear('blocks.0.adaln_modulation_self_attn.1.weight', 2048, 256, bias=False)
pt_in = torch.from_numpy(test_in).to(torch.float32)
pt_out = pt_lin.forward(pt_in)
pt_out_np = pt_out.float().cpu().numpy()

# ND: call vk_run_gemm directly
vk_ops._lib.vk_reset_pool()
nd_out = np.zeros((M, N), dtype=np.float32)
ok = vk_ops._lib.vk_run_gemm(b'blocks.0.adaln_modulation_self_attn.1.weight',
                              test_in.ctypes.data, nd_out.ctypes.data, M, N, K)
print(f"vk_run_gemm: ok={ok}")

err = np.abs(nd_out - pt_out_np).max()
print(f"\nmax_err: {err:.8f}")
if err > 1e-6:
    print(f"  PT[0,:5] = {pt_out_np[0,:5]}")
    print(f"  ND[0,:5] = {nd_out[0,:5]}")
    print(f"  PT range: [{pt_out_np.min():.6f},{pt_out_np.max():.6f}]")
    print(f"  ND range: [{nd_out.min():.6f},{nd_out.max():.6f}]")
print(f"RESULT: {'PASS' if err < 1e-6 else 'FAIL'}")

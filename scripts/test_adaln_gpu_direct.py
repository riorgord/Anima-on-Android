"""Direct test: GPU adaln_gpu output vs CPU reference (with lora)."""
import ctypes, numpy as np, torch, torch.nn.functional as F, time, sys

lib = ctypes.CDLL("/data/local/tmp/libdit_vk.so")
lib.dit_init.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
lib.dit_init.restype = ctypes.c_bool
lib.dit_write_lora.argtypes = [ctypes.c_void_p]
lib.dit_write_lora.restype = None
lib.dit_record_adaln_gpu_test.argtypes = [ctypes.c_int]
lib.dit_record_adaln_gpu_test.restype = ctypes.c_bool
lib.dit_read_buf.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_size_t]
lib.dit_read_buf.restype = ctypes.c_bool
lib.dit_forward.argtypes = [ctypes.c_void_p]*5 + [ctypes.c_int]*4
lib.dit_forward.restype = ctypes.c_bool
lib.dit_destroy.argtypes = []
lib.dit_destroy.restype = None

MS, D, M, S = 512, 2048, 2, 256
n_elem = MS * D

print("init...")
t0 = time.time()
ok = lib.dit_init(b"/data/local/tmp/diffusion_weights.bin", b"/data/local/tmp")
print(f"  init={ok} ({time.time()-t0:.1f}s)")
if not ok: sys.exit(1)

# Load t_emb and lora
t_emb = np.fromfile("/sdcard/anima_on_android/output/t_step0.bin", dtype=np.float16).reshape(M, D)
lora = np.fromfile("/sdcard/anima_on_android/output/lora_step0.bin", dtype=np.float16).reshape(3, M, D)

lib.dit_write_lora(lora.ctypes.data_as(ctypes.c_void_p))
print(f"lora: {lora.shape} uploaded")

# Record GPU adaln for block 0 self-attn
ok = lib.dit_record_adaln_gpu_test(0)
print(f"record adaln = {ok}")

# Run forward (just submits cmd[0] — GPU computes adaln)
x = np.random.randn(MS, D).astype(np.float16)
ctx = np.zeros((M, 512, 1024), dtype=np.float16)
out = np.zeros((MS, D), dtype=np.float16)

print("running GPU adaln...")
ok = lib.dit_forward(
    x.ctypes.data_as(ctypes.c_void_p),
    t_emb.ctypes.data_as(ctypes.c_void_p),
    ctx.ctypes.data_as(ctypes.c_void_p),
    out.ctypes.data_as(ctypes.c_void_p),
    MS, D, M, 512, 1024)
print(f"  forward={ok}")

# Read GPU bcBuf — contains scale/shift/gate at slots [0,1,2]
# bcBuf layout: 9 components × MS*D = 9 × 512×2048 × 2 = 18MB total
# Slots 0=scale, 1=shift, 2=gate (each MS*D fp16)
bc = np.zeros(n_elem * 9, dtype=np.uint16)
lib.dit_read_buf(4, bc.ctypes.data_as(ctypes.c_void_p), bc.nbytes)

def read_comp(slot):
    return bc[slot*n_elem:(slot+1)*n_elem].view(np.float16).reshape(MS, D).astype(np.float32)

gpu_scale = read_comp(0)
gpu_shift = read_comp(1)
gpu_gate  = read_comp(2)

print(f"GPU adaln: scale mean={gpu_scale.mean():.4f} shift={gpu_shift.mean():.4f} gate={gpu_gate.mean():.4f}")
print(f"  scale[0,0:4]={gpu_scale[0,:4]}")
print(f"  shift[0,0:4]={gpu_shift[0,:4]}")

# CPU reference: with lora
t32 = torch.from_numpy(t_emb.astype(np.float32))  # [M, D]
sd = torch.load("/sdcard/anima_on_android/models/diffusion_weights_fp16.pt", weights_only=True, map_location="cpu")
w1 = sd["blocks.0.adaln_modulation_self_attn.1.weight"].float()
w2 = sd["blocks.0.adaln_modulation_self_attn.2.weight"].float()
del sd

# CPU: SiLU → GEMM(D,256) → GEMM(256,3D) → +lora → scale+1 → broadcast
h = F.silu(t32)
h = F.linear(h, w1)
h = F.linear(h, w2)
lora_t = torch.from_numpy(lora.astype(np.float32))  # [3, M, D]
lora_cat = torch.cat([lora_t[0], lora_t[1], lora_t[2]], dim=-1)  # [M, 3D]
h = h + lora_cat
sh, sc, ga = torch.chunk(h, 3, dim=-1)
sc = sc + 1.0
sc_b = sc.repeat_interleave(S, 0).numpy().astype(np.float32)
sh_b = sh.repeat_interleave(S, 0).numpy().astype(np.float32)
ga_b = ga.repeat_interleave(S, 0).numpy().astype(np.float32)

print(f"CPU adaln: scale mean={sc_b.mean():.4f} shift={sh_b.mean():.4f} gate={ga_b.mean():.4f}")
print(f"  scale[0,0:4]={sc_b[0,:4]}")

err_s = np.abs(gpu_scale - sc_b).max()
err_h = np.abs(gpu_shift - sh_b).max()
err_g = np.abs(gpu_gate - ga_b).max()
print(f"\nmax_err: scale={err_s:.5f} shift={err_h:.5f} gate={err_g:.5f}")

if max(err_s, err_h, err_g) < 0.1:
    print("PASS — GPU adaln_gpu matches CPU (with lora)")
else:
    print("CHECK — review errors")

lib.dit_destroy()
print("DONE")

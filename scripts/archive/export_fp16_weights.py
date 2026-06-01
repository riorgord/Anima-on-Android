"""Convert BF16 safetensors to FP16 .pt for phone-equivalent weights."""
import sys, time, gc, os, glob
import safetensors.torch
import torch
import numpy as np

DEV = "cuda"  # GPU for fast conversion
SRC_DIR = "/mnt/d/AI"
OUTDIR = "/mnt/d/AI/anima_phone/models"
os.makedirs(OUTDIR, exist_ok=True)

# Find safetensors files using Python glob (handles Unicode paths correctly)
print("Searching for safetensors files...")
all_safetensors = glob.glob(os.path.join(SRC_DIR, "**", "*.safetensors"), recursive=True)
for f in all_safetensors:
    print(f"  {f}")

# Identify the key model files
diff_path = None
vae_path = None
for f in all_safetensors:
    if "anima-base" in f and "diffusion" in f:
        diff_path = f
    elif "qwen_image_vae" in f:
        vae_path = f

if not diff_path:
    print("ERROR: diffusion model safetensors not found!")
    print("Looking for anima-base-v1.0 in:", all_safetensors)
    sys.exit(1)
if not vae_path:
    print("ERROR: VAE safetensors not found!")
    sys.exit(1)

# ── 1. Diffusion model: BF16 → FP16 ──
print(f"\nConverting diffusion model...")
print(f"  Source: {diff_path}")
t0 = time.time()
diff_sd = safetensors.torch.load_file(diff_path, device="cpu")
print(f"  Loaded {len(diff_sd)} tensors in {time.time()-t0:.0f}s")

# Convert to FP16 and add "net." prefix (matching phone format)
print("  Converting BF16 → FP16...")
fp16_sd = {}
total_params = 0
for k, v in diff_sd.items():
    # Phone format: keys have "net." prefix
    new_k = "net." + k
    fp16_sd[new_k] = v.to(torch.float16)
    total_params += v.numel()
del diff_sd; gc.collect()

out_path = f"{OUTDIR}/diffusion_weights_fp16.pt"
torch.save(fp16_sd, out_path)
size_gb = os.path.getsize(out_path) / (1024**3)
print(f"  Saved: {out_path} ({size_gb:.1f}GB, {total_params/1e6:.0f}M params)")
del fp16_sd; gc.collect()

# ── 2. VAE: BF16 → FP16 ──
print(f"\nConverting VAE...")
print(f"  Source: {vae_path}")
t0 = time.time()
vae_sd = safetensors.torch.load_file(vae_path, device="cpu")
print(f"  Loaded {len(vae_sd)} tensors in {time.time()-t0:.0f}s")

print("  Converting BF16 → FP16...")
vae_fp16 = {}
for k, v in vae_sd.items():
    vae_fp16[k] = v.to(torch.float16)
del vae_sd; gc.collect()

out_path = f"{OUTDIR}/vae_weights_fp16.pt"
torch.save(vae_fp16, out_path)
size_mb = os.path.getsize(out_path) / (1024**2)
print(f"  Saved: {out_path} ({size_mb:.0f}MB)")
del vae_fp16; gc.collect()

print("\nDone! Files ready for PC reference pipeline:")
print(f"  {OUTDIR}/diffusion_weights_fp16.pt")
print(f"  {OUTDIR}/vae_weights_fp16.pt")

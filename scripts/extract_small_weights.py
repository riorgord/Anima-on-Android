"""Extract only non-block weights (~20MB) from full DiT checkpoint (3.9GB)."""
import torch, sys

src = sys.argv[1] if len(sys.argv) > 1 else "diffusion_weights_fp16.pt"
dst = sys.argv[2] if len(sys.argv) > 2 else "diffusion_weights_small.pt"

print(f"Loading {src}...")
# Handle both .safetensors and .pt
if src.endswith(".safetensors"):
    from safetensors.torch import load_file
    full = load_file(src)
else:
    full = torch.load(src, weights_only=True)
print(f"Total keys: {len(full)}")

# Keep only non-block keys (x_embedder, t_embedder, final_layer, pos_embedder, llm_adapter)
small = {}
for k, v in full.items():
    # Strip 'net.' prefix if present
    key = k[4:] if k.startswith("net.") else k
    if not key.startswith("blocks."):
        small[key] = v
        print(f"  keep: {key}  {list(v.shape)}")

total_mb = sum(v.numel() * v.element_size() for v in small.values()) / 1e6
print(f"\nExtracted {len(small)} keys, {total_mb:.1f} MB → {dst}")
torch.save(small, dst)
print("Done. Push to phone:")
print(f"  adb push {dst} /sdcard/anima_on_android/models/")

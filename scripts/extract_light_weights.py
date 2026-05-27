"""Extract only x_embedder, t_embedder, final_layer weights from full DiT state dict."""
import torch, sys
src = "/sdcard/anima_on_android/models/diffusion_weights_fp16.pt"
dst = "/sdcard/anima_on_android/models/dit_light.pt"

sd = torch.load(src, weights_only=True, map_location="cpu")
light = {}
for k, v in sd.items():
    # Keep: x_embedder, t_embedder, t_embedding_norm, final_layer, pos_embed, extra_pos
    if any(k.startswith(p) for p in [
        "x_embedder.", "t_embedder.", "t_embedding_norm.",
        "final_layer.", "pos_embedder.", "extra_pos_embedder."
    ]):
        light[k] = v
        print(f"  keep: {k}  {list(v.shape)}")

torch.save(light, dst)
print(f"Saved {len(light)} keys → {dst}")

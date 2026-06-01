import torch, safetensors.torch
from diffsynth.core.loader.config import ModelConfig
from diffsynth.pipelines.anima_image import AnimaImagePipeline

sd = safetensors.torch.load_file(
    "/mnt/d/AI/手坤的anima/models/vae/qwen_image_vae.safetensors", device="cpu")
cfgs = [ModelConfig(model_id="wan_video_vae",
                    path="/mnt/d/AI/手坤的anima/models/vae/qwen_image_vae.safetensors",
                    state_dict=sd, skip_download=True)]
pipe = AnimaImagePipeline(device="cpu", torch_dtype=torch.float16)
pool = pipe.download_and_load_models(cfgs)
vae = pool.fetch_model("wan_video_vae")
vae_sd = vae.state_dict()
# Strip "model." prefix to match our WanVAE
bare_sd = {k[6:]: v.to(torch.float16) for k, v in vae_sd.items()}
torch.save(bare_sd, "/mnt/d/diffsynth_vae_sd.pt")
print(f"Saved {len(bare_sd)} keys (no prefix) — ready for our WanVAE")

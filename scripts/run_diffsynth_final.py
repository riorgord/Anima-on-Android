"""DiffSynth native pipeline: base vs turbo on cuda:1 (3060 12G)."""
import sys
from pathlib import Path
from collections import defaultdict
sys.path.insert(0, "/home/riorg/anima-work/src/anima_min")

import torch
import gc
import safetensors.torch
from diffsynth.core.loader.config import ModelConfig
from diffsynth.pipelines.anima_image import AnimaImagePipeline
from transformers import AutoTokenizer

DEVICE = "cuda:1"  # RTX 3060 12GB
DTYPE = torch.bfloat16
PROMPT = "1girl, anime style, blue eyes"
SEED = 6666
OUTDIR = Path("/home/riorg/anima-work/output")
OUTDIR.mkdir(exist_ok=True)

MODEL_ROOT = "/mnt/d/AI/手坤的anima/models"
QWEN_CKPT = f"{MODEL_ROOT}/text_encoder/qwen_3_06b_base.safetensors"
DIFF_CKPT = f"{MODEL_ROOT}/diffusion_model/anima-base-v1.0.safetensors"
VAE_CKPT  = f"{MODEL_ROOT}/vae/qwen_image_vae.safetensors"
LORA_CKPT = f"{MODEL_ROOT}/lora/anima-turbo-lora-v0.1.safetensors"
QWEN_TOK  = "/mnt/d/AI/prompts/画师串小助手/engines/tokenizers/qwen3"
T5_TOK    = "/mnt/e/sbkuake/ComfyUI-aki-v2/ComfyUI-aki-v2/ComfyUI/comfy/text_encoders/t5_tokenizer"


def merge_lora(diff_state):
    lora = safetensors.torch.load_file(LORA_CKPT, device="cpu")
    pairs = defaultdict(dict)
    for key, value in lora.items():
        parts = key.split(".")
        if parts[-2] in ("lora_A", "lora_B") and parts[-1] == "weight":
            pairs[".".join(parts[:-2])][parts[-2]] = value
    merged = dict(diff_state)
    count = 0
    for lora_key, pair in pairs.items():
        base_key = lora_key.replace("diffusion_model.", "net.", 1) + ".weight"
        if base_key in merged:
            delta = (pair["lora_B"].to(torch.float32) @ pair["lora_A"].to(torch.float32)).to(torch.bfloat16)
            merged[base_key] = merged[base_key] + delta
            count += 1
    del lora, pairs
    print(f"  LoRA merged: {count} keys")
    return merged


def run(label, diff_state, steps, cfg):
    print(f"\n{'='*60}")
    print(f"  {label}: {steps} steps, CFG={cfg}")
    print(f"{'='*60}")

    qwen_sd = safetensors.torch.load_file(QWEN_CKPT, device="cpu")
    vae_sd  = safetensors.torch.load_file(VAE_CKPT, device="cpu")

    configs = [
        ModelConfig(model_id="z_image_text_encoder", path=QWEN_CKPT, state_dict=qwen_sd, skip_download=True),
        ModelConfig(model_id="anima_dit",              path=DIFF_CKPT, state_dict=diff_state, skip_download=True),
        ModelConfig(model_id="wan_video_vae",          path=VAE_CKPT,  state_dict=vae_sd, skip_download=True),
    ]

    pipe = AnimaImagePipeline(device=DEVICE, torch_dtype=DTYPE)
    pool = pipe.download_and_load_models(configs)
    pipe.text_encoder = pool.fetch_model("z_image_text_encoder")
    pipe.dit = pool.fetch_model("anima_dit")
    pipe.vae = pool.fetch_model("wan_video_vae")
    pipe.tokenizer = AutoTokenizer.from_pretrained(QWEN_TOK)
    pipe.tokenizer_t5xxl = AutoTokenizer.from_pretrained(T5_TOK)

    # Free unused refs
    del qwen_sd, vae_sd, configs, pool
    gc.collect()
    torch.cuda.empty_cache()

    print("  Running full native pipeline (including DiffSynth VAE)...")
    image = pipe(
        prompt=PROMPT,
        negative_prompt="",
        cfg_scale=cfg,
        height=1024, width=1024,
        seed=SEED,
        rand_device="cpu",
        num_inference_steps=steps,
        progress_bar_cmd=lambda x: x,
    )

    out_path = OUTDIR / f"{label}.png"
    image.save(str(out_path))
    print(f"  saved: {out_path}")

    del pipe, image
    gc.collect()
    torch.cuda.empty_cache()


def main():
    print(f"Device: {DEVICE}")

    # Load base diffusion weights
    base_state = safetensors.torch.load_file(DIFF_CKPT, device="cpu")
    print(f"Diffusion checkpoint: {len(base_state)} tensors")

    # === BASE ===
    run("base", base_state, steps=30, cfg=5.0)

    # === TURBO ===
    turbo_state = merge_lora(base_state)
    run("turbo", turbo_state, steps=12, cfg=1.0)
    del turbo_state

    print(f"\n{'='*60}")
    print("  Done! Compare:")
    print(f"    Base:  {OUTDIR / 'base.png'}")
    print(f"    Turbo: {OUTDIR / 'turbo.png'}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

"""PC reference: DiffSynth Anima pipeline, 3-step 256×256 matching phone params."""
import sys, os
from pathlib import Path
import torch, gc, time
import safetensors.torch
from diffsynth.core.loader.config import ModelConfig
from diffsynth.pipelines.anima_image import AnimaImagePipeline
from transformers import AutoTokenizer
from collections import defaultdict

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16  # DiffSynth uses bf16 internally for the pipeline
PROMPT = "1girl, anime style, blue eyes"
SEED = 6666
OUTDIR = Path("/mnt/d/AI/anima_phone/output")

MODEL_ROOT = "/mnt/d/AI/手坤的anima/models"
QWEN_CKPT = f"{MODEL_ROOT}/text_encoder/qwen_3_06b_base.safetensors"
DIFF_CKPT = f"{MODEL_ROOT}/diffusion_model/anima-base-v1.0.safetensors"
VAE_CKPT  = f"{MODEL_ROOT}/vae/qwen_image_vae.safetensors"
LORA_CKPT = f"{MODEL_ROOT}/lora/anima-turbo-lora-v0.1.safetensors"
QWEN_TOK  = "/mnt/d/AI/prompts/画师串小助手/engines/tokenizers/qwen3"
T5_TOK    = "/mnt/e/sbkuake/ComfyUI-aki-v2/ComfyUI-aki-v2/ComfyUI/comfy/text_encoders/t5_tokenizer"

# ── Config (match phone pipeline) ──
H, W = 256, 256
STEPS = 3
CFG = 5.0


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


def run(label, diff_state, steps, cfg, height, width, seed, sigma_shift=None):
    print(f"\n{'='*60}")
    print(f"  {label}: {steps} steps, CFG={cfg}, {height}x{width}, sigma_shift={sigma_shift}")
    print(f"{'='*60}")

    t0 = time.time()
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

    del qwen_sd, vae_sd, configs, pool
    gc.collect()
    torch.cuda.empty_cache()

    print(f"  Load time: {time.time()-t0:.0f}s")
    print(f"  Running pipeline...")
    t1 = time.time()

    image = pipe(
        prompt=PROMPT,
        negative_prompt="",
        cfg_scale=cfg,
        height=height, width=width,
        seed=seed,
        rand_device="cpu",
        num_inference_steps=steps,
        sigma_shift=sigma_shift,
        progress_bar_cmd=lambda x: x,
    )

    dt = time.time() - t1
    out_path = OUTDIR / f"ds_ref_{label}.png"
    image.save(str(out_path))
    print(f"  Inference: {dt:.0f}s")
    print(f"  saved: {out_path}")

    del pipe, image
    gc.collect()
    torch.cuda.empty_cache()
    return out_path


def main():
    print(f"Device: {DEVICE}, Dtype: {DTYPE}")

    # Load base diffusion weights
    base_state = safetensors.torch.load_file(DIFF_CKPT, device="cpu")
    print(f"Diffusion checkpoint: {len(base_state)} tensors")

    # Merge LoRA (turbo improves speed-quality tradeoff for few-step)
    turbo_state = merge_lora(base_state)

    # Test 1: phone-matching params — base model (no LoRA)
    run("3step_256_cfg5_base", base_state,
        steps=STEPS, cfg=CFG, height=H, width=W, seed=SEED, sigma_shift=3.0)

    # Test 2: phone-matching params — turbo LoRA
    run("3step_256_cfg5_turbo", turbo_state,
        steps=STEPS, cfg=CFG, height=H, width=W, seed=SEED, sigma_shift=3.0)

    # Test 3: without sigma_shift (DiffSynth default scheduler)
    run("3step_256_cfg5_base_noshift", base_state,
        steps=STEPS, cfg=CFG, height=H, width=W, seed=SEED, sigma_shift=None)

    del turbo_state, base_state

    print(f"\n{'='*60}")
    print(f"  Done! Outputs in {OUTDIR}/")
    print(f"  ds_ref_3step_256_cfg5_base.png")
    print(f"  ds_ref_3step_256_cfg5_turbo.png")
    print(f"  ds_ref_3step_256_cfg5_base_noshift.png")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

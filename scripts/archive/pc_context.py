"""Generate context files for PC reference pipeline (WSL2 single GPU)."""
import sys, torch, safetensors.torch, gc, os
sys.path.insert(0, "/home/riorg/anima-work/src/anima_min")
from llm_adapter import LLMAdapter, adapted_context
from transformers import Qwen3Model, Qwen3Config, Qwen2Tokenizer, T5TokenizerFast

DEV = "cuda"
MODEL_ROOT = "/mnt/d/AI/手坤的anima/models"
OUTDIR = "/mnt/d/AI/anima_phone/models"
os.makedirs(OUTDIR, exist_ok=True)

print("Loading Qwen text encoder...")
qwen_sd = safetensors.torch.load_file(f"{MODEL_ROOT}/text_encoder/qwen_3_06b_base.safetensors")
cfg = Qwen3Config(hidden_size=1024, num_hidden_layers=28, num_attention_heads=16,
    num_key_value_heads=8, intermediate_size=3072, vocab_size=151936, head_dim=128,
    hidden_act="silu", rms_norm_eps=1e-6, rope_theta=1000000.0, attention_bias=False,
    tie_word_embeddings=True, max_position_embeddings=40960, use_cache=False)
qwen = Qwen3Model(cfg).to(DEV).eval()
qwen.load_state_dict({k[6:]: v for k, v in qwen_sd.items()})
del qwen_sd; gc.collect()

print("Loading LLMAdapter...")
diff_sd = safetensors.torch.load_file(f"{MODEL_ROOT}/diffusion_model/anima-base-v1.0.safetensors")
adapter_sd = {}
for k, v in diff_sd.items():
    if k.startswith("net.llm_adapter."):
        adapter_sd[k[len("net.llm_adapter."):]] = v.float()
adapter = LLMAdapter(device=DEV, dtype=torch.float32).to(DEV).eval()
adapter.load_state_dict(adapter_sd, strict=False)
del adapter_sd, diff_sd; gc.collect()

print("Loading tokenizers...")
qw_tok = Qwen2Tokenizer.from_pretrained(
    "/mnt/d/AI/prompts/画师串小助手/engines/tokenizers/qwen3")
t5_tok = T5TokenizerFast.from_pretrained(
    "/mnt/e/sbkuake/ComfyUI-aki-v2/ComfyUI-aki-v2/ComfyUI/comfy/text_encoders/t5_tokenizer")

for name, text in [("cond", "1girl, anime style, blue eyes"), ("uncond", ".")]:
    q_ids = qw_tok(text, add_special_tokens=False, return_tensors="pt").input_ids.to(DEV)
    q_mask = qw_tok(text, add_special_tokens=False, return_tensors="pt").attention_mask.to(DEV)
    with torch.no_grad():
        hidden = qwen(q_ids, attention_mask=q_mask, output_hidden_states=True).last_hidden_state
    t_text = text if text != "." else ""
    t_ids = t5_tok(t_text, add_special_tokens=True, return_tensors="pt").input_ids.to(DEV)
    t_w = torch.ones_like(t_ids, dtype=torch.float32, device=DEV).unsqueeze(-1)
    with torch.no_grad():
        ctx = adapted_context(adapter, hidden, t_ids, t_w)
    out_path = f"{OUTDIR}/context_{name}.pt"
    torch.save(ctx.cpu().half(), out_path)
    print(f"  {name}: {list(ctx.shape)} → {out_path}")

print("Done.")

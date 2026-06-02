"""Sampler / Scheduler / PipelineConfig — ComfyUI KSampler-style interface.
Zero torch dependency. All tensors are numpy arrays (BF16 storage, FP32 compute).

== Porting guide for future schedulers/samplers from ComfyUI/torch ==

1. Find the original Python code in ComfyUI's `comfy/samplers.py` or `comfy/k_diffusion/`
2. Replace torch ops with numpy equivalents using this table:
   torch.linspace(a,b,n)  → np.linspace(a,b,n)
   torch.exp(x)           → np.exp(x)
   torch.sin/cos(x)       → np.sin/cos(x)
   torch.randn(shape)     → rng.standard_normal(shape, dtype=np.float32)
   torch.tensor([x])      → np.array([x], dtype=np.float32)
   torch.cat([a,b], dim)  → np.concatenate([a,b], axis=dim)
   torch.stack([a,b])     → np.stack([a,b])
   torch.where(cond,a,b)  → np.where(cond,a,b)
   torch.clamp(x,lo,hi)   → np.clip(x,lo,hi)
   tensor.reshape(...)    → same
   tensor.float()         → tensor.astype(np.float32)
   tensor.to(dtype)       → tensor.astype(dtype)
3. Do NOT "invent" new math — copy the exact formula from the torch source.
4. Register with a unique name in SCHEDULERS or SAMPLERS dict below.
5. PipelineConfig.scheduler/sampler string selects the implementation at runtime.
"""
from dataclasses import dataclass
from abc import ABC, abstractmethod
import numpy as np


# ═══════════════════════════════════════════════════════════════
# PipelineConfig — all tunable parameters (future frontend ↔ this)
# ═══════════════════════════════════════════════════════════════

@dataclass
class PipelineConfig:
    # Model
    H: int = 32              # latent patch height (32 → 256×256 image)
    W: int = 32              # latent patch width
    T: int = 1               # frames (1 = image)

    # Sampling
    seed: int = 6666
    steps: int = 3
    cfg: float = 5.0
    denoise: float = 1.0

    # Backend selection (string → registry lookup)
    sampler: str = "euler"
    scheduler: str = "z_image"


# ═══════════════════════════════════════════════════════════════
# Scheduler — noise schedule σ(t)
# ═══════════════════════════════════════════════════════════════

class Scheduler(ABC):
    """调度器: 给定步数 → 输出 sigma 序列 [steps+1].
    Steps to port a new scheduler (e.g. beta57):
    1. Find the sigma schedule formula in ComfyUI/k-diffusion source
    2. Replace torch.linspace/torch.exp with numpy equivalents
    3. Subclass Scheduler, implement get_sigmas()
    4. Register in SCHEDULERS dict
    """
    @abstractmethod
    def get_sigmas(self, steps: int, denoise: float = 1.0) -> np.ndarray:
        """Return sigma array shape [steps+1], descending from ~1.0 to ~0.0."""
        ...


class ZImageScheduler(Scheduler):
    """Z-Image sigma schedule: sigmas = shift * t / (1 + (shift-1)*t).
    Forked from FlowMatchScheduler.set_timesteps (src/flow_match.py, Apache 2.0).
    torch: sigmas = shift * torch.linspace(...) / (1+(shift-1)*torch.linspace(...))
    numpy: sigmas = shift * np.linspace(...) / (1+(shift-1)*np.linspace(...))"""

    def __init__(self, sigma_shift: float = 3.0):
        self.sigma_shift = sigma_shift

    def get_sigmas(self, steps: int, denoise: float = 1.0) -> np.ndarray:
        """Mirrors FlowMatchScheduler.set_timesteps: linspace→shift formula."""
        sigma_min = 0.0
        sigma_max = 1.0
        shift = self.sigma_shift
        sigma_start = sigma_min + (sigma_max - sigma_min) * denoise
        sigmas = np.linspace(sigma_start, sigma_min, steps + 1, dtype=np.float32)
        sigmas = shift * sigmas / (1.0 + (shift - 1.0) * sigmas)
        return sigmas.astype(np.float32)


# ═══════════════════════════════════════════════════════════════
# Sampler — denoising loop strategy
# ═══════════════════════════════════════════════════════════════

class Sampler(ABC):
    """采样器: 给定 model+noise+sigmas+context → 去噪 latent.
    Steps to port a new sampler (e.g. er_sde, dpm_2):
    1. Find the step() function in ComfyUI/k-diffusion source
    2. Replace all torch ops with numpy (see porting table at top)
    3. The model is called via model.forward(x_np, sigma, ctx_np) — numpy in/out
    4. Subclass Sampler, implement sample()
    5. Register in SAMPLERS dict
    """
    @abstractmethod
    def sample(self, model, noise: np.ndarray, sigmas: np.ndarray,
               ctx_cond: np.ndarray, ctx_uncond: np.ndarray,
               cfg: float, seed: int) -> np.ndarray:
        """Run full denoising loop. Returns final latent [1,C,H,W] fp32 numpy."""
        ...


class EulerSampler(Sampler):
    """Euler flow-match sampler.
    Forked from FlowMatchScheduler.step (src/flow_match.py).
    Step formula: x = x + v_cfg * (sigma_next - sigma)"""

    def sample(self, model, noise: np.ndarray, sigmas: np.ndarray,
               ctx_cond: np.ndarray, ctx_uncond: np.ndarray,
               cfg: float, seed: int) -> np.ndarray:
        """Mirrors the denoising loop in phone_pipeline_nd.py."""
        x = noise.astype(np.float32)  # [1, C, H, W]

        for i in range(len(sigmas) - 1):
            sigma = float(sigmas[i])
            sigma_next = float(sigmas[i + 1])

            # CFG batch: [uncond, cond] → B=2
            x_b = np.tile(np.expand_dims(x, 2), (2, 1, 1, 1, 1))  # [2,C,1,H,W]
            ctx_b = np.concatenate([ctx_uncond, ctx_cond], axis=0)  # [2,N,1024]
            sigma_b = np.array([sigma, sigma], dtype=np.float32)     # [2]

            v_b = model.forward(x_b, sigma_b, ctx_b)  # [2,C,1,H,W]

            # CFG mix: v_cfg = v_uncond + cfg*(v_cond - v_uncond)
            v_cond = v_b[1:2]
            v_uncond = v_b[0:1]
            v_cfg = v_uncond + cfg * (v_cond - v_uncond)

            # Euler step: x = x + v*(sigma_next - sigma)
            x = x + v_cfg[:, :, 0, :, :] * (sigma_next - sigma)

        return x


# ═══════════════════════════════════════════════════════════════
# Registry — name → implementation class
# ═══════════════════════════════════════════════════════════════
#
# To add a new scheduler/sampler:
# 1. Subclass Scheduler or Sampler above
# 2. Implement get_sigmas() or sample()
# 3. Add to the dict below with a unique name
# 4. Set PipelineConfig.scheduler / .sampler to the new name
#
# Example future additions:
#   "beta57"  Scheduler — Anima model card's recommended scheduler
#   "er_sde"  Sampler   — stochastic SDE sampler for better quality
#   "simple"  Scheduler — standard linear sigma schedule
#   "dpm_2"   Sampler   — second-order DPM solver

SCHEDULERS: dict[str, type] = {
    "z_image": ZImageScheduler,
}

SAMPLERS: dict[str, type] = {
    "euler": EulerSampler,
}


# ═══════════════════════════════════════════════════════════════
# run_ksampler — ComfyUI KSampler's "Generate" button equivalent
# ═══════════════════════════════════════════════════════════════
#
# This function mirrors ComfyUI's KSampler node wiring:
#   KSampler(model, seed, steps, cfg, sampler_name, scheduler,
#            positive, negative, latent_image, denoise) → latent
#
# Frontend usage:
#   from sampling import PipelineConfig, run_ksampler
#   cfg = PipelineConfig(H=32, steps=3, cfg=5.0, seed=6666,
#                        sampler="euler", scheduler="z_image")
#   latent = run_ksampler(model, cfg, ctx_cond, ctx_uncond)
#   # → VAE.decode(latent) → PNG

def run_ksampler(model, config: PipelineConfig,
                 ctx_cond: np.ndarray, ctx_uncond: np.ndarray) -> np.ndarray:
    """Run the full denoising loop — ComfyUI KSampler equivalent.
    Args:
        model:     NumpyDiT instance (model.forward(x_np, sigma, ctx_np) → v_np)
        config:    PipelineConfig (seed, steps, cfg, sampler, scheduler, denoise, H, W)
        ctx_cond:  float32 numpy [1, 512, 1024] positive prompt conditioning
        ctx_uncond: float32 numpy [1, 512, 1024] negative prompt conditioning
    Returns:
        latent: float32 numpy [1, 16, H, W] denoised latent (ready for VAE)
    """
    # 1. Scheduler → sigma sequence
    scheduler_cls = SCHEDULERS[config.scheduler]
    scheduler = scheduler_cls()
    sigmas = scheduler.get_sigmas(config.steps, config.denoise)

    # 2. Noise (ComfyUI "latent_image" — empty latent with noise)
    rng = np.random.default_rng(config.seed)
    noise = rng.standard_normal((1, 16, config.H, config.W), dtype=np.float32)

    # 3. Sampler → denoising loop
    sampler_cls = SAMPLERS[config.sampler]
    sampler = sampler_cls()
    latent = sampler.sample(model, noise, sigmas, ctx_cond, ctx_uncond,
                            config.cfg, config.seed)

    return latent.astype(np.float32)

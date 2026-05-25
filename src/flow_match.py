# Adapted from DiffSynth-Studio (Apache 2.0) — diffsynth/diffusion/flow_match.py
import torch


class FlowMatchScheduler:

    def __init__(self, sigma_shift=3.0):
        self.sigma_shift = sigma_shift
        self.num_train_timesteps = 1000

    def set_timesteps(self, num_inference_steps=100, denoising_strength=1.0):
        sigma_min = 0.0
        sigma_max = 1.0
        shift = self.sigma_shift
        sigma_start = sigma_min + (sigma_max - sigma_min) * denoising_strength
        sigmas = torch.linspace(sigma_start, sigma_min, num_inference_steps + 1)[:-1]
        sigmas = shift * sigmas / (1 + (shift - 1) * sigmas)
        timesteps = sigmas * self.num_train_timesteps
        self.sigmas = sigmas
        self.timesteps = timesteps

    def step(self, model_output, timestep, sample):
        if isinstance(timestep, torch.Tensor):
            timestep = timestep.cpu()
        timestep_id = torch.argmin((self.timesteps - timestep).abs())
        sigma = self.sigmas[timestep_id]
        if timestep_id + 1 >= len(self.timesteps):
            sigma_ = 0.0
        else:
            sigma_ = self.sigmas[timestep_id + 1]
        prev_sample = sample + model_output * (sigma_ - sigma)
        return prev_sample

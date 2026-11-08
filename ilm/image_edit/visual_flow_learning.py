"""One-forward supervised flow matching for the native image generator.

No lexical class targets are used. Clean target latents are supervision, not
generation inputs. Inference must still begin from independent noise and be
assessed separately; this training loss alone does not prove readable output.
"""
from dataclasses import dataclass

import torch


@dataclass
class FlowExample:
    noisy_latents: torch.Tensor
    velocity: torch.Tensor
    timesteps: torch.Tensor
    sigmas: torch.Tensor
    indices: torch.Tensor


def interpolate(clean, noise, sigmas, timesteps, indices):
    if clean.shape != noise.shape or clean.ndim != 4:
        raise ValueError("matching BCHW clean/noise latent tensors required")
    if clean.device != noise.device or clean.dtype != noise.dtype or not clean.is_floating_point():
        raise ValueError("latent and noise device/dtype must match")
    if sigmas.shape != (clean.shape[0], 1, 1, 1) or sigmas.device != clean.device or sigmas.dtype != clean.dtype:
        raise ValueError("one native noise level per image required")
    if timesteps.shape != (clean.shape[0],) or timesteps.device != clean.device:
        raise ValueError("native model timesteps must match the latent batch/device")
    if not torch.isfinite(sigmas).all() or (sigmas <= 0).any() or (sigmas > 1).any():
        raise ValueError("training noise levels must be in (0, 1]")
    if clean.requires_grad or noise.requires_grad:
        raise ValueError("target-codec latents and Gaussian noise must remain supervision")
    return FlowExample(noisy_latents=(1.0 - sigmas) * clean + sigmas * noise,
        velocity=noise - clean, timesteps=timesteps, sigmas=sigmas, indices=indices)


def sample_flow(clean, scheduler, *, time_generator, noise_generator):
    """Use the release's uniform-index, shifted discrete training schedule.

    A separate fresh training scheduler is required. A sampler-mutated 50-step
    scheduler cannot silently supply the 1000-step training distribution.
    Explicit generators avoid coupling target noise to reader/codec RNG use.
    """
    total = int(scheduler.config.num_train_timesteps)
    if (scheduler.config.use_dynamic_shifting or scheduler.config.stochastic_sampling or
        len(scheduler.timesteps) != total or len(scheduler.sigmas) != total):
        raise ValueError("fresh native fixed-shift training scheduler required")
    if time_generator.device != torch.device("cpu"):
        raise ValueError("use a separate CPU generator for uniform timestep sampling")
    if noise_generator.device != clean.device:
        raise ValueError("Gaussian generator must use the latent device")
    u = torch.rand((clean.shape[0],), generator=time_generator, device="cpu")
    indices = (u * total).long()
    timesteps = scheduler.timesteps[indices].to(clean.device)
    sigmas = scheduler.sigmas[indices].to(device=clean.device, dtype=clean.dtype).reshape(-1, 1, 1, 1)
    noise = torch.randn(clean.shape, generator=noise_generator, device=clean.device, dtype=clean.dtype)
    return interpolate(clean, noise, sigmas, timesteps, indices)


def velocity_loss(prediction, target):
    if prediction.shape != target.shape:
        raise ValueError("native predicted and target velocity shapes differ")
    return (prediction.float() - target.float()).square().flatten(1).mean(1).mean()


def denoising_loss(transformer, example, *, sequence_condition, pooled_condition, reference_latents):
    """One actual denoiser call, with image-derived conditions supplied upstream.

    The caller must obtain these conditions from input images alone. No clean
    answer or noisy target is used as the reference-image condition.
    """
    prediction = transformer(hidden_states=example.noisy_latents,
        cond_hidden_states=reference_latents, encoder_hidden_states=sequence_condition,
        pooled_projections=pooled_condition, timestep=example.timesteps, return_dict=False)[0]
    loss = velocity_loss(prediction, example.velocity)
    if not torch.isfinite(loss):
        raise FloatingPointError("nonfinite native supervised flow loss")
    return loss, prediction

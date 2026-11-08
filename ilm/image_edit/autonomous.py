"""Differentiable, source-matched autonomous generation from independent noise.

The sampler has no target/label argument. Targets enter a separate final-image
loss. Exact native forward equivalence is required before training.
"""
import inspect
from pathlib import Path

import numpy as np
import torch
from torch.utils.checkpoint import checkpoint

from .baseline import open_page, sha


def native_implementation(pipe):
    return {name: dict(file=Path(inspect.getfile(type(obj))).name,
        sha256=sha(inspect.getfile(type(obj)))) for name, obj in
        (("pipeline", pipe), ("scheduler", pipe.scheduler), ("image_processor", pipe.image_processor))}


@torch.no_grad()
def encode_reference(pipe, path):
    """Match the native image branch inside its BF16 autocast context.

    The historical paired cache was encoded outside autocast. It remains
    historical evidence, but is not a bit-exact native sampling reference.
    """
    page = open_page(path)
    pixels = pipe.image_processor.preprocess(page, height=page.height,
        width=page.width, resize_mode="crop").to(device=pipe.device, dtype=pipe.vae.dtype)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        return pipe._encode_vae_image(pixels, generator=None).detach()


def sample_page(pipe, input_features, positive, *, seed, differentiable=False,
                capture=False, steps=50, width=512, height=256):
    from diffusers.pipelines.flux2.pipeline_flux2_klein import compute_empirical_mu, retrieve_timesteps
    if pipe.scheduler.config.stochastic_sampling or pipe.config.is_distilled:
        raise ValueError("the pinned deterministic undistilled source is required")
    if steps != 50 or (width, height) != (512, 256):
        raise ValueError("the declared native 50-step 512x256 interface is fixed")
    device = pipe.device
    condition = positive.to(device)
    text_ids = pipe._prepare_text_ids(condition).to(device)
    features = input_features.to(device)
    reference = pipe._pack_latents(features)
    reference_ids = pipe._prepare_image_ids([features]).to(device)
    z, latent_ids = pipe.prepare_latents(batch_size=1,
        num_latents_channels=pipe.transformer.config.in_channels // 4,
        height=height, width=width, dtype=condition.dtype, device=device,
        generator=torch.Generator(device=device).manual_seed(seed))
    all_ids = torch.cat((latent_ids, reference_ids), dim=1)
    sigmas = np.linspace(1., 1 / steps, steps)
    if getattr(pipe.scheduler.config, "use_flow_sigmas", False):
        sigmas = None
    times, count = retrieve_timesteps(pipe.scheduler, steps, device,
        sigmas=sigmas, mu=compute_empirical_mu(image_seq_len=z.shape[1], num_steps=steps))
    if len(times) != count or count != 50:
        raise ValueError("native sampling schedule differs")
    # Each checkpoint recomputes only the pure neural field. The mutable native
    # scheduler is never called from a recomputed closure. Reproduce its exact
    # deterministic FP32 addition and output-dtype cast below.
    def field(value, time):
        model_input = torch.cat((value, reference), dim=1).to(pipe.transformer.dtype)
        with pipe.transformer.cache_context("cond"), torch.autocast("cuda", dtype=torch.bfloat16):
            prediction = pipe.transformer(hidden_states=model_input,
                timestep=time, guidance=None, encoder_hidden_states=condition,
                txt_ids=text_ids, img_ids=all_ids, joint_attention_kwargs=None,
                return_dict=False)[0][:, :value.shape[1]]
        return prediction

    traces, probes = [], []
    for index, t in enumerate(times):
        time = t.expand(z.shape[0]).to(z.dtype) / 1000
        prediction = checkpoint(field, z, time, use_reentrant=False) if differentiable else field(z, time)
        if differentiable and capture and index in (0, 24, 49):
            prediction.retain_grad(); probes.append((index, prediction))
        dt = pipe.scheduler.sigmas[index + 1] - pipe.scheduler.sigmas[index]
        z = (z.float() + dt * prediction).to(prediction.dtype)
        if capture:
            traces.append(z.detach().cpu().clone())
    features_out = pipe._unpack_latents_with_ids(z, latent_ids, height // 16, width // 16)

    def decode(value):
        mean = pipe.vae.bn.running_mean.view(1, -1, 1, 1).to(value)
        std = (pipe.vae.bn.running_var.view(1, -1, 1, 1)
            + pipe.vae.config.batch_norm_eps).sqrt().to(value)
        native = pipe._unpatchify_latents(value * std + mean)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return pipe.vae.decode(native, return_dict=False)[0]

    decoded = checkpoint(decode, features_out, use_reentrant=False) if differentiable else decode(features_out)
    rgb = pipe.image_processor.postprocess(decoded, output_type="pt")
    return dict(rgb=rgb, unclipped_rgb=decoded.float() / 2 + .5,
        traces=traces, probes=probes)


def final_image_loss(result, target_rgb, target_text, reader):
    target = target_rgb.float()
    prediction = result["unclipped_rgb"]
    if target.shape != prediction.shape or target.shape != (1, 3, 256, 512):
        raise ValueError("the canonical complete target page must match declared geometry")
    ink = target.mean(dim=1, keepdim=True) < .9
    if not bool(ink.any()) or not bool((~ink).any()):
        raise ValueError("both target ink and background are required")
    error = (prediction - target).abs().mean(dim=1, keepdim=True)
    balanced = .5 * (error[ink].mean() + error[~ink].mean())
    with torch.autocast("cuda", dtype=torch.bfloat16):
        recognition = reader.loss(result["rgb"], target_text)
    return recognition + 10 * balanced, dict(recognition=float(recognition.detach()),
        balanced_pixel_l1=float(balanced.detach()), target_ink_pixels=int(ink.sum()))

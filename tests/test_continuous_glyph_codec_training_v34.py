from __future__ import annotations

import torch

from ilm.visual_lm.continuous_glyph_codec import (
    ContinuousGlyphCodec,
    ContinuousGlyphCodecConfig,
    ContinuousGlyphCodecOutput,
)
from ilm.visual_lm.continuous_glyph_codec_training import (
    continuous_glyph_codec_loss,
    fixed_latent_noise,
    training_latent_noise,
)


def output_for_logits(logits: torch.Tensor) -> ContinuousGlyphCodecOutput:
    batch = logits.shape[0]
    latents = torch.zeros(batch, 768)
    return ContinuousGlyphCodecOutput(
        logits=logits,
        latents=latents,
        decoder_latents=latents,
    )


def test_codec_loss_prefers_correct_binary_reconstruction() -> None:
    targets = torch.randint(0, 2, (8, 1, 32, 32)).float()
    correct = continuous_glyph_codec_loss(
        output_for_logits((targets * 2.0 - 1.0) * 12.0),
        targets,
    )
    inverted = continuous_glyph_codec_loss(
        output_for_logits((1.0 - targets) * 24.0 - 12.0),
        targets,
    )
    assert correct.loss < 1e-3
    assert inverted.loss > correct.loss
    assert correct.patches == 8


def test_codec_loss_backpropagates_through_encoder_and_decoder() -> None:
    model = ContinuousGlyphCodec(ContinuousGlyphCodecConfig())
    targets = torch.randint(0, 2, (4, 1, 32, 32)).float()
    encoded = model.encode(targets)
    noise = training_latent_noise(encoded, seed=34, update=1)
    output = model(targets, latent_noise=noise)
    losses = continuous_glyph_codec_loss(output, targets)
    losses.loss.backward()
    assert torch.isfinite(losses.loss)
    assert model.encoder.stem.weight.grad is not None
    assert model.decoder.output.weight.grad is not None


def test_training_noise_is_stateless_and_selects_about_half() -> None:
    latents = torch.zeros(4096, 768)
    first = training_latent_noise(latents, seed=20263400, update=19)
    second = training_latent_noise(latents, seed=20263400, update=19)
    different = training_latent_noise(latents, seed=20263400, update=20)
    selected_fraction = float(first.abs().sum(dim=1).gt(0).float().mean())
    assert torch.equal(first, second)
    assert not torch.equal(first, different)
    assert 0.47 < selected_fraction < 0.53
    assert float(first.std()) < 0.05


def test_fixed_noise_is_reproducible_and_uses_requested_scale() -> None:
    latents = torch.zeros(2048, 768)
    first = fixed_latent_noise(latents, sigma=0.03, seed=7)
    second = fixed_latent_noise(latents, sigma=0.03, seed=7)
    assert torch.equal(first, second)
    assert abs(float(first.std()) - 0.03) < 5e-4

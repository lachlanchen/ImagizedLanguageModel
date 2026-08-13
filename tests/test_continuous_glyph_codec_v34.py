from __future__ import annotations

import pytest
import torch

from ilm.visual_lm.continuous_glyph_codec import (
    ContinuousGlyphCodec,
    ContinuousGlyphCodecConfig,
    continuous_glyph_codec_boundary_receipt,
)


def test_codec_shapes_and_normalized_latents() -> None:
    model = ContinuousGlyphCodec(ContinuousGlyphCodecConfig())
    pixels = torch.randint(0, 2, (3, 1, 32, 32)).float()
    output = model(pixels)
    assert output.logits.shape == pixels.shape
    assert output.latents.shape == (3, 768)
    assert torch.allclose(output.latents.mean(dim=1), torch.zeros(3), atol=1e-5)
    assert torch.allclose(
        output.latents.var(dim=1, unbiased=False),
        torch.ones(3),
        atol=1e-2,
    )


def test_codec_accepts_continuous_noise_and_backpropagates() -> None:
    model = ContinuousGlyphCodec(ContinuousGlyphCodecConfig())
    pixels = torch.randint(0, 2, (2, 1, 32, 32)).float()
    noise = torch.randn(2, 768) * 0.03
    output = model(pixels, latent_noise=noise)
    output.logits.square().mean().backward()
    assert model.encoder.stem.weight.grad is not None
    assert model.decoder.output.weight.grad is not None


def test_codec_reconstruct_and_boundary_are_pixel_only() -> None:
    model = ContinuousGlyphCodec(ContinuousGlyphCodecConfig())
    pixels = torch.ones(2, 1, 32, 32)
    reconstructed = model.reconstruct(pixels)
    receipt = continuous_glyph_codec_boundary_receipt(model)
    assert reconstructed.shape == pixels.shape
    assert set(reconstructed.unique().tolist()).issubset({0.0, 1.0})
    assert receipt["parameter_names_with_forbidden_fragments"] == []
    assert receipt["encode_parameters"] == ["pixels"]
    assert receipt["decode_parameters"] == ["latents"]
    assert receipt["uses_visual_codebook"] is False


def test_codec_rejects_non_pixel_inputs() -> None:
    model = ContinuousGlyphCodec(ContinuousGlyphCodecConfig())
    with pytest.raises(ValueError, match="shape"):
        model(torch.ones(1, 1, 32, 64))
    with pytest.raises(TypeError, match="floating"):
        model(torch.ones(1, 1, 32, 32, dtype=torch.uint8))
    with pytest.raises(ValueError, match="align"):
        model(
            torch.ones(1, 1, 32, 32),
            latent_noise=torch.zeros(2, 768),
        )

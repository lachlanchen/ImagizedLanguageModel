from __future__ import annotations

import torch

from ilm.visual_lm.visual_posterior_language_v51 import (
    VisualGlyphScorePrior,
    VisualPosteriorLanguageConfig,
    cosine_diffusion_schedule,
    visual_posterior_language_boundary_receipt,
)


def _model() -> VisualGlyphScorePrior:
    return VisualGlyphScorePrior(
        VisualPosteriorLanguageConfig(
            diffusion_steps=20,
            base_channels=16,
            unet_depth=1,
            time_dimension=32,
            image_to_image_start=6,
            sampling_steps=4,
            guidance_strength=0.05,
        )
    )


def test_cosine_schedule_is_finite_and_decreasing() -> None:
    betas, alpha_bars = cosine_diffusion_schedule(20, offset=0.008)
    assert betas.shape == alpha_bars.shape == (20,)
    assert bool(torch.isfinite(alpha_bars).all())
    assert bool((alpha_bars[1:] < alpha_bars[:-1]).all())
    assert bool(((betas > 0.0) & (betas < 1.0)).all())


def test_prior_loss_backpropagates_without_symbolic_inputs() -> None:
    torch.manual_seed(3)
    model = _model().train()
    images = (torch.rand(4, 1, 32, 32) > 0.82).float()
    timesteps = torch.tensor([0, 3, 9, 18], dtype=torch.long)
    noise = torch.randn_like(images)
    result = model.training_loss(images, timesteps, noise)
    result.loss.backward()
    assert torch.isfinite(result.loss)
    assert model.denoiser.out_conv.weight.grad is not None


def test_posterior_sampler_emits_binary_visible_images() -> None:
    torch.manual_seed(5)
    model = _model().eval()
    plans = (torch.rand(2, 1, 32, 32) > 0.8).float()
    anchors = model.visual_field.encode_unit(plans)
    generator = torch.Generator().manual_seed(7)
    sampled, metadata = model.posterior_sample(
        plans,
        anchors,
        generator=generator,
    )
    assert sampled.shape == plans.shape
    assert set(torch.unique(sampled).tolist()) <= {0.0, 1.0}
    assert torch.isfinite(metadata["continuous"]).all()
    assert metadata["visual_fields"].shape == (2, 1024)


def test_boundary_excludes_symbolic_and_lookup_paths() -> None:
    receipt = visual_posterior_language_boundary_receipt(_model())
    assert receipt["text_tokenizer"] is False
    assert receipt["unicode_or_character_ids_in_model"] is False
    assert receipt["glyph_vocabulary_or_candidate_bank_in_model"] is False
    assert receipt["prior_receives_sequence_or_character_labels"] is False

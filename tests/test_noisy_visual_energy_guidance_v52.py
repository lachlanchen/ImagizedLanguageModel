from __future__ import annotations

import torch

from ilm.visual_lm.canonical_glyph_language import CanonicalGlyphLanguageConfig
from ilm.visual_lm.hierarchical_raster_language import CausalRasterReader
from ilm.visual_lm.noisy_visual_energy_guidance_v52 import (
    NoisyVisualEnergyConfig,
    NoisyVisualEnergyCritic,
    noisy_visual_energy_boundary_receipt,
)
from ilm.visual_lm.noisy_visual_energy_guidance_v52_training import (
    exact_raster_positive_mask,
    multi_positive_log_mass,
)


def test_v52_critic_scores_noisy_images_and_backpropagates_to_pixels() -> None:
    critic = NoisyVisualEnergyCritic(NoisyVisualEnergyConfig())
    hidden = torch.randn(3, 384)
    anchors = torch.nn.functional.normalize(torch.randn(3, 1024), dim=-1)
    images = torch.randn(3, 1, 32, 32, requires_grad=True)
    timesteps = torch.full((3,), 47, dtype=torch.long)
    logits = critic.score_matrix(hidden, anchors, images, timesteps)
    assert logits.shape == (3, 3)
    gradient = torch.autograd.grad(logits.diagonal().sum(), images)[0]
    assert gradient.shape == images.shape
    assert torch.isfinite(gradient).all()
    assert float(gradient.abs().sum()) > 0.0


def test_v52_exact_positive_mask_groups_duplicate_rasters() -> None:
    reader = CausalRasterReader(CanonicalGlyphLanguageConfig())
    images = torch.zeros(3, 1, 32, 32)
    images[0, :, 8:24, 15:17] = 1.0
    images[1] = images[0]
    images[2, :, 15:17, 8:24] = 1.0
    positives, _ = exact_raster_positive_mask(reader, images)
    assert positives.tolist() == [
        [True, True, False],
        [True, True, False],
        [False, False, True],
    ]
    logits = torch.eye(3) * 2.0
    log_mass = multi_positive_log_mass(logits, positives)
    assert log_mass.shape == (3,)
    assert torch.isfinite(log_mass).all()


def test_v52_boundary_has_no_symbolic_runtime_path() -> None:
    critic = NoisyVisualEnergyCritic(NoisyVisualEnergyConfig())
    receipt = noisy_visual_energy_boundary_receipt(critic)
    assert receipt["critic_parameters"] <= 6_000_000
    assert receipt["parameter_names_with_forbidden_fragments"] == []
    assert receipt["text_tokenizer"] is False
    assert receipt["unicode_or_character_ids_in_model"] is False
    assert receipt["glyph_vocabulary_or_candidate_bank_in_model"] is False

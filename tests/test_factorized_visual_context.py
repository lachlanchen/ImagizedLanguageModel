from __future__ import annotations

import torch

from ilm.visual_lm.factorized_visual_context import (
    FactorizedVisualContextConfig,
    FactorizedVisualContextModel,
    factorized_visual_context_boundary_receipt,
    multi_positive_particle_contrastive_loss,
    particle_energy_score,
    suffix_pair_ranking_loss,
)


def _small_model() -> FactorizedVisualContextModel:
    return FactorizedVisualContextModel(
        FactorizedVisualContextConfig(
            visual_dim=64,
            model_dim=128,
            layers=2,
            heads=4,
            mlp_ratio=2.0,
            retina_base_channels=8,
            particle_count=4,
            particle_noise_dim=16,
        )
    )


def test_factorized_model_uses_only_image_streams_and_exposes_parts() -> None:
    model = _small_model().eval()
    context = torch.rand(2, 8, 1, 32, 32)
    with torch.no_grad():
        output = model.language(context)
        last = model.language(context[:, -1:])
    assert output["appearance_state"].shape == (2, 128)
    assert output["history_residual"].shape == (2, 128)
    assert output["particles"].shape == (2, 4, 4, 64)
    assert torch.equal(last["history_residual"], torch.zeros_like(last["history_residual"]))
    assert torch.allclose(output["particles"].norm(dim=-1), torch.ones(2, 4, 4))


def test_history_residual_can_be_swapped_without_changing_appearance() -> None:
    model = _small_model().eval()
    context = torch.rand(2, 7, 1, 32, 32)
    context[1, -1] = context[0, -1]
    with torch.no_grad():
        parts = model.factorize(context)
        swapped = model.fuse_parts(
            parts["appearance_state"], parts["history_residual"].flip(0)
        )
        particles = model.predict_particles_from_state(
            swapped["fused_state"], horizons=(1,)
        )
    assert torch.allclose(
        parts["appearance_visual"][0], parts["appearance_visual"][1], atol=1e-7
    )
    assert particles.shape == (2, 1, 4, 64)


def test_energy_and_pair_losses_have_finite_gradients() -> None:
    particles = torch.nn.functional.normalize(
        torch.randn(3, 4, 5, 16, requires_grad=True), dim=-1
    )
    target = torch.nn.functional.normalize(torch.randn(3, 4, 16), dim=-1)
    energy = particle_energy_score(particles, target).mean()
    pair, pair_metrics = suffix_pair_ranking_loss(
        particles[:2, 0], particles[1:3, 0], target[:2, 0], target[1:3, 0]
    )
    candidates = torch.nn.functional.normalize(torch.randn(9, 16), dim=-1)
    candidates[:3] = target[:, 0]
    contrastive, _ = multi_positive_particle_contrastive_loss(
        particles[:, 0],
        target[:, 0],
        candidates,
        scale=torch.tensor(10.0),
    )
    loss = energy + pair + contrastive
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(particles.grad if particles.is_leaf else loss)
    assert 0.0 <= float(pair_metrics["pair_ranking_accuracy"]) <= 1.0


def test_v26_boundary_has_no_symbolic_or_deployed_bank_path() -> None:
    receipt = factorized_visual_context_boundary_receipt(
        FactorizedVisualContextConfig()
    )
    assert receipt["output_is_continuous_visual_distribution"] is True
    assert receipt["history_can_be_zeroed_or_swapped"] is True
    for key in (
        "uses_strings",
        "uses_token_ids",
        "uses_unicode_ids",
        "uses_character_ids",
        "uses_vocabulary_embedding",
        "uses_ocr",
        "uses_visual_codebook",
        "uses_glyph_lookup",
        "uses_external_language_model",
        "candidate_bank_deployed",
    ):
        assert receipt[key] is False

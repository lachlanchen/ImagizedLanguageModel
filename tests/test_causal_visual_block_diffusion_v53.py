from __future__ import annotations

import torch

from ilm.visual_lm.canonical_glyph_language import CanonicalGlyphLanguageConfig
from ilm.visual_lm.causal_visual_block_diffusion_v53 import (
    CausalVisualBlockDiffusion,
    CausalVisualBlockDiffusionConfig,
    ConditionalStripDenoiser,
    causal_visual_block_boundary_receipt,
)
from ilm.visual_lm.causal_visual_block_diffusion_v53_training import (
    _visual_fields,
    stratified_future_timesteps,
    visual_block_diffusion_loss,
)
from ilm.visual_lm.canonical_glyph_language_evaluation import _shuffle_prefix
from ilm.visual_lm.hierarchical_raster_language import CausalRasterReader
from ilm.visual_lm.visual_posterior_language_v51 import (
    VisualGlyphScorePrior,
    VisualPosteriorLanguageConfig,
)


def _model() -> CausalVisualBlockDiffusion:
    reader = CausalRasterReader(CanonicalGlyphLanguageConfig())
    denoiser = ConditionalStripDenoiser(CausalVisualBlockDiffusionConfig())
    return CausalVisualBlockDiffusion(reader, denoiser)


def test_v53_velocity_round_trip_recovers_clean_and_noise() -> None:
    model = _model()
    clean = torch.randint(0, 2, (2, 4, 1, 32, 32)).float()
    noise = torch.randn_like(clean)
    times = torch.tensor([[0, 27, 65, 99], [12, 43, 81, 94]])
    noisy = model.add_noise(clean, times, noise)
    velocity = model.velocity_target(clean, times, noise)
    recovered_clean, recovered_noise = model.velocity_to_clean_noise(
        noisy, velocity, times
    )
    assert torch.allclose(recovered_clean, clean * 2.0 - 1.0, atol=1e-5)
    assert torch.allclose(recovered_noise, noise, atol=1e-5)


def test_v53_denoiser_supports_independent_cell_times_and_gradients() -> None:
    denoiser = ConditionalStripDenoiser(CausalVisualBlockDiffusionConfig())
    noisy = torch.randn(2, 4, 1, 32, 32, requires_grad=True)
    hidden = torch.randn(2, 384)
    anchors = torch.nn.functional.normalize(torch.randn(2, 1024), dim=-1)
    plan = torch.randint(0, 2, (2, 1, 32, 32)).float()
    times = torch.tensor([[0, 30, 70, 99], [99, 70, 30, 0]])
    velocity, representation = denoiser(noisy, hidden, anchors, plan, times)
    assert velocity.shape == noisy.shape
    assert representation.shape == (2, 4, 1024)
    gradient = torch.autograd.grad(velocity.square().mean(), noisy)[0]
    assert torch.isfinite(gradient).all()
    assert float(gradient.abs().sum()) > 0.0


def test_v53_initialization_transfers_v51_function_to_the_strip() -> None:
    torch.manual_seed(7)
    prior = VisualGlyphScorePrior(VisualPosteriorLanguageConfig())
    denoiser = ConditionalStripDenoiser(CausalVisualBlockDiffusionConfig())
    denoiser.initialize_from_prior(prior)
    cells = torch.randn(2, 4, 1, 32, 32)
    times = torch.full((2, 4), 37, dtype=torch.long)
    zeros_hidden = torch.zeros(2, 384)
    zeros_anchor = torch.zeros(2, 1024)
    zeros_plan = torch.zeros(2, 1, 32, 32)
    block_velocity, _ = denoiser(cells, zeros_hidden, zeros_anchor, zeros_plan, times)
    local_velocity = prior.predict_noise(
        cells.flatten(0, 1), times.flatten()
    ).reshape_as(cells)
    difference = (block_velocity - local_velocity).abs().mean()
    cosine = torch.nn.functional.cosine_similarity(
        block_velocity.flatten(1), local_velocity.flatten(1)
    ).mean()
    # Strip convolutions deliberately see adjacent cells at their boundaries,
    # so the transferred function is close rather than bitwise identical.
    assert float(difference.detach()) < 0.05
    assert float(cosine.detach()) > 0.97


def test_v53_boundary_has_no_symbolic_runtime_path() -> None:
    model = _model()
    receipt = causal_visual_block_boundary_receipt(model)
    assert receipt["denoiser_parameters"] <= 8_000_000
    assert receipt["combined_parameters"] <= 26_000_000
    assert receipt["parameter_names_with_forbidden_fragments"] == []
    assert receipt["text_tokenizer"] is False
    assert receipt["unicode_or_character_ids_in_model"] is False
    assert receipt["glyph_vocabulary_or_candidate_bank_in_model"] is False


def test_v53_stratified_times_cover_all_noise_ranges() -> None:
    generator = torch.Generator().manual_seed(53)
    times = stratified_future_timesteps(
        8,
        device=torch.device("cpu"),
        generator=generator,
    )
    assert times.shape == (8, 4)
    for row in times:
        assert int((row <= 24).sum()) == 1
        assert int(((row >= 25) & (row <= 59)).sum()) == 1
        assert int(((row >= 60) & (row <= 89)).sum()) == 1
        assert int((row >= 90).sum()) == 1


def test_v53_full_loss_is_finite_and_updates_the_conditional_writer() -> None:
    torch.manual_seed(530)
    model = _model()
    context = torch.randint(0, 2, (2, 60, 1, 32, 32)).float()
    shuffled = _shuffle_prefix(context, first_index=0)
    targets = torch.randint(0, 2, (2, 4, 1, 32, 32)).float()
    generator = torch.Generator().manual_seed(531)
    times = stratified_future_timesteps(
        2,
        device=torch.device("cpu"),
        generator=generator,
    )
    loss = visual_block_diffusion_loss(
        model,
        context=context,
        shuffled_context=shuffled,
        target_cells=targets,
        timesteps=times,
        noise=torch.randn(targets.shape, generator=generator),
        drop_condition=torch.tensor([False, True]),
    )
    loss.loss.backward()
    assert torch.isfinite(loss.loss)
    assert model.denoiser.condition[-1].weight.grad is not None
    assert torch.isfinite(model.denoiser.condition[-1].weight.grad).all()
    assert all(parameter.grad is None for parameter in model.reader.parameters())


def test_v53_visual_field_loss_reaches_predicted_pixels() -> None:
    model = _model()
    signed = torch.randn(2, 4, 1, 32, 32, requires_grad=True)
    target = torch.nn.functional.normalize(torch.randn(2, 4, 1024), dim=-1)
    loss = (1.0 - (_visual_fields(model, signed) * target).sum(-1)).mean()
    gradient = torch.autograd.grad(loss, signed)[0]
    assert torch.isfinite(gradient).all()
    assert gradient.abs().sum() > 0
    binary = torch.randint(0, 2, (2, 4, 1, 32, 32)).float()
    expected = model.reader.field.encode_unit(binary.flatten(0, 1)).reshape(2, 4, -1)
    assert torch.allclose(_visual_fields(model, binary * 2 - 1), expected, atol=1e-6)


def test_v53_synchronous_noise_matches_inference_support() -> None:
    times = stratified_future_timesteps(
        32,
        device=torch.device("cpu"),
        generator=torch.Generator().manual_seed(532),
        synchronous=True,
    )
    assert torch.equal(times, times[:, :1].expand_as(times))
    assert bool((times[:, 0] >= 60).any())

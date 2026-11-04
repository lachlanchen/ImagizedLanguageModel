from __future__ import annotations

import torch

from ilm.visual_lm.canonical_glyph_language import (
    CanonicalGlyphLanguageConfig,
    CanonicalGlyphLanguageModel,
)
from ilm.visual_lm.hierarchical_raster_language import (
    HierarchicalRasterLanguageConfig,
    HierarchicalRasterLanguageModel,
    RasterMixtureOutput,
    hierarchical_raster_language_boundary_receipt,
    micro_patches_to_raster,
    raster_to_generation_units,
    raster_to_micro_patches,
)
from ilm.visual_lm.hierarchical_raster_language_training import (
    hierarchical_raster_language_loss,
    raster_mixture_nll,
)
from ilm.visual_lm.hierarchical_raster_language_evaluation import (
    evaluate_raster_generation,
    evaluate_raster_likelihood,
    evaluate_visible_feedback,
)


def _reader_config() -> CanonicalGlyphLanguageConfig:
    return CanonicalGlyphLanguageConfig(
        model_dim=128,
        layers=1,
        heads=4,
        mlp_ratio=2.0,
        dropout=0.0,
        noise_dim=16,
        generator_layers=1,
    )


def _model() -> HierarchicalRasterLanguageModel:
    return HierarchicalRasterLanguageModel(
        _reader_config(),
        HierarchicalRasterLanguageConfig(
            patch_size=8,
            local_dim=64,
            local_layers=1,
            local_heads=4,
            local_mlp_ratio=2.0,
            local_dropout=0.0,
            mixtures=2,
        ),
    )


def _plan_preserving_model() -> HierarchicalRasterLanguageModel:
    return HierarchicalRasterLanguageModel(
        _reader_config(),
        HierarchicalRasterLanguageConfig(
            patch_size=8,
            local_dim=64,
            local_layers=1,
            local_heads=4,
            local_mlp_ratio=2.0,
            local_dropout=0.0,
            mixtures=2,
            plan_preserving_residual=True,
        ),
    )


def test_micro_patch_roundtrip_is_exact() -> None:
    cells = (torch.rand(3, 1, 32, 32) > 0.7).float()
    patches = raster_to_micro_patches(cells, patch_size=8)
    restored = micro_patches_to_raster(
        patches,
        cell_size=32,
        patch_size=8,
    )
    assert patches.shape == (3, 16, 64)
    assert torch.equal(restored, cells)
    units = raster_to_generation_units(cells, config=_model().config)
    assert units.shape == (3, 20, 64)
    assert torch.equal(units[:, _model().config.coarse_units :], patches)


def test_reader_loads_v42_without_failed_generator() -> None:
    source = CanonicalGlyphLanguageModel(_reader_config())
    model = _model()
    model.load_v42_reader(source.state_dict())
    assert not any(name.startswith("reader.generator") for name, _ in model.named_parameters())
    assert torch.equal(
        model.reader.input_projection.weight,
        source.input_projection.weight,
    )


def test_teacher_forcing_and_generation_are_image_native() -> None:
    torch.manual_seed(3)
    model = _model().eval()
    context = (torch.rand(2, 4, 1, 32, 32) > 0.8).float()
    target = (torch.rand(2, 1, 32, 32) > 0.8).float()
    language = model.language(context)
    output = model.teacher_force(
        language["hidden_states"][:, -1],
        language["anchor_fields"][:, -1],
        target,
    )
    assert output.mixture_logits.shape == (2, 20, 2)
    assert output.pixel_logits.shape == (2, 20, 2, 64)
    generated, metadata = model.sample_next(context, stochastic=False)
    assert generated.shape == (2, 1, 32, 32)
    assert set(torch.unique(generated).tolist()) <= {0.0, 1.0}
    assert torch.isfinite(metadata["log_probability"]).all()
    boundary = hierarchical_raster_language_boundary_receipt(model)
    assert boundary["text_tokenizer"] is False
    assert boundary["glyph_vocabulary_or_candidate_bank_in_model"] is False


def test_mixture_likelihood_rewards_matching_raster() -> None:
    target = torch.zeros(2, 3, 4)
    target[0, :, 0] = 1.0
    matching = torch.where(target[:, :, None] > 0.5, 8.0, -8.0).expand(-1, -1, 2, -1)
    wrong = -matching
    mixture = torch.zeros(2, 3, 2)
    matching_output = RasterMixtureOutput(mixture, matching, torch.empty(0))
    wrong_output = RasterMixtureOutput(mixture, wrong, torch.empty(0))
    _, matching_nll = raster_mixture_nll(matching_output, target)
    _, wrong_nll = raster_mixture_nll(wrong_output, target)
    assert bool((matching_nll < wrong_nll).all())


def test_plan_preserving_writer_starts_at_visual_plan() -> None:
    torch.manual_seed(5)
    model = _plan_preserving_model().eval()
    context = (torch.rand(2, 4, 1, 32, 32) > 0.8).float()
    language = model.language(context)
    hidden = language["hidden_states"][:, -1]
    anchor = language["anchor_fields"][:, -1]
    plan = model.plan_units(anchor)
    generated, _ = model.writer.generate(
        hidden,
        anchor,
        plan,
        stochastic=False,
    )
    expected = model.reader.field.binary(anchor)
    assert torch.equal(generated, expected)


def test_confident_plan_pixels_cannot_be_flipped_by_bounded_residual() -> None:
    torch.manual_seed(6)
    model = _plan_preserving_model().eval()
    context = (torch.rand(2, 4, 1, 32, 32) > 0.8).float()
    language = model.language(context)
    hidden = language["hidden_states"][:, -1]
    anchor = language["anchor_fields"][:, -1]
    plan = model.plan_units(anchor)
    targets = torch.zeros_like(plan)
    output = model.writer(hidden, anchor, targets, plan)
    confident = (plan <= 1e-4) | (plan >= 1.0 - 1e-4)
    if bool(confident.any()):
        predicted = output.pixel_logits.mean(dim=2) >= 0.0
        assert torch.equal(predicted[confident], (plan >= 0.5)[confident])


def test_joint_loss_backpropagates_through_reader_and_writer() -> None:
    torch.manual_seed(7)
    model = _model().train()
    context = (torch.rand(2, 4, 1, 32, 32) > 0.84).float()
    target = (torch.rand(2, 4, 1, 32, 32) > 0.84).float()
    language = model.language(context)
    generator = torch.Generator().manual_seed(11)
    loss = hierarchical_raster_language_loss(
        model,
        language,
        target,
        positions_per_stream=2,
        generator=generator,
    )
    loss.loss.backward()
    assert torch.isfinite(loss.loss)
    assert model.reader.input_projection.weight.grad is not None
    assert model.writer.output.weight.grad is not None
    assert loss.selected_positions == 4


def test_evaluator_compares_likelihood_generation_and_feedback() -> None:
    torch.manual_seed(13)
    model = _model().eval()
    context = (torch.rand(2, 8, 1, 32, 32) > 0.84).float()
    continuation = (torch.rand(2, 4, 1, 32, 32) > 0.84).float()
    bank = continuation[:, 0].clone()
    loader = [
        {
            "context": context,
            "continuation": continuation,
            "target_index": torch.arange(2),
        }
    ]
    likelihood = evaluate_raster_likelihood(
        model,
        loader,
        device=torch.device("cpu"),
        precision="fp32",
        maximum_examples=2,
    )
    generation = evaluate_raster_generation(
        model,
        loader,
        bank,
        device=torch.device("cpu"),
        precision="fp32",
        maximum_examples=2,
        samples=2,
    )
    feedback = evaluate_visible_feedback(
        model,
        loader,
        bank,
        device=torch.device("cpu"),
        precision="fp32",
        maximum_examples=2,
        steps=2,
    )
    assert likelihood["examples"] == 2
    assert generation["ordered"]["examples"] == 2
    assert "plan_identity_top1" in generation["ordered"]
    assert "greedy_identity_retention_of_plan" in generation["ordered"]
    assert len(feedback["ordered"]["identity_top1_by_step"]) == 2

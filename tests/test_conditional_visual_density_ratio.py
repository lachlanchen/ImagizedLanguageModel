from __future__ import annotations

import pytest
import torch

from ilm.visual_lm.conditional_visual_density_ratio import (
    ConditionalVisualDensityRatioConfig,
    ConditionalVisualDensityRatioModel,
    conditional_visual_density_ratio_boundary_receipt,
    per_row_assignment_margin,
    row_center_scores,
)
from ilm.visual_lm.dense_visual_future_energy import (
    DenseVisualFutureConfig,
    DenseVisualFutureModel,
)


def _small_config() -> ConditionalVisualDensityRatioConfig:
    return ConditionalVisualDensityRatioConfig(
        visual_dim=64,
        semantic_dim=64,
        model_dim=128,
        layers=2,
        heads=4,
        mlp_ratio=2.0,
        retina_base_channels=8,
        semantic_hidden_dim=96,
        evidence_layers=2,
        evidence_heads=4,
        evidence_mlp_ratio=2.0,
        relation_hidden_dim=128,
        score_chunk_size=3,
    )


def _small_model() -> ConditionalVisualDensityRatioModel:
    return ConditionalVisualDensityRatioModel(_small_config())


def test_v29_scores_shared_and_paired_arbitrary_candidate_images() -> None:
    model = _small_model().eval()
    context = torch.rand(2, 7, 1, 32, 32)
    candidates = torch.rand(5, 1, 32, 32)
    paired_context = context[:, None].expand(-1, 2, -1, -1, -1, -1)
    paired_candidates = torch.rand(2, 2, 1, 32, 32)
    with torch.no_grad():
        state = model.encode_context(context)
        shared = model.score_shared_candidates(context, candidates)
        paired = model.score_paired_candidates(
            paired_context, paired_candidates
        )
    assert state.shape == (2, 7, 128)
    assert shared.shape == (2, 5)
    assert paired.shape == (2, 2, 2)
    assert torch.isfinite(shared).all()
    assert torch.isfinite(paired).all()


def test_v29_shared_scoring_is_chunk_invariant() -> None:
    model = _small_model().eval()
    context = torch.rand(2, 8, 1, 32, 32)
    candidates = torch.rand(7, 1, 32, 32)
    with torch.no_grad():
        state = model.encode_context(context)
        raw, semantic = model.encode_image_parts(candidates, target=True)
        one = model.score_encoded_shared(
            state, raw, semantic, chunk_size=1
        )
        full = model.score_encoded_shared(
            state, raw, semantic, chunk_size=7
        )
    assert torch.allclose(one, full, atol=1e-6, rtol=1e-6)


def test_v29_candidate_column_permutation_is_equivariant() -> None:
    model = _small_model().eval()
    contexts = torch.rand(3, 2, 8, 1, 32, 32)
    candidates = torch.rand(3, 2, 1, 32, 32)
    with torch.no_grad():
        original = model.score_paired_candidates(contexts, candidates)
        swapped = model.score_paired_candidates(contexts, candidates.flip(1))
    assert torch.allclose(swapped, original.flip(-1), atol=1e-6, rtol=1e-6)


def test_v29_identical_suffix_rows_have_identical_scores_and_zero_delta() -> None:
    model = _small_model().eval()
    suffix = torch.rand(2, 4, 1, 32, 32)
    candidates = torch.rand(2, 2, 1, 32, 32)
    contexts = suffix[:, None].expand(-1, 2, -1, -1, -1, -1).clone()
    with torch.no_grad():
        score = model.score_exact_suffix_paired(contexts, candidates)
    assert torch.equal(score[:, 0], score[:, 1])
    assert torch.equal(row_center_scores(score)[:, 0], row_center_scores(score)[:, 1])
    assert torch.equal(score - score, torch.zeros_like(score))


def test_v29_row_centering_removes_only_row_offsets() -> None:
    scores = torch.tensor([[[1.0, 3.0], [8.0, 2.0]]])
    offsets = torch.tensor([[[50.0], [-12.0]]])
    centered = row_center_scores(scores)
    shifted = row_center_scores(scores + offsets)
    assert torch.allclose(centered, shifted)
    assert torch.allclose(centered.mean(dim=-1), torch.zeros(1, 2))


def test_v29_per_row_margin_cannot_hide_one_wrong_assignment() -> None:
    logits = torch.tensor([[[8.0, 0.0], [9.0, 10.0]]])
    assignments = torch.tensor([[0, 0]])
    margins = per_row_assignment_margin(logits, assignments)
    assert margins.shape == (1, 2)
    assert float(margins[0, 0]) > 0
    assert float(margins[0, 1]) < 0
    with pytest.raises(ValueError, match="assignments"):
        per_row_assignment_margin(logits, assignments.float())


def test_v29_loads_only_the_v28_visual_context_backbone() -> None:
    v28 = DenseVisualFutureModel(
        DenseVisualFutureConfig(
            visual_dim=64,
            semantic_dim=64,
                model_dim=128,
            layers=2,
            heads=4,
            mlp_ratio=2.0,
            retina_base_channels=8,
            semantic_hidden_dim=96,
            hypotheses=2,
        )
    )
    model = _small_model()
    receipt = model.load_v28_backbone_state(v28.state_dict())
    assert receipt["discarded_future_heads"] is True
    for name in model._BACKBONE_MODULES:
        source = getattr(v28, name).state_dict()
        target = getattr(model, name).state_dict()
        assert source.keys() == target.keys()
        for key in source:
            assert torch.equal(source[key], target[key])
    assert all(not parameter.requires_grad for parameter in model.retina.parameters())
    assert all(
        not parameter.requires_grad
        for parameter in model.semantic_adapter.parameters()
    )


def test_v29_rejects_symbolic_inputs() -> None:
    model = _small_model()
    with pytest.raises(TypeError, match="floating image"):
        model.encode_context(torch.ones(1, 4, 1, 32, 32, dtype=torch.long))
    with pytest.raises(ValueError, match="at least two candidates"):
        row_center_scores(torch.ones(3, 1))


def test_v29_boundary_has_no_symbolic_or_deployed_bank_route() -> None:
    receipt = conditional_visual_density_ratio_boundary_receipt(_small_config())
    for key in (
        "input_is_continuous_image_stream",
        "candidate_is_arbitrary_image",
        "output_is_candidate_conditioned_visual_energy",
        "retina_is_frozen",
        "semantic_adapters_are_frozen",
    ):
        assert receipt[key] is True
    for key in (
        "uses_strings",
        "uses_token_ids",
        "uses_unicode_ids",
        "uses_character_ids",
        "uses_vocabulary_embedding",
        "uses_vocabulary_output",
        "uses_ocr",
        "uses_visual_codebook",
        "uses_glyph_lookup",
        "uses_external_language_model",
        "candidate_bank_deployed",
        "candidate_bank_in_model_state",
    ):
        assert receipt[key] is False

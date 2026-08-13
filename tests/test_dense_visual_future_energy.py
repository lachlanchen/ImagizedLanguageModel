from __future__ import annotations

import pytest
import torch

from ilm.visual_lm.dense_visual_future_energy import (
    DenseVisualFutureConfig,
    DenseVisualFutureModel,
    assignment_margin,
    dense_visual_future_boundary_receipt,
    mixture_energy_score,
    weighted_multi_positive_nce,
)


def _small_model() -> DenseVisualFutureModel:
    return DenseVisualFutureModel(
        DenseVisualFutureConfig(
            visual_dim=64,
            semantic_dim=64,
            model_dim=128,
            layers=2,
            heads=4,
            mlp_ratio=2.0,
            retina_base_channels=8,
            semantic_hidden_dim=128,
            hypotheses=3,
        )
    )


def test_v28_scores_shared_and_paired_arbitrary_images() -> None:
    model = _small_model().eval()
    context = torch.rand(2, 7, 1, 32, 32)
    candidates = torch.rand(5, 1, 32, 32)
    paired_context = context[:, None].expand(-1, 2, -1, -1, -1, -1)
    paired_candidates = torch.rand(2, 2, 1, 32, 32)
    with torch.no_grad():
        state = model.encode_context(context)
        distribution = model.future_distribution(state, horizon=2)
        shared = model.score_shared_candidates(context, candidates)
        paired = model.score_paired_candidates(
            paired_context, paired_candidates
        )
    assert state.shape == (2, 7, 128)
    assert distribution["raw_queries"].shape == (2, 7, 3, 64)
    assert distribution["semantic_queries"].shape == (2, 7, 3, 64)
    assert distribution["mixture_logits"].shape == (2, 7, 3)
    assert shared.shape == (2, 5)
    assert paired.shape == (2, 2, 2)


def test_v28_candidate_column_permutation_is_exactly_equivariant() -> None:
    model = _small_model().eval()
    contexts = torch.rand(3, 2, 8, 1, 32, 32)
    candidates = torch.rand(3, 2, 1, 32, 32)
    with torch.no_grad():
        original = model.score_paired_candidates(contexts, candidates)
        swapped = model.score_paired_candidates(contexts, candidates.flip(1))
    assert torch.allclose(swapped, original.flip(-1), atol=1e-6)


def test_v28_frozen_retina_and_identity_initialized_semantic_adapter() -> None:
    model = _small_model().eval()
    images = torch.rand(4, 1, 32, 32)
    raw, semantic = model.encode_image_parts(images, target=False)
    target_raw, target_semantic = model.encode_image_parts(images, target=True)
    assert all(not parameter.requires_grad for parameter in model.retina.parameters())
    assert torch.allclose(semantic, raw, atol=1e-7)
    assert torch.allclose(target_raw, raw, atol=1e-7)
    assert torch.allclose(target_semantic, raw, atol=1e-7)


def test_v28_weighted_multi_positive_nce_supports_duplicate_pixels() -> None:
    logits = torch.tensor(
        [[3.0, -1.0, 2.0], [-2.0, 4.0, -3.0]], requires_grad=True
    )
    query_groups = torch.tensor([7, 8])
    candidate_groups = torch.tensor([7, 8, 7])
    weights = torch.tensor([2.0, 1.0])
    loss, metrics = weighted_multi_positive_nce(
        logits,
        query_groups,
        candidate_groups,
        weights=weights,
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(logits.grad).all()
    assert float(metrics["top1"]) == 1.0


def test_v28_energy_score_rewards_a_matching_continuous_future() -> None:
    target = torch.nn.functional.normalize(torch.randn(4, 16), dim=-1)
    matching = target[:, None].expand(-1, 3, -1).clone().requires_grad_(True)
    unrelated = torch.nn.functional.normalize(torch.randn(4, 3, 16), dim=-1)
    mixture = torch.zeros(4, 3)
    match_score = mixture_energy_score(matching, mixture, target).mean()
    unrelated_score = mixture_energy_score(unrelated, mixture, target).mean()
    match_score.backward()
    assert float(match_score.detach()) < float(unrelated_score)
    assert torch.isfinite(matching.grad).all()


def test_v28_assignment_margin_uses_visible_candidate_positions() -> None:
    logits = torch.tensor(
        [[[1.0, 5.0], [4.0, 0.0]], [[6.0, 0.0], [1.0, 7.0]]]
    )
    assignments = torch.tensor([[1, 0], [0, 1]])
    margin = assignment_margin(logits, assignments)
    assert torch.all(margin > 0)
    with pytest.raises(ValueError, match="assignments"):
        assignment_margin(logits, assignments.float())


def test_v28_rejects_symbolic_inputs_and_unknown_horizons() -> None:
    model = _small_model()
    with pytest.raises(TypeError, match="floating image"):
        model.encode_context(torch.ones(1, 4, 1, 32, 32, dtype=torch.long))
    with pytest.raises(ValueError, match="unknown V28 horizon"):
        model.future_distribution(torch.rand(1, 128), horizon=3)


def test_v28_boundary_has_no_symbolic_or_deployed_bank_route() -> None:
    receipt = dense_visual_future_boundary_receipt(DenseVisualFutureConfig())
    for key in (
        "input_is_continuous_image_stream",
        "candidate_is_arbitrary_image",
        "output_is_continuous_future_distribution",
        "retina_is_frozen",
        "target_semantic_route_is_ema",
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
    ):
        assert receipt[key] is False

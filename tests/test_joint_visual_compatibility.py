from __future__ import annotations

import pytest
import torch

from ilm.visual_lm.joint_visual_compatibility import (
    JointVisualCompatibilityConfig,
    JointVisualCompatibilityModel,
    exact_image_identity_mask,
    joint_visual_compatibility_boundary_receipt,
    multi_positive_nce,
    paired_assignment_loss,
    vicreg_loss,
)


def _small_model() -> JointVisualCompatibilityModel:
    return JointVisualCompatibilityModel(
        JointVisualCompatibilityConfig(
            visual_dim=64,
            model_dim=128,
            layers=2,
            heads=4,
            mlp_ratio=2.0,
            retina_base_channels=8,
            candidate_hidden_dim=128,
        )
    )


def test_joint_model_scores_shared_and_per_example_candidate_images() -> None:
    model = _small_model().eval()
    context = torch.rand(2, 7, 1, 32, 32)
    shared = torch.rand(5, 1, 32, 32)
    paired_context = context[:, None].expand(-1, 2, -1, -1, -1, -1)
    paired_candidates = torch.rand(2, 2, 1, 32, 32)
    with torch.no_grad():
        query = model.encode_context(context)
        shared_logits = model.score_shared_candidates(context, shared)
        pair_logits = model.score_paired_candidates(
            paired_context, paired_candidates
        )
    assert query.shape == (2, 64)
    assert torch.allclose(query.norm(dim=-1), torch.ones(2), atol=1e-5)
    assert shared_logits.shape == (2, 5)
    assert pair_logits.shape == (2, 2, 2)


def test_candidate_projector_initially_preserves_retinal_geometry() -> None:
    model = _small_model().eval()
    images = torch.rand(3, 1, 32, 32)
    with torch.no_grad():
        retinal = torch.nn.functional.normalize(model.retina(images).float(), dim=-1)
        projected = model.encode_candidates(images, target=False)
        target = model.encode_candidates(images, target=True)
    assert torch.allclose(projected, retinal, atol=1e-7)
    assert torch.allclose(target, retinal, atol=1e-7)


def test_joint_model_rejects_non_image_symbolic_inputs() -> None:
    model = _small_model()
    with pytest.raises(TypeError, match="floating image"):
        model.encode_context(torch.ones(1, 4, 1, 32, 32, dtype=torch.long))
    with pytest.raises(ValueError, match="shape"):
        model.score_shared_candidates(
            torch.rand(1, 4, 1, 32, 32), torch.rand(2, 32, 32)
        )


def test_exact_image_mask_and_multi_positive_nce_use_pixels() -> None:
    images = torch.zeros(3, 1, 32, 32)
    images[1] = 1.0
    images[2] = images[0]
    positives = exact_image_identity_mask(images)
    assert torch.equal(
        positives,
        torch.tensor(
            [[True, False, True], [False, True, False], [True, False, True]]
        ),
    )
    logits = torch.tensor(
        [[2.0, -1.0, 3.0], [-1.0, 4.0, -2.0], [3.0, 0.0, 2.0]],
        requires_grad=True,
    )
    loss, metrics = multi_positive_nce(logits, positives)
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(logits.grad).all()
    assert float(metrics["multi_positive_top1"]) == 1.0


def test_pair_assignment_supports_swapped_candidate_columns() -> None:
    logits = torch.tensor(
        [[[1.0, 5.0], [4.0, 0.0]], [[6.0, 0.0], [1.0, 7.0]]],
        requires_grad=True,
    )
    assignments = torch.tensor([[1, 0], [0, 1]], dtype=torch.long)
    loss, metrics = paired_assignment_loss(logits, assignments)
    loss.backward()
    assert float(metrics["pair_arm_accuracy"]) == 1.0
    assert float(metrics["pair_both_correct_rate"]) == 1.0
    assert torch.isfinite(logits.grad).all()


def test_identical_pair_rows_receive_exact_tie_aware_chance() -> None:
    logits = torch.tensor([[[2.0, 1.0], [2.0, 1.0]]])
    _, metrics = paired_assignment_loss(
        logits, torch.tensor([[0, 1]], dtype=torch.long)
    )
    assert float(metrics["pair_arm_accuracy"]) == 0.5
    assert float(metrics["pair_both_correct_rate"]) == 0.0


def test_vicreg_has_finite_gradients_for_small_visual_batches() -> None:
    first = torch.randn(4, 16, requires_grad=True)
    second = torch.randn(4, 16, requires_grad=True)
    loss, metrics = vicreg_loss(first, second)
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(first.grad).all()
    assert torch.isfinite(second.grad).all()
    assert set(metrics) == {
        "vicreg_loss",
        "vicreg_invariance",
        "vicreg_variance",
        "vicreg_covariance",
    }


def test_v27_boundary_has_no_symbolic_or_deployed_candidate_bank() -> None:
    receipt = joint_visual_compatibility_boundary_receipt(
        JointVisualCompatibilityConfig()
    )
    assert receipt["candidate_is_arbitrary_image"] is True
    assert receipt["target_route_is_ema"] is True
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

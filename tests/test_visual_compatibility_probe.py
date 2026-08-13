from __future__ import annotations

import torch

from ilm.visual_lm.visual_compatibility_probe import (
    VisualCandidateCompatibilityProbe,
    VisualCompatibilityProbeConfig,
    paired_compatibility_loss,
    visual_compatibility_probe_boundary_receipt,
)


def _probe() -> VisualCandidateCompatibilityProbe:
    return VisualCandidateCompatibilityProbe(
        VisualCompatibilityProbeConfig(
            context_dim=16,
            candidate_dim=12,
            hidden_dim=24,
            projection_dim=8,
        )
    )


def test_visual_candidate_probe_scores_arbitrary_image_derived_pairs() -> None:
    probe = _probe()
    contexts = torch.randn(5, 2, 16)
    candidates = torch.randn(5, 2, 12)
    logits = probe(contexts, candidates)
    assert logits.shape == (5, 2, 2)
    assert torch.isfinite(logits).all()


def test_paired_compatibility_loss_prefers_the_diagonal_and_backpropagates() -> None:
    logits = torch.tensor(
        [[[3.0, -1.0], [-2.0, 4.0]], [[2.0, 0.0], [-1.0, 2.0]]],
        requires_grad=True,
    )
    loss, metrics = paired_compatibility_loss(logits)
    loss.backward()
    assert float(metrics["arm_accuracy"]) == 1.0
    assert float(metrics["strict_arm_accuracy"]) == 1.0
    assert float(metrics["arm_tie_rate"]) == 0.0
    assert float(metrics["both_correct_rate"]) == 1.0
    assert float(metrics["mean_margin"]) > 0.0
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_identical_context_rows_are_an_exact_chance_control() -> None:
    probe = _probe().eval()
    one_context = torch.randn(7, 1, 16)
    contexts = one_context.expand(-1, 2, -1)
    candidates = torch.randn(7, 2, 12)
    with torch.no_grad():
        logits = probe(contexts, candidates)
        _, metrics = paired_compatibility_loss(logits)
    assert torch.equal(logits[:, 0], logits[:, 1])
    assert float(metrics["arm_accuracy"]) == 0.5
    assert float(metrics["both_correct_rate"]) == 0.0
    assert abs(float(metrics["mean_margin"])) < 1e-6


def test_exact_ties_receive_chance_credit_instead_of_counting_as_errors() -> None:
    logits = torch.zeros(3, 2, 2)
    _, metrics = paired_compatibility_loss(logits)
    assert float(metrics["arm_accuracy"]) == 0.5
    assert float(metrics["strict_arm_accuracy"]) == 0.0
    assert float(metrics["arm_tie_rate"]) == 1.0


def test_probe_boundary_is_visual_and_explicitly_diagnostic() -> None:
    receipt = visual_compatibility_probe_boundary_receipt()
    assert receipt["input_context_is_image_derived"] is True
    assert receipt["input_candidates_are_images"] is True
    assert receipt["v26_backbone_is_frozen"] is True
    assert receipt["diagnostic_candidate_images_required"] is True
    assert receipt["deployed_language_model"] is False
    for key in (
        "uses_strings",
        "uses_token_ids",
        "uses_unicode_ids",
        "uses_character_ids",
        "uses_ocr",
        "uses_external_language_model",
    ):
        assert receipt[key] is False

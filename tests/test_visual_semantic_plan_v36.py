from __future__ import annotations

import inspect

import torch

from ilm.visual_lm.visual_semantic_plan import (
    VisualSemanticPlanConfig,
    VisualSemanticPlanModel,
    VisualSentenceImageTeacher,
    visual_semantic_plan_boundary_receipt,
)
from ilm.visual_lm.visual_semantic_plan_data import V36_PATCHES, V36_WIDTH


def _tiny_config() -> VisualSemanticPlanConfig:
    return VisualSemanticPlanConfig(
        reader_hidden_size=64,
        reader_layers=2,
        reader_heads=4,
        reader_intermediate_size=128,
        reader_dropout=0.0,
        planner_dim=48,
        planner_layers=2,
        planner_heads=4,
        planner_mlp_dim=96,
        planner_dropout=0.0,
        plan_dim=64,
        length_hidden_size=24,
    )


def test_v36_model_emits_normalized_continuous_plans() -> None:
    torch.manual_seed(3)
    model = VisualSemanticPlanModel(_tiny_config()).eval()
    pixels = torch.rand(2, 3, 16, V36_WIDTH)
    mask = torch.zeros(2, V36_PATCHES)
    mask[:, :7] = 1.0
    output = model.generate_plan(pixels, mask)
    assert output.plans.shape == (2, 5, 64)
    assert output.length.shape == (2,)
    assert torch.allclose(output.plans.norm(dim=-1), torch.ones(2, 5), atol=1e-5)
    assert torch.all(output.length >= 0.0)
    assert torch.isfinite(output.plans).all()


def test_v36_blank_prompt_is_finite_and_uses_no_answer_argument() -> None:
    model = VisualSemanticPlanModel(_tiny_config()).eval()
    pixels = torch.ones(1, 3, 16, V36_WIDTH)
    mask = torch.zeros(1, V36_PATCHES)
    output = model.generate_plan(pixels, mask)
    assert torch.isfinite(output.plans).all()
    assert list(inspect.signature(model.generate_plan).parameters) == [
        "prompt_pixels",
        "prompt_mask",
    ]


def test_v36_teacher_uses_only_pixels_and_patch_masks() -> None:
    teacher = VisualSentenceImageTeacher(_tiny_config()).eval()
    pixels = torch.rand(2, 3, 16, V36_WIDTH)
    mask = torch.zeros(2, V36_PATCHES)
    mask[:, :5] = 1.0
    embedding = teacher(pixels, mask)
    assert embedding.shape == (2, 64)
    assert torch.allclose(embedding.norm(dim=-1), torch.ones(2), atol=1e-5)


def test_v36_boundary_rejects_runtime_teacher_and_candidates() -> None:
    model = VisualSemanticPlanModel(_tiny_config())
    receipt = visual_semantic_plan_boundary_receipt(model)
    assert receipt["parameter_cap_pass"] is True
    assert receipt["forbidden_parameter_names"] == []
    assert receipt["uses_answer_teacher_at_runtime"] is False
    assert receipt["candidate_bank_deployed"] is False
    assert receipt["generates_raster"] is False

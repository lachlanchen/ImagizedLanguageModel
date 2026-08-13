from __future__ import annotations

import torch

from ilm.visual_lm.visual_semantic_distillation import (
    VisualSemanticDistillationConfig,
    VisualSemanticDistillationModel,
    visual_semantic_distillation_boundary_receipt,
)
from ilm.visual_lm.visual_semantic_distillation_data import (
    V37_PATCHES,
    V37_PATCH_SIZE,
    V37_WIDTH,
)


def tiny_config() -> VisualSemanticDistillationConfig:
    return VisualSemanticDistillationConfig(
        reader_hidden_size=64,
        reader_layers=2,
        reader_heads=4,
        reader_intermediate_size=128,
        reader_dropout=0.0,
        projection_hidden_size=96,
        semantic_dim=64,
        projection_dropout=0.0,
        plan_hidden_size=32,
        length_hidden_size=16,
    )


def test_tiny_model_has_complete_image_only_forward() -> None:
    model = VisualSemanticDistillationModel(tiny_config()).eval()
    pixels = torch.ones(2, 3, V37_PATCH_SIZE, V37_WIDTH)
    pixels[:, :, :, :32] = 0
    mask = torch.zeros(2, V37_PATCHES)
    mask[:, :2] = 1

    output = model.generate_plan(pixels, mask)

    assert output.semantic_state.shape == (2, 64)
    assert output.answer_plan.shape == (2, 64)
    assert output.length.shape == (2,)
    assert torch.allclose(output.semantic_state.norm(dim=-1), torch.ones(2), atol=1e-5)
    assert torch.allclose(output.answer_plan.norm(dim=-1), torch.ones(2), atol=1e-5)
    assert bool(((output.length >= 0) & (output.length <= V37_PATCHES)).all())
    assert torch.isfinite(output.semantic_features).all()


def test_residual_scale_is_bounded_and_initialized() -> None:
    model = VisualSemanticDistillationModel(tiny_config())

    assert abs(float(model.residual_scale.detach()) - 0.05) < 1e-6
    model.residual_logit.data.fill_(100)
    assert 0.49 < float(model.residual_scale.detach()) <= 0.5
    model.residual_logit.data.fill_(-100)
    assert 0 <= float(model.residual_scale.detach()) < 1e-6


def test_reader_trainability_stages_are_explicit() -> None:
    model = VisualSemanticDistillationModel(tiny_config())

    model.freeze_reader()
    model.train()
    assert not model.reader.training
    assert not any(parameter.requires_grad for parameter in model.reader.parameters())
    assert all(
        parameter.requires_grad for parameter in model.semantic_head.parameters()
    )

    model.unfreeze_reader()
    model.train()
    assert model.reader.training
    assert all(parameter.requires_grad for parameter in model.reader.parameters())


def test_boundary_has_no_symbolic_or_teacher_parameters() -> None:
    model = VisualSemanticDistillationModel(tiny_config())
    receipt = visual_semantic_distillation_boundary_receipt(model)

    assert receipt["parameter_cap_pass"]
    assert receipt["forbidden_parameter_names"] == []
    assert not receipt["uses_strings"]
    assert not receipt["uses_token_ids"]
    assert not receipt["uses_unicode_ids"]
    assert not receipt["uses_ocr"]
    assert not receipt["uses_bge_at_runtime"]
    assert not receipt["candidate_bank_deployed"]
    assert not receipt["generates_raster"]


def test_production_model_stays_below_parameter_cap() -> None:
    model = VisualSemanticDistillationModel(VisualSemanticDistillationConfig())
    receipt = visual_semantic_distillation_boundary_receipt(model)

    assert receipt["total_parameters"] < 100_000_000
    assert receipt["total_parameters"] > 80_000_000

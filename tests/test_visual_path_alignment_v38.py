from __future__ import annotations

import torch
import torch.nn.functional as F

from ilm.visual_lm.visual_path_alignment import (
    VisualPathAlignmentConfig,
    VisualPathAlignmentModel,
    visual_path_alignment_boundary_receipt,
)
from ilm.visual_lm.visual_semantic_distillation_data import (
    V37_PATCHES,
    V37_PATCH_SIZE,
    V37_WIDTH,
)


def tiny_config() -> VisualPathAlignmentConfig:
    return VisualPathAlignmentConfig(
        reader_hidden_size=64,
        reader_layers=2,
        reader_heads=4,
        reader_intermediate_size=128,
        reader_dropout=0.0,
        projection_hidden_size=96,
        semantic_dim=64,
        projection_dropout=0.0,
        answer_hidden_size=32,
        length_hidden_size=16,
    )


def visual_batch(batch: int = 2) -> tuple[torch.Tensor, torch.Tensor]:
    pixels = torch.ones(batch, 3, V37_PATCH_SIZE, V37_WIDTH)
    pixels[:, :, :, :48] = 0
    mask = torch.zeros(batch, V37_PATCHES)
    mask[:, :3] = 1
    return pixels, mask


def test_v38_has_complete_one_pass_image_only_forward() -> None:
    model = VisualPathAlignmentModel(tiny_config()).eval()
    pixels, mask = visual_batch()

    output = model.generate_plan(pixels, mask)

    assert output.prompt_state.shape == (2, 64)
    assert output.answer_state.shape == (2, 64)
    assert output.length.shape == (2,)
    assert torch.allclose(output.prompt_state.norm(dim=-1), torch.ones(2), atol=1e-5)
    assert torch.allclose(output.answer_state.norm(dim=-1), torch.ones(2), atol=1e-5)
    assert bool(((output.length >= 0) & (output.length <= V37_PATCHES)).all())
    assert torch.isfinite(output.pooled_visual_state).all()


def test_answer_map_can_leave_the_prompt_direction_without_an_angle_cap() -> None:
    model = VisualPathAlignmentModel(tiny_config()).eval()
    model.initialize_answer_rotation(-torch.eye(64))
    pixels, mask = visual_batch()

    output = model(pixels, mask)

    cosine = F.cosine_similarity(output.prompt_state, output.answer_state, dim=-1)
    assert torch.allclose(cosine, -torch.ones_like(cosine), atol=1e-5)


def test_reader_trainability_is_explicit() -> None:
    model = VisualPathAlignmentModel(tiny_config())

    model.freeze_reader()
    model.train()
    assert not model.reader.training
    assert not any(parameter.requires_grad for parameter in model.reader.parameters())
    assert all(parameter.requires_grad for parameter in model.prompt_head.parameters())

    model.unfreeze_reader()
    model.train()
    assert model.reader.training
    assert all(parameter.requires_grad for parameter in model.reader.parameters())


def test_v38_boundary_is_independent_and_below_cap() -> None:
    receipt = visual_path_alignment_boundary_receipt(
        VisualPathAlignmentModel(VisualPathAlignmentConfig())
    )

    assert 80_000_000 < receipt["total_parameters"] < 100_000_000
    assert receipt["forbidden_parameter_names"] == []
    assert receipt["parameter_cap_pass"]
    assert not receipt["uses_token_ids"]
    assert not receipt["uses_unicode_ids"]
    assert not receipt["uses_ocr"]
    assert not receipt["uses_bge_at_runtime"]
    assert not receipt["uses_qwen_at_runtime"]
    assert not receipt["answer_map_angle_capped"]


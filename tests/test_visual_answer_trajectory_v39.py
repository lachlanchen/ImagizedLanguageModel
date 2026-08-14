from __future__ import annotations

import torch
import torch.nn.functional as F

from ilm.visual_lm.visual_answer_trajectory import (
    V39_MAX_SEGMENTS,
    VisualAnswerTrajectoryConfig,
    VisualAnswerTrajectoryModel,
    visual_segment_count_distribution,
    visual_answer_trajectory_boundary_receipt,
)
from ilm.visual_lm.visual_semantic_distillation_data import (
    V37_PATCHES,
    V37_PATCH_SIZE,
    V37_WIDTH,
)


def tiny_config() -> VisualAnswerTrajectoryConfig:
    return VisualAnswerTrajectoryConfig(
        reader_hidden_size=64,
        reader_layers=2,
        reader_heads=4,
        reader_intermediate_size=128,
        reader_dropout=0.0,
        projection_hidden_size=96,
        semantic_dim=64,
        projection_dropout=0.0,
        answer_hidden_size=32,
        planner_hidden_size=64,
        planner_layers=2,
        planner_heads=4,
        planner_intermediate_size=128,
        planner_dropout=0.0,
        length_hidden_size=16,
    )


def visual_batch(batch: int = 2) -> tuple[torch.Tensor, torch.Tensor]:
    pixels = torch.ones(batch, 3, V37_PATCH_SIZE, V37_WIDTH)
    pixels[:, :, :, :64] = 0
    mask = torch.zeros(batch, V37_PATCHES)
    mask[:, :4] = 1
    return pixels, mask


def test_v39_image_only_forward_emits_ordered_trajectory() -> None:
    model = VisualAnswerTrajectoryModel(tiny_config()).eval()
    pixels, mask = visual_batch()

    output = model.generate_plan(pixels, mask)

    assert output.read_state.shape == (2, 64)
    assert output.answer_state.shape == (2, 64)
    assert output.segment_states.shape == (2, V39_MAX_SEGMENTS, 64)
    assert output.stop_logits.shape == (2, V39_MAX_SEGMENTS)
    assert output.active_probabilities.shape == (2, V39_MAX_SEGMENTS)
    assert output.lengths.shape == (2, V39_MAX_SEGMENTS)
    assert torch.allclose(output.read_state.norm(dim=-1), torch.ones(2), atol=1e-5)
    assert torch.allclose(output.answer_state.norm(dim=-1), torch.ones(2), atol=1e-5)
    assert torch.allclose(
        output.segment_states.norm(dim=-1),
        torch.ones(2, V39_MAX_SEGMENTS),
        atol=1e-5,
    )
    assert bool(
        (output.active_probabilities[:, 1:] <= output.active_probabilities[:, :-1])
        .all()
    )
    assert bool(((output.lengths >= 0) & (output.lengths <= V37_PATCHES)).all())
    assert torch.allclose(output.lengths, torch.full_like(output.lengths, 20.0))


def test_stop_hazards_define_a_normalized_first_stop_distribution() -> None:
    logits = torch.full((2, V39_MAX_SEGMENTS), -20.0)
    logits[:, 1] = 20.0
    alternate_last = logits.clone()
    alternate_last[:, -1] = 20.0

    distribution = visual_segment_count_distribution(logits)
    alternate = visual_segment_count_distribution(alternate_last)

    assert torch.allclose(
        distribution.probabilities.sum(dim=1),
        torch.ones(2),
        atol=1e-6,
    )
    assert distribution.mode.tolist() == [2, 2]
    assert distribution.median.tolist() == [2, 2]
    assert torch.allclose(distribution.expected, torch.full((2,), 2.0), atol=1e-4)
    assert torch.allclose(distribution.probabilities, alternate.probabilities)


def test_zero_initialized_planner_preserves_v38_answer_baseline() -> None:
    model = VisualAnswerTrajectoryModel(tiny_config()).eval()
    pixels, mask = visual_batch()

    output = model(pixels, mask)

    assert torch.allclose(output.answer_state, output.baseline_answer_state, atol=1e-6)
    assert torch.allclose(
        output.segment_states,
        output.baseline_answer_state[:, None].expand_as(output.segment_states),
        atol=1e-6,
    )


def test_planner_corrections_can_create_distinct_ordered_states() -> None:
    model = VisualAnswerTrajectoryModel(tiny_config()).eval()
    final = model.final_correction[-1]
    assert isinstance(final, torch.nn.Linear)
    torch.nn.init.normal_(final.weight, std=0.02)
    pixels, mask = visual_batch()

    output = model(pixels, mask)

    adjacent = F.cosine_similarity(
        output.segment_states[:, :-1],
        output.segment_states[:, 1:],
        dim=-1,
    )
    assert bool((adjacent < 0.99999).any())


def test_blank_visual_mask_stays_finite() -> None:
    model = VisualAnswerTrajectoryModel(tiny_config()).eval()
    pixels, mask = visual_batch()
    mask.zero_()

    output = model(pixels, mask)

    assert all(
        bool(torch.isfinite(value).all())
        for value in (
            output.read_state,
            output.answer_state,
            output.segment_states,
            output.stop_logits,
            output.active_probabilities,
            output.lengths,
        )
    )


def test_v39_boundary_is_independent_and_below_cap() -> None:
    receipt = visual_answer_trajectory_boundary_receipt(
        VisualAnswerTrajectoryModel(VisualAnswerTrajectoryConfig())
    )

    assert 90_000_000 < receipt["total_parameters"] < 120_000_000
    assert receipt["forbidden_parameter_names"] == []
    assert receipt["parameter_cap_pass"]
    assert receipt["deployable_inputs"] == ["prompt_pixels", "prompt_mask"]
    assert not receipt["uses_strings"]
    assert not receipt["uses_token_ids"]
    assert not receipt["uses_unicode_ids"]
    assert not receipt["uses_ocr"]
    assert not receipt["uses_bge_at_runtime"]
    assert not receipt["uses_candidate_bank_at_runtime"]
    assert not receipt["generates_raster"]

from __future__ import annotations

import torch
import torch.nn.functional as F

from ilm.visual_lm.visual_answer_trajectory import (
    VisualAnswerTrajectoryConfig,
    VisualAnswerTrajectoryModel,
)
from ilm.visual_lm.visual_answer_trajectory_training import (
    VisualAnswerTrajectoryEMA,
    VisualAnswerTrajectoryTargetBank,
    set_v39_stage_trainability,
    variance_covariance_loss,
    visual_answer_trajectory_loss,
    visual_answer_trajectory_optimizer_groups,
)
from ilm.visual_lm.visual_semantic_distillation_data import (
    V37_PATCHES,
    V37_PATCH_SIZE,
    V37_WIDTH,
)


DIMENSION = 64


def tiny_config() -> VisualAnswerTrajectoryConfig:
    return VisualAnswerTrajectoryConfig(
        reader_hidden_size=64,
        reader_layers=2,
        reader_heads=4,
        reader_intermediate_size=128,
        reader_dropout=0.0,
        projection_hidden_size=96,
        semantic_dim=DIMENSION,
        projection_dropout=0.0,
        answer_hidden_size=32,
        planner_hidden_size=64,
        planner_layers=2,
        planner_heads=4,
        planner_intermediate_size=128,
        planner_dropout=0.0,
        length_hidden_size=16,
    )


def target_bank(records: int = 10) -> VisualAnswerTrajectoryTargetBank:
    generator = torch.Generator().manual_seed(39)
    prompt = F.normalize(torch.randn(records, DIMENSION, generator=generator), dim=-1)
    answer = F.normalize(torch.randn(records, DIMENSION, generator=generator), dim=-1)
    counts = torch.tensor([(index % 4) + 1 for index in range(records)])
    offsets = torch.cat((torch.zeros(1, dtype=torch.long), counts.cumsum(0)))
    segments = F.normalize(
        torch.randn(int(offsets[-1]), DIMENSION, generator=generator),
        dim=-1,
    )
    return VisualAnswerTrajectoryTargetBank(
        identifiers=tuple(f"record-{index}" for index in range(records)),
        prompt_targets=prompt.half(),
        answer_targets=answer.half(),
        segment_targets=segments.half(),
        segment_offsets=offsets,
        segment_lengths=torch.linspace(1, V37_PATCHES, len(segments)).half(),
        teacher_mean=torch.zeros(DIMENSION).half(),
        receipt={"split": "train"},
    )


def visual_batch(batch: int = 2) -> tuple[torch.Tensor, torch.Tensor]:
    pixels = torch.rand(batch, 3, V37_PATCH_SIZE, V37_WIDTH)
    mask = torch.zeros(batch, V37_PATCHES)
    mask[:, :8] = 1
    return pixels, mask


def test_target_lookup_and_candidate_sampling_are_deterministic() -> None:
    bank = target_bank()
    targets = bank.lookup(
        ("record-2", "record-5"),
        torch.tensor((1, 0)),
        device="cpu",
    )
    first_global = bank.global_candidate_set(
        targets.bank_indices,
        count=8,
        seed=123,
        device="cpu",
    )
    second_global = bank.global_candidate_set(
        targets.bank_indices,
        count=8,
        seed=123,
        device="cpu",
    )
    first_segments = bank.segment_candidate_set(
        targets,
        count=12,
        seed=456,
        device="cpu",
    )
    second_segments = bank.segment_candidate_set(
        targets,
        count=12,
        seed=456,
        device="cpu",
    )

    assert targets.segment_mask.sum(dim=1).tolist() == [3, 2]
    assert torch.equal(first_global.bank_indices, second_global.bank_indices)
    assert torch.equal(first_global.positive_labels, second_global.positive_labels)
    assert torch.equal(first_segments.bank_indices, second_segments.bank_indices)
    assert torch.equal(first_segments.positive_labels, second_segments.positive_labels)
    assert torch.equal(first_segments.sampled_labels, second_segments.sampled_labels)


def test_target_bank_round_trip_preserves_flat_geometry() -> None:
    bank = target_bank()

    restored = VisualAnswerTrajectoryTargetBank.from_state_dict(bank.state_dict())

    assert restored.identifiers == bank.identifiers
    assert torch.equal(restored.segment_offsets, bank.segment_offsets)
    assert torch.equal(restored.segment_targets, bank.segment_targets)


def test_complete_v39_loss_is_finite_and_backpropagates() -> None:
    model = VisualAnswerTrajectoryModel(tiny_config())
    bank = target_bank()
    targets = bank.lookup(
        ("record-2", "record-5"),
        torch.tensor((1, 0)),
        device="cpu",
    )
    global_candidates = bank.global_candidate_set(
        targets.bank_indices,
        count=8,
        seed=123,
        device="cpu",
    )
    segment_candidates = bank.segment_candidate_set(
        targets,
        count=12,
        seed=456,
        device="cpu",
    )
    pixels, mask = visual_batch()
    prompt_anchor = model(pixels, mask)
    prompt_view = model(pixels + 0.001, mask)
    segment_anchor = model.encode_visual(pixels + 0.002, mask)
    segment_view = model.encode_visual(pixels + 0.003, mask)

    losses = visual_answer_trajectory_loss(
        prompt_anchor,
        prompt_view,
        segment_anchor,
        segment_view,
        targets,
        global_candidates,
        segment_candidates,
    )
    losses.loss.backward()

    assert torch.isfinite(losses.loss)
    assert losses.loss > 0
    assert model.prompt_head[-1].weight.grad is not None
    assert model.answer_transform.weight.grad is not None
    assert model.position_queries.grad is not None
    assert model.final_correction[-1].weight.grad is not None
    assert model.stop_head[-1].weight.grad is not None
    assert model.length_head[-1].weight.grad is not None
    assert model.reader.embeddings.patch_embeddings.projection.weight.grad is not None
    assert all(
        torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
        if parameter.grad is not None
    )


def test_variance_control_detects_collapse() -> None:
    collapsed = F.normalize(torch.ones(24, DIMENSION), dim=-1)
    diverse = F.normalize(
        torch.randn(24, DIMENSION, generator=torch.Generator().manual_seed(4)),
        dim=-1,
    )

    collapsed_variance, _ = variance_covariance_loss(collapsed)
    diverse_variance, _ = variance_covariance_loss(diverse)

    assert collapsed_variance > diverse_variance


def test_stage_trainability_and_optimizer_partition_are_explicit() -> None:
    model = VisualAnswerTrajectoryModel(tiny_config())

    set_v39_stage_trainability(model, "trajectory-head")
    assert not any(parameter.requires_grad for parameter in model.reader.parameters())
    assert not any(parameter.requires_grad for parameter in model.prompt_head.parameters())
    groups = visual_answer_trajectory_optimizer_groups(
        model,
        head_learning_rate=1e-4,
        reader_learning_rate=0.0,
        weight_decay=0.01,
    )
    covered = [id(parameter) for group in groups for parameter in group["params"]]
    expected = [id(parameter) for parameter in model.parameters() if parameter.requires_grad]
    assert len(covered) == len(set(covered))
    assert set(covered) == set(expected)

    set_v39_stage_trainability(model, "full-path-adaptation")
    assert all(parameter.requires_grad for parameter in model.reader.parameters())
    assert all(parameter.requires_grad for parameter in model.prompt_head.parameters())


def test_all_parameter_ema_updates_serializes_and_copies() -> None:
    model = VisualAnswerTrajectoryModel(tiny_config())
    names = tuple(name for name, _parameter in model.named_parameters())
    ema = VisualAnswerTrajectoryEMA(model, names, decay=0.5)
    first_name, first_parameter = next(iter(model.named_parameters()))
    original = ema.shadow[first_name].clone()

    with torch.no_grad():
        first_parameter.add_(2.0)
    ema.update(model)

    assert ema.shadow[first_name].device == first_parameter.device
    assert torch.allclose(ema.shadow[first_name], original + 1.0)
    state = ema.state_dict()
    assert state["shadow"][first_name].device.type == "cpu"

    destination = VisualAnswerTrajectoryModel(tiny_config())
    ema.copy_to(destination)
    assert torch.equal(dict(destination.named_parameters())[first_name], ema.shadow[first_name])

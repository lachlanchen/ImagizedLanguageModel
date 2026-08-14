from __future__ import annotations

import torch
import torch.nn.functional as F

from ilm.visual_lm.visual_path_alignment import (
    VisualPathAlignmentConfig,
    VisualPathAlignmentModel,
)
from ilm.visual_lm.visual_path_alignment_training import (
    VisualPathAlignmentTargetBank,
    orthogonal_prompt_answer_rotation,
    set_v38_stage_trainability,
    variance_covariance_loss,
    visual_path_alignment_loss,
)
from ilm.visual_lm.visual_semantic_distillation_data import (
    V37_PATCHES,
    V37_PATCH_SIZE,
    V37_WIDTH,
)


DIMENSION = 64


def tiny_config() -> VisualPathAlignmentConfig:
    return VisualPathAlignmentConfig(
        reader_hidden_size=64,
        reader_layers=2,
        reader_heads=4,
        reader_intermediate_size=128,
        reader_dropout=0.0,
        projection_hidden_size=96,
        semantic_dim=DIMENSION,
        projection_dropout=0.0,
        answer_hidden_size=32,
        length_hidden_size=16,
    )


def target_bank(records: int = 40) -> VisualPathAlignmentTargetBank:
    generator = torch.Generator().manual_seed(38)
    prompt = F.normalize(torch.randn(records, DIMENSION, generator=generator), dim=-1)
    rotation, _ = torch.linalg.qr(
        torch.randn(DIMENSION, DIMENSION, generator=generator)
    )
    answer = F.normalize(prompt @ rotation, dim=-1)
    return VisualPathAlignmentTargetBank(
        identifiers=tuple(f"record-{index}" for index in range(records)),
        prompt_targets=prompt.half(),
        answer_targets=answer.half(),
        lengths=torch.linspace(1, 32, records).half(),
        teacher_mean=torch.zeros(DIMENSION).half(),
        receipt={"split": "train"},
    )


def visual_batch(batch: int = 2) -> tuple[torch.Tensor, torch.Tensor]:
    pixels = torch.rand(batch, 3, V37_PATCH_SIZE, V37_WIDTH)
    mask = torch.zeros(batch, V37_PATCHES)
    mask[:, :8] = 1
    return pixels, mask


def test_candidate_set_injects_every_nearest_negative_deterministically() -> None:
    bank = target_bank()
    targets = bank.lookup(("record-2", "record-9"), device="cpu")
    first = bank.candidate_set(
        targets.bank_indices,
        count=20,
        seed=123,
        neighbors=3,
        teacher_ceiling=0.99,
        device="cpu",
    )
    second = bank.candidate_set(
        targets.bank_indices,
        count=20,
        seed=123,
        neighbors=3,
        teacher_ceiling=0.99,
        device="cpu",
    )

    assert torch.equal(first.bank_indices, second.bank_indices)
    assert torch.equal(first.nearest_labels, second.nearest_labels)
    assert first.nearest_labels.shape == (2, 3)
    assert len(first.bank_indices.unique()) == 20
    nearest_bank_indices = first.bank_indices[first.nearest_labels]
    expected = bank.nearest_answer_indices(neighbors=3, teacher_ceiling=0.99)[
        targets.bank_indices
    ]
    assert torch.equal(nearest_bank_indices, expected)


def test_train_only_rotation_recovers_an_orthogonal_relation() -> None:
    bank = target_bank()

    rotation, receipt = orthogonal_prompt_answer_rotation(bank)

    assert rotation.shape == (DIMENSION, DIMENSION)
    assert receipt["orthogonality_max_error"] < 1e-4
    assert receipt["train_top1"] == 1.0
    assert receipt["train_cosine"] > 0.999


def test_complete_v38_loss_is_finite_and_backpropagates() -> None:
    model = VisualPathAlignmentModel(tiny_config())
    bank = target_bank()
    rotation, _ = orthogonal_prompt_answer_rotation(bank)
    model.initialize_answer_rotation(rotation)
    targets = bank.lookup(("record-2", "record-9"), device="cpu")
    candidates = bank.candidate_set(
        targets.bank_indices,
        count=20,
        seed=123,
        neighbors=3,
        teacher_ceiling=0.99,
        device="cpu",
    )
    pixels, mask = visual_batch()
    outputs = [model(pixels + index * 0.001, mask) for index in range(5)]

    losses = visual_path_alignment_loss(*outputs, targets, candidates)
    losses.loss.backward()

    assert torch.isfinite(losses.loss)
    assert losses.loss > 0
    assert model.prompt_head[-1].weight.grad is not None
    assert model.answer_transform.weight.grad is not None
    assert model.answer_adapter[-1].weight.grad is not None
    assert model.reader.embeddings.patch_embeddings.projection.weight.grad is not None
    assert all(
        torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
        if parameter.grad is not None
    )


def test_variance_control_detects_collapse() -> None:
    collapsed = F.normalize(torch.ones(24, 64), dim=-1)
    diverse = F.normalize(
        torch.randn(24, 64, generator=torch.Generator().manual_seed(4)), dim=-1
    )

    collapsed_variance, _ = variance_covariance_loss(collapsed)
    diverse_variance, _ = variance_covariance_loss(diverse)

    assert collapsed_variance > diverse_variance


def test_stage_trainability_unfreezes_complete_reader() -> None:
    model = VisualPathAlignmentModel(tiny_config())

    set_v38_stage_trainability(model, "head-realignment")
    assert not any(parameter.requires_grad for parameter in model.reader.parameters())
    set_v38_stage_trainability(model, "full-path-adaptation")
    assert all(parameter.requires_grad for parameter in model.reader.parameters())


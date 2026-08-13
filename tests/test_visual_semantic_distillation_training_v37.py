from __future__ import annotations

import torch
import torch.nn.functional as F

from ilm.visual_lm.visual_semantic_distillation import (
    VisualSemanticDistillationConfig,
    VisualSemanticDistillationModel,
)
from ilm.visual_lm.visual_semantic_distillation_data import (
    V37_PATCHES,
    V37_PATCH_SIZE,
    V37_SEMANTIC_DIM,
    V37_WIDTH,
)
from ilm.visual_lm.visual_semantic_distillation_training import (
    VisualSemanticDistillationTargetBank,
    centered_effective_rank,
    set_v37_stage_trainability,
    vicreg_variance_covariance,
    visual_semantic_distillation_loss,
)


def tiny_config() -> VisualSemanticDistillationConfig:
    return VisualSemanticDistillationConfig(
        reader_hidden_size=64,
        reader_layers=2,
        reader_heads=4,
        reader_intermediate_size=128,
        reader_dropout=0.0,
        projection_hidden_size=96,
        semantic_dim=V37_SEMANTIC_DIM,
        projection_dropout=0.0,
        plan_hidden_size=32,
        length_hidden_size=16,
    )


def target_bank(records: int = 520) -> VisualSemanticDistillationTargetBank:
    generator = torch.Generator().manual_seed(37)
    prompt = F.normalize(
        torch.randn(records, V37_SEMANTIC_DIM, generator=generator), dim=-1
    )
    answer = F.normalize(
        prompt + 0.2 * torch.randn(records, V37_SEMANTIC_DIM, generator=generator),
        dim=-1,
    )
    return VisualSemanticDistillationTargetBank(
        identifiers=tuple(f"record-{index}" for index in range(records)),
        prompt_targets=prompt.half(),
        answer_targets=answer.half(),
        lengths=torch.linspace(1, 32, records).half(),
        teacher_mean=torch.zeros(V37_SEMANTIC_DIM).half(),
        receipt={"split": "train"},
    )


def visual_batch(batch: int = 2) -> tuple[torch.Tensor, torch.Tensor]:
    pixels = torch.rand(batch, 3, V37_PATCH_SIZE, V37_WIDTH)
    mask = torch.zeros(batch, V37_PATCHES)
    mask[:, :8] = 1
    return pixels, mask


def test_target_bank_lookup_and_candidates_are_deterministic() -> None:
    bank = target_bank()
    targets = bank.lookup(("record-2", "record-9"), device="cpu")
    first = bank.candidate_set(
        targets.bank_indices,
        count=512,
        seed=123,
        device="cpu",
    )
    second = bank.candidate_set(
        targets.bank_indices,
        count=512,
        seed=123,
        device="cpu",
    )

    assert torch.equal(first.bank_indices, second.bank_indices)
    assert torch.equal(first.positive_labels, torch.tensor([0, 1]))
    assert len(first.bank_indices.unique()) == 512
    assert torch.allclose(first.prompt[first.positive_labels], targets.prompt)
    assert torch.allclose(first.answer[first.positive_labels], targets.answer)


def test_complete_loss_is_finite_and_backpropagates() -> None:
    model = VisualSemanticDistillationModel(tiny_config())
    bank = target_bank()
    targets = bank.lookup(("record-2", "record-9"), device="cpu")
    candidates = bank.candidate_set(
        targets.bank_indices,
        count=512,
        seed=123,
        device="cpu",
    )
    pixels, mask = visual_batch()
    outputs = [model(pixels + index * 0.001, mask) for index in range(4)]

    losses = visual_semantic_distillation_loss(*outputs, targets, candidates)
    losses.loss.backward()

    assert torch.isfinite(losses.loss)
    assert losses.loss > 0
    assert all(
        torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
        if parameter.grad is not None
    )
    assert model.semantic_head[-1].weight.grad is not None
    assert model.reader.embeddings.patch_embeddings.projection.weight.grad is not None


def test_vicreg_penalizes_collapsed_states() -> None:
    collapsed = F.normalize(torch.ones(24, 64), dim=-1)
    diverse = F.normalize(
        torch.randn(24, 64, generator=torch.Generator().manual_seed(4)), dim=-1
    )

    collapsed_variance, _ = vicreg_variance_covariance(collapsed)
    diverse_variance, _ = vicreg_variance_covariance(diverse)

    assert collapsed_variance > diverse_variance


def test_stage_trainability_unfreezes_complete_reader() -> None:
    model = VisualSemanticDistillationModel(tiny_config())

    set_v37_stage_trainability(model, "projection-warmup")
    assert not any(parameter.requires_grad for parameter in model.reader.parameters())
    set_v37_stage_trainability(model, "full-visual-adaptation")
    assert all(parameter.requires_grad for parameter in model.reader.parameters())


def test_effective_rank_detects_collapse() -> None:
    generator = torch.Generator().manual_seed(9)
    diverse = torch.randn(64, 48, generator=generator)
    collapsed = torch.ones(64, 48) + 0.001 * torch.randn(64, 48, generator=generator)

    diverse_rank = centered_effective_rank(diverse)
    collapsed_rank = centered_effective_rank(collapsed)
    assert diverse_rank > 20
    assert collapsed_rank < 2
    assert diverse_rank > 10 * collapsed_rank
    exact = torch.ones(64, 48)
    assert centered_effective_rank(exact) < 1e-6

from __future__ import annotations

import torch

from ilm.visual_lm.visual_semantic_plan import (
    VisualSemanticPlanConfig,
    VisualSemanticPlanModel,
    VisualSentenceImageTeacher,
)
from ilm.visual_lm.visual_semantic_plan_data import (
    V36_CHUNKS,
    V36_PATCHES,
    V36_WIDTH,
)
from ilm.visual_lm.visual_semantic_plan_training import (
    SelectiveExponentialMovingAverage,
    VisualSemanticPlanTargetBank,
    encode_visual_semantic_teacher_targets,
    set_v36_stage_trainability,
    v36_optimizer_receipt,
    visual_semantic_plan_loss,
    visual_semantic_plan_optimizer_groups,
)


def _config() -> VisualSemanticPlanConfig:
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


def _batch(batch_size: int = 4) -> dict[str, torch.Tensor]:
    prompt = torch.rand(batch_size, 3, 16, V36_WIDTH)
    prompt_view = (prompt + torch.randn_like(prompt) * 0.01).clamp(0, 1)
    answer = torch.rand(batch_size, 3, 16, V36_WIDTH)
    mask = torch.zeros(batch_size, V36_PATCHES)
    mask[:, :8] = 1.0
    chunk_pixels = torch.rand(batch_size, V36_CHUNKS, 3, 16, V36_WIDTH)
    chunk_mask = torch.zeros(batch_size, V36_CHUNKS, V36_PATCHES)
    chunk_mask[:, :2, :8] = 1.0
    return {
        "prompt_pixels": prompt,
        "prompt_mask": mask,
        "prompt_view_pixels": prompt_view,
        "prompt_view_mask": mask.clone(),
        "answer_pixels": answer,
        "answer_mask": mask.clone(),
        "answer_chunk_pixels": chunk_pixels,
        "answer_chunk_mask": chunk_mask,
        "answer_length": torch.full((batch_size,), 8.0),
    }


def test_v36_teacher_targets_and_loss_backpropagate() -> None:
    torch.manual_seed(5)
    model = VisualSemanticPlanModel(_config())
    teacher = VisualSentenceImageTeacher(_config()).requires_grad_(False).eval()
    batch = _batch()
    targets = encode_visual_semantic_teacher_targets(teacher, batch)
    primary = model(batch["prompt_pixels"], batch["prompt_mask"])
    alternate = model(batch["prompt_view_pixels"], batch["prompt_view_mask"])
    loss = visual_semantic_plan_loss(primary, alternate, targets)
    loss.loss.backward()
    assert torch.isfinite(loss.loss)
    assert loss.active_chunks.item() == 8
    assert model.plan_queries.grad is not None
    assert torch.isfinite(model.plan_queries.grad).all()


def test_v36_stage_trainability_and_optimizer_receipt() -> None:
    model = VisualSemanticPlanModel(_config())
    set_v36_stage_trainability(model, "plan-alignment")
    assert not any(parameter.requires_grad for parameter in model.reader.parameters())
    assert model.plan_queries.requires_grad
    set_v36_stage_trainability(model, "semantic-adaptation")
    assert any(parameter.requires_grad for parameter in model.reader.encoder.layer[-2:].parameters())
    assert not any(parameter.requires_grad for parameter in model.reader.encoder.layer[:-2].parameters())
    groups = visual_semantic_plan_optimizer_groups(
        model,
        head_learning_rate=1e-4,
        reader_learning_rate=1e-5,
    )
    receipt = v36_optimizer_receipt(model, groups)
    assert {row["role"] for row in receipt["groups"]} == {"head", "reader"}
    assert not any("teacher" in name for name in receipt["optimized_parameter_names"])


def test_v36_selective_ema_and_target_bank_round_trip() -> None:
    model = VisualSemanticPlanModel(_config())
    names = ["plan_queries", "plan_scale", "plan_bias"]
    ema = SelectiveExponentialMovingAverage(model, names, decay=0.9)
    with torch.no_grad():
        model.plan_queries.add_(1.0)
    ema.update(model)
    state = ema.state_dict()
    restored = SelectiveExponentialMovingAverage(model, names, decay=0.9)
    restored.load_state_dict(state)
    assert torch.equal(restored.shadow["plan_queries"], state["shadow"]["plan_queries"])

    identifiers = ("a", "b")
    bank = VisualSemanticPlanTargetBank(
        identifiers=identifiers,
        global_plans=torch.randn(2, 64).half(),
        chunk_plans=torch.randn(2, V36_CHUNKS, 64).half(),
        chunk_active=torch.ones(2, V36_CHUNKS),
        lengths=torch.tensor([3.0, 7.0]),
        receipt={"test": True},
    )
    loaded = VisualSemanticPlanTargetBank.from_state_dict(bank.state_dict())
    targets = loaded.lookup(["b", "a"], device="cpu")
    assert torch.equal(targets.length, torch.tensor([7.0, 3.0]))
    assert targets.global_plan.dtype == torch.float32

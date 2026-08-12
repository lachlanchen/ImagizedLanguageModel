from __future__ import annotations

import torch
import torch.nn.functional as F

from ilm.visual_lm.saccade_lm import FovealRetina, VisualSaccadeConfig
from ilm.visual_lm.visual_motor_plan import (
    ContinuousVisualMotorPlan,
    VisualMotorPlanConfig,
    evaluate_visual_motor_plan_batch,
    visual_motor_plan_config_from_payload,
    visual_motor_plan_config_payload,
    visual_motor_plan_loss,
)


def _tiny_planner() -> ContinuousVisualMotorPlan:
    torch.manual_seed(13)
    return ContinuousVisualMotorPlan(
        VisualMotorPlanConfig(
            fovea_size=16,
            visual_dim=64,
            style_dim=16,
            style_base_channels=8,
            plan_base_channels=32,
            context_dim=64,
            dropout=0.0,
        )
    )


def _tiny_retina() -> FovealRetina:
    return FovealRetina(
        VisualSaccadeConfig(
            fovea_size=16,
            visual_dim=64,
            state_dim=128,
            state_layers=1,
            retina_base_channels=16,
            dropout=0.0,
            visual_hypotheses=1,
        )
    ).eval().requires_grad_(False)


def test_visual_motor_plan_has_continuous_image_boundary() -> None:
    planner = _tiny_planner().eval()
    intended = F.normalize(torch.randn(3, 64), dim=-1)
    style = torch.rand(3, 1, 16, 16)

    logits = planner(intended, style)
    ink = planner.plan(intended, style)

    assert logits.shape == style.shape
    assert ink.shape == style.shape
    assert float(ink.min().detach()) >= 0.0
    assert float(ink.max().detach()) <= 1.0
    assert not hasattr(planner, "embedding")
    assert not hasattr(planner, "classifier")
    assert not hasattr(planner, "vocabulary")


def test_visual_motor_plan_loss_backpropagates_without_training_retina() -> None:
    planner = _tiny_planner().train()
    retina = _tiny_retina()
    target = torch.rand(5, 1, 16, 16)
    semantic = (0.8 * target + 0.1).clamp(0, 1)
    style = torch.rand_like(target)

    loss, metrics, trace = visual_motor_plan_loss(
        planner,
        retina,
        target,
        semantic,
        style,
        duplicate_similarity=0.99,
    )
    loss.backward()

    assert torch.isfinite(loss)
    assert all(torch.isfinite(value) for value in metrics.values())
    assert trace["correct_ink"].requires_grad
    assert planner.style_encoder.field[0].weight.grad is not None
    assert planner.seed.weight.grad is not None
    assert planner.output[-1].weight.grad is not None
    assert all(parameter.grad is None for parameter in retina.parameters())


def test_intended_state_changes_motor_plan_with_fixed_style() -> None:
    planner = _tiny_planner().eval()
    intended = F.normalize(torch.randn(4, 64), dim=-1)
    style = torch.rand(4, 1, 16, 16)

    correct = planner.plan(intended, style)
    shuffled = planner.plan(intended.roll(1, dims=0), style)

    assert not torch.allclose(correct, shuffled)


def test_visual_motor_plan_evaluator_reports_causal_control() -> None:
    planner = _tiny_planner().eval()
    retina = _tiny_retina()
    target = torch.rand(4, 1, 16, 16)
    semantic = (0.8 * target + 0.1).clamp(0, 1)
    style = torch.rand_like(target)

    metrics, trace = evaluate_visual_motor_plan_batch(
        planner,
        retina,
        target,
        semantic,
        style,
        duplicate_similarity=0.99,
    )

    assert all(torch.isfinite(value) for value in metrics.values())
    assert float(metrics["condition_pixel_l1"]) > 0.0
    assert trace["correct_ink"].shape == target.shape
    assert trace["shuffled_ink"].shape == target.shape


def test_visual_motor_plan_config_round_trip() -> None:
    config = _tiny_planner().config
    restored = visual_motor_plan_config_from_payload(
        visual_motor_plan_config_payload(config)
    )
    assert restored == config

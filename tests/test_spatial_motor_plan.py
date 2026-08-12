from __future__ import annotations

import torch
import torch.nn.functional as F

from ilm.visual_lm.saccade_lm import FovealRetina, VisualSaccadeConfig
from ilm.visual_lm.spatial_motor_plan import (
    SpatialMotorPlanConfig,
    SpatialRetinalMotorPlan,
    evaluate_spatial_motor_plan_batch,
    spatial_motor_plan_config_from_payload,
    spatial_motor_plan_config_payload,
    spatial_motor_plan_loss,
    visual_complexity_masks,
    visual_complexity_score,
)
from ilm.visual_lm.visual_motor_plan import ContinuousVisualMotorPlan


def _tiny_config() -> SpatialMotorPlanConfig:
    return SpatialMotorPlanConfig(
        fovea_size=16,
        visual_dim=64,
        style_dim=16,
        style_base_channels=8,
        plan_base_channels=32,
        context_dim=64,
        dropout=0.0,
        spatial_channels=48,
        spatial_hidden_channels=32,
        spatial_blocks=1,
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


def test_retina_exposes_global_state_and_continuous_spatial_field() -> None:
    retina = _tiny_retina()
    image = torch.rand(3, 1, 16, 16)

    visual, field = retina.forward_with_field(image)

    assert visual.shape == (3, 64)
    assert field.shape == (3, 48, 2, 2)
    assert torch.allclose(visual, retina(image))


def test_zero_initialized_spatial_plan_exactly_matches_global_plan() -> None:
    torch.manual_seed(17)
    config = _tiny_config()
    global_plan = ContinuousVisualMotorPlan(config.global_config()).eval()
    spatial = SpatialRetinalMotorPlan(config).eval()
    spatial.load_global_plan(global_plan.state_dict())
    intended = F.normalize(torch.randn(4, 64), dim=-1)
    field = torch.randn(4, 48, 2, 2)
    style = torch.rand(4, 1, 16, 16)

    expected = global_plan.plan(intended, style)
    actual = spatial.plan(intended, field, style)

    assert torch.equal(expected, actual)
    assert all(not parameter.requires_grad for parameter in spatial.global_plan.parameters())
    assert any(parameter.requires_grad for parameter in spatial.spatial_adapter.parameters())


def test_spatial_loss_trains_only_residual_and_gate() -> None:
    torch.manual_seed(19)
    planner = SpatialRetinalMotorPlan(_tiny_config()).train()
    retina = _tiny_retina()
    target = torch.rand(6, 1, 16, 16)
    semantic = (0.8 * target + 0.1).clamp(0, 1)
    style = torch.rand_like(target)

    loss, metrics, trace = spatial_motor_plan_loss(
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
    assert planner.spatial_adapter.output.weight.grad is not None
    assert planner.spatial_gate_logit.grad is not None
    assert all(parameter.grad is None for parameter in planner.global_plan.parameters())
    assert all(parameter.grad is None for parameter in retina.parameters())


def test_spatial_evaluator_reports_all_causal_branches_and_strata() -> None:
    planner = SpatialRetinalMotorPlan(_tiny_config()).eval()
    retina = _tiny_retina()
    target = torch.rand(8, 1, 16, 16)
    semantic = (0.8 * target + 0.1).clamp(0, 1)
    style = torch.rand_like(target)

    metrics, trace = evaluate_spatial_motor_plan_batch(
        planner,
        retina,
        target,
        semantic,
        style,
        duplicate_similarity=0.99,
    )

    for branch in (
        "correct",
        "spatial_shuffled",
        "global_shuffled",
        "both_shuffled",
        "zero_field",
    ):
        assert trace[f"{branch}_ink"].shape == target.shape
        assert f"{branch}_pixel_f1" in metrics
        assert f"{branch}_pixel_f1_dense" in metrics
    assert float(metrics["semantic_target_pixel_l1"]) > 0.0
    assert sum(int(metrics[f"{name}_examples"]) for name in ("simple", "medium", "dense")) == 8


def test_visual_complexity_is_image_only_and_orders_density() -> None:
    simple = torch.zeros(1, 1, 16, 16)
    simple[:, :, 7:9, 3:13] = 1.0
    dense = torch.zeros_like(simple)
    dense[:, :, ::2] = 1.0
    dense[:, :, :, ::2] = 1.0
    images = torch.cat((simple, dense))

    scores = visual_complexity_score(images)
    masks = visual_complexity_masks(images)

    assert scores.shape == (2,)
    assert scores[1] > scores[0]
    assert bool(masks["simple"][0])
    assert bool(masks["dense"][1])


def test_spatial_motor_plan_config_round_trip() -> None:
    config = _tiny_config()
    restored = spatial_motor_plan_config_from_payload(
        spatial_motor_plan_config_payload(config)
    )
    assert restored == config

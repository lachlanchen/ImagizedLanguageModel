from __future__ import annotations

import torch
import torch.nn.functional as F

from ilm.visual_lm.retinal_topology_router import (
    FIELD_ROUTE,
    GLOBAL_CONTROL_ROUTE,
    RetinalTopologyRouter,
    RetinalTopologyRouterConfig,
    block_downsample,
    block_upsample,
    evaluate_retinal_topology_router_batch,
    quadrant_field_mask,
    retinal_topology_router_config_from_payload,
    retinal_topology_router_config_payload,
    retinal_topology_router_loss,
    zero_block_mean,
)
from ilm.visual_lm.saccade_lm import FovealRetina, VisualSaccadeConfig


def _tiny_config(route_mode: str = FIELD_ROUTE) -> RetinalTopologyRouterConfig:
    return RetinalTopologyRouterConfig(
        fovea_size=16,
        field_size=2,
        visual_dim=96,
        spatial_channels=96,
        style_dim=16,
        style_base_channels=8,
        hidden_channels=32,
        pointwise_blocks=1,
        coarse_hidden_dim=64,
        dropout=0.0,
        route_mode=route_mode,
    )


def _tiny_retina() -> FovealRetina:
    return FovealRetina(
        VisualSaccadeConfig(
            fovea_size=16,
            visual_dim=96,
            state_dim=128,
            state_layers=1,
            retina_base_channels=32,
            dropout=0.0,
            visual_hypotheses=1,
        )
    ).eval().requires_grad_(False)


def _inputs(batch: int = 6) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    intended = F.normalize(torch.randn(batch, 96), dim=-1)
    field = torch.randn(batch, 96, 2, 2)
    style = torch.rand(batch, 1, 16, 16)
    return intended, field, style


def _parameter_count(module: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


def test_block_projection_has_exact_zero_block_mean() -> None:
    detail = torch.randn(5, 1, 16, 16)

    projected = zero_block_mean(detail, field_size=2)
    reconstructed = block_upsample(block_downsample(detail, 2), patch_size=8)

    assert torch.allclose(projected, detail - reconstructed)
    assert float(block_downsample(projected, 2).abs().max()) < 1e-6


def test_field_router_enforces_coarse_detail_decomposition() -> None:
    torch.manual_seed(20)
    planner = RetinalTopologyRouter(_tiny_config()).eval()
    intended, field, style = _inputs()

    logits, trace = planner.logits_with_trace(intended, field, style)

    assert logits.shape == (6, 1, 16, 16)
    assert trace["coarse_cell_logits"].shape == (6, 1, 2, 2)
    assert torch.equal(trace["combined_logits"], logits)
    assert torch.allclose(
        logits,
        trace["coarse_logits"] + trace["detail_logits"],
        atol=1e-6,
    )
    assert (
        float(block_downsample(trace["detail_logits"], 2).detach().abs().max())
        < 1e-6
    )
    coarse_residual = trace["coarse_logits"] - block_upsample(
        block_downsample(trace["coarse_logits"], 2),
        8,
    )
    assert float(coarse_residual.detach().abs().max()) < 1e-6


def test_field_router_has_no_global_to_detail_path() -> None:
    torch.manual_seed(21)
    planner = RetinalTopologyRouter(_tiny_config()).eval()
    intended, field, style = _inputs()
    changed_intended = F.normalize(torch.randn_like(intended), dim=-1)

    _, original = planner.logits_with_trace(intended, field, style)
    _, changed = planner.logits_with_trace(changed_intended, field, style)

    assert torch.equal(original["raw_detail_logits"], changed["raw_detail_logits"])
    assert torch.equal(original["detail_logits"], changed["detail_logits"])
    assert not torch.equal(original["coarse_logits"], changed["coarse_logits"])


def test_global_control_is_exact_capacity_and_ignores_field() -> None:
    torch.manual_seed(22)
    candidate = RetinalTopologyRouter(_tiny_config(FIELD_ROUTE)).eval()
    control = RetinalTopologyRouter(_tiny_config(GLOBAL_CONTROL_ROUTE)).eval()
    control.load_state_dict(candidate.state_dict(), strict=True)
    intended, field, style = _inputs()

    _, first = control.logits_with_trace(intended, field, style)
    _, second = control.logits_with_trace(intended, field.roll(1, dims=0), style)

    assert _parameter_count(candidate) == _parameter_count(control)
    assert torch.equal(first["detail_logits"], second["detail_logits"])
    assert torch.equal(first["combined_logits"], second["combined_logits"])


def test_quadrant_occlusion_is_topographically_local() -> None:
    torch.manual_seed(23)
    planner = RetinalTopologyRouter(_tiny_config()).eval()
    intended, field, style = _inputs(batch=4)
    correct = planner.plan(intended, field, style)

    for quadrant in range(4):
        occluded = planner.plan(
            intended,
            field,
            style,
            field_mask=quadrant_field_mask(field, quadrant, field_size=2),
        )
        delta = (correct - occluded).abs()
        row, column = divmod(quadrant, 2)
        outside = delta.clone()
        outside[
            :,
            :,
            row * 8 : (row + 1) * 8,
            column * 8 : (column + 1) * 8,
        ] = 0.0
        assert float(delta.detach().sum()) > 0.0
        assert float(outside.detach().max()) == 0.0


def test_router_loss_updates_writer_but_not_frozen_retina() -> None:
    torch.manual_seed(24)
    planner = RetinalTopologyRouter(_tiny_config()).train()
    retina = _tiny_retina()
    target = torch.rand(6, 1, 16, 16)
    semantic = (0.8 * target + 0.1).clamp(0, 1)
    style = torch.rand_like(target)

    loss, metrics, trace = retinal_topology_router_loss(
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
    assert planner.detail.output.weight.grad is not None
    assert planner.coarse[-1].weight.grad is not None
    assert all(parameter.grad is None for parameter in retina.parameters())


def test_control_loss_omits_inapplicable_field_margins() -> None:
    planner = RetinalTopologyRouter(_tiny_config(GLOBAL_CONTROL_ROUTE)).train()
    retina = _tiny_retina()
    target = torch.rand(4, 1, 16, 16)
    semantic = torch.rand_like(target)
    style = torch.rand_like(target)

    _, metrics, _ = retinal_topology_router_loss(
        planner,
        retina,
        target,
        semantic,
        style,
        duplicate_similarity=0.99,
    )

    assert float(metrics["field_margin_loss"]) == 0.0
    assert float(metrics["zero_margin_loss"]) == 0.0
    assert float(metrics["field_condition_pixel_l1"]) == 0.0
    assert float(metrics["zero_field_condition_pixel_l1"]) == 0.0


def test_evaluator_reports_interventions_strata_and_invariants() -> None:
    torch.manual_seed(25)
    planner = RetinalTopologyRouter(_tiny_config()).eval()
    retina = _tiny_retina()
    target = torch.rand(8, 1, 16, 16)
    semantic = (0.8 * target + 0.1).clamp(0, 1)
    style = torch.rand_like(target)

    metrics, trace = evaluate_retinal_topology_router_batch(
        planner,
        retina,
        target,
        semantic,
        style,
        duplicate_similarity=0.99,
    )

    for branch in (
        "correct",
        "field_shuffled",
        "global_shuffled",
        "both_shuffled",
        "zero_field",
    ):
        assert trace[f"{branch}_ink"].shape == target.shape
        assert f"{branch}_pixel_f1_dense" in metrics
    for quadrant in range(4):
        assert trace[f"occluded_q{quadrant}_ink"].shape == target.shape
    assert float(metrics["coarse_within_block_max"]) < 1e-6
    assert float(metrics["detail_block_mean_abs_max"]) < 1e-6
    assert float(metrics["decomposition_error_max"]) < 1e-6
    assert float(metrics["occlusion_locality"]) > 0.99
    assert sum(
        int(metrics[f"{name}_examples"])
        for name in ("simple", "medium", "dense")
    ) == 8


def test_router_config_round_trip() -> None:
    config = _tiny_config()
    restored = retinal_topology_router_config_from_payload(
        retinal_topology_router_config_payload(config)
    )
    assert restored == config

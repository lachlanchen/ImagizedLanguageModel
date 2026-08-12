from __future__ import annotations

import torch
import torch.nn.functional as F

from ilm.visual_lm.field_complete_writer import (
    FIELD_COMPLETE_ROUTE,
    TILED_GLOBAL_CONTROL_ROUTE,
    FieldCompleteWriter,
    FieldCompleteWriterConfig,
    evaluate_field_complete_writer_batch,
    field_complete_writer_config_from_payload,
    field_complete_writer_config_payload,
    field_complete_writer_loss,
    image_patch_cell_variation,
    sylvester_hadamard,
    zero_dc_hadamard_basis,
)
from ilm.visual_lm.retinal_topology_router import block_downsample
from ilm.visual_lm.saccade_lm import FovealRetina, VisualSaccadeConfig


def _tiny_config(
    route_mode: str = FIELD_COMPLETE_ROUTE,
) -> FieldCompleteWriterConfig:
    return FieldCompleteWriterConfig(
        fovea_size=16,
        field_size=2,
        visual_dim=96,
        spatial_channels=96,
        style_dim=16,
        style_base_channels=8,
        hidden_channels=32,
        context_dim=32,
        pointwise_blocks=1,
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


def test_hadamard_basis_is_exact_zero_dc_and_orthonormal() -> None:
    hadamard = sylvester_hadamard(64)
    basis = zero_dc_hadamard_basis(64)

    assert hadamard.shape == (64, 64)
    assert basis.shape == (64, 63)
    assert float(basis.sum(dim=0).abs().max()) == 0.0
    assert torch.equal(basis.abs(), torch.full_like(basis, 0.125))
    assert torch.allclose(basis.T @ basis, torch.eye(63), atol=1e-6)


def test_default_arms_have_preregistered_equal_parameter_count() -> None:
    candidate = FieldCompleteWriter(FieldCompleteWriterConfig())
    control = FieldCompleteWriter(
        FieldCompleteWriterConfig(route_mode=TILED_GLOBAL_CONTROL_ROUTE)
    )

    assert _parameter_count(candidate) == 582_336
    assert _parameter_count(control) == 582_336
    assert {
        name: tuple(value.shape) for name, value in candidate.state_dict().items()
    } == {name: tuple(value.shape) for name, value in control.state_dict().items()}
    assert not any("position" in name for name, _ in candidate.named_parameters())


def test_field_complete_writer_enforces_patch_decomposition() -> None:
    torch.manual_seed(21)
    writer = FieldCompleteWriter(_tiny_config()).eval()
    intended, field, style = _inputs()

    logits, trace = writer.logits_with_trace(intended, field, style)

    assert logits.shape == (6, 1, 16, 16)
    assert trace["coarse_cell_logits"].shape == (6, 1, 2, 2)
    assert trace["detail_coefficients"].shape == (6, 63, 2, 2)
    assert trace["global_context"].shape == (6, 32)
    assert torch.equal(trace["combined_logits"], logits)
    assert torch.allclose(
        logits,
        trace["coarse_logits"] + trace["detail_logits"],
        atol=1e-6,
    )
    assert (
        float(
            block_downsample(trace["detail_logits"].double(), 2)
            .detach()
            .abs()
            .max()
        )
        < 5e-6
    )


def test_zero_source_cannot_create_cell_specific_spatial_plan() -> None:
    torch.manual_seed(22)
    writer = FieldCompleteWriter(_tiny_config()).eval()
    intended, field, style = _inputs(batch=4)
    zero_mask = torch.zeros(4, 1, 2, 2)

    logits, _ = writer.logits_with_trace(
        intended,
        field,
        style,
        field_mask=zero_mask,
    )
    changed_global = F.normalize(torch.randn_like(intended), dim=-1)
    changed_logits, _ = writer.logits_with_trace(
        changed_global,
        field,
        style,
        field_mask=zero_mask,
    )

    assert float(image_patch_cell_variation(logits, 2).detach()) < 1e-6
    assert float(image_patch_cell_variation(changed_logits, 2).detach()) < 1e-6
    assert not torch.equal(logits, changed_logits)


def test_one_cell_and_quadrant_interventions_are_exactly_local() -> None:
    torch.manual_seed(23)
    writer = FieldCompleteWriter(_tiny_config()).eval()
    intended, field, style = _inputs(batch=4)
    correct = writer.plan(intended, field, style)

    one_cell = torch.ones(4, 1, 2, 2)
    one_cell[:, :, 0, 1] = 0.0
    changed = writer.plan(intended, field, style, field_mask=one_cell)
    delta = (correct - changed).abs()
    outside = delta.clone()
    outside[:, :, 0:8, 8:16] = 0.0
    assert float(delta.sum().detach()) > 0.0
    assert float(outside.max().detach()) == 0.0

    for quadrant in range(4):
        mask = torch.ones(4, 1, 2, 2)
        row, column = divmod(quadrant, 2)
        mask[:, :, row, column] = 0.0
        changed = writer.plan(intended, field, style, field_mask=mask)
        delta = (correct - changed).abs()
        outside = delta.clone()
        outside[
            :,
            :,
            row * 8 : (row + 1) * 8,
            column * 8 : (column + 1) * 8,
        ] = 0.0
        assert float(delta.sum().detach()) > 0.0
        assert float(outside.max().detach()) == 0.0


def test_tiled_control_is_equal_capacity_repeated_and_ignores_field() -> None:
    torch.manual_seed(24)
    candidate = FieldCompleteWriter(_tiny_config()).eval()
    control = FieldCompleteWriter(
        _tiny_config(TILED_GLOBAL_CONTROL_ROUTE)
    ).eval()
    control.load_state_dict(candidate.state_dict(), strict=True)
    intended, field, style = _inputs()

    first = control(intended, field, style)
    second = control(intended, field.roll(1, dims=0), style)
    zero_mask = torch.zeros(6, 1, 2, 2)
    masked = control(intended, field, style, field_mask=zero_mask)

    assert _parameter_count(candidate) == _parameter_count(control)
    assert torch.equal(first, second)
    assert torch.equal(first, masked)
    assert float(image_patch_cell_variation(first, 2).detach()) < 1e-6


def test_candidate_uses_field_and_global_context_without_position_map() -> None:
    torch.manual_seed(25)
    writer = FieldCompleteWriter(_tiny_config()).eval()
    intended, field, style = _inputs(batch=3)
    intended.requires_grad_(True)
    field.requires_grad_(True)

    output = writer(intended, field, style)
    output.square().mean().backward()

    assert field.grad is not None
    assert float(field.grad.abs().sum()) > 0.0
    assert intended.grad is not None
    assert float(intended.grad.abs().sum()) > 0.0
    assert not any("position" in name for name, _ in writer.named_parameters())


def test_writer_loss_updates_both_patch_heads_but_not_retina() -> None:
    torch.manual_seed(26)
    writer = FieldCompleteWriter(_tiny_config()).train()
    retina = _tiny_retina()
    target = torch.rand(6, 1, 16, 16)
    semantic = (0.8 * target + 0.1).clamp(0, 1)
    style = torch.rand_like(target)

    loss, metrics, trace = field_complete_writer_loss(
        writer,
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
    assert writer.coarse_output.weight.grad is not None
    assert writer.detail_output.weight.grad is not None
    assert writer.global_conditioner.network[1].weight.grad is not None
    assert all(parameter.grad is None for parameter in retina.parameters())


def test_control_loss_omits_field_margins_and_field_effects() -> None:
    writer = FieldCompleteWriter(
        _tiny_config(TILED_GLOBAL_CONTROL_ROUTE)
    ).train()
    retina = _tiny_retina()
    target = torch.rand(4, 1, 16, 16)
    semantic = torch.rand_like(target)
    style = torch.rand_like(target)

    _, metrics, _ = field_complete_writer_loss(
        writer,
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


def test_evaluator_reports_interventions_strata_and_structural_metrics() -> None:
    torch.manual_seed(27)
    writer = FieldCompleteWriter(_tiny_config()).eval()
    retina = _tiny_retina()
    target = torch.rand(8, 1, 16, 16)
    semantic = (0.8 * target + 0.1).clamp(0, 1)
    style = torch.rand_like(target)

    metrics, trace = evaluate_field_complete_writer_batch(
        writer,
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
    assert float(metrics["basis_dc_leakage_max"]) == 0.0
    assert float(metrics["basis_gram_error_max"]) < 1e-6
    assert float(metrics["detail_block_mean_abs_max"]) < 5e-6
    assert float(metrics["decomposition_error_max"]) < 1e-6
    assert float(metrics["zero_source_cell_variation_max"]) < 1e-6
    assert float(metrics["occlusion_locality"]) > 0.99
    assert sum(
        int(metrics[f"{name}_examples"])
        for name in ("simple", "medium", "dense")
    ) == 8


def test_writer_config_round_trip() -> None:
    config = _tiny_config()
    restored = field_complete_writer_config_from_payload(
        field_complete_writer_config_payload(config)
    )
    assert restored == config

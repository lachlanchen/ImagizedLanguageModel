from __future__ import annotations

import torch

from ilm.visual_lm.visual_semantic_raster_training import (
    V32_LOSS_WEIGHTS,
    diagonal_gaussian_state_nll,
    latent_variance_floor_loss,
    raster_edge_loss,
    raster_ink_dice_loss,
    raster_pixel_bce,
    raster_warmup_microstep,
    stop_position_loss,
    visual_semantic_raster_training_microstep,
    weighted_visual_semantic_raster_loss,
)
from ilm.visual_lm.visual_semantic_raster_transducer import (
    VisualSemanticRasterConfig,
    VisualSemanticRasterTransducer,
)


def _config() -> VisualSemanticRasterConfig:
    return VisualSemanticRasterConfig(
        maximum_prompt_patches=8,
        maximum_answer_cells=4,
        reader_hidden_size=32,
        reader_layers=1,
        reader_heads=4,
        reader_intermediate_size=64,
        reader_dropout=0.0,
        planner_dim=32,
        planner_layers=1,
        planner_heads=4,
        planner_mlp_dim=64,
        planner_dropout=0.0,
        cell_retina_channels=8,
        target_width=32,
        target_blocks=1,
        latent_dim=8,
        decoder_width=32,
        decoder_layers=1,
        decoder_heads=4,
        decoder_mlp_dim=64,
        decoder_dropout=0.0,
        feedback_noise_probability=0.0,
        feedback_ground_truth_probability=0.0,
    )


def _batch(config: VisualSemanticRasterConfig) -> dict[str, torch.Tensor]:
    answer_mask = torch.tensor([[1.0, 1.0, 0.0, 0.0]])
    stop_targets = torch.tensor([[0.0, 0.0, 1.0, 0.0, 0.0]])
    stop_mask = torch.tensor([[1.0, 1.0, 1.0, 0.0, 0.0]])
    return {
        "prompt_pixels": torch.rand(1, 3, 16, config.prompt_width),
        "prompt_mask": torch.ones(1, config.maximum_prompt_patches),
        "answer_cells": torch.rand(1, 4, 1, 24, 24),
        "answer_mask": answer_mask,
        "stop_targets": stop_targets,
        "stop_mask": stop_mask,
    }


def test_v32_atomic_losses_are_zero_or_near_zero_for_exact_predictions() -> None:
    target = torch.zeros(1, 2, 1, 24, 24)
    target[:, 0, :, 6:18, 9:15] = 1.0
    mask = torch.tensor([[1.0, 0.0]])
    logits = torch.where(target > 0.5, 20.0, -20.0)
    assert raster_pixel_bce(logits, target, mask) < 1e-6
    assert raster_edge_loss(logits, target, mask) < 1e-6
    assert raster_ink_dice_loss(logits, target, mask) < 1e-6
    stop_target = torch.tensor([[0.0, 1.0, 0.0]])
    stop_logits = torch.tensor([[-20.0, 20.0, 20.0]])
    stop_mask = torch.tensor([[1.0, 1.0, 0.0]])
    assert stop_position_loss(stop_logits, stop_target, stop_mask) < 1e-6


def test_v32_state_nll_detaches_target_encoder_states() -> None:
    target = torch.randn(2, 3, 5, requires_grad=True)
    mean = torch.zeros_like(target, requires_grad=True)
    log_scale = torch.zeros_like(target, requires_grad=True)
    loss = diagonal_gaussian_state_nll(
        mean,
        log_scale,
        target,
        torch.ones(2, 3),
    )
    loss.backward()
    assert target.grad is None
    assert mean.grad is not None
    assert log_scale.grad is not None


def test_v32_variance_floor_penalizes_collapse() -> None:
    mask = torch.ones(2, 3)
    collapsed = torch.zeros(2, 3, 4)
    varied = torch.tensor(
        [[[-1.0, -1.0, -1.0, -1.0]] * 3, [[1.0, 1.0, 1.0, 1.0]] * 3]
    )
    collapsed_loss, collapsed_std = latent_variance_floor_loss(collapsed, mask)
    varied_loss, varied_std = latent_variance_floor_loss(varied, mask)
    assert collapsed_loss > 0.03
    assert varied_loss == 0.0
    assert collapsed_std.max() < varied_std.min()


def test_v32_weighted_objective_uses_preregistered_coefficients() -> None:
    terms = {
        "state": torch.tensor(1.0),
        "pixel": torch.tensor(2.0),
        "edge": torch.tensor(3.0),
        "ink": torch.tensor(4.0),
        "stop": torch.tensor(5.0),
        "variance": torch.tensor(6.0),
    }
    expected = 1.0 + 2.0 + 0.25 * 3.0 + 0.25 * 4.0 + 0.2 * 5.0 + 0.05 * 6.0
    assert weighted_visual_semantic_raster_loss(terms) == expected
    assert V32_LOSS_WEIGHTS.state == 1.0


def test_v32_full_microstep_is_finite_and_updates_both_learning_routes() -> None:
    config = _config()
    model = VisualSemanticRasterTransducer(config).train()
    loss, metrics = visual_semantic_raster_training_microstep(model, _batch(config))
    loss.backward()
    assert torch.isfinite(loss)
    assert set(metrics) >= {
        "loss",
        "loss_state",
        "loss_pixel",
        "loss_edge",
        "loss_ink",
        "loss_stop",
        "loss_variance",
        "raster_mae",
        "stop_binary_accuracy",
    }
    assert any(parameter.grad is not None for parameter in model.planner.parameters())
    assert any(parameter.grad is not None for parameter in model.target_encoder.parameters())
    assert any(parameter.grad is not None for parameter in model.raster_decoder.parameters())


def test_v32_raster_warmup_does_not_touch_reader_or_planner() -> None:
    config = _config()
    model = VisualSemanticRasterTransducer(config).train()
    loss, metrics = raster_warmup_microstep(model, _batch(config))
    loss.backward()
    assert torch.isfinite(loss)
    assert "loss_pixel" in metrics
    assert all(parameter.grad is None for parameter in model.reader.parameters())
    assert all(parameter.grad is None for parameter in model.planner.parameters())
    assert any(parameter.grad is not None for parameter in model.target_encoder.parameters())
    assert any(parameter.grad is not None for parameter in model.raster_decoder.parameters())

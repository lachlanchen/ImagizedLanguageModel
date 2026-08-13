from __future__ import annotations

import torch

from ilm.visual_lm.dense_visual_future_energy import (
    DenseVisualFutureConfig,
    DenseVisualFutureModel,
)
from scripts.train_dense_visual_future_energy_v28 import (
    _set_trainable,
    build_optimizer,
    training_microstep,
)


def _small_model() -> DenseVisualFutureModel:
    model = DenseVisualFutureModel(
        DenseVisualFutureConfig(
            visual_dim=64,
            semantic_dim=64,
            model_dim=128,
            layers=2,
            heads=4,
            mlp_ratio=2.0,
            retina_base_channels=8,
            semantic_hidden_dim=128,
            hypotheses=3,
        )
    )
    _set_trainable(model)
    return model


def _natural_batch(batch: int = 2) -> dict[str, torch.Tensor]:
    return {
        "first_view": torch.rand(batch, 68, 1, 32, 32),
        "second_view": torch.rand(batch, 68, 1, 32, 32),
        "pixel_groups": torch.arange(batch * 68).reshape(batch, 68),
    }


def _pair_batch(batch: int = 2) -> dict[str, torch.Tensor]:
    context = torch.rand(batch, 2, 64, 1, 32, 32)
    context[:, 1, -4:] = context[:, 0, -4:]
    reference = torch.rand(batch, 2, 64, 1, 32, 32)
    reference[:, 1, -4:] = reference[:, 0, -4:]
    return {
        "contexts": context,
        "candidates": torch.rand(batch, 2, 1, 32, 32),
        "assignment": torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
        "reference_contexts": reference,
        "reference_candidates": torch.rand(batch, 2, 1, 32, 32),
        "reference_assignment": torch.tensor(
            [[1, 0], [0, 1]], dtype=torch.long
        ),
    }


def test_v28_microstep_trains_language_and_adapter_but_not_fixed_paths() -> None:
    torch.manual_seed(17)
    model = _small_model()
    loss, metrics = training_microstep(
        model,
        _natural_batch(),
        _pair_batch(),
        context_noise_maximum=0.0,
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(model.semantic_adapter.up.weight.grad).all()
    assert torch.isfinite(model.context_input.weight.grad).all()
    assert torch.isfinite(model.raw_queries.weight.grad).all()
    assert all(parameter.grad is None for parameter in model.retina.parameters())
    assert all(
        parameter.grad is None
        for parameter in model.target_semantic_adapter.parameters()
    )
    assert set(metrics) >= {
        "loss",
        "dense_loss",
        "energy_score",
        "identity_loss",
        "natural_order_loss",
        "natural_order_advantage",
        "pair_loss",
        "pair_order_loss",
        "pair_full_minus_suffix_margin",
        "pair_full_minus_shuffled_margin",
        "h1_top1",
        "h2_top1",
        "h4_top1",
    }


def test_v28_optimizer_excludes_frozen_retina_and_ema_adapter() -> None:
    model = _small_model()
    optimizer = build_optimizer(
        model,
        learning_rate=3e-4,
        weight_decay=0.05,
        device=torch.device("cpu"),
    )
    optimized = {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    }
    frozen = {
        id(parameter)
        for module in (model.retina, model.target_semantic_adapter)
        for parameter in module.parameters()
    }
    assert not optimized.intersection(frozen)
    assert id(model.context_input.weight) in optimized
    assert id(model.semantic_adapter.up.weight) in optimized


def test_v28_ema_updates_only_target_semantic_adapter() -> None:
    model = _small_model()
    online = next(model.semantic_adapter.parameters())
    target = next(model.target_semantic_adapter.parameters())
    target_before = target.detach().clone()
    with torch.no_grad():
        online.add_(2.0)
    model.update_target_adapter(0.5)
    assert torch.allclose(target, 0.5 * target_before + 0.5 * online)
    assert model.target_semantic_adapter.training is False
    assert model.retina.training is False


def test_fixed_v28_model_stays_inside_preregistered_parameter_caps() -> None:
    model = DenseVisualFutureModel(DenseVisualFutureConfig())
    _set_trainable(model)
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    assert total == 17_859_142
    assert trainable == 16_377_990
    assert total < 24_000_000
    assert trainable < 20_000_000

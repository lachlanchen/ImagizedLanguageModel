from __future__ import annotations

import torch

from ilm.visual_lm.joint_visual_compatibility import (
    JointVisualCompatibilityConfig,
    JointVisualCompatibilityModel,
)
from scripts.train_joint_visual_compatibility_v27 import (
    _set_trainable,
    build_optimizer,
    training_microstep,
)


def _small_model() -> JointVisualCompatibilityModel:
    model = JointVisualCompatibilityModel(
        JointVisualCompatibilityConfig(
            visual_dim=64,
            model_dim=128,
            layers=2,
            heads=4,
            mlp_ratio=2.0,
            retina_base_channels=8,
            candidate_hidden_dim=128,
        )
    )
    _set_trainable(model)
    return model


def _natural_batch(batch: int = 4) -> dict[str, torch.Tensor]:
    canonical = torch.zeros(batch, 1, 1, 32, 32)
    for index in range(batch):
        canonical[index, 0, 0, index, index] = 1.0
    return {
        "context": torch.rand(batch, 8, 1, 32, 32),
        "target": torch.rand(batch, 1, 1, 32, 32),
        "reference_context": torch.rand(batch, 8, 1, 32, 32),
        "reference_target": torch.rand(batch, 1, 1, 32, 32),
        "canonical_target": canonical,
    }


def _pair_batch(batch: int = 2) -> dict[str, torch.Tensor]:
    return {
        "contexts": torch.rand(batch, 2, 8, 1, 32, 32),
        "candidates": torch.rand(batch, 2, 1, 32, 32),
        "assignment": torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
        "reference_contexts": torch.rand(batch, 2, 8, 1, 32, 32),
        "reference_candidates": torch.rand(batch, 2, 1, 32, 32),
        "reference_assignment": torch.tensor(
            [[1, 0], [0, 1]], dtype=torch.long
        ),
    }


def test_v27_microstep_trains_retina_context_and_candidate_without_ema_gradients() -> None:
    model = _small_model()
    loss, metrics = training_microstep(
        model,
        _natural_batch(),
        _pair_batch(),
        context_noise_maximum=0.0,
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(model.retina.stem[1].weight.grad).all()
    assert torch.isfinite(model.context_input.weight.grad).all()
    assert torch.isfinite(model.candidate_projector.up.weight.grad).all()
    assert all(parameter.grad is None for parameter in model.target_retina.parameters())
    assert all(
        parameter.grad is None
        for parameter in model.target_candidate_projector.parameters()
    )
    assert set(metrics) >= {
        "loss",
        "natural_loss",
        "natural_top1",
        "identity_loss",
        "identity_top1",
        "pair_loss",
        "pair_arm_accuracy",
        "vicreg_loss",
    }


def test_v27_optimizer_has_fixed_separate_retina_rate() -> None:
    model = _small_model()
    optimizer = build_optimizer(
        model,
        learning_rate=3e-4,
        retina_learning_rate=3e-5,
        weight_decay=0.05,
        device=torch.device("cpu"),
    )
    groups = {group["name"]: group for group in optimizer.param_groups}
    assert set(groups) == {"compatibility", "retina"}
    assert groups["compatibility"]["lr"] == 3e-4
    assert groups["retina"]["lr"] == 3e-5
    retina_ids = {id(parameter) for parameter in model.retina.parameters()}
    assert {id(parameter) for parameter in groups["retina"]["params"]} == retina_ids
    assert not retina_ids.intersection(
        id(parameter) for parameter in groups["compatibility"]["params"]
    )


def test_v27_ema_updates_only_target_visual_modules() -> None:
    model = _small_model()
    online = next(model.candidate_projector.parameters())
    target = next(model.target_candidate_projector.parameters())
    target_before = target.detach().clone()
    with torch.no_grad():
        online.add_(2.0)
    model.update_target(0.5)
    assert torch.allclose(target, 0.5 * target_before + 0.5 * online)
    assert model.target_candidate_projector.training is False
    assert model.target_retina.training is False


def test_fixed_v27_model_stays_below_parameter_cap() -> None:
    model = JointVisualCompatibilityModel(JointVisualCompatibilityConfig())
    _set_trainable(model)
    trainable = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    assert trainable == 17_118_401
    assert trainable < 20_000_000

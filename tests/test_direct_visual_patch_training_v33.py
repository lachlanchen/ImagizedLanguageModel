from __future__ import annotations

import torch

from ilm.visual_lm.direct_visual_patch_lm import (
    DirectVisualPatchConfig,
    DirectVisualPatchLM,
)
from ilm.visual_lm.direct_visual_patch_training import (
    direct_visual_patch_loss,
    direct_visual_patch_optimizer_groups,
    module_state_sha256,
    set_core_trainable,
    stage_cosine_learning_rate,
    strip_to_patches,
)


def tiny_model() -> DirectVisualPatchLM:
    return DirectVisualPatchLM(
        DirectVisualPatchConfig(
            patch_size=32,
            maximum_patches=8,
            hidden_size=64,
            layers=2,
            attention_heads=4,
            key_value_heads=4,
            intermediate_size=128,
        )
    )


def batch() -> dict[str, torch.Tensor]:
    pixels = torch.ones(2, 1, 32, 32 * 6)
    pixels[:, :, 7:25, 5:27] = 0.0
    patch_mask = torch.ones(2, 6)
    next_mask = torch.zeros(2, 6)
    next_mask[:, 2:5] = 1.0
    stop_targets = torch.zeros(2, 6)
    stop_targets[:, 5] = 1.0
    stop_mask = torch.zeros(2, 6)
    stop_mask[:, 2:6] = 1.0
    return {
        "pixels": pixels,
        "patch_mask": patch_mask,
        "next_patch_mask": next_mask,
        "reconstruction_mask": patch_mask,
        "stop_targets": stop_targets,
        "stop_mask": stop_mask,
    }


def test_strip_patchify_is_lossless() -> None:
    pixels = batch()["pixels"]
    patches = strip_to_patches(pixels, 32)
    restored = patches.permute(0, 2, 3, 1, 4).reshape_as(pixels)
    assert torch.equal(restored, pixels)


def test_calibration_and_causal_losses_are_finite() -> None:
    model = tiny_model()
    data = batch()
    output = model(data["pixels"], data["patch_mask"])
    calibration = direct_visual_patch_loss(output, data, mode="calibration")
    causal = direct_visual_patch_loss(output, data, mode="causal")
    assert torch.isfinite(calibration.loss)
    assert torch.isfinite(causal.loss)
    assert calibration.active_patches.item() == 12
    assert causal.active_patches.item() == 6
    causal.loss.backward()
    assert model.output_projection.weight.grad is not None


def test_optimizer_covers_frozen_core_for_later_unfreeze() -> None:
    model = tiny_model()
    set_core_trainable(model, False)
    groups = direct_visual_patch_optimizer_groups(
        model,
        adapter_learning_rate=3e-4,
        core_learning_rate=0.0,
    )
    grouped = {id(parameter) for group in groups for parameter in group["params"]}
    assert grouped == {id(parameter) for parameter in model.parameters()}
    assert all(not parameter.requires_grad for parameter in model.backbone.parameters())
    set_core_trainable(model, True)
    assert all(parameter.requires_grad for parameter in model.backbone.parameters())


def test_core_hash_changes_only_when_core_changes() -> None:
    model = tiny_model()
    before = module_state_sha256(model.backbone)
    with torch.no_grad():
        model.output_projection.weight.add_(1.0)
    assert module_state_sha256(model.backbone) == before
    with torch.no_grad():
        next(model.backbone.parameters()).add_(1.0)
    assert module_state_sha256(model.backbone) != before


def test_stage_schedule_warms_and_decays() -> None:
    values = [
        stage_cosine_learning_rate(i, peak=1e-3, warmup=2, total=10)
        for i in range(1, 11)
    ]
    assert values[0] == 5e-4
    assert values[1] == 1e-3
    assert values[-1] == 1e-4
    assert all(value > 0 for value in values)


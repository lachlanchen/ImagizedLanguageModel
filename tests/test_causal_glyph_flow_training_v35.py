from __future__ import annotations

import torch
import torch.nn as nn

from ilm.visual_lm.causal_glyph_flow import CausalGlyphFlowConfig, CausalGlyphFlowLM
from ilm.visual_lm.causal_glyph_flow_training import (
    causal_glyph_flow_loss,
    causal_glyph_flow_optimizer_groups,
    set_v35_optimizer_learning_rates,
    set_v35_stage_trainability,
    v35_optimizer_receipt,
    visual_interface_alignment_loss,
)


def tiny_model() -> CausalGlyphFlowLM:
    return CausalGlyphFlowLM(
        CausalGlyphFlowConfig(
            maximum_patches=8,
            hidden_size=64,
            layers=2,
            attention_heads=4,
            key_value_heads=2,
            intermediate_size=128,
            flow_width=64,
            flow_depth=2,
            codec_channels=(8, 16, 24, 32),
            codec_group_norm_groups=8,
        )
    )


def sample_batch() -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(3501)
    pixels = (torch.rand(2, 1, 32, 6 * 32, generator=generator) > 0.84).float()
    patch_mask = torch.tensor([[1, 1, 1, 1, 1, 0], [1, 1, 1, 1, 0, 0.0]])
    next_patch_mask = torch.tensor(
        [[1, 1, 1, 1, 0, 0], [1, 1, 1, 0, 0, 0.0]]
    )
    stop_targets = torch.tensor(
        [[0, 0, 0, 0, 1, 0], [0, 0, 0, 1, 0, 0.0]]
    )
    stop_mask = patch_mask.clone()
    return {
        "pixels": pixels,
        "patch_mask": patch_mask,
        "next_patch_mask": next_patch_mask,
        "stop_targets": stop_targets,
        "stop_mask": stop_mask,
    }


def test_alignment_loss_targets_frozen_visual_projection() -> None:
    torch.manual_seed(3502)
    model = tiny_model()
    batch = sample_batch()
    teacher = nn.Conv2d(1, 64, 32, stride=32, bias=False)
    teacher.requires_grad_(False)
    loss = visual_interface_alignment_loss(
        model,
        batch["pixels"],
        batch["patch_mask"],
        teacher,
    )
    assert torch.isfinite(loss.loss)
    assert -1.0 <= float(loss.cosine_similarity.detach()) <= 1.0
    assert int(loss.active_patches) == 9

    set_v35_stage_trainability(model, "visual-interface-alignment")
    loss.loss.backward()
    assert any(parameter.grad is not None for parameter in model.input_adapter.parameters())
    assert all(parameter.grad is None for parameter in model.backbone.parameters())
    assert all(parameter.grad is None for parameter in model.codec.parameters())


def test_causal_loss_trains_density_anchor_visual_and_stop_heads() -> None:
    torch.manual_seed(3503)
    model = tiny_model()
    set_v35_stage_trainability(model, "causal")
    batch = sample_batch()
    output = model(batch["pixels"], batch["patch_mask"])
    generator = torch.Generator().manual_seed(3504)
    losses = causal_glyph_flow_loss(
        model,
        output,
        batch,
        generator=generator,
        maximum_density_patches=5,
    )
    assert torch.isfinite(losses.loss)
    assert losses.density_patches == 5
    assert int(losses.active_patches) == 7
    assert float(losses.flow.detach()) > 0
    assert float(losses.anchor.detach()) > 0
    assert float(losses.visual.detach()) > 0
    assert float(losses.stop.detach()) > 0
    losses.loss.backward()
    assert model.flow_head.output_projection.weight.grad is not None
    assert model.anchor_head.layers[-1].weight.grad is not None
    assert model.stop_head.weight.grad is not None
    assert any(parameter.grad is not None for parameter in model.backbone.parameters())
    assert all(parameter.grad is None for parameter in model.input_adapter.parameters())
    assert all(parameter.grad is None for parameter in model.codec.parameters())


def test_density_sampling_is_reproducible() -> None:
    torch.manual_seed(3505)
    first_model = tiny_model()
    second_model = tiny_model()
    second_model.load_state_dict(first_model.state_dict())
    batch = sample_batch()
    first_output = first_model(batch["pixels"], batch["patch_mask"])
    second_output = second_model(batch["pixels"], batch["patch_mask"])
    first = causal_glyph_flow_loss(
        first_model,
        first_output,
        batch,
        generator=torch.Generator().manual_seed(3506),
        maximum_density_patches=4,
    )
    second = causal_glyph_flow_loss(
        second_model,
        second_output,
        batch,
        generator=torch.Generator().manual_seed(3506),
        maximum_density_patches=4,
    )
    assert torch.equal(first.loss, second.loss)


def test_optimizer_groups_exclude_codec_and_switch_rates() -> None:
    model = tiny_model()
    groups = causal_glyph_flow_optimizer_groups(
        model,
        adapter_learning_rate=3e-4,
        head_learning_rate=0.0,
        core_learning_rate=0.0,
    )
    receipt = v35_optimizer_receipt(model, groups)
    assert receipt["codec_parameter_names_optimized"] == []
    assert {group["role"] for group in groups} == {"adapter", "head", "core"}
    optimizer = torch.optim.AdamW(groups)
    set_v35_optimizer_learning_rates(
        optimizer,
        adapter=0.0,
        head=8e-5,
        core=8e-6,
    )
    rates = {group["role"]: group["lr"] for group in optimizer.param_groups}
    assert rates == {"adapter": 0.0, "head": 8e-5, "core": 8e-6}

from __future__ import annotations

import copy

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from ilm.visual_lm.visual_binding_stream import (
    QUERY_AWARE_ROUTE,
    QUERY_BLIND_ROUTE,
    OverlapLocalWriter,
    VisualBindingStream,
    VisualBindingStreamConfig,
    visual_binding_config_from_payload,
    visual_binding_config_payload,
    visual_binding_stream_loss,
)


class TinySpatialRetina(nn.Module):
    def __init__(self, visual_dim: int = 192) -> None:
        super().__init__()
        self.visual_dim = visual_dim
        self.field_projection = nn.Conv2d(1, visual_dim, 1, bias=False)
        self.visual_projection = nn.Linear(16, visual_dim, bias=False)

    def forward_with_field(
        self,
        images: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        pooled = F.adaptive_avg_pool2d(images, (4, 4))
        field = self.field_projection(pooled)
        visual = self.visual_projection(pooled.flatten(1))
        return visual, field

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        visual, _ = self.forward_with_field(images)
        return visual


def build_model(route_mode: str = QUERY_AWARE_ROUTE) -> VisualBindingStream:
    torch.manual_seed(5)
    config = VisualBindingStreamConfig(
        model_dim=64,
        transformer_blocks=2,
        attention_heads=4,
        feedforward_dim=128,
        writer_hidden_channels=64,
        writer_context_dim=64,
        writer_blocks=2,
        dropout=0.0,
        route_mode=route_mode,
    )
    return VisualBindingStream(config, TinySpatialRetina())


def random_prompt(batch: int = 3) -> torch.Tensor:
    return torch.rand(batch, 6, 1, 32, 32)


def test_stream_shapes_and_image_only_boundary() -> None:
    model = build_model()
    prompt = random_prompt()
    output = model(prompt)
    assert output.shape == (3, 1, 1, 32, 32)
    assert output.dtype.is_floating_point
    receipt = model.boundary_receipt()
    assert receipt["input_is_continuous_image"] is True
    assert receipt["output_is_continuous_image"] is True
    for forbidden in (
        "uses_strings",
        "uses_token_ids",
        "uses_unicode_ids",
        "uses_ocr",
        "uses_character_labels",
        "uses_operation_ids",
        "uses_slot_indices",
        "uses_visual_codebook",
        "uses_glyph_lookup",
        "uses_external_language_model",
        "retina_trainable",
    ):
        assert receipt[forbidden] is False
    with pytest.raises(TypeError):
        model(torch.zeros(1, 6, 1, 32, 32, dtype=torch.int64))


def test_attention_is_finite_and_normalized() -> None:
    model = build_model().eval()
    _, trace = model.logits_with_trace(random_prompt())
    attention = trace["selection_attention"]
    assert attention.shape == (3, 6)
    assert torch.isfinite(attention).all()
    torch.testing.assert_close(attention.sum(dim=1), torch.ones(3))


def test_query_blind_control_is_exactly_invariant_to_final_frame() -> None:
    model = build_model(QUERY_BLIND_ROUTE).eval()
    first = random_prompt()
    second = first.clone()
    second[:, -1] = torch.rand_like(second[:, -1])
    first_logits, first_trace = model.logits_with_trace(first)
    second_logits, second_trace = model.logits_with_trace(second)
    torch.testing.assert_close(first_logits, second_logits, rtol=0.0, atol=0.0)
    torch.testing.assert_close(
        first_trace["query_context"],
        second_trace["query_context"],
        rtol=0.0,
        atol=0.0,
    )


def test_query_aware_route_changes_with_final_frame() -> None:
    model = build_model(QUERY_AWARE_ROUTE).eval()
    first = random_prompt()
    second = first.clone()
    second[:, -1] = 1.0 - second[:, -1]
    _, first_trace = model.logits_with_trace(first)
    _, second_trace = model.logits_with_trace(second)
    assert not torch.equal(
        first_trace["query_context"],
        second_trace["query_context"],
    )


def test_candidate_and_control_have_equal_parameter_shapes() -> None:
    candidate = build_model(QUERY_AWARE_ROUTE)
    control = build_model(QUERY_BLIND_ROUTE)
    candidate_shapes = [
        (name, tuple(parameter.shape), parameter.requires_grad)
        for name, parameter in candidate.named_parameters()
    ]
    control_shapes = [
        (name, tuple(parameter.shape), parameter.requires_grad)
        for name, parameter in control.named_parameters()
    ]
    assert candidate_shapes == control_shapes


def test_full_configuration_is_below_parameter_budget() -> None:
    model = VisualBindingStream(
        VisualBindingStreamConfig(),
        TinySpatialRetina(),
    )
    trainable = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    assert trainable == 3_410_128
    assert trainable < 4_000_000


def test_overlap_weights_cover_every_pixel() -> None:
    writer = OverlapLocalWriter(build_model().config)
    assert writer.overlap_weights.shape == (1, 1, 32, 32)
    assert float(writer.overlap_weights.min()) > 0.0
    assert torch.isfinite(writer.overlap_weights).all()


def test_one_cell_field_intervention_has_bounded_support() -> None:
    model = build_model().eval()
    writer = model.writer
    visual = torch.randn(1, model.config.visual_dim)
    context = torch.randn(1, model.config.model_dim)
    field = torch.zeros(
        1,
        model.config.spatial_channels,
        model.config.field_size,
        model.config.field_size,
    )
    changed = field.clone()
    changed[:, :, 1, 2] = torch.linspace(-1.0, 1.0, model.config.spatial_channels)
    first = writer(visual, field, context)
    second = writer(visual, changed, context)
    difference = (second - first).abs()[0, 0]

    support = torch.zeros_like(difference, dtype=torch.bool)
    top = 1 * model.config.writer_stride - model.config.writer_padding
    left = 2 * model.config.writer_stride - model.config.writer_padding
    bottom = top + model.config.writer_patch_size
    right = left + model.config.writer_patch_size
    support[max(0, top) : min(32, bottom), max(0, left) : min(32, right)] = True
    assert float(difference[~support].max().detach()) == 0.0
    assert float(difference[support].max().detach()) > 0.0


def test_tiled_source_emits_equal_local_patch_logits() -> None:
    model = build_model().eval()
    writer = model.writer
    visual = torch.randn(2, model.config.visual_dim)
    context = torch.randn(2, model.config.model_dim)
    source = torch.randn(2, model.config.spatial_channels, 1, 1).expand(
        -1,
        -1,
        4,
        4,
    )
    _, trace = writer.logits_with_trace(visual, source, context)
    patches = trace["patch_logits"].flatten(2)
    assert float((patches - patches[:, :, :1]).abs().max().detach()) == 0.0


def test_generated_pixels_backpropagate_to_source_and_query_frames() -> None:
    model = build_model().train()
    prompt = random_prompt(batch=2).requires_grad_(True)
    output = model(prompt)
    output.mean().backward()
    assert prompt.grad is not None
    assert float(prompt.grad[:, 1].abs().sum()) > 0.0
    assert float(prompt.grad[:, -1].abs().sum()) > 0.0
    assert all(parameter.grad is None for parameter in model.retina.parameters())


def test_loss_is_finite_and_preserves_answer_stream() -> None:
    model = build_model().train()
    prompt = random_prompt(batch=2)
    counterfactual = prompt.clone()
    counterfactual[:, -1] = torch.rand_like(counterfactual[:, -1])
    target = torch.rand(2, 1, 32, 32)
    counter_target = torch.rand(2, 1, 32, 32)
    loss, metrics, trace = visual_binding_stream_loss(
        model,
        prompt,
        target,
        counterfactual,
        counter_target,
        prompt[:, 1],
        prompt[:, 3],
    )
    assert torch.isfinite(loss)
    assert all(torch.isfinite(value) for value in metrics.values())
    assert trace["generated"].shape == (4, 1, 32, 32)
    loss.backward()


def test_configuration_and_state_round_trip() -> None:
    model = build_model().eval()
    payload = visual_binding_config_payload(model.config)
    restored_config = visual_binding_config_from_payload(payload)
    restored = VisualBindingStream(restored_config, TinySpatialRetina()).eval()
    restored.load_state_dict(copy.deepcopy(model.state_dict()))
    prompt = random_prompt()
    torch.testing.assert_close(model(prompt), restored(prompt), rtol=0.0, atol=0.0)

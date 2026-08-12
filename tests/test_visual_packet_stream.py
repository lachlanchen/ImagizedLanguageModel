from __future__ import annotations

import copy

import pytest
import torch
import torch.nn as nn

from ilm.visual_lm.visual_packet_stream import (
    HEADER_BLIND_ROUTE,
    HISTORY_BLIND_ROUTE,
    OPERATION_BLIND_ROUTE,
    PACKET_AWARE_ROUTE,
    QUERY_BLIND_ROUTE,
    ROUTE_MODES,
    VisualPacketRereadStream,
    VisualPacketStreamConfig,
    visual_packet_stream_config_from_payload,
    visual_packet_stream_config_payload,
)


class CodeRetina(nn.Module):
    def forward_with_field(
        self, images: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        visual = images.flatten(1)[:, :192]
        field = visual[:, :, None, None].expand(-1, -1, 4, 4)
        return visual, field


class IdentityCanonicalizer(nn.Module):
    def logits_with_trace(
        self, image: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        logits = 8.0 * (image - 0.5)
        return logits, {"input_logits": logits, "residual_logits": torch.zeros_like(logits)}

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.logits_with_trace(image)[0].sigmoid()


class CodeOperationReader(nn.Module):
    def forward(self, visual: torch.Tensor) -> torch.Tensor:
        return 4.0 * (visual[:, 14:15] - visual[:, 15:16])


def code_image(index: int | None) -> torch.Tensor:
    image = torch.zeros(1, 32, 32)
    if index is not None:
        image.flatten()[index] = 1.0
    return image


def packet(header: int | None, a: int | None, b: int | None) -> torch.Tensor:
    return torch.stack((code_image(header), code_image(a), code_image(b)))


def build_prompt(*, same: bool = True, query_first: bool = True) -> torch.Tensor:
    packets = [
        packet(0, 10, 12),
        packet(0, 11, 13),
        packet(1, 14 if same else 15, None),
        packet(2, 10 if query_first else 11, None),
        packet(20, None, None),
    ]
    return torch.cat(packets).unsqueeze(0)


def build_model(route_mode: str) -> VisualPacketRereadStream:
    torch.manual_seed(113)
    model = VisualPacketRereadStream(
        VisualPacketStreamConfig(route_mode=route_mode),
        CodeRetina(),
        IdentityCanonicalizer(),
        CodeOperationReader(),
        match_temperature=16.0,
    )
    with torch.no_grad():
        model.role_prototypes.zero_()
        model.role_prototypes[0, 0] = 1.0
        model.role_prototypes[1, 1] = 1.0
        model.role_prototypes[2, 2] = 1.0
    return model


def trainable_parameters(model: nn.Module) -> int:
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


def test_v24_parameter_shapes_boundary_and_dynamic_contract() -> None:
    model = build_model(PACKET_AWARE_ROUTE).eval()
    assert trainable_parameters(model) == 1_347
    prompt = build_prompt()
    output, trace = model.logits_with_trace(prompt)
    assert output.shape == (1, 2, 1, 32, 32)
    assert trace["pair_indices"].shape == (1, 2)
    assert trace["query_indices"].shape == (1, 1)
    assert trace["operation_indices"].shape == (1, 1)
    assert set(trace["pair_indices"][0].tolist()) == {0, 1}
    assert trace["query_indices"].item() == 3
    assert trace["operation_indices"].item() == 2
    receipt = model.boundary_receipt()
    assert receipt["uses_absolute_frame_roles"] is False
    assert receipt["uses_relative_packet_offsets"] is True
    assert receipt["uses_padding_mask"] is False
    assert receipt["uses_token_ids"] is False
    assert receipt["rereads_generated_pixels"] is True

    padded = torch.cat((prompt, torch.zeros(1, 9, 1, 32, 32)), dim=1)
    assert model(padded).shape == (1, 2, 1, 32, 32)
    with pytest.raises(TypeError, match="image tensors"):
        model(prompt.to(torch.int64))
    with pytest.raises(ValueError, match="packet aligned"):
        model(torch.zeros(1, 16, 1, 32, 32))


def test_v24_visual_relation_and_reread_emit_two_correct_images() -> None:
    model = build_model(PACKET_AWARE_ROUTE).eval()
    prompt = build_prompt(same=True, query_first=True)
    output = model(prompt)
    assert output[0, 0].flatten().argmax().item() == 12
    assert output[0, 1].flatten().argmax().item() == 10

    other = build_prompt(same=False, query_first=True)
    other_output = model(other)
    assert other_output[0, 0].flatten().argmax().item() == 13
    assert other_output[0, 1].flatten().argmax().item() == 11

    override = code_image(13).unsqueeze(0)
    intervened = model.rollout_with_first_frame(prompt, override)
    assert intervened[0, 0].flatten().argmax().item() == 12
    assert intervened[0, 1].flatten().argmax().item() == 11


def test_v24_packet_permutation_is_structural() -> None:
    model = build_model(PACKET_AWARE_ROUTE).eval()
    prompt = build_prompt()
    packets = prompt.reshape(1, 5, 3, 1, 32, 32)
    permuted = packets[:, (3, 1, 4, 0, 2)].reshape_as(prompt)
    output, trace = model.logits_with_trace(prompt)
    changed, changed_trace = model.logits_with_trace(permuted)
    torch.testing.assert_close(output, changed, rtol=0.0, atol=1e-6)
    torch.testing.assert_close(
        trace["routed_source"],
        changed_trace["routed_source"],
        rtol=0.0,
        atol=1e-6,
    )
    torch.testing.assert_close(
        trace["routed_label"],
        changed_trace["routed_label"],
        rtol=0.0,
        atol=1e-6,
    )


def test_v24_blind_controls_remove_exact_visual_causes() -> None:
    prompt = build_prompt()
    query_changed = build_prompt(query_first=False)
    operation_changed = build_prompt(same=False)
    query_blind = build_model(QUERY_BLIND_ROUTE).eval()
    operation_blind = build_model(OPERATION_BLIND_ROUTE).eval()
    torch.testing.assert_close(
        query_blind(prompt), query_blind(query_changed), rtol=0.0, atol=0.0
    )
    torch.testing.assert_close(
        operation_blind(prompt),
        operation_blind(operation_changed),
        rtol=0.0,
        atol=0.0,
    )

    history_blind = build_model(HISTORY_BLIND_ROUTE).eval()
    first = code_image(12).unsqueeze(0)
    other = code_image(13).unsqueeze(0)
    first_output = history_blind.rollout_with_first_frame(prompt, first)
    other_output = history_blind.rollout_with_first_frame(prompt, other)
    torch.testing.assert_close(
        first_output[:, 1], other_output[:, 1], rtol=0.0, atol=0.0
    )


def test_v24_candidate_gradients_cross_visual_causes_and_feedback() -> None:
    model = build_model(PACKET_AWARE_ROUTE).train()
    prompt = build_prompt().requires_grad_(True)
    output = model(prompt)
    output_weights = torch.linspace(0.1, 1.0, output.numel()).reshape_as(output)
    (output * output_weights).sum().backward(retain_graph=True)
    packet_gradients = prompt.grad.reshape(1, 5, 3, 1, 32, 32).abs().sum(
        dim=(0, 3, 4, 5)
    )
    assert packet_gradients[:, 0].sum() > 0
    assert packet_gradients[2, 1] > 0
    assert packet_gradients[3, 1] > 0
    assert packet_gradients[:2, 1].sum() > 0
    assert packet_gradients[:2, 2].sum() > 0
    assert model.role_prototypes.grad is not None

    prompt.grad = None
    _, trace = model.logits_with_trace(prompt)
    trace["generated_first_frame"].retain_grad()
    label_weights = torch.linspace(
        0.1, 1.0, trace["routed_label"].numel()
    ).reshape_as(trace["routed_label"])
    (trace["routed_label"] * label_weights).sum().backward()
    assert trace["generated_first_frame"].grad is not None
    assert trace["generated_first_frame"].grad.abs().sum() > 0


def test_v24_arms_and_configuration_round_trip() -> None:
    models = [build_model(route_mode) for route_mode in ROUTE_MODES]
    shapes = [
        [
            (name, tuple(parameter.shape))
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        ]
        for model in models
    ]
    assert all(shape == shapes[0] for shape in shapes[1:])

    config = VisualPacketStreamConfig(route_mode=HEADER_BLIND_ROUTE)
    assert visual_packet_stream_config_from_payload(
        visual_packet_stream_config_payload(config)
    ) == config
    restored = copy.deepcopy(models[0]).eval()
    restored.load_state_dict(models[0].state_dict())
    prompt = build_prompt()
    torch.testing.assert_close(
        models[0].eval()(prompt), restored(prompt), rtol=0.0, atol=0.0
    )

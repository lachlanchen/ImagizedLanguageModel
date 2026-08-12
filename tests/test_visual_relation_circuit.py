from __future__ import annotations

import copy

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from ilm.visual_lm.visual_relation_circuit import (
    OPERATION_BLIND_ROUTE,
    QUERY_BLIND_ROUTE,
    RELATION_AWARE_ROUTE,
    VisualCanonicalizer,
    VisualRelationCircuit,
    VisualRelationCircuitConfig,
    canonicalizer_loss,
    relation_circuit_config_from_payload,
    relation_circuit_config_payload,
)


class TinyRetina(nn.Module):
    def __init__(self, visual_dim: int = 192) -> None:
        super().__init__()
        self.projection = nn.Linear(16, visual_dim, bias=False)

    def forward_with_field(
        self, images: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        pooled = F.adaptive_avg_pool2d(images, (4, 4))
        visual = self.projection(pooled.flatten(1))
        return visual, visual[:, :, None, None].expand(-1, -1, 4, 4)


def build_model(route_mode: str) -> VisualRelationCircuit:
    torch.manual_seed(73)
    return VisualRelationCircuit(
        VisualRelationCircuitConfig(route_mode=route_mode),
        TinyRetina(),
        VisualCanonicalizer(),
    )


def trainable_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def test_v23_parameter_counts_and_image_boundary() -> None:
    canonicalizer = VisualCanonicalizer()
    assert sum(parameter.numel() for parameter in canonicalizer.parameters()) == 1_122_081
    model = build_model(RELATION_AWARE_ROUTE)
    assert trainable_parameters(model) == 25_602
    assert all(not parameter.requires_grad for parameter in model.retina.parameters())
    assert all(
        not parameter.requires_grad for parameter in model.canonicalizer.parameters()
    )
    receipt = model.boundary_receipt()
    assert receipt["input_is_continuous_image"] is True
    assert receipt["output_is_continuous_image"] is True
    assert receipt["uses_token_ids"] is False
    assert receipt["uses_target_indices"] is False


def test_relation_weights_and_pair_swap_are_structural() -> None:
    model = build_model(RELATION_AWARE_ROUTE).eval()
    prompt = torch.rand(3, 6, 1, 32, 32)
    output, trace = model.logits_with_trace(prompt)
    assert output.shape == (3, 1, 1, 32, 32)
    assert trace["route_weights"].shape == (3, 2)
    assert torch.isfinite(trace["route_weights"]).all()
    assert (trace["route_weights"] >= 0).all()
    torch.testing.assert_close(
        trace["route_weights"].sum(dim=1),
        torch.ones_like(trace["route_weights"].sum(dim=1)),
        rtol=1e-6,
        atol=1e-6,
    )
    swapped = prompt[:, (2, 3, 0, 1, 4, 5)].clone()
    swapped_output, swapped_trace = model.logits_with_trace(swapped)
    torch.testing.assert_close(
        trace["routed_source"],
        swapped_trace["routed_source"],
        rtol=1e-5,
        atol=1e-6,
    )
    torch.testing.assert_close(output, swapped_output, rtol=1e-5, atol=1e-6)


def test_blind_controls_remove_exact_visual_causes() -> None:
    prompt = torch.rand(2, 6, 1, 32, 32)
    changed_query = prompt.clone()
    changed_query[:, 5] = torch.rand_like(changed_query[:, 5])
    query_blind = build_model(QUERY_BLIND_ROUTE).eval()
    torch.testing.assert_close(
        query_blind(prompt), query_blind(changed_query), rtol=0.0, atol=0.0
    )

    changed_operation = prompt.clone()
    changed_operation[:, 4] = torch.rand_like(changed_operation[:, 4])
    operation_blind = build_model(OPERATION_BLIND_ROUTE).eval()
    torch.testing.assert_close(
        operation_blind(prompt),
        operation_blind(changed_operation),
        rtol=0.0,
        atol=0.0,
    )


def test_candidate_gradients_reach_every_visual_cause_but_not_frozen_weights() -> None:
    model = build_model(RELATION_AWARE_ROUTE).train()
    prompt = torch.rand(2, 6, 1, 32, 32, requires_grad=True)
    model(prompt).sum().backward()
    for frame in range(6):
        assert prompt.grad[:, frame].abs().sum() > 0
    assert all(parameter.grad is None for parameter in model.retina.parameters())
    assert all(
        parameter.grad is None for parameter in model.canonicalizer.parameters()
    )
    assert model.operation_reader[1].weight.grad is not None


def test_canonicalizer_loss_and_config_round_trip() -> None:
    model = VisualCanonicalizer()
    source = torch.rand(2, 1, 32, 32)
    target = torch.rand(2, 1, 32, 32)
    loss, metrics = canonicalizer_loss(model, source, target)
    loss.backward()
    assert torch.isfinite(loss)
    assert 0 <= metrics["pixel_f1"] <= 1

    config = VisualRelationCircuitConfig(route_mode=OPERATION_BLIND_ROUTE)
    assert relation_circuit_config_from_payload(
        relation_circuit_config_payload(config)
    ) == config
    with pytest.raises(TypeError, match="continuous"):
        model(torch.ones(1, 1, 32, 32, dtype=torch.int64))


def test_relation_arms_have_identical_trainable_shapes() -> None:
    models = [
        build_model(route_mode)
        for route_mode in (
            RELATION_AWARE_ROUTE,
            QUERY_BLIND_ROUTE,
            OPERATION_BLIND_ROUTE,
        )
    ]
    shapes = [
        [(name, tuple(parameter.shape)) for name, parameter in model.named_parameters() if parameter.requires_grad]
        for model in models
    ]
    assert shapes[0] == shapes[1] == shapes[2]

    restored = copy.deepcopy(models[0]).eval()
    restored.load_state_dict(models[0].state_dict())
    prompt = torch.rand(1, 6, 1, 32, 32)
    torch.testing.assert_close(models[0].eval()(prompt), restored(prompt))

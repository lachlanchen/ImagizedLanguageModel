from __future__ import annotations

import copy
from argparse import Namespace

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from ilm.visual_lm.visual_packet_data import PARTITION_SALT
from ilm.visual_lm.visual_packet_stream import (
    HEADER_BLIND_ROUTE,
    HISTORY_BLIND_ROUTE,
    OPERATION_BLIND_ROUTE,
    PACKET_AWARE_ROUTE,
    QUERY_BLIND_ROUTE,
    VisualPacketRereadStream,
    VisualPacketStreamConfig,
)
from ilm.visual_lm.visual_relation_circuit import VisualCanonicalizer
from scripts.train_visual_packet_stream_v24 import (
    EXPECTED_PARAMETERS,
    FIXED_EVIDENCE_ARGUMENTS,
    FIXED_LOSS_ARGUMENTS,
    FIXED_MODEL_ARGUMENTS,
    FIXED_OPTIMIZATION_ARGUMENTS,
    _parameter_shapes,
    _require_fixed_arguments,
    candidate_selection_gate_report,
    control_selection_gate_report,
    load_packet_state,
    packet_state_dict,
    packet_stream_loss,
    selection_rank,
    student_boundary_is_clean,
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


def build_model(route_mode: str) -> VisualPacketRereadStream:
    torch.manual_seed(151)
    operation_reader = nn.Sequential(
        nn.LayerNorm(192), nn.Linear(192, 128), nn.SiLU(), nn.Linear(128, 1)
    )
    return VisualPacketRereadStream(
        VisualPacketStreamConfig(route_mode=route_mode),
        TinyRetina(),
        VisualCanonicalizer(),
        operation_reader,
        match_temperature=9.8,
    )


def fixed_args(*, smoke: bool = False) -> Namespace:
    return Namespace(
        partition_salt=PARTITION_SALT,
        smoke=smoke,
        **FIXED_MODEL_ARGUMENTS,
        **FIXED_LOSS_ARGUMENTS,
        **FIXED_OPTIMIZATION_ARGUMENTS,
        **FIXED_EVIDENCE_ARGUMENTS,
    )


def passing_candidate_metrics() -> dict[str, float]:
    return {
        "frame1_binary_choice_accuracy": 0.99,
        "query_switch_accuracy": 0.95,
        "operation_switch_accuracy": 0.96,
        "heldout_combination_minimum_switch_accuracy": 0.90,
        "frame1_identity_top1": 0.90,
        "identity_bank_identities": 88.0,
        "frame1_pixel_f1": 0.75,
        "frame1_target_cosine": 0.90,
        "frame2_label_top1": 0.99,
        "frame2_pixel_f1": 0.65,
        "frame2_target_cosine": 0.90,
        "teacher_forced_label_agreement": 0.99,
        "frame2_generated_history_consistency": 0.97,
        "history_switch_accuracy": 0.95,
        "history_output_pixel_l1": 0.20,
        "query_output_pixel_l1": 0.20,
        "operation_output_pixel_l1": 0.20,
        "header_output_pixel_l1": 0.20,
        "query_header_localization_accuracy": 1.0,
        "operation_header_localization_accuracy": 1.0,
        "pair_header_localization_accuracy": 1.0,
        "packet_permutation_frame1_identity_consistency": 1.0,
        "packet_permutation_frame2_identity_consistency": 1.0,
        "packet_permutation_frame1_output_pixel_l1": 0.0,
        "packet_permutation_frame2_output_pixel_l1": 0.0,
        "distractor_frame1_identity_consistency": 0.99,
        "distractor_frame2_identity_consistency": 0.99,
        "heldout_length_frame1_identity_top1": 0.95,
        "heldout_length_frame2_label_top1": 0.96,
        "student_boundary_clean": 1.0,
        "frozen_images_instantiated": 0.0,
        "step": 400.0,
    }


def test_v24_evidence_arguments_are_fixed_and_smoke_is_bounded() -> None:
    _require_fixed_arguments(fixed_args())
    changed = fixed_args()
    changed.localization_weight = 0.5
    with pytest.raises(ValueError, match="localization-weight"):
        _require_fixed_arguments(changed)

    smoke = fixed_args(smoke=True)
    smoke.maximum_steps = 20
    smoke.batch_size = 2
    _require_fixed_arguments(smoke)
    smoke.maximum_steps = 21
    with pytest.raises(ValueError, match="1--20"):
        _require_fixed_arguments(smoke)


def test_v24_candidate_and_control_gates_are_strict() -> None:
    metrics = passing_candidate_metrics()
    assert all(candidate_selection_gate_report(metrics).values())
    metrics["frame2_pixel_f1"] = 0.58
    assert candidate_selection_gate_report(metrics)["frame2_pixel_f1"] is False

    intervention = {
        HEADER_BLIND_ROUTE: "header_output_pixel_l1",
        QUERY_BLIND_ROUTE: "query_output_pixel_l1",
        OPERATION_BLIND_ROUTE: "operation_output_pixel_l1",
        HISTORY_BLIND_ROUTE: "history_output_pixel_l1",
    }
    for route_mode, metric_name in intervention.items():
        control = passing_candidate_metrics()
        control[metric_name] = 0.0
        assert all(control_selection_gate_report(control, route_mode).values())
        control[metric_name] = 1e-7
        assert not all(control_selection_gate_report(control, route_mode).values())


def test_v24_selection_rank_prefers_weakest_causal_switch() -> None:
    first = passing_candidate_metrics()
    later = dict(first, step=500.0)
    assert selection_rank(first, PACKET_AWARE_ROUTE) > selection_rank(
        later, PACKET_AWARE_ROUTE
    )
    later["query_switch_accuracy"] = 0.97
    later["operation_switch_accuracy"] = 0.98
    later["history_switch_accuracy"] = 0.97
    assert selection_rank(later, PACKET_AWARE_ROUTE) > selection_rank(
        first, PACKET_AWARE_ROUTE
    )


def test_v24_packet_state_excludes_frozen_modules_and_round_trips() -> None:
    model = build_model(PACKET_AWARE_ROUTE).eval()
    assert sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    ) == EXPECTED_PARAMETERS
    state = packet_state_dict(model)
    assert state
    assert all(
        not name.startswith(("retina.", "canonicalizer.", "operation_reader."))
        for name in state
    )
    restored = build_model(PACKET_AWARE_ROUTE).eval()
    restored.retina.load_state_dict(copy.deepcopy(model.retina.state_dict()))
    restored.canonicalizer.load_state_dict(
        copy.deepcopy(model.canonicalizer.state_dict())
    )
    restored.operation_reader.load_state_dict(
        copy.deepcopy(model.operation_reader.state_dict())
    )
    load_packet_state(restored, copy.deepcopy(state))
    prompt = torch.rand(2, 15, 1, 32, 32)
    torch.testing.assert_close(model(prompt), restored(prompt), rtol=0.0, atol=0.0)
    assert _parameter_shapes(model) == _parameter_shapes(restored)


def test_v24_loss_accepts_only_image_streams_and_backpropagates() -> None:
    model = build_model(PACKET_AWARE_ROUTE).train()
    batch_size = 2
    prompt = torch.rand(batch_size, 15, 1, 32, 32)
    target = torch.rand(batch_size, 2, 1, 32, 32)
    localization = torch.rand(batch_size, 4, 1, 32, 32)
    batch = {
        "prompt": prompt,
        "query_counterfactual_prompt": torch.rand_like(prompt),
        "operation_counterfactual_prompt": torch.rand_like(prompt),
        "target_stream": target,
        "query_counterfactual_target_stream": torch.rand_like(target),
        "operation_counterfactual_target_stream": torch.rand_like(target),
        "localization_target": localization,
        "query_counterfactual_localization_target": torch.rand_like(localization),
        "operation_counterfactual_localization_target": torch.rand_like(
            localization
        ),
    }
    loss, metrics = packet_stream_loss(
        model,
        batch,
        stroke_weight=4.0,
        generated_visual_weight=0.10,
        localization_weight=0.25,
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(metrics["localization_loss"])
    assert model.role_prototypes.grad is not None
    assert all(parameter.grad is None for parameter in model.retina.parameters())
    assert all(
        parameter.grad is None for parameter in model.canonicalizer.parameters()
    )
    assert all(
        parameter.grad is None for parameter in model.operation_reader.parameters()
    )

    bad = dict(batch, prompt=prompt.to(torch.int64))
    with pytest.raises(TypeError, match="continuous image"):
        packet_stream_loss(
            model,
            bad,
            stroke_weight=4.0,
            generated_visual_weight=0.10,
            localization_weight=0.25,
        )


def test_v24_boundary_receipt_is_strict() -> None:
    model = build_model(PACKET_AWARE_ROUTE)
    receipt = model.boundary_receipt()
    assert student_boundary_is_clean(receipt, PACKET_AWARE_ROUTE)
    changed = dict(receipt, uses_padding_mask=True)
    assert not student_boundary_is_clean(changed, PACKET_AWARE_ROUTE)

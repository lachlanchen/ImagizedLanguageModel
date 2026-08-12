from __future__ import annotations

import copy

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from ilm.visual_lm.ink_jepa_data import retinal_font_manifest
from ilm.visual_lm.visual_binding_stream import (
    QUERY_AWARE_ROUTE,
    QUERY_BLIND_ROUTE,
    VisualBindingStream,
    VisualBindingStreamConfig,
    visual_binding_config_payload,
)
from scripts.eval_visual_binding_stream_development import validate_pair_metadata
from scripts.train_visual_binding_stream import (
    ARCHITECTURE,
    EXPECTED_PARAMETERS,
    EXPECTED_PARTITION,
    EXPECTED_PVF_SHA256,
    FIXED_EVIDENCE_ARGUMENTS,
    FIXED_LOSS_ARGUMENTS,
    FIXED_MODEL_ARGUMENTS,
    FIXED_OPTIMIZATION_ARGUMENTS,
    PROTOCOL_DOCUMENT,
    _parameter_shapes,
    candidate_selection_gate_report,
    control_selection_gate_report,
    load_student_state,
    paired_gate_report,
    student_boundary_is_clean,
    student_state_dict,
)
from scripts.train_visual_state_actuator import file_sha256


class TinySpatialRetina(nn.Module):
    def __init__(self, visual_dim: int = 192) -> None:
        super().__init__()
        self.field_projection = nn.Conv2d(1, visual_dim, 1, bias=False)
        self.visual_projection = nn.Linear(16, visual_dim, bias=False)

    def forward_with_field(
        self,
        images: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        pooled = F.adaptive_avg_pool2d(images, (4, 4))
        return (
            self.visual_projection(pooled.flatten(1)),
            self.field_projection(pooled),
        )


def build_model(route_mode: str) -> VisualBindingStream:
    torch.manual_seed(73)
    return VisualBindingStream(
        VisualBindingStreamConfig(dropout=0.0, route_mode=route_mode),
        TinySpatialRetina(),
    )


def passing_candidate_metrics() -> dict[str, float]:
    return {
        "binary_choice_accuracy": 0.90,
        "counterfactual_switch_accuracy": 0.86,
        "heldout_combination_switch_accuracy": 0.82,
        "identity_top1": 0.55,
        "identity_bank_identities": 104.0,
        "query_shuffled_identity_top1": 0.30,
        "target_cosine": 0.82,
        "pixel_f1": 0.62,
        "oracle_pixel_f1": 0.68,
        "paired_output_pixel_l1": 0.10,
        "target_margin_over_operation": 0.18,
        "target_margin_over_query_label": 0.17,
        "frozen_images_instantiated": 0.0,
        "student_boundary_clean": 1.0,
    }


def passing_control_metrics() -> dict[str, float]:
    return {
        "binary_choice_accuracy": 0.50,
        "counterfactual_switch_accuracy": 0.25,
        "identity_top1": 0.20,
        "pixel_f1": 0.40,
        "paired_output_pixel_l1": 0.0,
        "frozen_images_instantiated": 0.0,
        "student_boundary_clean": 1.0,
    }


def test_candidate_gates_use_strict_preregistered_thresholds() -> None:
    metrics = passing_candidate_metrics()
    assert all(candidate_selection_gate_report(metrics).values())
    metrics["counterfactual_switch_accuracy"] = 0.80
    report = candidate_selection_gate_report(metrics)
    assert report["counterfactual_switch_accuracy"] is False


def test_control_and_paired_gates_enforce_causal_differences() -> None:
    candidate = passing_candidate_metrics()
    control = passing_control_metrics()
    assert all(control_selection_gate_report(control).values())
    report = paired_gate_report(
        candidate,
        control,
        candidate_parameters=EXPECTED_PARAMETERS,
        control_parameters=EXPECTED_PARAMETERS,
        parameter_shapes_equal=True,
    )
    assert all(report.values())
    control["identity_top1"] = 0.35
    assert paired_gate_report(
        candidate,
        control,
        candidate_parameters=EXPECTED_PARAMETERS,
        control_parameters=EXPECTED_PARAMETERS,
        parameter_shapes_equal=True,
    )["candidate_identity_gain"] is False


def test_student_boundary_check_rejects_symbolic_or_trainable_retina() -> None:
    model = build_model(QUERY_AWARE_ROUTE)
    receipt = model.boundary_receipt()
    assert student_boundary_is_clean(receipt, QUERY_AWARE_ROUTE)
    symbolic = dict(receipt)
    symbolic["uses_token_ids"] = True
    assert not student_boundary_is_clean(symbolic, QUERY_AWARE_ROUTE)
    trainable_retina = dict(receipt)
    trainable_retina["retina_trainable"] = True
    assert not student_boundary_is_clean(trainable_retina, QUERY_AWARE_ROUTE)


def test_student_checkpoint_excludes_retina_and_round_trips() -> None:
    model = build_model(QUERY_AWARE_ROUTE).eval()
    state = student_state_dict(model)
    assert state
    assert all(not key.startswith("retina.") for key in state)
    restored = VisualBindingStream(
        model.config,
        copy.deepcopy(model.retina),
    ).eval()
    load_student_state(restored, copy.deepcopy(state))
    prompt = torch.rand(2, 6, 1, 32, 32)
    torch.testing.assert_close(model(prompt), restored(prompt), rtol=0.0, atol=0.0)


def checkpoint_metadata(
    model: VisualBindingStream,
    route_mode: str,
) -> dict[str, object]:
    best = passing_candidate_metrics()
    best["step"] = 1_000.0
    return {
        "architecture": ARCHITECTURE,
        "route_mode": route_mode,
        "smoke_only": False,
        "step": 1_000,
        "best_development": best,
        "pvf_sha256": EXPECTED_PVF_SHA256,
        "manifest_sha256": "manifest",
        "trainable_parameters": EXPECTED_PARAMETERS,
        "trainable_parameter_shapes": _parameter_shapes(model),
        "model_config": visual_binding_config_payload(model.config),
        "boundary_receipt": model.boundary_receipt(),
        "retinal_fonts": retinal_font_manifest(),
        "partition": dict(EXPECTED_PARTITION),
        "protocol": {
            "protocol_sha256": file_sha256(PROTOCOL_DOCUMENT),
            "fixed_model_arguments": FIXED_MODEL_ARGUMENTS,
            "fixed_loss_arguments": FIXED_LOSS_ARGUMENTS,
            "fixed_optimization_arguments": FIXED_OPTIMIZATION_ARGUMENTS,
            "fixed_evidence_arguments": FIXED_EVIDENCE_ARGUMENTS,
        },
        "args": {
            **FIXED_MODEL_ARGUMENTS,
            **FIXED_LOSS_ARGUMENTS,
            **FIXED_OPTIMIZATION_ARGUMENTS,
            **FIXED_EVIDENCE_ARGUMENTS,
            "manifest": "manifest.jsonl",
            "partition_salt": "visual-binding-stream-v22",
            "pvf_checkpoint": "pvf.pt",
        },
    }


def test_pair_metadata_accepts_equal_arms_and_refuses_smoke() -> None:
    candidate_model = build_model(QUERY_AWARE_ROUTE)
    control_model = build_model(QUERY_BLIND_ROUTE)
    candidate = checkpoint_metadata(candidate_model, QUERY_AWARE_ROUTE)
    control = checkpoint_metadata(control_model, QUERY_BLIND_ROUTE)
    validate_pair_metadata(candidate, control)
    control["smoke_only"] = True
    with pytest.raises(ValueError, match="smoke-only"):
        validate_pair_metadata(candidate, control)

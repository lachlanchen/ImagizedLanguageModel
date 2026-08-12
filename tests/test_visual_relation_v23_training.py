from __future__ import annotations

import copy
from argparse import Namespace

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from ilm.visual_lm.ink_jepa_data import retinal_font_manifest
from ilm.visual_lm.visual_relation_circuit import (
    OPERATION_BLIND_ROUTE,
    QUERY_BLIND_ROUTE,
    RELATION_AWARE_ROUTE,
    VisualCanonicalizer,
    VisualRelationCircuit,
    VisualRelationCircuitConfig,
    relation_circuit_config_payload,
)
from ilm.visual_lm.visual_relation_data import PARTITION_SALT
from scripts.eval_visual_relation_circuit_development_v23 import (
    validate_pair_metadata,
)
from scripts.train_visual_relation_circuit_v23 import (
    ARCHITECTURE,
    EXPECTED_CANONICALIZER_SHA256,
    EXPECTED_MANIFEST_SHA256,
    EXPECTED_PARAMETERS,
    EXPECTED_PARTITION,
    EXPECTED_PVF_SHA256,
    FIXED_EVIDENCE_ARGUMENTS,
    FIXED_LOSS_ARGUMENTS,
    FIXED_MODEL_ARGUMENTS,
    FIXED_OPTIMIZATION_ARGUMENTS,
    PROTOCOL_DOCUMENT,
    SOURCE_FILES,
    _parameter_shapes,
    _require_fixed_arguments,
    _visual_variants,
    candidate_selection_gate_report,
    control_selection_gate_report,
    load_relation_state,
    paired_gate_report,
    relation_loss,
    relation_state_dict,
    selection_rank,
    student_boundary_is_clean,
)
from scripts.train_visual_state_actuator import file_sha256


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
    torch.manual_seed(81)
    return VisualRelationCircuit(
        VisualRelationCircuitConfig(route_mode=route_mode),
        TinyRetina(),
        VisualCanonicalizer(),
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
        "binary_choice_accuracy": 0.99,
        "query_switch_accuracy": 0.95,
        "operation_switch_accuracy": 0.96,
        "heldout_combination_minimum_switch_accuracy": 0.90,
        "pair_swap_identity_consistency": 1.0,
        "pair_swap_output_pixel_l1": 0.0,
        "identity_top1": 0.90,
        "identity_bank_identities": 109.0,
        "pixel_f1": 0.75,
        "target_cosine": 0.90,
        "query_output_pixel_l1": 0.20,
        "operation_output_pixel_l1": 0.21,
        "query_label_match_accuracy": 1.0,
        "operation_gate_accuracy": 1.0,
        "operation_gate_separation": 0.90,
        "student_boundary_clean": 1.0,
        "frozen_images_instantiated": 0.0,
        "step": 400.0,
    }


def passing_query_blind_metrics() -> dict[str, float]:
    return {
        "query_output_pixel_l1": 0.0,
        "operation_output_pixel_l1": 0.18,
        "query_switch_accuracy": 0.40,
        "operation_switch_accuracy": 0.72,
        "identity_top1": 0.30,
        "pixel_f1": 0.45,
        "student_boundary_clean": 1.0,
        "frozen_images_instantiated": 0.0,
        "step": 400.0,
    }


def passing_operation_blind_metrics() -> dict[str, float]:
    return {
        "query_output_pixel_l1": 0.18,
        "operation_output_pixel_l1": 0.0,
        "query_switch_accuracy": 0.70,
        "operation_switch_accuracy": 0.40,
        "identity_top1": 0.30,
        "pixel_f1": 0.45,
        "student_boundary_clean": 1.0,
        "frozen_images_instantiated": 0.0,
        "step": 400.0,
    }


def test_evidence_arguments_are_fixed_and_smoke_is_bounded() -> None:
    _require_fixed_arguments(fixed_args())
    changed = fixed_args()
    changed.generated_visual_weight = 0.2
    with pytest.raises(ValueError, match="generated-visual-weight"):
        _require_fixed_arguments(changed)

    smoke = fixed_args(smoke=True)
    smoke.maximum_steps = 20
    smoke.batch_size = 2
    _require_fixed_arguments(smoke)
    smoke.maximum_steps = 21
    with pytest.raises(ValueError, match="1--20"):
        _require_fixed_arguments(smoke)


def test_candidate_and_control_gates_are_strict() -> None:
    candidate = passing_candidate_metrics()
    assert all(candidate_selection_gate_report(candidate).values())
    candidate["operation_gate_accuracy"] = 0.98
    assert candidate_selection_gate_report(candidate)["operation_gate_accuracy"] is False

    query_blind = passing_query_blind_metrics()
    operation_blind = passing_operation_blind_metrics()
    assert all(
        control_selection_gate_report(query_blind, QUERY_BLIND_ROUTE).values()
    )
    assert all(
        control_selection_gate_report(
            operation_blind, OPERATION_BLIND_ROUTE
        ).values()
    )
    query_blind["query_output_pixel_l1"] = 1e-7
    assert (
        control_selection_gate_report(query_blind, QUERY_BLIND_ROUTE)[
            "query_blind_invariant"
        ]
        is False
    )


def test_paired_gate_requires_both_visual_causes_and_equal_arms() -> None:
    report = paired_gate_report(
        passing_candidate_metrics(),
        passing_query_blind_metrics(),
        passing_operation_blind_metrics(),
        candidate_parameters=EXPECTED_PARAMETERS,
        query_blind_parameters=EXPECTED_PARAMETERS,
        operation_blind_parameters=EXPECTED_PARAMETERS,
        parameter_shapes_equal=True,
    )
    assert all(report.values())

    operation_blind = passing_operation_blind_metrics()
    operation_blind["operation_switch_accuracy"] = 0.60
    report = paired_gate_report(
        passing_candidate_metrics(),
        passing_query_blind_metrics(),
        operation_blind,
        candidate_parameters=EXPECTED_PARAMETERS,
        query_blind_parameters=EXPECTED_PARAMETERS,
        operation_blind_parameters=EXPECTED_PARAMETERS,
        parameter_shapes_equal=True,
    )
    assert report["candidate_operation_switch_gain"] is False


def test_selection_rank_prefers_minimum_switch_then_quality() -> None:
    first = passing_candidate_metrics()
    later = dict(first, step=500.0)
    assert selection_rank(first, RELATION_AWARE_ROUTE) > selection_rank(
        later, RELATION_AWARE_ROUTE
    )
    later["query_switch_accuracy"] = 0.97
    assert selection_rank(later, RELATION_AWARE_ROUTE) > selection_rank(
        first, RELATION_AWARE_ROUTE
    )


def test_relation_state_excludes_frozen_modules_and_round_trips() -> None:
    model = build_model(RELATION_AWARE_ROUTE).eval()
    state = relation_state_dict(model)
    assert state
    assert all(
        not name.startswith(("retina.", "canonicalizer.")) for name in state
    )
    restored = build_model(RELATION_AWARE_ROUTE).eval()
    restored.retina.load_state_dict(copy.deepcopy(model.retina.state_dict()))
    restored.canonicalizer.load_state_dict(
        copy.deepcopy(model.canonicalizer.state_dict())
    )
    load_relation_state(restored, copy.deepcopy(state))
    prompt = torch.rand(2, 6, 1, 32, 32)
    torch.testing.assert_close(model(prompt), restored(prompt), rtol=0.0, atol=0.0)


def test_visual_variant_order_and_relation_loss_are_image_only() -> None:
    def image(value: float, prompt: bool = False) -> torch.Tensor:
        shape = (2, 6, 1, 32, 32) if prompt else (2, 1, 32, 32)
        return torch.full(shape, value)

    batch = {
        "prompt": image(1.0, True),
        "query_counterfactual_prompt": image(2.0, True),
        "operation_counterfactual_prompt": image(3.0, True),
        "pair_swapped_prompt": image(4.0, True),
        "target": image(5.0),
        "query_counterfactual_target": image(6.0),
        "operation_counterfactual_target": image(7.0),
        "pair_swapped_target": image(8.0),
        "distractor_target": image(9.0),
    }
    prompts, targets, distractors = _visual_variants(batch)
    assert prompts[:, 0, 0, 0, 0].tolist() == [1.0] * 2 + [2.0] * 2 + [3.0] * 2 + [4.0] * 2
    assert targets[:, 0, 0, 0].tolist() == [5.0] * 2 + [6.0] * 2 + [7.0] * 2 + [8.0] * 2
    assert distractors[:, 0, 0, 0].tolist() == [9.0] * 2 + [5.0] * 4 + [9.0] * 2

    random_batch = {
        key: torch.rand_like(value) for key, value in batch.items()
    }
    model = build_model(RELATION_AWARE_ROUTE)
    loss, metrics = relation_loss(
        model,
        random_batch,
        stroke_weight=4.0,
        generated_visual_weight=0.10,
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(metrics["generated_visual_loss"])
    assert all(parameter.grad is None for parameter in model.retina.parameters())
    assert all(
        parameter.grad is None for parameter in model.canonicalizer.parameters()
    )


def test_boundary_rejects_symbolic_or_trainable_frozen_modules() -> None:
    model = build_model(RELATION_AWARE_ROUTE)
    receipt = model.boundary_receipt()
    assert student_boundary_is_clean(receipt, RELATION_AWARE_ROUTE)
    symbolic = dict(receipt)
    symbolic["uses_target_indices"] = True
    assert not student_boundary_is_clean(symbolic, RELATION_AWARE_ROUTE)
    trainable_writer = dict(receipt)
    trainable_writer["canonicalizer_trainable"] = True
    assert not student_boundary_is_clean(trainable_writer, RELATION_AWARE_ROUTE)


def arm_checkpoint(route_mode: str) -> dict[str, object]:
    model = build_model(route_mode)
    if route_mode == RELATION_AWARE_ROUTE:
        metrics = passing_candidate_metrics()
    elif route_mode == QUERY_BLIND_ROUTE:
        metrics = passing_query_blind_metrics()
    else:
        metrics = passing_operation_blind_metrics()
    metrics["step"] = 400.0
    return {
        "architecture": ARCHITECTURE,
        "route_mode": route_mode,
        "smoke_only": False,
        "step": 400,
        "best_development": metrics,
        "trainable_parameters": EXPECTED_PARAMETERS,
        "trainable_parameter_shapes": _parameter_shapes(model),
        "model_config": relation_circuit_config_payload(model.config),
        "boundary_receipt": model.boundary_receipt(),
        "pvf_sha256": EXPECTED_PVF_SHA256,
        "canonicalizer_sha256": EXPECTED_CANONICALIZER_SHA256,
        "manifest_sha256": EXPECTED_MANIFEST_SHA256,
        "partition": dict(EXPECTED_PARTITION),
        "retinal_fonts": retinal_font_manifest(),
        "protocol": {
            "protocol_sha256": file_sha256(PROTOCOL_DOCUMENT),
            "source_files_sha256": {
                path: file_sha256(path) for path in SOURCE_FILES
            },
            "fixed_model_arguments": FIXED_MODEL_ARGUMENTS,
            "fixed_loss_arguments": FIXED_LOSS_ARGUMENTS,
            "fixed_optimization_arguments": FIXED_OPTIMIZATION_ARGUMENTS,
            "fixed_evidence_arguments": FIXED_EVIDENCE_ARGUMENTS,
            "canonicalizer_protocol": {"receipt": "same"},
        },
    }


def test_fresh_audit_metadata_refuses_smoke_and_unequal_shapes() -> None:
    checkpoints = {
        route_mode: arm_checkpoint(route_mode)
        for route_mode in (
            RELATION_AWARE_ROUTE,
            QUERY_BLIND_ROUTE,
            OPERATION_BLIND_ROUTE,
        )
    }
    validate_pair_metadata(checkpoints)

    smoke = copy.deepcopy(checkpoints)
    smoke[QUERY_BLIND_ROUTE]["smoke_only"] = True
    with pytest.raises(ValueError, match="smoke-only"):
        validate_pair_metadata(smoke)

    unequal = copy.deepcopy(checkpoints)
    unequal[OPERATION_BLIND_ROUTE]["trainable_parameter_shapes"][0]["shape"] = [1]
    with pytest.raises(ValueError, match="parameter shapes differ"):
        validate_pair_metadata(unequal)

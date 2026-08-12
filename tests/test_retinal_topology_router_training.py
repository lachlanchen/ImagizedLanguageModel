from __future__ import annotations

import argparse
import copy

import pytest
import torch

from ilm.visual_lm.retinal_topology_router import (
    FIELD_ROUTE,
    GLOBAL_CONTROL_ROUTE,
    RetinalTopologyRouter,
    RetinalTopologyRouterConfig,
    retinal_topology_router_config_payload,
)
from scripts.train_retinal_topology_router import (
    ARCHITECTURE,
    EXPECTED_PVF_SHA256,
    FIXED_EVIDENCE_ARGUMENTS,
    FIXED_LOSS_ARGUMENTS,
    FIXED_MODEL_ARGUMENTS,
    FIXED_OPTIMIZATION_ARGUMENTS,
    PARTITION_SALT,
    _require_fixed_arguments,
    better_development_candidate,
    checkpoint_payload,
    config_from_pvf,
    paired_gate_report,
    selection_eligible,
    selection_gate_report,
    selection_rule,
    validate_resume_checkpoint,
)
from scripts.eval_retinal_topology_router_development import validate_pair_metadata


def _partition() -> dict[str, object]:
    return {
        "algorithm": "sha256(salt + NUL + identifier) first 64 bits",
        "salt": PARTITION_SALT,
        "holdout_fraction": 0.06,
        "development_share_of_holdout": 0.5,
        "train_records": 90,
        "development_records": 5,
        "frozen_records": 5,
        "frozen_identifiers_sha256": "abc",
        "frozen_images_instantiated_during_training": False,
        "frozen_evaluator_permitted_during_development": False,
    }


def _passing_metrics() -> dict[str, float]:
    return {
        "correct_pixel_f1": 0.70,
        "correct_pixel_f1_dense": 0.72,
        "field_shuffled_pixel_f1_dense": 0.58,
        "zero_field_pixel_f1_dense": 0.60,
        "correct_identity_top1": 0.75,
        "both_shuffled_identity_top1": 0.70,
        "correct_target_cosine": 0.85,
        "both_shuffled_target_cosine": 0.80,
        "field_condition_pixel_l1": 0.10,
        "zero_field_condition_pixel_l1": 0.10,
        "occlusion_pixel_change": 0.03,
        "occlusion_locality": 0.50,
        "semantic_target_pixel_l1": 0.10,
        "coarse_within_block_max": 0.0,
        "detail_block_mean_abs_max": 0.0,
        "frozen_images_instantiated": 0.0,
    }


def _fixed_args(*, smoke: bool = False) -> argparse.Namespace:
    values = {
        **FIXED_MODEL_ARGUMENTS,
        **FIXED_LOSS_ARGUMENTS,
        **FIXED_OPTIMIZATION_ARGUMENTS,
        **FIXED_EVIDENCE_ARGUMENTS,
        "partition_salt": PARTITION_SALT,
        "smoke": smoke,
        "route_mode": FIELD_ROUTE,
        "pvf_checkpoint": "pvf.pt",
        "manifest": "data.jsonl",
    }
    return argparse.Namespace(**values)


def _config(route_mode: str = FIELD_ROUTE) -> RetinalTopologyRouterConfig:
    return RetinalTopologyRouterConfig(
        fovea_size=16,
        field_size=2,
        visual_dim=96,
        spatial_channels=96,
        style_dim=16,
        style_base_channels=8,
        hidden_channels=32,
        pointwise_blocks=1,
        coarse_hidden_dim=64,
        dropout=0.0,
        route_mode=route_mode,
    )


def _paired_checkpoint(route_mode: str) -> dict[str, object]:
    config = _config(route_mode)
    arguments = vars(_fixed_args()).copy()
    arguments["route_mode"] = route_mode
    arguments["smoke"] = False
    return {
        "architecture": ARCHITECTURE,
        "route_mode": route_mode,
        "planner_config": retinal_topology_router_config_payload(config),
        "partition": _partition(),
        "pvf_sha256": EXPECTED_PVF_SHA256,
        "selection_rule": selection_rule(route_mode),
        "best_development": {"step": 200},
        "smoke_only": False,
        "frozen_images_instantiated_during_training": False,
        "arguments": arguments,
        "student_contract": {
            "route_mode": route_mode,
            "global_state_enters_detail_path": (
                route_mode == GLOBAL_CONTROL_ROUTE
            ),
            "target_spatial_pixels_enter_condition": False,
            "student_received_token_ids": False,
            "student_received_unicode_ids": False,
            "student_received_ocr": False,
            "student_received_character_labels": False,
            "student_used_visual_codebook": False,
            "student_used_candidate_classifier": False,
            "student_used_external_language_model": False,
            "retina_trainable": False,
        },
    }


def test_candidate_selection_requires_every_strict_gate() -> None:
    passing = _passing_metrics()
    assert selection_eligible(passing, FIELD_ROUTE)
    assert all(selection_gate_report(passing, FIELD_ROUTE).values())

    failures = {
        "correct_pixel_f1": (0.66, "overall_pixel_f1"),
        "correct_pixel_f1_dense": (0.68, "dense_pixel_f1"),
        "field_shuffled_pixel_f1_dense": (0.60, "dense_field_shuffle_margin"),
        "zero_field_pixel_f1_dense": (0.62, "dense_zero_field_margin"),
        "correct_identity_top1": (0.70, "identity_top1"),
        "both_shuffled_identity_top1": (0.75, "identity_top1"),
        "correct_target_cosine": (0.82, "target_cosine"),
        "both_shuffled_target_cosine": (0.85, "target_cosine"),
        "field_condition_pixel_l1": (0.08, "field_condition_pixel_l1"),
        "zero_field_condition_pixel_l1": (0.08, "zero_field_condition_pixel_l1"),
        "occlusion_pixel_change": (0.02, "occlusion_pixel_change"),
        "occlusion_locality": (0.40, "occlusion_locality"),
        "semantic_target_pixel_l1": (0.05, "semantic_target_pixel_l1"),
        "coarse_within_block_max": (1e-6, "coarse_invariant"),
        "detail_block_mean_abs_max": (1e-6, "detail_invariant"),
        "frozen_images_instantiated": (1.0, "frozen_bank_sealed"),
    }
    for metric, (value, gate) in failures.items():
        changed = {**passing, metric: value}
        report = selection_gate_report(changed, FIELD_ROUTE)
        assert not report[gate], metric
        assert not selection_eligible(changed, FIELD_ROUTE), metric


def test_control_selection_omits_field_and_occlusion_gates() -> None:
    metrics = _passing_metrics()
    for key in (
        "field_shuffled_pixel_f1_dense",
        "zero_field_pixel_f1_dense",
        "field_condition_pixel_l1",
        "zero_field_condition_pixel_l1",
        "occlusion_pixel_change",
        "occlusion_locality",
    ):
        metrics.pop(key)

    report = selection_gate_report(metrics, GLOBAL_CONTROL_ROUTE)
    assert selection_eligible(metrics, GLOBAL_CONTROL_ROUTE)
    assert all(report.values())
    assert not any("field" in key or "occlusion" in key for key in report)
    assert selection_rule(GLOBAL_CONTROL_ROUTE)["paired_control_only"] is True


def test_paired_gate_requires_gain_noninferiority_and_equal_capacity() -> None:
    candidate = {"correct_pixel_f1_dense": 0.72, "correct_pixel_f1": 0.69}
    control = {"correct_pixel_f1_dense": 0.68, "correct_pixel_f1": 0.70}
    assert all(
        paired_gate_report(
            candidate,
            control,
            candidate_parameters=123,
            control_parameters=123,
        ).values()
    )

    failures = (
        ({**candidate, "correct_pixel_f1_dense": 0.71}, 123, 123),
        ({**candidate, "correct_pixel_f1": 0.679}, 123, 123),
        (candidate, 123, 124),
    )
    for changed, candidate_parameters, control_parameters in failures:
        report = paired_gate_report(
            changed,
            control,
            candidate_parameters=candidate_parameters,
            control_parameters=control_parameters,
        )
        assert not all(report.values())


def test_selection_order_is_dense_then_overall_then_earlier() -> None:
    incumbent = {"correct_pixel_f1_dense": 0.72, "correct_pixel_f1": 0.70, "step": 400}
    assert better_development_candidate(
        {"correct_pixel_f1_dense": 0.73, "correct_pixel_f1": 0.60, "step": 600},
        incumbent,
    )
    assert better_development_candidate(
        {"correct_pixel_f1_dense": 0.72, "correct_pixel_f1": 0.71, "step": 600},
        incumbent,
    )
    assert better_development_candidate(
        {"correct_pixel_f1_dense": 0.72, "correct_pixel_f1": 0.70, "step": 200},
        incumbent,
    )
    assert not better_development_candidate(
        {"correct_pixel_f1_dense": 0.72, "correct_pixel_f1": 0.70, "step": 600},
        incumbent,
    )


@pytest.mark.parametrize(
    ("name", "value"),
    (
        ("partition_salt", "wrong"),
        ("hidden_channels", 64),
        ("field_margin", 0.04),
        ("seed", 1),
        ("batch_size", 16),
    ),
)
def test_evidence_arguments_are_fixed(name: str, value: object) -> None:
    args = _fixed_args()
    setattr(args, name, value)
    with pytest.raises(ValueError, match="V20"):
        _require_fixed_arguments(args)


def test_smoke_mode_is_bounded_but_allows_small_operational_inputs() -> None:
    args = _fixed_args(smoke=True)
    args.maximum_steps = 20
    args.batch_size = 2
    args.sequence_length = 4
    args.positions_per_sequence = 1
    args.samples_per_epoch = 8
    args.development_samples = 4
    args.num_workers = 0
    args.validate_every = 2
    args.validation_batches = 1
    args.save_every = 2
    _require_fixed_arguments(args)

    args.maximum_steps = 21
    with pytest.raises(ValueError, match="20 optimization steps"):
        _require_fixed_arguments(args)


def test_config_is_derived_from_the_frozen_visual_field() -> None:
    args = _fixed_args()
    checkpoint = {
        "model_config": {
            "fovea_size": 32,
            "visual_dim": 192,
            "retina_base_channels": 64,
        }
    }
    config = config_from_pvf(checkpoint, args)
    assert config.fovea_size == 32
    assert config.field_size == 4
    assert config.visual_dim == config.spatial_channels == 192
    assert config.route_mode == FIELD_ROUTE

    incompatible = copy.deepcopy(checkpoint)
    incompatible["model_config"]["visual_dim"] = 128
    with pytest.raises(ValueError, match="visual_dim must equal spatial_channels"):
        config_from_pvf(incompatible, args)


def test_resume_requires_same_partition_field_config_and_mode() -> None:
    config = _config()
    checkpoint = {
        "architecture": ARCHITECTURE,
        "partition": _partition(),
        "pvf_sha256": "pvf-sha",
        "planner_config": retinal_topology_router_config_payload(config),
        "route_mode": FIELD_ROUTE,
        "smoke_only": False,
    }
    validate_resume_checkpoint(
        checkpoint,
        expected_partition=_partition(),
        expected_pvf_sha256="pvf-sha",
        expected_config=config,
        smoke=False,
    )

    mutations = (
        ("architecture", "wrong"),
        ("partition", {**_partition(), "salt": "wrong"}),
        ("pvf_sha256", "wrong"),
        ("route_mode", GLOBAL_CONTROL_ROUTE),
        ("smoke_only", True),
        (
            "planner_config",
            {
                **retinal_topology_router_config_payload(config),
                "pointwise_blocks": 2,
            },
        ),
    )
    for key, value in mutations:
        changed = copy.deepcopy(checkpoint)
        changed[key] = value
        with pytest.raises(ValueError):
            validate_resume_checkpoint(
                changed,
                expected_partition=_partition(),
                expected_pvf_sha256="pvf-sha",
                expected_config=config,
                smoke=False,
            )


@pytest.mark.parametrize(
    ("route_mode", "global_enters_detail"),
    ((FIELD_ROUTE, False), (GLOBAL_CONTROL_ROUTE, True)),
)
def test_checkpoint_records_image_only_student_contract(
    route_mode: str,
    global_enters_detail: bool,
) -> None:
    planner = RetinalTopologyRouter(_config(route_mode))
    optimizer = torch.optim.AdamW(planner.parameters(), lr=3e-4)
    args = argparse.Namespace(
        pvf_checkpoint="pvf.pt",
        smoke=True,
        route_mode=route_mode,
    )
    payload = checkpoint_payload(
        planner,
        optimizer,
        args=args,
        pvf_checkpoint={
            "architecture": "predictive-visual-field-v1",
            "model_config": {"fovea_size": 16},
            "render_config": {"fovea_size": 16},
            "global_step": 2_200,
        },
        pvf_sha256="pvf-sha",
        partition=_partition(),
        best_development=None,
        epoch=0,
        step=2,
        elapsed_seconds=1.0,
    )
    contract = payload["student_contract"]
    assert contract["global_state_enters_detail_path"] is global_enters_detail
    assert contract["target_spatial_pixels_enter_condition"] is False
    assert contract["student_received_token_ids"] is False
    assert contract["student_received_unicode_ids"] is False
    assert contract["student_received_ocr"] is False
    assert contract["student_used_visual_codebook"] is False
    assert contract["student_used_external_language_model"] is False
    assert contract["retina_trainable"] is False
    assert payload["frozen_images_instantiated_during_training"] is False
    assert payload["paired_control_gate_passed"] is False
    assert payload["blinded_readability_gate_passed"] is False


def test_paired_audit_accepts_only_clean_equal_shape_arms() -> None:
    candidate = _paired_checkpoint(FIELD_ROUTE)
    control = _paired_checkpoint(GLOBAL_CONTROL_ROUTE)
    candidate_config, control_config = validate_pair_metadata(candidate, control)
    assert candidate_config.route_mode == FIELD_ROUTE
    assert control_config.route_mode == GLOBAL_CONTROL_ROUTE
    assert {
        **retinal_topology_router_config_payload(candidate_config),
        "route_mode": None,
    } == {
        **retinal_topology_router_config_payload(control_config),
        "route_mode": None,
    }


@pytest.mark.parametrize(
    ("arm", "mutation"),
    (
        ("candidate", lambda value: value.update(smoke_only=True)),
        ("candidate", lambda value: value.update(best_development=None)),
        ("candidate", lambda value: value.update(pvf_sha256="wrong")),
        (
            "candidate",
            lambda value: value["student_contract"].update(
                student_received_token_ids=True
            ),
        ),
        (
            "control",
            lambda value: value.update(
                partition={**_partition(), "salt": "wrong"}
            ),
        ),
        (
            "control",
            lambda value: value["arguments"].update(seed=1),
        ),
        (
            "control",
            lambda value: value["planner_config"].update(pointwise_blocks=2),
        ),
    ),
)
def test_paired_audit_rejects_noncomparable_or_contaminated_arms(
    arm: str,
    mutation,
) -> None:
    candidate = _paired_checkpoint(FIELD_ROUTE)
    control = _paired_checkpoint(GLOBAL_CONTROL_ROUTE)
    mutation(candidate if arm == "candidate" else control)
    with pytest.raises(ValueError):
        validate_pair_metadata(candidate, control)

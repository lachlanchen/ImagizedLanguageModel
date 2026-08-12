from __future__ import annotations

import copy

import pytest

from ilm.visual_lm.spatial_motor_plan import SpatialMotorPlanConfig
from scripts.train_spatial_motor_plan import (
    ARCHITECTURE,
    GLOBAL_ARCHITECTURE,
    PARTITION_SALT,
    better_development_candidate,
    spatial_selection_eligible,
    validate_global_baseline_checkpoint,
    validate_resume_checkpoint,
)


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
        "correct_pixel_f1": 0.72,
        "correct_pixel_f1_dense": 0.64,
        "spatial_shuffled_pixel_f1_dense": 0.48,
        "zero_field_pixel_f1_dense": 0.59,
        "correct_identity_top1": 0.80,
        "both_shuffled_identity_top1": 0.02,
        "correct_target_cosine": 0.87,
        "both_shuffled_target_cosine": 0.08,
        "condition_pixel_l1": 0.12,
        "semantic_target_pixel_l1": 0.20,
        "frozen_images_instantiated": 0.0,
    }


def _global_checkpoint(*, selected: bool = True) -> dict[str, object]:
    best = {
        "step": 200,
        "correct_pixel_f1": 0.70,
        "shuffled_pixel_f1": 0.30,
        "correct_identity_top1": 0.80,
        "shuffled_identity_top1": 0.01,
        "correct_target_cosine": 0.85,
        "shuffled_target_cosine": 0.05,
        "condition_pixel_l1": 0.12,
    }
    return {
        "architecture": GLOBAL_ARCHITECTURE,
        "partition": _partition(),
        "arguments": {
            "partition_salt": PARTITION_SALT,
            "warmstart_v17": None,
        },
        "style_warmstart": None,
        "pvf_sha256": "pvf-sha",
        "global_step": 200,
        "best_development": best if selected else None,
        "student_contract": {
            "student_received_token_ids": False,
            "student_received_unicode_ids": False,
        },
    }


def test_global_baseline_requires_clean_selected_v19_checkpoint() -> None:
    validate_global_baseline_checkpoint(
        _global_checkpoint(),
        expected_partition=_partition(),
        expected_pvf_sha256="pvf-sha",
        allow_unselected=False,
    )

    unselected = _global_checkpoint(selected=False)
    with pytest.raises(ValueError, match="selected"):
        validate_global_baseline_checkpoint(
            unselected,
            expected_partition=_partition(),
            expected_pvf_sha256="pvf-sha",
            allow_unselected=False,
        )
    validate_global_baseline_checkpoint(
        unselected,
        expected_partition=_partition(),
        expected_pvf_sha256="pvf-sha",
        allow_unselected=True,
    )


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        (lambda value: value.update(architecture="wrong"), "architecture"),
        (
            lambda value: value["arguments"].update(partition_salt="wrong"),
            "fixed V19 salt",
        ),
        (
            lambda value: value.update(style_warmstart={"step": 10}),
            "V17 warm-start",
        ),
        (lambda value: value.update(pvf_sha256="wrong"), "Predictive Visual Field"),
        (
            lambda value: value["student_contract"].update(
                student_received_token_ids=True
            ),
            "image-only student contract",
        ),
    ),
)
def test_global_baseline_rejects_contamination(mutation, message: str) -> None:
    checkpoint = _global_checkpoint()
    mutation(checkpoint)
    with pytest.raises(ValueError, match=message):
        validate_global_baseline_checkpoint(
            checkpoint,
            expected_partition=_partition(),
            expected_pvf_sha256="pvf-sha",
            allow_unselected=False,
        )


def test_spatial_selection_requires_every_fixed_gate() -> None:
    passing = _passing_metrics()
    assert spatial_selection_eligible(passing)
    failures = {
        "correct_pixel_f1": 0.68,
        "correct_pixel_f1_dense": 0.58,
        "spatial_shuffled_pixel_f1_dense": 0.53,
        "zero_field_pixel_f1_dense": 0.62,
        "correct_identity_top1": 0.75,
        "both_shuffled_identity_top1": 0.81,
        "correct_target_cosine": 0.84,
        "both_shuffled_target_cosine": 0.88,
        "condition_pixel_l1": 0.08,
        "semantic_target_pixel_l1": 0.05,
        "frozen_images_instantiated": 1.0,
    }
    for key, value in failures.items():
        metrics = {**passing, key: value}
        assert not spatial_selection_eligible(metrics), key


def test_spatial_selection_uses_dense_then_overall_then_earlier_step() -> None:
    incumbent = {"correct_pixel_f1_dense": 0.64, "correct_pixel_f1": 0.72, "step": 400}
    assert better_development_candidate(
        {"correct_pixel_f1_dense": 0.65, "correct_pixel_f1": 0.60, "step": 600},
        incumbent,
    )
    assert better_development_candidate(
        {"correct_pixel_f1_dense": 0.64, "correct_pixel_f1": 0.73, "step": 600},
        incumbent,
    )
    assert better_development_candidate(
        {"correct_pixel_f1_dense": 0.64, "correct_pixel_f1": 0.72, "step": 200},
        incumbent,
    )
    assert not better_development_candidate(
        {"correct_pixel_f1_dense": 0.64, "correct_pixel_f1": 0.72, "step": 600},
        incumbent,
    )


def test_resume_requires_same_partition_sources_config_and_mode() -> None:
    config = SpatialMotorPlanConfig()
    checkpoint = {
        "architecture": ARCHITECTURE,
        "partition": _partition(),
        "pvf_sha256": "pvf-sha",
        "global_baseline_sha256": "global-sha",
        "planner_config": config.__dict__,
        "smoke_only": False,
    }
    validate_resume_checkpoint(
        checkpoint,
        expected_partition=_partition(),
        expected_pvf_sha256="pvf-sha",
        expected_global_sha256="global-sha",
        expected_config=config,
        smoke=False,
    )
    mutations = (
        ("partition", {**_partition(), "salt": "wrong"}),
        ("pvf_sha256", "wrong"),
        ("global_baseline_sha256", "wrong"),
        ("planner_config", {**config.__dict__, "spatial_blocks": 3}),
        ("smoke_only", True),
    )
    for key, value in mutations:
        changed = copy.deepcopy(checkpoint)
        changed[key] = value
        with pytest.raises(ValueError):
            validate_resume_checkpoint(
                changed,
                expected_partition=_partition(),
                expected_pvf_sha256="pvf-sha",
                expected_global_sha256="global-sha",
                expected_config=config,
                smoke=False,
            )

from __future__ import annotations

from ilm.visual_lm.ink_jepa_data import VisualGrammarRecord
from scripts.train_visual_motor_plan import partition_records, selection_eligible


def _record(index: int) -> VisualGrammarRecord:
    return VisualGrammarRecord(
        identifier=f"record-{index:04d}",
        text="天地玄黄",
        language="zh",
        source="unit-test",
        rights="public-domain",
    )


def test_salted_motor_plan_partition_is_deterministic_and_disjoint() -> None:
    records = [_record(index) for index in range(1_000)]
    first = partition_records(
        records,
        salt="motor-plan-test",
        holdout_fraction=0.10,
        development_share=0.50,
    )
    second = partition_records(
        records,
        salt="motor-plan-test",
        holdout_fraction=0.10,
        development_share=0.50,
    )

    assert [[record.identifier for record in split] for split in first] == [
        [record.identifier for record in split] for split in second
    ]
    identifier_sets = [{record.identifier for record in split} for split in first]
    assert not identifier_sets[0] & identifier_sets[1]
    assert not identifier_sets[0] & identifier_sets[2]
    assert not identifier_sets[1] & identifier_sets[2]
    assert sum(len(split) for split in first) == len(records)


def test_motor_plan_selection_requires_topology_and_causal_control() -> None:
    passing = {
        "correct_pixel_f1": 0.72,
        "shuffled_pixel_f1": 0.31,
        "correct_identity_top1": 0.60,
        "shuffled_identity_top1": 0.01,
        "correct_target_cosine": 0.74,
        "shuffled_target_cosine": 0.08,
        "condition_pixel_l1": 0.12,
    }

    assert selection_eligible(passing)
    for key, failure in (
        ("correct_pixel_f1", 0.60),
        ("shuffled_pixel_f1", 0.58),
        ("correct_identity_top1", 0.01),
        ("correct_target_cosine", 0.60),
        ("shuffled_target_cosine", 0.80),
        ("condition_pixel_l1", 0.05),
    ):
        metrics = {**passing, key: failure}
        assert not selection_eligible(metrics)

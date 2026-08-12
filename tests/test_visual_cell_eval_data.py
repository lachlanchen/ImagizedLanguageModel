from __future__ import annotations

import pytest
import torch

from ilm.visual_lm.ink_jepa_data import VisualGrammarRecord
from ilm.visual_lm.visual_cell_data import visual_cell_partition
from ilm.visual_lm.visual_cell_eval_data import (
    VisualCellAuditDataset,
    VisualCharacterStatistics,
    build_visual_cell_audit_windows,
    build_visual_character_statistics,
    is_han_character,
    render_visual_character_bank,
    visual_cell_audit_collate,
)


def identifier_for(split: str) -> str:
    for index in range(100_000):
        identifier = f"visual-cell-eval-{index}"
        if visual_cell_partition(identifier) == split:
            return identifier
    raise AssertionError(f"could not build identifier for {split}")


def record(split: str, text: str) -> VisualGrammarRecord:
    return VisualGrammarRecord(
        identifier=identifier_for(split),
        text=text,
        language="zh-Hans",
        source="unit-test",
        rights="test-only",
    )


def test_han_detection_does_not_treat_punctuation_as_a_character_id() -> None:
    assert is_han_character("中")
    assert is_han_character("學")
    assert not is_han_character("。")
    assert not is_han_character("A")
    assert not is_han_character("语言")


def test_training_statistics_define_host_only_visual_controls() -> None:
    records = [
        record("train", "天地玄黄天地玄黄宇宙洪荒" * 8),
        record("development", "天地玄黄宇宙洪荒" * 20),
    ]
    statistics = build_visual_character_statistics(
        records,
        bank_size=8,
        script_views_mode="original",
    )
    assert statistics.characters[:4] == ("地", "天", "玄", "黄")
    assert len(statistics.counts) == 8
    assert sum(statistics.bigram_rows["天"]) > 0
    assert len(statistics.bigram_rows["天"]) == 8


def test_development_windows_are_deterministic_and_never_use_frozen_records() -> None:
    train = record("train", "天地玄黄宇宙洪荒" * 20)
    development = record("development", "天地玄黄宇宙洪荒" * 40)
    frozen = record("frozen", "海咸河淡鳞潜羽翔" * 40)
    statistics = build_visual_character_statistics(
        [train, development, frozen],
        bank_size=8,
        script_views_mode="original",
    )
    first = build_visual_cell_audit_windows(
        [train, development, frozen],
        statistics,
        count=12,
        continuation_cells=4,
        seed=31,
        script_views_mode="original",
    )
    second = build_visual_cell_audit_windows(
        [train, development, frozen],
        statistics,
        count=12,
        continuation_cells=4,
        seed=31,
        script_views_mode="original",
    )
    assert first == second
    assert {window.identifier for window in first} == {development.identifier}
    assert all(len(window.context) == 64 for window in first)
    assert all(len(window.continuation) == 4 for window in first)


def test_audit_dataset_keeps_labels_outside_image_model_inputs() -> None:
    statistics = VisualCharacterStatistics(
        characters=("天", "地"),
        counts=(9, 7),
        bigram_rows={"天": (0, 7), "地": (9, 0)},
        visible_character_count=16,
        han_character_count=16,
    )
    development = record("development", "天地" * 60)
    windows = build_visual_cell_audit_windows(
        [development],
        statistics,
        count=2,
        continuation_cells=3,
        script_views_mode="original",
    )
    dataset = VisualCellAuditDataset(windows, statistics)
    batch = visual_cell_audit_collate([dataset[0], dataset[1]])
    assert batch["context"].shape == (2, 64, 1, 32, 32)
    assert batch["continuation"].shape == (2, 3, 1, 32, 32)
    assert batch["reference_continuation"].shape == (2, 3, 1, 32, 32)
    assert batch["target_index"].dtype == torch.int64
    student_context = {"context": batch["context"]}
    assert set(student_context) == {"context"}
    assert torch.is_floating_point(student_context["context"])

    bank = render_visual_character_bank(statistics)
    assert bank.shape == (2, 2, 1, 32, 32)


def test_statistics_reject_an_unavailable_bank_size() -> None:
    with pytest.raises(ValueError, match="only"):
        build_visual_character_statistics(
            [record("train", "天地天地")],
            bank_size=3,
            script_views_mode="original",
        )

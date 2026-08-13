from __future__ import annotations

import torch
from torch.utils.data import Dataset

from scripts.train_continuous_glyph_codec_v34 import (
    DatasetWindow,
    tensors_are_finite,
    validate_resume_checkpoint,
)


class NumberDataset(Dataset[dict]):
    def __len__(self) -> int:
        return 10

    def __getitem__(self, index: int) -> dict:
        return {"value": index}


def test_dataset_window_preserves_resume_cursor() -> None:
    window = DatasetWindow(NumberDataset(), start=3, count=4)
    assert len(window) == 4
    assert [window[index]["value"] for index in range(4)] == [3, 4, 5, 6]


def test_recursive_finite_check_catches_optimizer_nan() -> None:
    assert tensors_are_finite({"state": [torch.tensor([1.0]), torch.tensor(2)]})
    assert not tensors_are_finite({"state": {"moment": torch.tensor([float("nan")])}})


def test_resume_checkpoint_requires_both_data_cursors() -> None:
    checkpoint = {
        "experiment": "continuous-glyph-representation-codec-v34",
        "architecture": "continuous-glyph-representation-codec-v34",
        "protocol": {
            "sha256": "c2370374f202714e217236f7634f464eb98bed6a0f8afe898b9658614df7ce51"
        },
        "update": 3,
        "rendered_examples_consumed": 24,
        "historic_examples_consumed": 384,
    }
    assert (
        validate_resume_checkpoint(
            checkpoint,
            planned_updates=6_000,
            rendered_batch_size=8,
            historic_batch_size=128,
        )
        == 3
    )
    checkpoint["historic_examples_consumed"] = 383
    try:
        validate_resume_checkpoint(
            checkpoint,
            planned_updates=6_000,
            rendered_batch_size=8,
            historic_batch_size=128,
        )
    except ValueError as error:
        assert "historical" in str(error)
    else:
        raise AssertionError("V34 accepted an inconsistent historical cursor")

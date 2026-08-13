from __future__ import annotations

import argparse

from scripts.train_visual_semantic_raster_v32 import (
    effective_arguments,
    instruction_microbatches_before,
    stage_stream,
    training_stages,
)


def _args(**overrides):
    values = {
        "smoke": False,
        "exploratory": False,
        "out": None,
        "num_workers": 4,
        "batch_size": 4,
        "gradient_accumulation": 16,
        "raster_warmup_updates": 2_000,
        "continuation_updates": 4_000,
        "instruction_updates": 6_000,
        "stage_warmup": 500,
        "reader_unfreeze_after": 3_000,
        "log_every": 10,
        "save_every": 500,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def test_v32_instruction_stream_is_exactly_three_to_one() -> None:
    assert [stage_stream("instruction", index) for index in range(8)] == [
        "instruction",
        "instruction",
        "instruction",
        "continuation",
        "instruction",
        "instruction",
        "instruction",
        "continuation",
    ]
    assert instruction_microbatches_before(0) == 0
    assert instruction_microbatches_before(3) == 3
    assert instruction_microbatches_before(4) == 3
    assert instruction_microbatches_before(16) == 12


def test_v32_smoke_exercises_all_three_stages() -> None:
    args = effective_arguments(_args(smoke=True))
    assert args.batch_size == 1
    assert args.gradient_accumulation == 1
    assert args.reader_unfreeze_after == 0
    assert [(stage.name, stage.updates) for stage in training_stages(args)] == [
        ("raster-warmup", 1),
        ("continuation", 1),
        ("instruction", 1),
    ]


def test_v32_evidence_schedule_contains_twelve_thousand_updates() -> None:
    stages = training_stages(_args())
    assert sum(stage.updates for stage in stages) == 12_000

from __future__ import annotations

import argparse
import random

import numpy as np
import torch

from scripts.train_causal_glyph_flow_v35 import (
    DatasetWindow,
    _capture_rng,
    _restore_rng,
    _stage_progress,
    _validate_resume,
    effective_arguments,
    training_stages,
)
from ilm.visual_lm.causal_glyph_flow import CausalGlyphFlowConfig


class _RangeDataset(torch.utils.data.Dataset):
    def __len__(self) -> int:
        return 10

    def __getitem__(self, index: int) -> dict[str, int]:
        return {"index": index}


def _args(**overrides) -> argparse.Namespace:
    values = {
        "smoke": False,
        "exploratory": False,
        "tiny_model": False,
        "random_foundations": False,
        "allow_failed_stage_a": False,
        "alignment_only": False,
        "num_workers": 4,
        "batch_size": 8,
        "gradient_accumulation": 8,
        "alignment_updates": 2_000,
        "continuation_updates": 8_000,
        "instruction_updates": 12_000,
        "alignment_adapter_lr": 3e-4,
        "continuation_head_lr": 1e-4,
        "continuation_core_lr": 1e-5,
        "instruction_head_lr": 8e-5,
        "instruction_core_lr": 8e-6,
        "stage_warmup": 500,
        "minimum_learning_rate_ratio": 0.10,
        "weight_decay": 0.05,
        "gradient_clip": 1.0,
        "ema_decay": 0.999,
        "seed": 20_263_500,
        "precision": "bf16",
        "gate_minimum_patches": 2_048,
        "log_every": 10,
        "save_every": 1_000,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def test_smoke_arguments_reduce_all_stages_without_changing_seed() -> None:
    args = effective_arguments(_args(smoke=True))
    assert args.alignment_updates == 1
    assert args.continuation_updates == 1
    assert args.instruction_updates == 1
    assert args.batch_size == 1
    assert args.gradient_accumulation == 1
    assert args.seed == 20_263_500
    assert args.allow_failed_stage_a is True


def test_stage_progress_maps_global_update_to_exact_stage_cursors() -> None:
    stages = training_stages(_args())
    progress = _stage_progress(2_137, stages)
    assert [(stage.name, completed) for stage, completed in progress] == [
        ("visual-interface-alignment", 2_000),
        ("public-causal-continuation", 137),
        ("instruction-and-copy", 0),
    ]


def test_dataset_window_resumes_at_exact_example() -> None:
    window = DatasetWindow(_RangeDataset(), start=4, count=3)
    assert len(window) == 3
    assert [window[index]["index"] for index in range(3)] == [4, 5, 6]


def test_rng_capture_and_restore_includes_dedicated_flow_generator() -> None:
    random.seed(35)
    np.random.seed(35)
    torch.manual_seed(35)
    flow = torch.Generator().manual_seed(35)
    state = _capture_rng(flow)
    expected = (
        random.random(),
        float(np.random.rand()),
        float(torch.rand(())),
        float(torch.rand((), generator=flow)),
    )
    _restore_rng(state, flow)
    observed = (
        random.random(),
        float(np.random.rand()),
        float(torch.rand(())),
        float(torch.rand((), generator=flow)),
    )
    assert observed == expected


def test_resume_rejects_changed_effective_arguments() -> None:
    arguments = vars(_args()) | {"resume": None}
    checkpoint = {
        "experiment": "causal-glyph-flow-v35",
        "architecture": "causal-glyph-flow-v35",
        "protocol": {
            "sha256": "d7a4d49270676cd82c55e22ddd73466966e0b96723970f76fe66fa2381bd3718"
        },
        "model_config": CausalGlyphFlowConfig().__dict__,
        "optimizer": {},
        "rng": {},
        "resumable": True,
        "run_receipt": {
            "source_sha256": {"source": "hash"},
            "data": {"data": "hash"},
            "arguments": arguments,
        },
    }
    _validate_resume(
        checkpoint,
        config=CausalGlyphFlowConfig(),
        source_hashes={"source": "hash"},
        data_receipt={"data": "hash"},
        arguments=arguments,
    )
    try:
        _validate_resume(
            checkpoint,
            config=CausalGlyphFlowConfig(),
            source_hashes={"source": "hash"},
            data_receipt={"data": "hash"},
            arguments=arguments | {"batch_size": 4},
        )
    except ValueError as error:
        assert "effective arguments" in str(error)
    else:
        raise AssertionError("V35 accepted a changed resume batch size")

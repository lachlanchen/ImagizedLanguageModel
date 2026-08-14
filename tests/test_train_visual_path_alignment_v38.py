from __future__ import annotations

import argparse

import pytest
import torch
import torch.nn.functional as F

from ilm.visual_lm.visual_path_alignment import VisualPathAlignmentOutput
from scripts.train_visual_path_alignment_v38 import (
    DatasetWindow,
    _stage_progress,
    candidate_seed,
    effective_arguments,
    require_preregistered_arguments,
    split_path_outputs,
    training_stages,
    validate_batch_geometry,
    validate_checkpoint_boundary,
)


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
        "random_foundation": False,
        "device": "cuda:0",
        "precision": "bf16",
        "num_workers": 4,
        "batch_size": 8,
        "realignment_updates": 500,
        "adaptation_updates": 7_500,
        "realignment_head_lr": 1e-4,
        "adaptation_head_lr": 5e-5,
        "adaptation_reader_lr": 5e-6,
        "realignment_effective_batch": 64,
        "adaptation_effective_batch": 64,
        "candidate_count": 512,
        "nearest_neighbors": 16,
        "negative_teacher_ceiling": 0.85,
        "stage_warmup": 200,
        "minimum_learning_rate_ratio": 0.10,
        "weight_decay": 0.05,
        "gradient_clip": 1.0,
        "ema_decay": 0.999,
        "seed": 20_263_800,
        "log_every": 10,
        "save_every": 500,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def _output(batch: int, dimension: int = 8) -> VisualPathAlignmentOutput:
    source = torch.randn(batch, dimension, requires_grad=True)
    prompt = F.normalize(source, dim=-1)
    answer = F.normalize(torch.roll(source, shifts=1, dims=-1), dim=-1)
    return VisualPathAlignmentOutput(
        prompt_state=prompt,
        answer_state=answer,
        length=source[:, 0].square(),
        prompt_features=source,
        answer_base=answer,
        answer_correction=source * 0.01,
        pooled_visual_state=source,
    )


def test_smoke_arguments_preserve_five_path_training() -> None:
    args = effective_arguments(_args(smoke=True))
    stages = training_stages(args)
    validate_batch_geometry(
        stages,
        batch_size=args.batch_size,
        candidate_count=args.candidate_count,
        nearest_neighbors=args.nearest_neighbors,
    )

    assert [stage.updates for stage in stages] == [1, 1]
    assert [stage.effective_batch for stage in stages] == [2, 2]
    assert args.candidate_count == 8
    assert args.nearest_neighbors == 1
    assert args.tiny_model and args.random_foundation


def test_evidence_is_blocked_until_protocol_is_frozen() -> None:
    with pytest.raises(RuntimeError, match="protocol has not been frozen"):
        require_preregistered_arguments(_args())


def test_stage_progress_and_dataset_window_resume_exactly() -> None:
    stages = training_stages(_args())
    progress = _stage_progress(637, stages)

    assert [(stage.name, complete) for stage, complete in progress] == [
        ("head-realignment", 500),
        ("full-path-adaptation", 137),
    ]
    window = DatasetWindow(_RangeDataset(), start=4, count=3)
    assert [window[index]["index"] for index in range(3)] == [4, 5, 6]


def test_concatenated_five_view_split_retains_gradients() -> None:
    combined = _output(10)
    views = split_path_outputs(combined, batch_size=2)

    assert len(views) == 5
    assert all(view.prompt_state.shape == (2, 8) for view in views)
    sum(view.answer_state.sum() for view in views).backward()
    assert combined.prompt_features.grad is not None


def test_candidate_seed_is_position_derived() -> None:
    assert candidate_seed(38, global_update=10, microbatch=2) == candidate_seed(
        38,
        global_update=10,
        microbatch=2,
    )
    assert candidate_seed(38, global_update=10, microbatch=2) != candidate_seed(
        38,
        global_update=10,
        microbatch=3,
    )


def test_checkpoint_boundary_rejects_all_detached_training_tensors() -> None:
    validate_checkpoint_boundary(
        {"model": {"reader.embeddings.patch_embeddings.weight": torch.ones(2)}}
    )
    for payload in (
        {"teacher_mean": torch.ones(2)},
        {"nearest_answer_indices": torch.ones(2)},
        {"candidate_bank": torch.ones(2)},
        {"answer_rotation": torch.eye(2)},
    ):
        with pytest.raises(ValueError):
            validate_checkpoint_boundary(payload)


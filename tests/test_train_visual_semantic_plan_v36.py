from __future__ import annotations

import argparse

import torch

from ilm.visual_lm.visual_semantic_plan import (
    VisualSemanticPlanConfig,
    VisualSemanticPlanOutput,
)
from ilm.visual_lm.visual_semantic_plan_training import VisualSemanticTeacherTargets
from scripts.train_visual_semantic_plan_v36 import (
    DatasetWindow,
    _stage_progress,
    concatenate_plan_outputs,
    concatenate_teacher_targets,
    effective_arguments,
    training_stages,
    validate_batch_geometry,
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
        "batch_size": 16,
        "alignment_updates": 2_000,
        "adaptation_updates": 4_000,
        "alignment_head_lr": 3e-4,
        "adaptation_head_lr": 8e-5,
        "adaptation_reader_lr": 8e-6,
        "alignment_effective_batch": 128,
        "adaptation_effective_batch": 64,
        "stage_warmup": 200,
        "minimum_learning_rate_ratio": 0.10,
        "weight_decay": 0.05,
        "gradient_clip": 1.0,
        "ema_decay": 0.999,
        "seed": 20_263_600,
        "log_every": 10,
        "save_every": 500,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def _output(batch: int, dim: int = 8) -> VisualSemanticPlanOutput:
    source = torch.randn(batch, 5, dim, requires_grad=True)
    plans = torch.nn.functional.normalize(source, dim=-1)
    length = source[:, 0, 0].square()
    return VisualSemanticPlanOutput(
        plans=plans,
        length=length,
        planner_hidden=source,
        reader_memory=source,
    )


def _targets(batch: int, dim: int = 8) -> VisualSemanticTeacherTargets:
    return VisualSemanticTeacherTargets(
        global_plan=torch.randn(batch, dim),
        chunk_plans=torch.randn(batch, 4, dim),
        chunk_active=torch.ones(batch, 4),
        length=torch.ones(batch),
    )


def test_smoke_arguments_preserve_two_example_contrastive_batch() -> None:
    args = effective_arguments(_args(smoke=True))
    stages = training_stages(args)
    validate_batch_geometry(stages, batch_size=args.batch_size)
    assert [stage.updates for stage in stages] == [1, 1]
    assert [stage.effective_batch for stage in stages] == [2, 2]
    assert args.tiny_model is True
    assert args.random_foundation is True


def test_stage_progress_and_dataset_window_resume_exactly() -> None:
    stages = training_stages(_args())
    progress = _stage_progress(2_137, stages)
    assert [(stage.name, complete) for stage, complete in progress] == [
        ("plan-alignment", 2_000),
        ("semantic-adaptation", 137),
    ]
    window = DatasetWindow(_RangeDataset(), start=4, count=3)
    assert [window[index]["index"] for index in range(3)] == [4, 5, 6]


def test_effective_batch_concatenation_retains_gradients_and_targets() -> None:
    first = _output(2)
    second = _output(2)
    combined = concatenate_plan_outputs([first, second])
    targets = concatenate_teacher_targets([_targets(2), _targets(2)])
    combined.plans.sum().backward()

    assert combined.plans.shape == (4, 5, 8)
    assert combined.length.shape == (4,)
    assert targets.global_plan.shape == (4, 8)
    assert first.planner_hidden.grad is not None
    assert second.planner_hidden.grad is not None


def test_production_configuration_stays_below_cap() -> None:
    config = VisualSemanticPlanConfig()
    assert config.production_reader is True

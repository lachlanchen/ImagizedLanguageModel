from __future__ import annotations

from argparse import Namespace

import pytest
import torch

from ilm.visual_lm.visual_answer_trajectory import (
    VisualAnswerEncoding,
    VisualAnswerTrajectoryConfig,
    VisualAnswerTrajectoryModel,
)
from ilm.visual_lm.visual_answer_trajectory_training import (
    visual_answer_trajectory_optimizer_groups,
)
from scripts.train_visual_answer_trajectory_v39 import (
    DatasetWindow,
    TrainingStage,
    _stage_progress,
    acquire_output_lock,
    candidate_seed,
    set_optimizer_learning_rates,
    split_visual_encoding,
    stage_cosine_learning_rate,
    training_stages,
    validate_training_geometry,
)


class IntegerDataset(torch.utils.data.Dataset):
    def __len__(self) -> int:
        return 20

    def __getitem__(self, index: int) -> int:
        return index


def test_dataset_window_and_stage_progress_resume_exactly() -> None:
    window = DatasetWindow(IntegerDataset(), start=5, count=4)
    stages = (
        TrainingStage("trajectory-head", 3, 1e-4, 0.0, 8),
        TrainingStage("full-path-adaptation", 7, 5e-5, 5e-6, 8),
    )

    assert [window[index] for index in range(len(window))] == [5, 6, 7, 8]
    assert [(stage.name, done) for stage, done in _stage_progress(5, stages)] == [
        ("trajectory-head", 3),
        ("full-path-adaptation", 2),
    ]


def test_learning_rate_warms_and_decays_to_floor() -> None:
    values = [
        stage_cosine_learning_rate(
            update,
            peak=1e-3,
            warmup=2,
            total=10,
            minimum_ratio=0.1,
        )
        for update in range(1, 11)
    ]

    assert values[0] == pytest.approx(5e-4)
    assert values[1] == pytest.approx(1e-3)
    assert values[-1] == pytest.approx(1e-4)
    assert all(left >= right for left, right in zip(values[1:], values[2:]))


def test_candidate_seed_is_position_deterministic() -> None:
    assert candidate_seed(39, global_update=4, microbatch=2) == candidate_seed(
        39,
        global_update=4,
        microbatch=2,
    )
    assert candidate_seed(39, global_update=4, microbatch=2) != candidate_seed(
        39,
        global_update=4,
        microbatch=3,
    )


def test_split_visual_encoding_preserves_view_order() -> None:
    batch = 2
    rows = 4 * batch
    encoding = VisualAnswerEncoding(
        read_state=torch.arange(rows * 8).reshape(rows, 8).float(),
        read_features=torch.arange(rows * 8).reshape(rows, 8).float(),
        patch_states=torch.arange(rows * 3 * 4).reshape(rows, 3, 4).float(),
        pooled_visual_state=torch.arange(rows * 4).reshape(rows, 4).float(),
    )

    views = split_visual_encoding(encoding, batch_size=batch)

    assert len(views) == 4
    assert torch.equal(views[0].read_state, encoding.read_state[:2])
    assert torch.equal(views[3].read_state, encoding.read_state[6:])


def test_optimizer_covers_frozen_then_unfrozen_reader() -> None:
    config = VisualAnswerTrajectoryConfig(
        reader_hidden_size=64,
        reader_layers=1,
        reader_heads=4,
        reader_intermediate_size=128,
        reader_dropout=0.0,
        projection_hidden_size=96,
        semantic_dim=64,
        projection_dropout=0.0,
        answer_hidden_size=32,
        planner_hidden_size=64,
        planner_layers=1,
        planner_heads=4,
        planner_intermediate_size=128,
        planner_dropout=0.0,
        length_hidden_size=16,
    )
    model = VisualAnswerTrajectoryModel(config)
    model.unfreeze_reader()
    groups = visual_answer_trajectory_optimizer_groups(
        model,
        head_learning_rate=1e-4,
        reader_learning_rate=1e-5,
        weight_decay=0.01,
    )
    optimizer = torch.optim.AdamW(groups)

    model.freeze_reader()
    set_optimizer_learning_rates(optimizer, head=2e-4, reader=0.0)
    assert any(group["role"] == "reader" for group in optimizer.param_groups)
    assert all(
        group["lr"] == 0.0
        for group in optimizer.param_groups
        if group["role"] in {"reader", "reader-head"}
    )
    model.unfreeze_reader()
    set_optimizer_learning_rates(optimizer, head=5e-5, reader=5e-6)
    assert all(
        group["lr"] == 5e-6
        for group in optimizer.param_groups
        if group["role"] in {"reader", "reader-head"}
    )


def test_training_geometry_rejects_undersized_span_candidates() -> None:
    stages = (
        TrainingStage("trajectory-head", 1, 1e-4, 0.0, 8),
        TrainingStage("full-path-adaptation", 1, 5e-5, 5e-6, 8),
    )

    validate_training_geometry(
        stages,
        physical_batch=4,
        global_candidates=64,
        segment_candidates=64,
        records=100,
        segments=1_000,
    )
    with pytest.raises(ValueError, match="segment candidate"):
        validate_training_geometry(
            stages,
            physical_batch=4,
            global_candidates=64,
            segment_candidates=63,
            records=100,
            segments=1_000,
        )


def test_training_stage_arguments_are_explicit() -> None:
    args = Namespace(
        head_updates=2,
        adaptation_updates=3,
        head_lr=2e-4,
        adaptation_head_lr=5e-5,
        adaptation_reader_lr=5e-6,
        head_effective_batch=16,
        adaptation_effective_batch=32,
    )

    stages = training_stages(args)

    assert [stage.name for stage in stages] == [
        "trajectory-head",
        "full-path-adaptation",
    ]
    assert [stage.effective_batch for stage in stages] == [16, 32]


def test_output_lock_rejects_concurrent_trainers(tmp_path) -> None:
    output = tmp_path / "run"
    first = acquire_output_lock(output)
    try:
        with pytest.raises(RuntimeError, match="already active"):
            acquire_output_lock(output)
    finally:
        first.close()

    second = acquire_output_lock(output)
    second.close()

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

from ilm.visual_lm.visual_semantic_distillation import (
    VisualSemanticDistillationOutput,
    file_sha256,
)
from ilm.visual_lm.visual_semantic_distillation_data import (
    load_v37_instruction_records,
)
from ilm.visual_lm.visual_semantic_distillation_training import (
    VisualSemanticDistillationTargetBank,
)
from scripts.train_visual_semantic_distillation_v37 import (
    EXPECTED_PROTOCOL_SHA256,
    DatasetWindow,
    _stage_progress,
    candidate_seed,
    effective_arguments,
    main,
    split_distillation_outputs,
    training_stages,
    validate_batch_geometry,
    validate_checkpoint_boundary,
    validate_development_target_bank,
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
        "warmup_updates": 500,
        "adaptation_updates": 7_500,
        "warmup_head_lr": 3e-4,
        "adaptation_head_lr": 1e-4,
        "adaptation_reader_lr": 1e-5,
        "warmup_effective_batch": 64,
        "adaptation_effective_batch": 64,
        "candidate_count": 512,
        "stage_warmup": 200,
        "minimum_learning_rate_ratio": 0.10,
        "weight_decay": 0.05,
        "gradient_clip": 1.0,
        "ema_decay": 0.999,
        "seed": 20_263_700,
        "log_every": 10,
        "save_every": 500,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def _output(batch: int, dimension: int = 8) -> VisualSemanticDistillationOutput:
    source = torch.randn(batch, dimension, requires_grad=True)
    normalized = F.normalize(source, dim=-1)
    return VisualSemanticDistillationOutput(
        semantic_state=normalized,
        answer_plan=normalized,
        length=source[:, 0].square(),
        semantic_features=source,
        scaled_residual=0.05 * normalized,
        pooled_visual_state=source,
    )


def _write_smoke_bank(path: Path, manifest: Path) -> None:
    records = load_v37_instruction_records(manifest)
    by_identifier = {record.identifier: record for record in records}
    selected = [
        by_identifier[identifier]
        for identifier in (
            "alpaca-zh:10",
            "alpaca-zh:13",
            "alpaca-zh:18",
            "alpaca-zh:33",
        )
    ]
    generator = torch.Generator().manual_seed(37)
    prompt = F.normalize(torch.randn(4, 1024, generator=generator), dim=-1)
    answer = F.normalize(torch.randn(4, 1024, generator=generator), dim=-1)
    bank = VisualSemanticDistillationTargetBank(
        identifiers=tuple(record.identifier for record in selected),
        prompt_targets=prompt.half(),
        answer_targets=answer.half(),
        lengths=torch.full((4,), 4.0, dtype=torch.float16),
        teacher_mean=torch.zeros(1024),
        receipt={
            "label": "smoke",
            "split": "train",
            "protocol": {"sha256": EXPECTED_PROTOCOL_SHA256},
            "data": {"sha256": file_sha256(manifest)},
            "teacher": {
                "route": "deterministic-synthetic-smoke-only",
                "evidence_eligible": False,
                "student_runtime_dependency": False,
            },
            "source_sha256": {},
            "strings_stored": False,
            "token_ids_stored": False,
        },
    )
    torch.save(bank.state_dict(), path)


def test_smoke_arguments_preserve_two_example_accumulation() -> None:
    args = effective_arguments(_args(smoke=True))
    stages = training_stages(args)
    validate_batch_geometry(
        stages,
        batch_size=args.batch_size,
        candidate_count=args.candidate_count,
    )
    assert [stage.updates for stage in stages] == [1, 1]
    assert [stage.effective_batch for stage in stages] == [2, 2]
    assert args.candidate_count == 4
    assert args.tiny_model is True
    assert args.random_foundation is True


def test_stage_progress_and_dataset_window_resume_exactly() -> None:
    stages = training_stages(_args())
    progress = _stage_progress(637, stages)
    assert [(stage.name, complete) for stage, complete in progress] == [
        ("projection-warmup", 500),
        ("full-visual-adaptation", 137),
    ]
    window = DatasetWindow(_RangeDataset(), start=4, count=3)
    assert [window[index]["index"] for index in range(3)] == [4, 5, 6]


def test_concatenated_view_split_retains_gradients() -> None:
    combined = _output(8)
    views = split_distillation_outputs(combined, batch_size=2)
    assert len(views) == 4
    assert all(view.semantic_state.shape == (2, 8) for view in views)
    sum(view.answer_plan.sum() for view in views).backward()
    assert combined.semantic_features.grad is not None


def test_candidate_seed_is_position_derived() -> None:
    assert candidate_seed(37, global_update=10, microbatch=2) == candidate_seed(
        37,
        global_update=10,
        microbatch=2,
    )
    assert candidate_seed(37, global_update=10, microbatch=2) != candidate_seed(
        37,
        global_update=10,
        microbatch=3,
    )


def test_checkpoint_boundary_allows_visual_embeddings_but_rejects_teacher() -> None:
    validate_checkpoint_boundary(
        {"model": {"reader.embeddings.patch_embeddings.weight": torch.ones(2)}}
    )
    try:
        validate_checkpoint_boundary({"teacher_mean": torch.ones(2)})
    except ValueError:
        pass
    else:
        raise AssertionError("V37 checkpoint accepted a teacher tensor")


def test_development_bank_is_bound_to_exact_train_bank(tmp_path: Path) -> None:
    manifest = Path("data/raw/alpaca_zh.json")
    train_path = tmp_path / "train.pt"
    _write_smoke_bank(train_path, manifest)
    train = VisualSemanticDistillationTargetBank.from_state_dict(
        torch.load(train_path, map_location="cpu", weights_only=False)
    )
    development = VisualSemanticDistillationTargetBank(
        identifiers=train.identifiers,
        prompt_targets=train.prompt_targets.clone(),
        answer_targets=train.answer_targets.clone(),
        lengths=train.lengths.clone(),
        teacher_mean=train.teacher_mean.clone(),
        receipt={
            "split": "development",
            "protocol": {"sha256": EXPECTED_PROTOCOL_SHA256},
            "data": {"sha256": file_sha256(manifest)},
            "train_bank_sha256": file_sha256(train_path),
        },
    )
    validate_development_target_bank(
        development,
        train,
        train_bank_sha256=file_sha256(train_path),
        manifest_sha256=file_sha256(manifest),
        evidence=False,
    )
    development.receipt["train_bank_sha256"] = "0" * 64
    try:
        validate_development_target_bank(
            development,
            train,
            train_bank_sha256=file_sha256(train_path),
            manifest_sha256=file_sha256(manifest),
            evidence=False,
        )
    except ValueError:
        pass
    else:
        raise AssertionError("V37 accepted a development bank centered elsewhere")


def test_smoke_training_and_exact_resume(tmp_path: Path, monkeypatch) -> None:
    manifest = Path("data/raw/alpaca_zh.json")
    bank = tmp_path / "targets.pt"
    output = tmp_path / "run"
    _write_smoke_bank(bank, manifest)
    arguments = [
        "train_visual_semantic_distillation_v37.py",
        "--smoke",
        "--instruction-manifest",
        str(manifest),
        "--target-bank",
        str(bank),
        "--out",
        str(output),
    ]
    monkeypatch.setattr(sys, "argv", arguments)
    main()

    checkpoint_path = output / "checkpoint_latest.pt"
    standalone_path = output / "student_ema.pt"
    assert checkpoint_path.is_file()
    assert standalone_path.is_file()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    standalone = torch.load(standalone_path, map_location="cpu", weights_only=False)
    assert checkpoint["global_update"] == 2
    assert checkpoint["data_positions"] == {
        "projection-warmup": 2,
        "full-visual-adaptation": 2,
    }
    assert checkpoint["candidate_sampling_position"] == 2
    assert standalone["global_update"] == 2
    validate_checkpoint_boundary(checkpoint)
    validate_checkpoint_boundary(standalone)

    monkeypatch.setattr(
        sys,
        "argv",
        arguments + ["--resume", str(checkpoint_path)],
    )
    main()
    resumed = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    assert resumed["global_update"] == 2

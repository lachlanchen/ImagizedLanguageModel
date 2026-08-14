#!/usr/bin/env python3
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import random
import signal
import tempfile
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from ilm.visual_lm.visual_answer_trajectory import (
    V39_ARCHITECTURE,
    VisualAnswerEncoding,
    VisualAnswerTrajectoryConfig,
    VisualAnswerTrajectoryModel,
    load_v39_v38_initialization,
    visual_answer_trajectory_boundary_receipt,
)
from ilm.visual_lm.visual_answer_trajectory_data import (
    VisualAnswerTrajectoryDataset,
    VisualAnswerTrajectoryRecord,
    load_v39_instruction_records,
    visual_answer_trajectory_collate,
    visual_answer_trajectory_data_boundary_receipt,
    visual_answer_trajectory_tensor_batch,
)
from ilm.visual_lm.visual_answer_trajectory_training import (
    VisualAnswerTrajectoryEMA,
    VisualAnswerTrajectoryLoss,
    VisualAnswerTrajectoryTargetBank,
    set_v39_stage_trainability,
    visual_answer_trajectory_loss,
    visual_answer_trajectory_optimizer_groups,
)
from ilm.visual_lm.visual_semantic_distillation import file_sha256
from ilm.visual_lm.visual_semantic_distillation_data import (
    V37_SEMANTIC_DIM,
    VisualSemanticDistillationRenderConfig,
)


EXPERIMENT = V39_ARCHITECTURE
DEFAULT_INSTRUCTION_MANIFEST = "data/raw/alpaca_zh.json"
EXPECTED_INSTRUCTION_SHA256 = (
    "6fcb98c6d79691d1f9a88ef513335da9124e7fdeef5103343f5d9f9a6e8f4903"
)
DEFAULT_TARGET_BANK = (
    "artifacts/visual_answer_trajectory_v39_targets/exploratory_512.pt"
)
DEFAULT_INITIALIZATION = "artifacts/visual_path_alignment_v38_20260814/student_ema.pt"
EXPECTED_INITIALIZATION_SHA256 = (
    "25e2fd2652db537455eec57502ffe9e4b51c9cf964311d681c7a2b6e429a8429"
)
SEED = 20_263_900
MAXIMUM_VRAM_BYTES = 20 * 1024**3
SOURCE_FILES = (
    "ilm/visual_lm/visual_answer_trajectory.py",
    "ilm/visual_lm/visual_answer_trajectory_data.py",
    "ilm/visual_lm/visual_answer_trajectory_training.py",
    "scripts/train_visual_answer_trajectory_v39.py",
)


@dataclass(frozen=True)
class TrainingStage:
    name: str
    updates: int
    head_learning_rate: float
    reader_learning_rate: float
    effective_batch: int


class DatasetWindow(Dataset[dict[str, Any]]):
    def __init__(
        self,
        dataset: Dataset[dict[str, Any]],
        *,
        start: int,
        count: int,
    ) -> None:
        if start < 0 or count < 1 or start + count > len(dataset):
            raise ValueError("V39 dataset window lies outside its deterministic stream")
        self.dataset = dataset
        self.start = int(start)
        self.count = int(count)

    def __len__(self) -> int:
        return self.count

    def __getitem__(self, index: int) -> dict[str, Any]:
        if not 0 <= index < self.count:
            raise IndexError(index)
        return self.dataset[self.start + index]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the V39 image-native visual answer trajectory model."
    )
    parser.add_argument("--instruction-manifest", default=DEFAULT_INSTRUCTION_MANIFEST)
    parser.add_argument("--target-bank", default=DEFAULT_TARGET_BANK)
    parser.add_argument("--initialization", default=DEFAULT_INITIALIZATION)
    parser.add_argument("--out", default="artifacts/visual_answer_trajectory_v39_exploratory")
    parser.add_argument("--resume")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--precision", choices=("fp32", "fp16", "bf16"), default="bf16")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--head-updates", type=int, default=500)
    parser.add_argument("--adaptation-updates", type=int, default=3_500)
    parser.add_argument("--head-lr", type=float, default=2e-4)
    parser.add_argument("--adaptation-head-lr", type=float, default=5e-5)
    parser.add_argument("--adaptation-reader-lr", type=float, default=5e-6)
    parser.add_argument("--head-effective-batch", type=int, default=64)
    parser.add_argument("--adaptation-effective-batch", type=int, default=64)
    parser.add_argument("--global-candidates", type=int, default=512)
    parser.add_argument("--segment-candidates", type=int, default=2_048)
    parser.add_argument("--stage-warmup", type=int, default=100)
    parser.add_argument("--minimum-learning-rate-ratio", type=float, default=0.10)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--save-every", type=int, default=250)
    parser.add_argument("--maximum-hours", type=float, default=24.0)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--exploratory", action="store_true")
    parser.add_argument("--tiny-model", action="store_true")
    parser.add_argument("--random-foundation", action="store_true")
    return parser.parse_args()


def effective_arguments(args: argparse.Namespace) -> argparse.Namespace:
    values = vars(args).copy()
    if args.smoke:
        values.update(
            {
                "device": "cpu",
                "precision": "fp32",
                "num_workers": 0,
                "batch_size": 2,
                "head_updates": 1,
                "adaptation_updates": 1,
                "head_effective_batch": 2,
                "adaptation_effective_batch": 2,
                "global_candidates": 8,
                "segment_candidates": 64,
                "stage_warmup": 0,
                "log_every": 1,
                "save_every": 1,
                "maximum_hours": 1.0,
                "tiny_model": True,
                "random_foundation": True,
            }
        )
    return argparse.Namespace(**values)


def training_stages(args: argparse.Namespace) -> tuple[TrainingStage, ...]:
    stages = (
        TrainingStage(
            "trajectory-head",
            args.head_updates,
            args.head_lr,
            0.0,
            args.head_effective_batch,
        ),
        TrainingStage(
            "full-path-adaptation",
            args.adaptation_updates,
            args.adaptation_head_lr,
            args.adaptation_reader_lr,
            args.adaptation_effective_batch,
        ),
    )
    if any(
        stage.updates < 1
        or stage.effective_batch < 2
        or min(stage.head_learning_rate, stage.reader_learning_rate) < 0
        for stage in stages
    ):
        raise ValueError("V39 training stages are invalid")
    return stages


def validate_training_geometry(
    stages: Sequence[TrainingStage],
    *,
    physical_batch: int,
    global_candidates: int,
    segment_candidates: int,
    records: int,
    segments: int,
) -> None:
    if physical_batch < 1:
        raise ValueError("V39 physical batch must be positive")
    if not physical_batch <= global_candidates <= records:
        raise ValueError("V39 global candidate geometry is invalid")
    if not 16 * physical_batch <= segment_candidates <= segments:
        raise ValueError("V39 segment candidate geometry is invalid")
    for stage in stages:
        if stage.effective_batch % physical_batch:
            raise ValueError(
                f"V39 {stage.name} effective batch must divide by physical batch"
            )


def v39_model_config(*, tiny: bool) -> VisualAnswerTrajectoryConfig:
    if not tiny:
        return VisualAnswerTrajectoryConfig()
    return VisualAnswerTrajectoryConfig(
        reader_hidden_size=64,
        reader_layers=2,
        reader_heads=4,
        reader_intermediate_size=128,
        reader_dropout=0.0,
        projection_hidden_size=128,
        semantic_dim=V37_SEMANTIC_DIM,
        projection_dropout=0.0,
        answer_hidden_size=64,
        planner_hidden_size=64,
        planner_layers=2,
        planner_heads=4,
        planner_intermediate_size=128,
        planner_dropout=0.0,
        length_hidden_size=32,
    )


def choose_device(value: str) -> torch.device:
    device = torch.device(value)
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda:0")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("V39 requested CUDA but CUDA is unavailable")
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("V39 supports CPU or CUDA training")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    return device


def autocast_context(device: torch.device, precision: str):
    if precision == "fp32" or device.type != "cuda":
        return nullcontext()
    dtype = torch.float16 if precision == "fp16" else torch.bfloat16
    return torch.autocast("cuda", dtype=dtype)


def atomic_write_json(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=path.name,
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_torch_save(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent,
        prefix=path.name,
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
    try:
        torch.save(dict(payload), temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def append_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def cpu_model_state(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().clone() for name, value in model.state_dict().items()
    }


def ema_model_state(
    model: VisualAnswerTrajectoryModel,
    ema: VisualAnswerTrajectoryEMA,
) -> dict[str, torch.Tensor]:
    state = cpu_model_state(model)
    for name, value in ema.shadow.items():
        state[name] = value.detach().cpu().clone()
    return state


def tensors_are_finite(value: Any) -> bool:
    if isinstance(value, torch.Tensor):
        return not value.is_floating_point() or bool(torch.isfinite(value).all())
    if isinstance(value, Mapping):
        return all(tensors_are_finite(item) for item in value.values())
    if isinstance(value, (tuple, list)):
        return all(tensors_are_finite(item) for item in value)
    return True


def _capture_rng(device: torch.device) -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state() if device.type == "cuda" else None,
        "torch_cuda_device": str(device) if device.type == "cuda" else None,
    }


def _restore_rng(state: Mapping[str, Any], device: torch.device) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if device.type == "cuda":
        if state.get("torch_cuda_device") != str(device):
            raise ValueError("V39 resume CUDA device changed")
        torch.cuda.set_rng_state(state["torch_cuda"])


def _stage_progress(
    global_update: int,
    stages: Sequence[TrainingStage],
) -> list[tuple[TrainingStage, int]]:
    total = sum(stage.updates for stage in stages)
    if not 0 <= global_update <= total:
        raise ValueError("V39 global update lies outside the training plan")
    result: list[tuple[TrainingStage, int]] = []
    prior = 0
    for stage in stages:
        completed = min(stage.updates, max(0, global_update - prior))
        result.append((stage, completed))
        prior += stage.updates
    return result


def stage_cosine_learning_rate(
    update: int,
    *,
    peak: float,
    warmup: int,
    total: int,
    minimum_ratio: float,
) -> float:
    if not 1 <= update <= total or not 0 <= warmup < total:
        raise ValueError("V39 learning-rate position is invalid")
    if peak < 0 or not 0 <= minimum_ratio <= 1:
        raise ValueError("V39 learning-rate values are invalid")
    if warmup and update <= warmup:
        return peak * update / warmup
    progress = (update - warmup) / max(1, total - warmup)
    cosine = 0.5 * (1 + math.cos(math.pi * progress))
    return peak * (minimum_ratio + (1 - minimum_ratio) * cosine)


def set_optimizer_learning_rates(
    optimizer: torch.optim.Optimizer,
    *,
    head: float,
    reader: float,
) -> None:
    for group in optimizer.param_groups:
        role = str(group.get("role", ""))
        if role == "trajectory":
            group["lr"] = head
        elif role in {"reader", "reader-head"}:
            group["lr"] = reader
        else:
            raise ValueError(f"V39 optimizer group has an unknown role: {role!r}")


def candidate_seed(seed: int, *, global_update: int, microbatch: int) -> int:
    if global_update < 0 or microbatch < 0:
        raise ValueError("V39 candidate position cannot be negative")
    return int(seed) + (global_update + 1) * 1_000_003 + microbatch * 97


def _stage_loader(
    dataset: Dataset[dict[str, Any]],
    *,
    stage: TrainingStage,
    completed_updates: int,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    seed: int,
) -> Iterator[dict[str, Any]]:
    consumed = completed_updates * stage.effective_batch
    remaining = (stage.updates - completed_updates) * stage.effective_batch
    window = DatasetWindow(dataset, start=consumed, count=remaining)
    return iter(
        DataLoader(
            window,
            batch_size=batch_size,
            shuffle=False,
            drop_last=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=num_workers > 0,
            collate_fn=visual_answer_trajectory_collate,
            generator=torch.Generator().manual_seed(seed),
        )
    )


def split_visual_encoding(
    encoding: VisualAnswerEncoding,
    *,
    batch_size: int,
) -> tuple[VisualAnswerEncoding, VisualAnswerEncoding, VisualAnswerEncoding, VisualAnswerEncoding]:
    if batch_size < 1 or encoding.read_state.shape[0] != 4 * batch_size:
        raise ValueError("V39 concatenated encoding has another batch geometry")

    def view(index: int) -> VisualAnswerEncoding:
        start = index * batch_size
        stop = start + batch_size
        return VisualAnswerEncoding(
            read_state=encoding.read_state[start:stop],
            read_features=encoding.read_features[start:stop],
            patch_states=encoding.patch_states[start:stop],
            pooled_visual_state=encoding.pooled_visual_state[start:stop],
        )

    return view(0), view(1), view(2), view(3)


def loss_metrics(loss: VisualAnswerTrajectoryLoss) -> dict[str, float]:
    result: dict[str, float] = {}
    for field in fields(loss):
        value = getattr(loss, field.name)
        if not isinstance(value, torch.Tensor) or value.numel() != 1:
            raise TypeError(f"V39 loss metric {field.name!r} is not scalar")
        result[field.name] = float(value.detach().cpu())
    return result


def load_target_bank(path: str | Path) -> VisualAnswerTrajectoryTargetBank:
    state = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(state, Mapping):
        raise TypeError("V39 target bank must contain a state mapping")
    return VisualAnswerTrajectoryTargetBank.from_state_dict(state)


def validate_target_bank(
    bank: VisualAnswerTrajectoryTargetBank,
    *,
    manifest_sha256: str,
) -> None:
    receipt = bank.receipt
    if receipt.get("split") != "train":
        raise ValueError("V39 training requires a train target bank")
    if receipt.get("data", {}).get("sha256") != manifest_sha256:
        raise ValueError("V39 target bank has another instruction corpus")
    if bank.prompt_targets.shape[1] != V37_SEMANTIC_DIM:
        raise ValueError("V39 target width differs from the model")
    if receipt.get("source_text_strings_stored", True):
        raise ValueError("V39 target bank declares stored source text")
    if receipt.get("token_ids_stored", True) or receipt.get("unicode_ids_stored", True):
        raise ValueError("V39 target bank declares symbolic target payloads")
    if receipt.get("student_runtime_teacher_dependency", True):
        raise ValueError("V39 target bank declares a runtime teacher")


def select_bank_records(
    records: Sequence[VisualAnswerTrajectoryRecord],
    bank: VisualAnswerTrajectoryTargetBank,
) -> list[VisualAnswerTrajectoryRecord]:
    by_identifier = {record.identifier: record for record in records}
    try:
        return [by_identifier[identifier] for identifier in bank.identifiers]
    except KeyError as error:
        raise KeyError(f"V39 corpus lacks target record {error.args[0]!r}") from error


def selected_identifier_sha256(identifiers: Sequence[str]) -> str:
    return hashlib.sha256("\n".join(identifiers).encode()).hexdigest()


def optimizer_receipt(
    model: nn.Module,
    groups: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    names = {id(parameter): name for name, parameter in model.named_parameters()}
    return {
        "type": "AdamW",
        "betas": [0.9, 0.95],
        "groups": [
            {
                "role": group["role"],
                "decay": bool(group["decay"]),
                "initial_lr": float(group["lr"]),
                "weight_decay": float(group["weight_decay"]),
                "parameters": len(group["params"]),
                "elements": sum(parameter.numel() for parameter in group["params"]),
                "names_sha256": selected_identifier_sha256(
                    tuple(names[id(parameter)] for parameter in group["params"])
                ),
            }
            for group in groups
        ],
    }


def validate_checkpoint_boundary(payload: Mapping[str, Any]) -> None:
    forbidden_true = (
        "contains_target_tensors",
        "contains_teacher_model",
        "contains_candidate_tensors",
        "contains_source_language_strings",
    )
    if any(bool(payload.get(name, True)) for name in forbidden_true):
        raise ValueError("V39 checkpoint boundary contains a forbidden payload")
    if "finite" in payload and not bool(payload["finite"]):
        raise FloatingPointError("V39 checkpoint finite audit failed")
    if not tensors_are_finite(payload.get("model", {})):
        raise FloatingPointError("V39 checkpoint model is non-finite")


def acquire_output_lock(output: Path):
    lock_path = output.with_name(f"{output.name}.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    handle = lock_path.open("a+", encoding="utf-8")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        handle.close()
        raise RuntimeError(f"V39 training output is already active: {output}") from error
    return handle


def _validate_resume(
    checkpoint: Mapping[str, Any],
    *,
    config: VisualAnswerTrajectoryConfig,
    arguments: Mapping[str, Any],
    stages: Sequence[TrainingStage],
    target_bank_sha256: str,
    instruction_sha256: str,
    source_hashes: Mapping[str, str],
) -> None:
    if checkpoint.get("architecture") != V39_ARCHITECTURE:
        raise ValueError("V39 resume checkpoint has another architecture")
    if checkpoint.get("model_config") != asdict(config):
        raise ValueError("V39 resume model configuration changed")
    run_receipt = checkpoint.get("run_receipt", {})
    prior_arguments = dict(run_receipt.get("arguments", {}))
    current_arguments = dict(arguments)
    prior_arguments["resume"] = None
    current_arguments["resume"] = None
    if prior_arguments != current_arguments:
        raise ValueError("V39 resume arguments changed")
    if run_receipt.get("stages") != [asdict(stage) for stage in stages]:
        raise ValueError("V39 resume stages changed")
    if run_receipt.get("source_sha256") != dict(source_hashes):
        raise ValueError("V39 resume source code changed")
    data = run_receipt.get("data", {})
    if data.get("target_bank_sha256") != target_bank_sha256:
        raise ValueError("V39 resume target bank changed")
    if data.get("instruction_sha256") != instruction_sha256:
        raise ValueError("V39 resume instruction corpus changed")


def main() -> None:
    args = effective_arguments(parse_args())
    if not args.smoke and not args.exploratory:
        raise RuntimeError("V39 has no frozen evidence protocol; use --exploratory")
    if args.exploratory and args.smoke:
        raise ValueError("V39 smoke and exploratory labels are mutually exclusive")
    positive_integer_settings = (
        args.batch_size,
        args.log_every,
        args.save_every,
    )
    if min(positive_integer_settings) < 1 or min(args.num_workers, args.stage_warmup) < 0:
        raise ValueError("V39 training arguments are invalid")
    if not 0 <= args.minimum_learning_rate_ratio <= 1 or args.gradient_clip <= 0:
        raise ValueError("V39 schedule or gradient settings are invalid")
    if not 0 < args.ema_decay < 1 or args.maximum_hours <= 0:
        raise ValueError("V39 EMA or wall-time settings are invalid")

    stages = training_stages(args)
    device = choose_device(args.device)
    if device.type == "cpu" and args.precision != "fp32":
        raise ValueError("V39 CPU training requires --precision fp32")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.cuda.reset_peak_memory_stats()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    instruction_sha256 = file_sha256(args.instruction_manifest)
    if not args.smoke and instruction_sha256 != EXPECTED_INSTRUCTION_SHA256:
        raise RuntimeError("V39 instruction corpus hash changed")
    target_bank_sha256 = file_sha256(args.target_bank)
    bank = load_target_bank(args.target_bank)
    validate_target_bank(bank, manifest_sha256=instruction_sha256)
    validate_training_geometry(
        stages,
        physical_batch=args.batch_size,
        global_candidates=args.global_candidates,
        segment_candidates=args.segment_candidates,
        records=len(bank.identifiers),
        segments=bank.segment_targets.shape[0],
    )
    all_records = load_v39_instruction_records(args.instruction_manifest)
    selected_records = select_bank_records(all_records, bank)
    identifier_hash = selected_identifier_sha256(bank.identifiers)
    render_config = VisualSemanticDistillationRenderConfig(augment=True)
    datasets = {
        stage.name: VisualAnswerTrajectoryDataset(
            selected_records,
            render_config=render_config,
            seed=args.seed + stage_index * 10_000_000,
            length=stage.updates * stage.effective_batch,
        )
        for stage_index, stage in enumerate(stages)
    }

    config = v39_model_config(tiny=args.tiny_model)
    model = VisualAnswerTrajectoryModel(config)
    if args.random_foundation:
        if not args.smoke:
            raise ValueError("V39 random foundation is smoke-only")
        initialization: dict[str, Any] = {
            "route": "random-tiny-smoke-with-identity-global-map",
            "evidence_eligible": False,
        }
    else:
        initialization = load_v39_v38_initialization(
            model,
            args.initialization,
            expected_sha256=EXPECTED_INITIALIZATION_SHA256,
        )
    boundary = visual_answer_trajectory_boundary_receipt(model)
    if boundary["forbidden_parameter_names"] or not boundary["parameter_cap_pass"]:
        raise RuntimeError("V39 deployable model boundary is invalid")

    model.to(device)
    model.unfreeze_reader()
    groups = visual_answer_trajectory_optimizer_groups(
        model,
        head_learning_rate=args.head_lr,
        reader_learning_rate=args.adaptation_reader_lr,
        weight_decay=args.weight_decay,
    )
    optimizer_definition = optimizer_receipt(model, groups)
    optimizer = torch.optim.AdamW(
        groups,
        betas=(0.9, 0.95),
        fused=device.type == "cuda",
    )
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=device.type == "cuda" and args.precision == "fp16",
    )
    ema_names = tuple(name for name, _parameter in model.named_parameters())
    ema = VisualAnswerTrajectoryEMA(model, ema_names, decay=args.ema_decay)
    set_v39_stage_trainability(model, "trajectory-head")

    output = Path(args.out)
    output_lock = acquire_output_lock(output)
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output / "checkpoint_latest.pt"
    standalone_path = output / "student_ema.pt"
    metrics_path = output / "metrics.jsonl"
    source_hashes = {path: file_sha256(path) for path in SOURCE_FILES}
    data_receipt = {
        "instruction_manifest": str(Path(args.instruction_manifest).resolve()),
        "instruction_sha256": instruction_sha256,
        "target_bank": str(Path(args.target_bank).resolve()),
        "target_bank_sha256": target_bank_sha256,
        "records": len(bank.identifiers),
        "segments": bank.segment_targets.shape[0],
        "identifiers_sha256": identifier_hash,
        "target_label": bank.receipt.get("label"),
        "target_strings_stored": bank.receipt.get("source_text_strings_stored"),
        "runtime_teacher_dependency": bank.receipt.get(
            "student_runtime_teacher_dependency"
        ),
    }
    run_receipt: dict[str, Any] = {
        "experiment": EXPERIMENT,
        "label": "smoke" if args.smoke else "exploratory",
        "architecture": V39_ARCHITECTURE,
        "arguments": vars(args) | {"resume": None},
        "stages": [asdict(stage) for stage in stages],
        "source_sha256": source_hashes,
        "data": data_receipt,
        "initialization": initialization,
        "model_boundary": boundary,
        "data_boundary": visual_answer_trajectory_data_boundary_receipt(),
        "optimizer": optimizer_definition,
        "ema_parameter_names_sha256": selected_identifier_sha256(ema_names),
        "target_tensors_in_checkpoint": False,
        "teacher_model_in_student_process": False,
        "candidate_tensors_in_checkpoint": False,
        "source_language_strings_in_checkpoint": False,
        "effective_batch_preserved_by_gradient_accumulation": True,
        "four_visual_paths_concatenated_for_one_reader_pass": True,
        "device": str(device),
        "device_name": (
            torch.cuda.get_device_name() if device.type == "cuda" else "cpu"
        ),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "maximum_vram_bytes": MAXIMUM_VRAM_BYTES,
        "derived_checkpoint_license": "local research only; no redistribution",
    }

    global_update = 0
    elapsed_before = 0.0
    stage_summaries: dict[str, dict[str, Any]] = {}
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        if not isinstance(checkpoint, Mapping):
            raise TypeError("V39 resume checkpoint must contain a mapping")
        _validate_resume(
            checkpoint,
            config=config,
            arguments=vars(args),
            stages=stages,
            target_bank_sha256=target_bank_sha256,
            instruction_sha256=instruction_sha256,
            source_hashes=source_hashes,
        )
        model.load_state_dict(checkpoint["model"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        scaler.load_state_dict(checkpoint.get("scaler", {}))
        ema.load_state_dict(checkpoint["ema"])
        global_update = int(checkpoint["global_update"])
        elapsed_before = float(checkpoint.get("training_elapsed_seconds", 0.0))
        stage_summaries = dict(checkpoint.get("stage_summaries", {}))
        _restore_rng(checkpoint["rng"], device)
        run_receipt = dict(checkpoint["run_receipt"])
    else:
        atomic_write_json(run_receipt, output / "run_receipt.json")
        print(json.dumps(run_receipt, ensure_ascii=False, indent=2), flush=True)

    stop_requested = False
    stop_reason: str | None = None

    def request_stop(signum: int, _frame: Any) -> None:
        nonlocal stop_requested, stop_reason
        stop_requested = True
        stop_reason = f"signal-{signum}"
        print(f"V39 received signal {signum}; saving after this update", flush=True)

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    started = time.monotonic()

    def elapsed_seconds() -> float:
        return elapsed_before + time.monotonic() - started

    def save_checkpoint() -> None:
        payload = {
            "experiment": EXPERIMENT,
            "architecture": V39_ARCHITECTURE,
            "model_config": asdict(config),
            "model": cpu_model_state(model),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "ema": ema.state_dict(),
            "global_update": global_update,
            "data_positions": {
                stage.name: completed * stage.effective_batch
                for stage, completed in _stage_progress(global_update, stages)
            },
            "candidate_sampling_position": global_update,
            "training_elapsed_seconds": elapsed_seconds(),
            "stage_summaries": stage_summaries,
            "run_receipt": run_receipt,
            "rng": _capture_rng(device),
            "finite": tensors_are_finite(model.state_dict())
            and tensors_are_finite(optimizer.state_dict())
            and tensors_are_finite(ema.state_dict(cpu=False)),
            "peak_allocated_vram_bytes": (
                torch.cuda.max_memory_allocated() if device.type == "cuda" else 0
            ),
            "resumable": True,
            "contains_target_tensors": False,
            "contains_teacher_model": False,
            "contains_candidate_tensors": False,
            "contains_source_language_strings": False,
        }
        validate_checkpoint_boundary(payload)
        atomic_torch_save(payload, checkpoint_path)

    for stage_index, (stage, completed_updates) in enumerate(
        _stage_progress(global_update, stages)
    ):
        if completed_updates >= stage.updates:
            continue
        set_v39_stage_trainability(model, stage.name)
        model.train()
        microbatches = stage.effective_batch // args.batch_size
        loader = _stage_loader(
            datasets[stage.name],
            stage=stage,
            completed_updates=completed_updates,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
            seed=args.seed + stage_index * 10_000,
        )
        stage_started = time.monotonic()
        latest_metrics: dict[str, Any] = {}
        stage_update = completed_updates
        for stage_update in range(completed_updates + 1, stage.updates + 1):
            if elapsed_seconds() >= args.maximum_hours * 3600:
                stop_requested = True
                stop_reason = "maximum-hours"
                break
            warmup = min(args.stage_warmup, max(0, stage.updates - 1))
            head_lr = stage_cosine_learning_rate(
                stage_update,
                peak=stage.head_learning_rate,
                warmup=warmup,
                total=stage.updates,
                minimum_ratio=args.minimum_learning_rate_ratio,
            )
            reader_lr = stage_cosine_learning_rate(
                stage_update,
                peak=stage.reader_learning_rate,
                warmup=warmup,
                total=stage.updates,
                minimum_ratio=args.minimum_learning_rate_ratio,
            )
            set_optimizer_learning_rates(optimizer, head=head_lr, reader=reader_lr)
            optimizer.zero_grad(set_to_none=True)
            accumulated: dict[str, float] = {}
            effective_identifiers: list[str] = []
            for microbatch_index in range(microbatches):
                raw_batch = next(loader)
                tensor_batch = visual_answer_trajectory_tensor_batch(raw_batch)
                batch_identifiers = tuple(str(value) for value in raw_batch["identifiers"])
                batch_size = len(batch_identifiers)
                targets = bank.lookup(
                    batch_identifiers,
                    tensor_batch["segment_indices"],
                    device=device,
                    dtype=torch.float32,
                )
                seed = candidate_seed(
                    args.seed,
                    global_update=global_update,
                    microbatch=microbatch_index,
                )
                global_candidates = bank.global_candidate_set(
                    targets.bank_indices,
                    count=args.global_candidates,
                    seed=seed,
                    device=device,
                    dtype=torch.float32,
                )
                segment_candidates = bank.segment_candidate_set(
                    targets,
                    count=args.segment_candidates,
                    seed=seed + 43,
                    device=device,
                    dtype=torch.float32,
                )
                view_keys = (
                    ("prompt_anchor_pixels", "prompt_anchor_mask"),
                    ("prompt_view_pixels", "prompt_view_mask"),
                    ("segment_anchor_pixels", "segment_anchor_mask"),
                    ("segment_view_pixels", "segment_view_mask"),
                )
                pixels = torch.cat([tensor_batch[pixel] for pixel, _ in view_keys]).to(
                    device,
                    non_blocking=device.type == "cuda",
                )
                masks = torch.cat([tensor_batch[mask] for _, mask in view_keys]).to(
                    device,
                    non_blocking=device.type == "cuda",
                )
                with autocast_context(device, args.precision):
                    combined = model.encode_visual(pixels, masks)
                    prompt_anchor_encoding, prompt_view_encoding, segment_anchor, segment_view = (
                        split_visual_encoding(combined, batch_size=batch_size)
                    )
                    prompt_anchor = model.plan_from_encoding(
                        prompt_anchor_encoding,
                        masks[:batch_size],
                    )
                    prompt_view = model.plan_from_encoding(
                        prompt_view_encoding,
                        masks[batch_size : 2 * batch_size],
                    )
                    losses = visual_answer_trajectory_loss(
                        prompt_anchor,
                        prompt_view,
                        segment_anchor,
                        segment_view,
                        targets,
                        global_candidates,
                        segment_candidates,
                    )
                    scaled_loss = losses.loss / microbatches
                if not bool(torch.isfinite(losses.loss)):
                    raise FloatingPointError("V39 encountered a non-finite loss")
                if scaler.is_enabled():
                    scaler.scale(scaled_loss).backward()
                else:
                    scaled_loss.backward()
                for name, value in loss_metrics(losses).items():
                    accumulated[name] = accumulated.get(name, 0.0) + value / microbatches
                effective_identifiers.extend(batch_identifiers)

            if len(set(effective_identifiers)) != len(effective_identifiers):
                raise RuntimeError("V39 effective batch contains duplicate positives")
            if scaler.is_enabled():
                scaler.unscale_(optimizer)
            trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                trainable,
                args.gradient_clip,
            )
            if not bool(torch.isfinite(gradient_norm)):
                raise FloatingPointError("V39 encountered a non-finite gradient")
            if scaler.is_enabled():
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            if not tensors_are_finite(
                {name: parameter for name, parameter in model.named_parameters() if parameter.requires_grad}
            ):
                raise FloatingPointError("V39 encountered a non-finite parameter")
            ema.update(model)
            global_update += 1
            peak_vram = torch.cuda.max_memory_allocated() if device.type == "cuda" else 0
            if peak_vram >= MAXIMUM_VRAM_BYTES:
                raise MemoryError("V39 reached the fixed 20 GiB VRAM stop rule")
            latest_metrics = {
                "global_update": global_update,
                "stage": stage.name,
                "stage_update": stage_update,
                "physical_batch": args.batch_size,
                "microbatches": microbatches,
                "effective_batch": stage.effective_batch,
                "global_candidates": args.global_candidates,
                "segment_candidates": args.segment_candidates,
                "head_lr": head_lr,
                "reader_lr": reader_lr,
                "gradient_norm": float(gradient_norm),
                "peak_allocated_vram_bytes": peak_vram,
                "elapsed_seconds": elapsed_seconds(),
                **accumulated,
            }
            append_jsonl(metrics_path, latest_metrics)
            if stage_update == 1 or stage_update % args.log_every == 0:
                print(json.dumps(latest_metrics, sort_keys=True), flush=True)
            if global_update % args.save_every == 0 or stop_requested:
                save_checkpoint()
            if stop_requested:
                break
        completed_now = dict(
            (item.name, completed) for item, completed in _stage_progress(global_update, stages)
        )[stage.name]
        stage_complete = completed_now == stage.updates and not stop_requested
        stage_summaries[stage.name] = {
            "updates_completed": completed_now,
            "updates_planned": stage.updates,
            "examples_consumed": completed_now * stage.effective_batch,
            "elapsed_seconds": time.monotonic() - stage_started,
            "latest_metrics": latest_metrics,
            "complete": stage_complete,
        }
        save_checkpoint()
        if stop_requested:
            break

    save_checkpoint()
    complete = global_update == sum(stage.updates for stage in stages)
    if complete:
        standalone = {
            "experiment": EXPERIMENT,
            "architecture": V39_ARCHITECTURE,
            "weight_route": "all-parameter-ema",
            "model_config": asdict(config),
            "model": ema_model_state(model, ema),
            "global_update": global_update,
            "boundary": visual_answer_trajectory_boundary_receipt(model),
            "source_sha256": source_hashes,
            "training_data_sha256": instruction_sha256,
            "training_target_bank_sha256": target_bank_sha256,
            "selected_identifier_sha256": identifier_hash,
            "initialization_sha256": (
                None if args.random_foundation else file_sha256(args.initialization)
            ),
            "contains_target_tensors": False,
            "contains_teacher_model": False,
            "contains_candidate_tensors": False,
            "contains_source_language_strings": False,
            "license": "local research only; no redistribution",
        }
        validate_checkpoint_boundary(standalone)
        atomic_torch_save(standalone, standalone_path)
    summary = {
        "experiment": EXPERIMENT,
        "global_update": global_update,
        "planned_updates": sum(stage.updates for stage in stages),
        "complete": complete,
        "stage_summaries": stage_summaries,
        "training_elapsed_seconds": elapsed_seconds(),
        "peak_allocated_vram_bytes": (
            torch.cuda.max_memory_allocated() if device.type == "cuda" else 0
        ),
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "standalone_checkpoint": str(standalone_path.resolve()) if complete else None,
        "standalone_checkpoint_sha256": (
            file_sha256(standalone_path) if complete else None
        ),
        "stopped": stop_requested,
        "stop_reason": stop_reason,
    }
    atomic_write_json(summary, output / "training_summary.json")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    output_lock.close()


if __name__ == "__main__":
    main()

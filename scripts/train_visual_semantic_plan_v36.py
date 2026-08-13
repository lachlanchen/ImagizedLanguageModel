#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import random
import signal
import tempfile
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from ilm.visual_lm.direct_visual_patch_training import stage_cosine_learning_rate
from ilm.visual_lm.visual_semantic_plan import (
    PIXEL_LINGUIST_REVISION,
    PIXEL_LINGUIST_WEIGHT_SHA256,
    V36_ARCHITECTURE,
    VisualSemanticPlanConfig,
    VisualSemanticPlanModel,
    VisualSemanticPlanOutput,
    file_sha256,
    load_pixel_linguist_reader,
    resolve_pixel_linguist_checkpoint,
    visual_semantic_plan_boundary_receipt,
)
from ilm.visual_lm.visual_semantic_plan_data import (
    VisualSemanticPlanPromptDataset,
    VisualSemanticPlanRenderConfig,
    load_v36_instruction_records,
    visual_semantic_plan_data_boundary_receipt,
    visual_semantic_plan_prompt_collate,
)
from ilm.visual_lm.visual_semantic_plan_training import (
    SelectiveExponentialMovingAverage,
    VisualSemanticPlanLoss,
    VisualSemanticPlanTargetBank,
    VisualSemanticTeacherTargets,
    set_v36_optimizer_learning_rates,
    set_v36_stage_trainability,
    v36_optimizer_receipt,
    visual_semantic_plan_loss,
    visual_semantic_plan_optimizer_groups,
)
from ilm.visual_lm.visual_semantic_raster_data import VisualRasterRecord


EXPERIMENT = "visual-semantic-plan-v36"
PROTOCOL_DOCUMENT = "references/visual_semantic_plan_v36_protocol.md"
EXPECTED_PROTOCOL_SHA256 = (
    "7e637698af08803c4ef509c564160ea63e5a952398a1e50cd924ec888167d6fb"
)
DEFAULT_INSTRUCTION_MANIFEST = "data/raw/alpaca_zh.json"
DEFAULT_TARGET_BANK = "artifacts/visual_semantic_plan_v36_targets/train.pt"
EXPECTED_INSTRUCTION_SHA256 = (
    "6fcb98c6d79691d1f9a88ef513335da9124e7fdeef5103343f5d9f9a6e8f4903"
)
SEED = 20_263_600
MAXIMUM_VRAM_BYTES = 20 * 1024**3
SOURCE_FILES = (
    "ilm/visual_lm/visual_semantic_plan.py",
    "ilm/visual_lm/visual_semantic_plan_data.py",
    "ilm/visual_lm/visual_semantic_plan_training.py",
    "scripts/train_visual_semantic_plan_v36.py",
)


@dataclass(frozen=True)
class TrainingStage:
    name: str
    updates: int
    head_learning_rate: float
    reader_learning_rate: float
    effective_batch: int


FIXED_STAGES = (
    TrainingStage("plan-alignment", 2_000, 3e-4, 0.0, 128),
    TrainingStage("semantic-adaptation", 4_000, 8e-5, 8e-6, 64),
)
FIXED_OPTIMIZATION: dict[str, Any] = {
    "precision": "bf16",
    "stage_warmup": 200,
    "minimum_learning_rate_ratio": 0.10,
    "weight_decay": 0.05,
    "gradient_clip": 1.0,
    "ema_decay": 0.999,
    "seed": SEED,
    "save_every": 500,
}


class DatasetWindow(Dataset[dict[str, Any]]):
    def __init__(self, dataset: Dataset[dict[str, Any]], *, start: int, count: int) -> None:
        if start < 0 or count < 1 or start + count > len(dataset):
            raise ValueError("V36 dataset window lies outside its deterministic stream")
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
        description="Train the preregistered V36 visual semantic planner."
    )
    parser.add_argument("--instruction-manifest", default=DEFAULT_INSTRUCTION_MANIFEST)
    parser.add_argument("--target-bank", default=DEFAULT_TARGET_BANK)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument(
        "--out",
        default="artifacts/visual_semantic_plan_v36_20260814",
    )
    parser.add_argument("--resume", default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--precision", choices=("fp32", "fp16", "bf16"), default="bf16")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--alignment-updates", type=int, default=2_000)
    parser.add_argument("--adaptation-updates", type=int, default=4_000)
    parser.add_argument("--alignment-head-lr", type=float, default=3e-4)
    parser.add_argument("--adaptation-head-lr", type=float, default=8e-5)
    parser.add_argument("--adaptation-reader-lr", type=float, default=8e-6)
    parser.add_argument("--alignment-effective-batch", type=int, default=128)
    parser.add_argument("--adaptation-effective-batch", type=int, default=64)
    parser.add_argument("--stage-warmup", type=int, default=200)
    parser.add_argument("--minimum-learning-rate-ratio", type=float, default=0.10)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--save-every", type=int, default=500)
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
                "alignment_updates": 1,
                "adaptation_updates": 1,
                "alignment_effective_batch": 2,
                "adaptation_effective_batch": 2,
                "stage_warmup": 0,
                "log_every": 1,
                "save_every": 1,
                "tiny_model": True,
                "random_foundation": True,
            }
        )
    return argparse.Namespace(**values)


def training_stages(args: argparse.Namespace) -> tuple[TrainingStage, ...]:
    stages = (
        TrainingStage(
            "plan-alignment",
            args.alignment_updates,
            args.alignment_head_lr,
            0.0,
            args.alignment_effective_batch,
        ),
        TrainingStage(
            "semantic-adaptation",
            args.adaptation_updates,
            args.adaptation_head_lr,
            args.adaptation_reader_lr,
            args.adaptation_effective_batch,
        ),
    )
    if any(
        stage.updates < 1
        or stage.effective_batch < 2
        or min(stage.head_learning_rate, stage.reader_learning_rate) < 0.0
        for stage in stages
    ):
        raise ValueError("V36 training stages are invalid")
    return stages


def require_preregistered_arguments(args: argparse.Namespace) -> None:
    if args.smoke or args.exploratory:
        return
    if args.tiny_model or args.random_foundation:
        raise ValueError("V36 evidence requires the pinned production foundation")
    expected = {
        **FIXED_OPTIMIZATION,
        "alignment_updates": FIXED_STAGES[0].updates,
        "adaptation_updates": FIXED_STAGES[1].updates,
        "alignment_head_lr": FIXED_STAGES[0].head_learning_rate,
        "adaptation_head_lr": FIXED_STAGES[1].head_learning_rate,
        "adaptation_reader_lr": FIXED_STAGES[1].reader_learning_rate,
        "alignment_effective_batch": FIXED_STAGES[0].effective_batch,
        "adaptation_effective_batch": FIXED_STAGES[1].effective_batch,
    }
    for name, expected_value in expected.items():
        if getattr(args, name) != expected_value:
            option = name.replace("_", "-")
            raise ValueError(f"V36 evidence requires --{option}={expected_value}")
    if args.batch_size > 16:
        raise ValueError("V36 evidence physical batch may only be reduced from 16")


def validate_batch_geometry(
    stages: Sequence[TrainingStage],
    *,
    batch_size: int,
) -> None:
    if batch_size < 2:
        raise ValueError("V36 contrastive microbatches require at least two examples")
    for stage in stages:
        if stage.effective_batch % batch_size:
            raise ValueError(
                f"V36 {stage.name} effective batch must divide by physical batch"
            )


def v36_model_config(*, tiny: bool) -> VisualSemanticPlanConfig:
    if not tiny:
        return VisualSemanticPlanConfig()
    return VisualSemanticPlanConfig(
        reader_hidden_size=64,
        reader_layers=2,
        reader_heads=4,
        reader_intermediate_size=128,
        reader_dropout=0.0,
        planner_dim=64,
        planner_layers=1,
        planner_heads=4,
        planner_mlp_dim=128,
        planner_dropout=0.0,
        plan_dim=64,
        length_hidden_size=32,
    )


def choose_device(value: str, *, evidence: bool) -> torch.device:
    device = torch.device(value)
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda:0")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("V36 requested CUDA but CUDA is unavailable")
    if evidence:
        if device.type != "cuda":
            raise RuntimeError("V36 evidence requires one CUDA device")
        torch.cuda.set_device(device)
        name = torch.cuda.get_device_name(device)
        if "4090" not in name:
            raise RuntimeError(f"V36 evidence requires an RTX 4090, found {name!r}")
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("V36 evidence requires CUDA BF16 support")
    return device


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def autocast_context(device: torch.device, precision: str):
    if device.type != "cuda" or precision == "fp32":
        return nullcontext()
    dtype = torch.float16 if precision == "fp16" else torch.bfloat16
    return torch.autocast(device_type="cuda", dtype=dtype)


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
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")


def cpu_model_state(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
    }


def tensors_are_finite(value: Any) -> bool:
    if isinstance(value, torch.Tensor):
        return not value.is_floating_point() or bool(torch.isfinite(value).all())
    if isinstance(value, Mapping):
        return all(tensors_are_finite(item) for item in value.values())
    if isinstance(value, (tuple, list)):
        return all(tensors_are_finite(item) for item in value)
    return True


def _capture_rng() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def _restore_rng(state: Mapping[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available() and state.get("torch_cuda"):
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def _stage_progress(
    global_update: int,
    stages: Sequence[TrainingStage],
) -> list[tuple[TrainingStage, int]]:
    if global_update < 0 or global_update > sum(stage.updates for stage in stages):
        raise ValueError("V36 global update lies outside the training plan")
    result: list[tuple[TrainingStage, int]] = []
    prior = 0
    for stage in stages:
        completed = min(stage.updates, max(0, global_update - prior))
        result.append((stage, completed))
        prior += stage.updates
    return result


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
            collate_fn=visual_semantic_plan_prompt_collate,
            generator=torch.Generator().manual_seed(seed),
        )
    )


def concatenate_plan_outputs(
    outputs: Sequence[VisualSemanticPlanOutput],
) -> VisualSemanticPlanOutput:
    if not outputs:
        raise ValueError("V36 cannot concatenate an empty plan output list")
    plans = torch.cat([output.plans for output in outputs])
    lengths = torch.cat([output.length for output in outputs])
    placeholder = plans.new_empty((0,))
    return VisualSemanticPlanOutput(
        plans=plans,
        length=lengths,
        planner_hidden=placeholder,
        reader_memory=placeholder,
    )


def concatenate_teacher_targets(
    targets: Sequence[VisualSemanticTeacherTargets],
) -> VisualSemanticTeacherTargets:
    if not targets:
        raise ValueError("V36 cannot concatenate an empty target list")
    result = VisualSemanticTeacherTargets(
        global_plan=torch.cat([target.global_plan for target in targets]),
        chunk_plans=torch.cat([target.chunk_plans for target in targets]),
        chunk_active=torch.cat([target.chunk_active for target in targets]),
        length=torch.cat([target.length for target in targets]),
    )
    result.validate()
    return result


def load_target_bank(path: str | Path) -> VisualSemanticPlanTargetBank:
    state = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(state, Mapping):
        raise TypeError("V36 target bank must contain a state mapping")
    return VisualSemanticPlanTargetBank.from_state_dict(state)


def validate_target_bank(
    bank: VisualSemanticPlanTargetBank,
    *,
    config: VisualSemanticPlanConfig,
    manifest_sha256: str,
    evidence: bool,
) -> None:
    receipt = bank.receipt
    if receipt.get("split") != "train":
        raise ValueError("V36 training requires a train target bank")
    if receipt.get("protocol", {}).get("sha256") != EXPECTED_PROTOCOL_SHA256:
        raise ValueError("V36 target bank has a different protocol")
    if receipt.get("model_config") != asdict(config):
        raise ValueError("V36 target bank has a different model configuration")
    if receipt.get("data", {}).get("sha256") != manifest_sha256:
        raise ValueError("V36 target bank has different instruction data")
    if bank.global_plans.shape[-1] != config.plan_dim:
        raise ValueError("V36 target plan width differs from the student")
    if not tensors_are_finite(bank.state_dict()):
        raise FloatingPointError("V36 target bank is non-finite")
    if evidence:
        data = receipt.get("data", {})
        initialization = receipt.get("initialization", {})
        if data.get("selected_records") != data.get("eligible_records"):
            raise ValueError("V36 evidence requires the complete eligible train bank")
        if initialization.get("sha256") != PIXEL_LINGUIST_WEIGHT_SHA256:
            raise ValueError("V36 target bank has a different visual foundation")
        if not initialization.get("evidence_eligible", False):
            raise ValueError("V36 target bank is not evidence eligible")
        if initialization.get("revision") != PIXEL_LINGUIST_REVISION:
            raise ValueError("V36 target bank has a different foundation revision")
        for path, digest in receipt.get("source_sha256", {}).items():
            if not Path(path).is_file() or file_sha256(path) != digest:
                raise ValueError(f"V36 target-bank source changed: {path}")


def select_bank_records(
    records: Sequence[VisualRasterRecord],
    bank: VisualSemanticPlanTargetBank,
) -> list[VisualRasterRecord]:
    by_identifier = {record.identifier: record for record in records}
    try:
        selected = [by_identifier[identifier] for identifier in bank.identifiers]
    except KeyError as error:
        raise KeyError(f"V36 manifest lacks bank record {error.args[0]!r}") from error
    return selected


def _validate_resume(
    checkpoint: Mapping[str, Any],
    *,
    config: VisualSemanticPlanConfig,
    source_hashes: Mapping[str, str],
    data_receipt: Mapping[str, Any],
    arguments: Mapping[str, Any],
) -> None:
    if checkpoint.get("experiment") != EXPERIMENT:
        raise ValueError("resume checkpoint is not V36")
    if checkpoint.get("architecture") != V36_ARCHITECTURE:
        raise ValueError("V36 resume has the wrong architecture")
    if checkpoint.get("protocol", {}).get("sha256") != EXPECTED_PROTOCOL_SHA256:
        raise ValueError("V36 resume has a different protocol")
    if checkpoint.get("model_config") != asdict(config):
        raise ValueError("V36 resume has a different model configuration")
    receipt = checkpoint.get("run_receipt", {})
    if receipt.get("source_sha256") != dict(source_hashes):
        raise ValueError("V36 source changed since the interrupted run")
    if receipt.get("data") != dict(data_receipt):
        raise ValueError("V36 resume has a different data receipt")
    if receipt.get("arguments") != (dict(arguments) | {"resume": None}):
        raise ValueError("V36 resume has different effective arguments")
    if checkpoint.get("optimizer") is None or checkpoint.get("rng") is None:
        raise ValueError("V36 resume lacks optimizer or RNG state")
    if not checkpoint.get("resumable", False):
        raise ValueError("V36 checkpoint is not resumable")


def _trainable_parameters_are_finite(model: nn.Module) -> bool:
    return all(
        not parameter.requires_grad
        or not parameter.is_floating_point()
        or bool(torch.isfinite(parameter).all())
        for parameter in model.parameters()
    )


def ema_model_state(
    model: nn.Module,
    ema: SelectiveExponentialMovingAverage,
) -> dict[str, torch.Tensor]:
    state = cpu_model_state(model)
    for name, value in ema.shadow.items():
        if name not in state or state[name].shape != value.shape:
            raise ValueError(f"V36 EMA cannot map deployable parameter {name!r}")
        state[name] = value.detach().cpu().to(dtype=state[name].dtype).clone()
    return state


def main() -> None:
    raw_args = parse_args()
    require_preregistered_arguments(raw_args)
    args = effective_arguments(raw_args)
    stages = training_stages(args)
    validate_batch_geometry(stages, batch_size=args.batch_size)
    evidence = not args.smoke and not args.exploratory
    if args.random_foundation != args.tiny_model:
        raise ValueError("V36 random foundations and tiny models must be used together")
    if not args.smoke and (args.tiny_model or args.random_foundation):
        raise ValueError("V36 tiny random training is restricted to smoke runs")
    if args.num_workers < 0 or min(args.log_every, args.save_every) < 1:
        raise ValueError("V36 worker, logging, and saving settings are invalid")
    if file_sha256(PROTOCOL_DOCUMENT) != EXPECTED_PROTOCOL_SHA256:
        raise RuntimeError("V36 protocol changed after preregistration")
    manifest_sha256 = file_sha256(args.instruction_manifest)
    if evidence and manifest_sha256 != EXPECTED_INSTRUCTION_SHA256:
        raise RuntimeError("V36 instruction data differs from preregistration")

    device = choose_device(args.device, evidence=evidence)
    seed_everything(args.seed)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)
    output = Path(args.out)
    if output.exists() and any(output.iterdir()) and not args.resume:
        raise FileExistsError(f"V36 output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    metrics_path = output / "training_metrics.jsonl"
    checkpoint_path = output / "checkpoint_latest.pt"
    standalone_path = output / "planner_ema.pt"
    protocol = {"path": PROTOCOL_DOCUMENT, "sha256": EXPECTED_PROTOCOL_SHA256}
    source_hashes = {path: file_sha256(path) for path in SOURCE_FILES}

    config = v36_model_config(tiny=args.tiny_model)
    bank = load_target_bank(args.target_bank)
    validate_target_bank(
        bank,
        config=config,
        manifest_sha256=manifest_sha256,
        evidence=evidence,
    )
    records = load_v36_instruction_records(args.instruction_manifest)
    selected_records = select_bank_records(records, bank)
    data_receipt = {
        "instruction_manifest": str(Path(args.instruction_manifest).resolve()),
        "instruction_sha256": manifest_sha256,
        "raw_records": len(records),
        "train_target_records": len(bank.identifiers),
        "target_bank": str(Path(args.target_bank).resolve()),
        "target_bank_sha256": file_sha256(args.target_bank),
        "target_bank_receipt": bank.receipt,
    }
    render_config = VisualSemanticPlanRenderConfig(augment=True)
    datasets = {
        stage.name: VisualSemanticPlanPromptDataset(
            selected_records,
            split="train",
            render_config=render_config,
            seed=args.seed + (index + 1) * 1_000_000,
            length=stage.updates * stage.effective_batch,
        )
        for index, stage in enumerate(stages)
    }
    if any(set(dataset.records) != set(selected_records) for dataset in datasets.values()):
        raise RuntimeError("V36 prompt stream and target bank records differ")

    model = VisualSemanticPlanModel(config)
    if args.random_foundation:
        initialization: dict[str, Any] = {
            "route": "random-tiny-smoke",
            "evidence_eligible": False,
        }
    else:
        external_checkpoint = resolve_pixel_linguist_checkpoint(
            args.checkpoint,
            local_files_only=args.checkpoint is None,
        )
        initialization = load_pixel_linguist_reader(model.reader, external_checkpoint)
        initialization["evidence_eligible"] = True

    set_v36_stage_trainability(model, "semantic-adaptation")
    ema_names = tuple(
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    )
    set_v36_stage_trainability(model, "plan-alignment")
    model.to(device)
    groups = visual_semantic_plan_optimizer_groups(
        model,
        head_learning_rate=args.alignment_head_lr,
        reader_learning_rate=0.0,
        weight_decay=args.weight_decay,
    )
    optimizer_receipt = v36_optimizer_receipt(model, groups)
    optimizer = torch.optim.AdamW(
        groups,
        betas=(0.9, 0.95),
        fused=device.type == "cuda",
    )
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=device.type == "cuda" and args.precision == "fp16",
    )
    ema = SelectiveExponentialMovingAverage(
        model,
        ema_names,
        decay=args.ema_decay,
    )
    global_update = 0
    elapsed_before = 0.0
    stage_summaries: dict[str, dict[str, Any]] = {}
    run_receipt = {
        "experiment": EXPERIMENT,
        "label": "smoke" if args.smoke else "exploratory" if args.exploratory else "evidence",
        "architecture": V36_ARCHITECTURE,
        "protocol": protocol,
        "arguments": vars(args) | {"resume": None},
        "stages": [asdict(stage) for stage in stages],
        "source_sha256": source_hashes,
        "data": data_receipt,
        "initialization": initialization,
        "model_boundary": visual_semantic_plan_boundary_receipt(model),
        "data_boundary": visual_semantic_plan_data_boundary_receipt(),
        "optimizer": optimizer_receipt,
        "ema_parameter_names": list(ema_names),
        "answer_teacher_retained": False,
        "target_tensors_in_checkpoint": False,
        "effective_contrastive_batch_preserved": True,
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "maximum_vram_bytes": MAXIMUM_VRAM_BYTES,
        "external_revision": PIXEL_LINGUIST_REVISION,
        "external_weight_sha256": PIXEL_LINGUIST_WEIGHT_SHA256,
        "derived_checkpoint_license": "local research only; no redistribution",
    }

    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        _validate_resume(
            checkpoint,
            config=config,
            source_hashes=source_hashes,
            data_receipt=data_receipt,
            arguments=vars(args),
        )
        model.load_state_dict(checkpoint["model"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        scaler.load_state_dict(checkpoint.get("scaler", {}))
        ema.load_state_dict(checkpoint["ema"])
        global_update = int(checkpoint["global_update"])
        elapsed_before = float(checkpoint.get("training_elapsed_seconds", 0.0))
        stage_summaries = dict(checkpoint.get("stage_summaries", {}))
        _restore_rng(checkpoint["rng"])
        run_receipt = dict(checkpoint["run_receipt"])
    else:
        atomic_write_json(run_receipt, output / "run_receipt.json")
        print(json.dumps(run_receipt, ensure_ascii=False, indent=2), flush=True)

    stop_requested = False

    def request_stop(signum: int, _frame: Any) -> None:
        nonlocal stop_requested
        stop_requested = True
        print(f"V36 received signal {signum}; saving after this update", flush=True)

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    started = time.monotonic()

    def save_checkpoint() -> None:
        elapsed = elapsed_before + time.monotonic() - started
        payload = {
            "experiment": EXPERIMENT,
            "architecture": V36_ARCHITECTURE,
            "protocol": protocol,
            "model_config": asdict(config),
            "model": cpu_model_state(model),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "ema": ema.state_dict(),
            "global_update": global_update,
            "training_elapsed_seconds": elapsed,
            "stage_summaries": stage_summaries,
            "run_receipt": run_receipt,
            "rng": _capture_rng(),
            "finite": tensors_are_finite(model.state_dict())
            and tensors_are_finite(optimizer.state_dict())
            and tensors_are_finite(ema.state_dict(cpu=False)),
            "peak_allocated_vram_bytes": (
                torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
            ),
            "resumable": True,
            "contains_target_tensors": False,
            "contains_answer_teacher": False,
        }
        atomic_torch_save(payload, checkpoint_path)

    for stage_index, (stage, completed_updates) in enumerate(
        _stage_progress(global_update, stages)
    ):
        if completed_updates >= stage.updates:
            continue
        set_v36_stage_trainability(model, stage.name)
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
        final_plan_variance_max = float("nan")
        for stage_update in range(completed_updates + 1, stage.updates + 1):
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
            set_v36_optimizer_learning_rates(
                optimizer,
                head=head_lr,
                reader=reader_lr,
            )
            optimizer.zero_grad(set_to_none=True)
            primary_outputs: list[VisualSemanticPlanOutput] = []
            alternate_outputs: list[VisualSemanticPlanOutput] = []
            target_batches: list[VisualSemanticTeacherTargets] = []
            active_prompt_patches = 0.0
            identifiers: list[str] = []
            for _ in range(microbatches):
                raw_batch = next(loader)
                prompt_pixels = raw_batch["prompt_pixels"].to(
                    device, non_blocking=device.type == "cuda"
                )
                prompt_mask = raw_batch["prompt_mask"].to(
                    device, non_blocking=device.type == "cuda"
                )
                view_pixels = raw_batch["prompt_view_pixels"].to(
                    device, non_blocking=device.type == "cuda"
                )
                view_mask = raw_batch["prompt_view_mask"].to(
                    device, non_blocking=device.type == "cuda"
                )
                batch_identifiers = [
                    str(metadata["identifier"]) for metadata in raw_batch["metadata"]
                ]
                with autocast_context(device, args.precision):
                    primary_outputs.append(model(prompt_pixels, prompt_mask))
                    alternate_outputs.append(model(view_pixels, view_mask))
                target_batches.append(
                    bank.lookup(batch_identifiers, device=device, dtype=torch.float32)
                )
                identifiers.extend(batch_identifiers)
                active_prompt_patches += float(prompt_mask.sum())
            if len(set(identifiers)) != len(identifiers):
                raise RuntimeError("V36 effective contrastive batch contains duplicates")
            primary = concatenate_plan_outputs(primary_outputs)
            alternate = concatenate_plan_outputs(alternate_outputs)
            targets = concatenate_teacher_targets(target_batches)
            losses: VisualSemanticPlanLoss = visual_semantic_plan_loss(
                primary,
                alternate,
                targets,
            )
            if not bool(torch.isfinite(losses.loss)):
                raise FloatingPointError("V36 encountered a non-finite loss")
            if scaler.is_enabled():
                scaler.scale(losses.loss).backward()
                scaler.unscale_(optimizer)
            else:
                losses.loss.backward()
            trainable = [
                parameter for parameter in model.parameters() if parameter.requires_grad
            ]
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                trainable,
                args.gradient_clip,
            )
            if not bool(torch.isfinite(gradient_norm)):
                raise FloatingPointError("V36 encountered a non-finite gradient")
            if scaler.is_enabled():
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            if not _trainable_parameters_are_finite(model):
                raise FloatingPointError("V36 encountered a non-finite parameter")
            ema.update(model)
            global_update += 1
            per_dimension_variance = primary.plans[:, 0].detach().float().var(
                dim=0,
                unbiased=False,
            )
            final_plan_variance_max = float(per_dimension_variance.max())
            peak_vram = (
                torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
            )
            if peak_vram >= MAXIMUM_VRAM_BYTES:
                raise MemoryError("V36 reached the preregistered 20 GiB VRAM stop rule")
            latest_metrics = {
                "global_update": global_update,
                "stage": stage.name,
                "stage_update": stage_update,
                "physical_batch": args.batch_size,
                "microbatches": microbatches,
                "effective_batch": stage.effective_batch,
                "head_lr": head_lr,
                "reader_lr": reader_lr,
                "gradient_norm": float(gradient_norm),
                "active_prompt_patches": active_prompt_patches,
                "plan_variance_min": float(per_dimension_variance.min()),
                "plan_variance_max": final_plan_variance_max,
                "peak_allocated_vram_bytes": peak_vram,
                "elapsed_seconds": elapsed_before + time.monotonic() - started,
                **losses.detached_metrics(),
            }
            append_jsonl(metrics_path, latest_metrics)
            if stage_update == 1 or stage_update % args.log_every == 0:
                print(json.dumps(latest_metrics, sort_keys=True), flush=True)
            if global_update % args.save_every == 0 or stop_requested:
                save_checkpoint()
            if stop_requested:
                break
        stage_summaries[stage.name] = {
            "updates_completed": stage_update,
            "updates_planned": stage.updates,
            "examples_consumed": stage_update * stage.effective_batch,
            "elapsed_seconds": time.monotonic() - stage_started,
            "latest_metrics": latest_metrics,
        }
        save_checkpoint()
        if final_plan_variance_max < 1e-4:
            raise RuntimeError(
                f"V36 {stage.name} collapsed below the plan-variance stop rule"
            )
        if stop_requested:
            break

    elapsed = elapsed_before + time.monotonic() - started
    save_checkpoint()
    complete = global_update == sum(stage.updates for stage in stages)
    if complete:
        standalone = {
            "experiment": EXPERIMENT,
            "architecture": V36_ARCHITECTURE,
            "weight_route": "selective-ema",
            "model_config": asdict(config),
            "model": ema_model_state(model, ema),
            "global_update": global_update,
            "protocol": protocol,
            "boundary": visual_semantic_plan_boundary_receipt(model),
            "external_revision": PIXEL_LINGUIST_REVISION,
            "external_weight_sha256": PIXEL_LINGUIST_WEIGHT_SHA256,
            "contains_target_tensors": False,
            "contains_answer_teacher": False,
            "license": "local research only; no redistribution",
        }
        atomic_torch_save(standalone, standalone_path)
    summary = {
        "experiment": EXPERIMENT,
        "global_update": global_update,
        "planned_updates": sum(stage.updates for stage in stages),
        "complete": complete,
        "stage_summaries": stage_summaries,
        "training_elapsed_seconds": elapsed,
        "peak_allocated_vram_bytes": (
            torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
        ),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "standalone_checkpoint": str(standalone_path) if complete else None,
        "standalone_checkpoint_sha256": (
            file_sha256(standalone_path) if complete else None
        ),
        "stopped_by_signal": stop_requested,
    }
    atomic_write_json(summary, output / "training_summary.json")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

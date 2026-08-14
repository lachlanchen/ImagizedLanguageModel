#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
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
from ilm.visual_lm.visual_path_alignment import (
    V38_ARCHITECTURE,
    VisualPathAlignmentConfig,
    VisualPathAlignmentModel,
    VisualPathAlignmentOutput,
    file_sha256,
    load_v38_v37_initialization,
    visual_path_alignment_boundary_receipt,
)
from ilm.visual_lm.visual_path_alignment_data import (
    V38_TRAIN_FONTS,
    VisualPathAlignmentDataset,
    VisualPathAlignmentParaphrase,
    load_v37_instruction_records,
    load_v38_paraphrases,
    visual_path_alignment_collate,
    visual_path_alignment_data_boundary_receipt,
    visual_path_alignment_pixel_batch,
)
from ilm.visual_lm.visual_path_alignment_training import (
    VisualPathAlignmentEMA,
    VisualPathAlignmentLoss,
    VisualPathAlignmentTargetBank,
    centered_effective_rank,
    orthogonal_prompt_answer_rotation,
    set_v38_optimizer_learning_rates,
    set_v38_stage_trainability,
    visual_path_alignment_loss,
    visual_path_alignment_optimizer_groups,
)
from ilm.visual_lm.visual_semantic_distillation_data import (
    V37_SEMANTIC_DIM,
    VisualSemanticDistillationRenderConfig,
)
from ilm.visual_lm.visual_semantic_raster_data import VisualRasterRecord


EXPERIMENT = V38_ARCHITECTURE
PROTOCOL_DOCUMENT = "references/visual_path_alignment_v38_protocol.md"
# Set only after implementation and smoke validation, before evidence training.
EXPECTED_PROTOCOL_SHA256: str | None = None
DEFAULT_INSTRUCTION_MANIFEST = "data/raw/alpaca_zh.json"
DEFAULT_PARAPHRASE_MANIFEST = (
    "data/teacher/visual_path_alignment_paraphrases_v38.jsonl"
)
DEFAULT_PARAPHRASE_RECEIPT = (
    "data/teacher/visual_path_alignment_paraphrases_v38.receipt.json"
)
DEFAULT_TARGET_BANK = "artifacts/visual_semantic_distillation_v37_targets/train.pt"
DEFAULT_DEVELOPMENT_TARGET_BANK = (
    "artifacts/visual_semantic_distillation_v37_targets/development.pt"
)
DEFAULT_INITIALIZATION = (
    "artifacts/visual_semantic_distillation_v37_20260814/student_ema.pt"
)
EXPECTED_INSTRUCTION_SHA256 = (
    "6fcb98c6d79691d1f9a88ef513335da9124e7fdeef5103343f5d9f9a6e8f4903"
)
EXPECTED_PARAPHRASE_SHA256 = (
    "25db6abd4eb266a2ae05b5d8b8e9cf23caa9c523f61e89b08cc52e542fc2a68b"
)
EXPECTED_INITIALIZATION_SHA256 = (
    "367b0dfd5b54c537aaa4cb41305c6c63c08862921228b706bd1dc37c5c2170f8"
)
EXPECTED_TRAIN_BANK_SHA256 = (
    "3cd73f0818d65fd45c7700470cd010e292f359eed5aa3e62859bdf50d301711d"
)
EXPECTED_DEVELOPMENT_BANK_SHA256 = (
    "6dac8ea8df5afbb1fbe032ab6f1dd8b196ea67ef549475025adbe7bb04706b8f"
)
EXPECTED_BGE_MANIFEST_SHA256 = (
    "7907646426070047a77226ac3e684fbbe8410524f7b4a74d02837e43f2146bab"
)
EXPECTED_BGE_MODEL_SHA256 = (
    "daec91ffb5dd0c27411bd71f29932917c49cf529a641d0168496c3a501e3062c"
)
SEED = 20_263_800
MAXIMUM_VRAM_BYTES = 20 * 1024**3
SOURCE_FILES = (
    "ilm/visual_lm/visual_path_alignment.py",
    "ilm/visual_lm/visual_path_alignment_data.py",
    "ilm/visual_lm/visual_path_alignment_training.py",
    "scripts/train_visual_path_alignment_v38.py",
)


@dataclass(frozen=True)
class TrainingStage:
    name: str
    updates: int
    head_learning_rate: float
    reader_learning_rate: float
    effective_batch: int


FIXED_STAGES = (
    TrainingStage("head-realignment", 500, 1e-4, 0.0, 64),
    TrainingStage("full-path-adaptation", 7_500, 5e-5, 5e-6, 64),
)
FIXED_OPTIMIZATION: dict[str, Any] = {
    "precision": "bf16",
    "candidate_count": 512,
    "nearest_neighbors": 16,
    "negative_teacher_ceiling": 0.85,
    "stage_warmup": 200,
    "minimum_learning_rate_ratio": 0.10,
    "weight_decay": 0.05,
    "gradient_clip": 1.0,
    "ema_decay": 0.999,
    "seed": SEED,
    "save_every": 500,
}


class DatasetWindow(Dataset[dict[str, Any]]):
    def __init__(
        self,
        dataset: Dataset[dict[str, Any]],
        *,
        start: int,
        count: int,
    ) -> None:
        if start < 0 or count < 1 or start + count > len(dataset):
            raise ValueError("V38 dataset window lies outside its deterministic stream")
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
        description="Train the V38 image-native visual-path alignment student."
    )
    parser.add_argument("--instruction-manifest", default=DEFAULT_INSTRUCTION_MANIFEST)
    parser.add_argument("--paraphrase-manifest", default=DEFAULT_PARAPHRASE_MANIFEST)
    parser.add_argument("--paraphrase-receipt", default=DEFAULT_PARAPHRASE_RECEIPT)
    parser.add_argument("--target-bank", default=DEFAULT_TARGET_BANK)
    parser.add_argument(
        "--development-target-bank",
        default=DEFAULT_DEVELOPMENT_TARGET_BANK,
    )
    parser.add_argument("--initialization", default=DEFAULT_INITIALIZATION)
    parser.add_argument(
        "--out",
        default="artifacts/visual_path_alignment_v38_20260814",
    )
    parser.add_argument("--resume", default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--precision", choices=("fp32", "fp16", "bf16"), default="bf16")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--realignment-updates", type=int, default=500)
    parser.add_argument("--adaptation-updates", type=int, default=7_500)
    parser.add_argument("--realignment-head-lr", type=float, default=1e-4)
    parser.add_argument("--adaptation-head-lr", type=float, default=5e-5)
    parser.add_argument("--adaptation-reader-lr", type=float, default=5e-6)
    parser.add_argument("--realignment-effective-batch", type=int, default=64)
    parser.add_argument("--adaptation-effective-batch", type=int, default=64)
    parser.add_argument("--candidate-count", type=int, default=512)
    parser.add_argument("--nearest-neighbors", type=int, default=16)
    parser.add_argument("--negative-teacher-ceiling", type=float, default=0.85)
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
                "realignment_updates": 1,
                "adaptation_updates": 1,
                "realignment_effective_batch": 2,
                "adaptation_effective_batch": 2,
                "candidate_count": 8,
                "nearest_neighbors": 1,
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
            "head-realignment",
            args.realignment_updates,
            args.realignment_head_lr,
            0.0,
            args.realignment_effective_batch,
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
        raise ValueError("V38 training stages are invalid")
    return stages


def require_preregistered_arguments(args: argparse.Namespace) -> None:
    if args.smoke or args.exploratory:
        return
    if EXPECTED_PROTOCOL_SHA256 is None:
        raise RuntimeError("V38 evidence protocol has not been frozen")
    if args.tiny_model or args.random_foundation:
        raise ValueError("V38 evidence requires the pinned production initialization")
    expected = {
        **FIXED_OPTIMIZATION,
        "realignment_updates": FIXED_STAGES[0].updates,
        "adaptation_updates": FIXED_STAGES[1].updates,
        "realignment_head_lr": FIXED_STAGES[0].head_learning_rate,
        "adaptation_head_lr": FIXED_STAGES[1].head_learning_rate,
        "adaptation_reader_lr": FIXED_STAGES[1].reader_learning_rate,
        "realignment_effective_batch": FIXED_STAGES[0].effective_batch,
        "adaptation_effective_batch": FIXED_STAGES[1].effective_batch,
    }
    for name, expected_value in expected.items():
        if getattr(args, name) != expected_value:
            raise ValueError(
                f"V38 evidence requires --{name.replace('_', '-')}={expected_value}"
            )
    if args.batch_size > 8:
        raise ValueError("V38 evidence physical batch may only be reduced from 8")


def validate_batch_geometry(
    stages: Sequence[TrainingStage],
    *,
    batch_size: int,
    candidate_count: int,
    nearest_neighbors: int,
) -> None:
    if batch_size < 2 or nearest_neighbors < 1:
        raise ValueError("V38 batch geometry is invalid")
    if candidate_count < batch_size * (nearest_neighbors + 1):
        raise ValueError("V38 candidates cannot contain worst-case injected negatives")
    for stage in stages:
        if stage.effective_batch % batch_size:
            raise ValueError(
                f"V38 {stage.name} effective batch must divide by physical batch"
            )


def v38_model_config(*, tiny: bool) -> VisualPathAlignmentConfig:
    if not tiny:
        return VisualPathAlignmentConfig()
    return VisualPathAlignmentConfig(
        reader_hidden_size=64,
        reader_layers=2,
        reader_heads=4,
        reader_intermediate_size=128,
        reader_dropout=0.0,
        projection_hidden_size=128,
        semantic_dim=V37_SEMANTIC_DIM,
        projection_dropout=0.0,
        answer_hidden_size=64,
        length_hidden_size=32,
    )


def choose_device(value: str, *, evidence: bool) -> torch.device:
    device = torch.device(value)
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda:0")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("V38 requested CUDA but CUDA is unavailable")
    if evidence:
        if device.type != "cuda" or device.index != 0:
            raise RuntimeError("V38 evidence requires CUDA device 0")
        torch.cuda.set_device(device)
        name = torch.cuda.get_device_name(device)
        if "4090" not in name:
            raise RuntimeError(f"V38 evidence requires an RTX 4090, found {name!r}")
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("V38 evidence requires CUDA BF16 support")
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
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def cpu_model_state(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().clone() for name, value in model.state_dict().items()
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
        raise ValueError("V38 global update lies outside the training plan")
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
            collate_fn=visual_path_alignment_collate,
            generator=torch.Generator().manual_seed(seed),
        )
    )


def split_path_outputs(
    output: VisualPathAlignmentOutput,
    *,
    batch_size: int,
) -> tuple[
    VisualPathAlignmentOutput,
    VisualPathAlignmentOutput,
    VisualPathAlignmentOutput,
    VisualPathAlignmentOutput,
    VisualPathAlignmentOutput,
]:
    if batch_size < 1 or output.prompt_state.shape[0] != 5 * batch_size:
        raise ValueError("V38 concatenated view output has an invalid batch")

    def view(index: int) -> VisualPathAlignmentOutput:
        start = index * batch_size
        stop = start + batch_size
        return VisualPathAlignmentOutput(
            prompt_state=output.prompt_state[start:stop],
            answer_state=output.answer_state[start:stop],
            length=output.length[start:stop],
            prompt_features=output.prompt_features[start:stop],
            answer_base=output.answer_base[start:stop],
            answer_correction=output.answer_correction[start:stop],
            pooled_visual_state=output.pooled_visual_state[start:stop],
        )

    return view(0), view(1), view(2), view(3), view(4)


def candidate_seed(seed: int, *, global_update: int, microbatch: int) -> int:
    if global_update < 0 or microbatch < 0:
        raise ValueError("V38 candidate position cannot be negative")
    return int(seed) + (global_update + 1) * 1_000_003 + microbatch * 97


def load_target_bank(path: str | Path) -> VisualPathAlignmentTargetBank:
    state = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(state, Mapping):
        raise TypeError("V38 target bank must contain a state mapping")
    return VisualPathAlignmentTargetBank.from_v37_state_dict(state)


def validate_target_bank(
    bank: VisualPathAlignmentTargetBank,
    *,
    split: str,
    manifest_sha256: str,
    evidence: bool,
) -> None:
    receipt = bank.receipt
    if receipt.get("split") != split:
        raise ValueError(f"V38 requires a {split} target bank")
    if receipt.get("data", {}).get("sha256") != manifest_sha256:
        raise ValueError("V38 target bank has different instruction data")
    if bank.prompt_targets.shape[1] != V37_SEMANTIC_DIM:
        raise ValueError("V38 target width differs from the student")
    if not tensors_are_finite(
        (bank.prompt_targets, bank.answer_targets, bank.lengths, bank.teacher_mean)
    ):
        raise FloatingPointError("V38 target bank is non-finite")
    if evidence:
        expected_count = 5_822 if split == "train" else 196
        if len(bank.identifiers) != expected_count:
            raise ValueError(f"V38 evidence requires {expected_count} {split} targets")
        teacher = receipt.get("teacher", {})
        if teacher.get("manifest_sha256") != EXPECTED_BGE_MANIFEST_SHA256:
            raise ValueError("V38 target bank has a different BGE manifest")
        if teacher.get("model_layer_sha256") != EXPECTED_BGE_MODEL_SHA256:
            raise ValueError("V38 target bank has a different BGE model")
        if not teacher.get("evidence_eligible", False):
            raise ValueError("V38 detached target teacher is not evidence eligible")
        if teacher.get("student_runtime_dependency", True):
            raise ValueError("V38 target bank declares a runtime teacher")
        if receipt.get("strings_stored", True) or receipt.get("token_ids_stored", True):
            raise ValueError("V38 detached target bank contains symbolic payloads")


def select_bank_records(
    records: Sequence[VisualRasterRecord],
    bank: VisualPathAlignmentTargetBank,
) -> list[VisualRasterRecord]:
    by_identifier = {record.identifier: record for record in records}
    try:
        return [by_identifier[identifier] for identifier in bank.identifiers]
    except KeyError as error:
        raise KeyError(f"V38 manifest lacks bank record {error.args[0]!r}") from error


def selected_identifier_sha256(identifiers: Sequence[str]) -> str:
    return hashlib.sha256("\n".join(identifiers).encode()).hexdigest()


def tensor_sha256(value: torch.Tensor) -> str:
    array = value.detach().cpu().contiguous().numpy()
    return hashlib.sha256(array.tobytes()).hexdigest()


def _iter_tensor_paths(value: Any, path: tuple[str, ...] = ()):
    if isinstance(value, torch.Tensor):
        yield path, value
    elif isinstance(value, Mapping):
        for key, item in value.items():
            yield from _iter_tensor_paths(item, path + (str(key),))
    elif isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            yield from _iter_tensor_paths(item, path + (str(index),))


def validate_checkpoint_boundary(payload: Mapping[str, Any]) -> None:
    forbidden_tensor_fragments = (
        "target",
        "teacher",
        "candidate",
        "nearest",
        "bge",
        "rotation",
    )
    for path, _tensor in _iter_tensor_paths(payload):
        joined = ".".join(path).lower()
        if any(fragment in joined for fragment in forbidden_tensor_fragments):
            raise ValueError(f"V38 checkpoint contains forbidden tensor {joined!r}")
    forbidden_source_keys = {
        "prompt_text",
        "answer_text",
        "source_text",
        "source_strings",
        "raw_strings",
        "paraphrase_text",
    }

    def inspect(value: Any) -> None:
        if isinstance(value, Mapping):
            if forbidden_source_keys.intersection(str(key) for key in value):
                raise ValueError("V38 checkpoint contains source-language strings")
            for item in value.values():
                inspect(item)
        elif isinstance(value, (tuple, list)):
            for item in value:
                inspect(item)

    inspect(payload)


def _validate_resume(
    checkpoint: Mapping[str, Any],
    *,
    config: VisualPathAlignmentConfig,
    source_hashes: Mapping[str, str],
    data_receipt: Mapping[str, Any],
    arguments: Mapping[str, Any],
    stages: Sequence[TrainingStage],
    protocol: Mapping[str, Any],
) -> None:
    if checkpoint.get("experiment") != EXPERIMENT:
        raise ValueError("resume checkpoint is not V38")
    if checkpoint.get("architecture") != V38_ARCHITECTURE:
        raise ValueError("V38 resume has the wrong architecture")
    if checkpoint.get("protocol") != dict(protocol):
        raise ValueError("V38 resume has a different protocol")
    if checkpoint.get("model_config") != asdict(config):
        raise ValueError("V38 resume has a different model configuration")
    receipt = checkpoint.get("run_receipt", {})
    if receipt.get("source_sha256") != dict(source_hashes):
        raise ValueError("V38 source changed since the interrupted run")
    if receipt.get("data") != dict(data_receipt):
        raise ValueError("V38 resume has a different data receipt")
    if receipt.get("arguments") != (dict(arguments) | {"resume": None}):
        raise ValueError("V38 resume has different effective arguments")
    if checkpoint.get("optimizer") is None or checkpoint.get("rng") is None:
        raise ValueError("V38 resume lacks optimizer or RNG state")
    if not checkpoint.get("resumable", False):
        raise ValueError("V38 checkpoint is not resumable")
    global_update = int(checkpoint.get("global_update", -1))
    expected_positions = {
        stage.name: completed * stage.effective_batch
        for stage, completed in _stage_progress(global_update, stages)
    }
    if checkpoint.get("data_positions") != expected_positions:
        raise ValueError("V38 resume data position is inconsistent")
    if checkpoint.get("candidate_sampling_position") != global_update:
        raise ValueError("V38 resume candidate position is inconsistent")
    validate_checkpoint_boundary(checkpoint)


def _trainable_parameters_are_finite(model: nn.Module) -> bool:
    return all(
        not parameter.requires_grad
        or not parameter.is_floating_point()
        or bool(torch.isfinite(parameter).all())
        for parameter in model.parameters()
    )


def ema_model_state(
    model: nn.Module,
    ema: VisualPathAlignmentEMA,
) -> dict[str, torch.Tensor]:
    state = cpu_model_state(model)
    for name, value in ema.shadow.items():
        if name not in state or state[name].shape != value.shape:
            raise ValueError(f"V38 EMA cannot map deployable parameter {name!r}")
        state[name] = value.detach().cpu().to(dtype=state[name].dtype).clone()
    return state


def optimizer_receipt(
    model: nn.Module,
    groups: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    named = {id(parameter): name for name, parameter in model.named_parameters()}
    summaries: list[dict[str, Any]] = []
    seen: set[int] = set()
    for group in groups:
        parameters = list(group["params"])
        identifiers = {id(parameter) for parameter in parameters}
        if seen.intersection(identifiers):
            raise ValueError("V38 optimizer groups overlap")
        seen.update(identifiers)
        summaries.append(
            {
                "role": group["role"],
                "decay": bool(group["decay"]),
                "parameters": len(parameters),
                "elements": sum(parameter.numel() for parameter in parameters),
                "names_sha256": selected_identifier_sha256(
                    sorted(named[id(parameter)] for parameter in parameters)
                ),
            }
        )
    if seen != set(named):
        raise ValueError("V38 optimizer does not cover every parameter")
    return {"algorithm": "AdamW", "betas": [0.9, 0.95], "groups": summaries}


@torch.no_grad()
def deterministic_rank_probe(
    model: VisualPathAlignmentModel,
    records: Sequence[VisualRasterRecord],
    paraphrases: Mapping[str, VisualPathAlignmentParaphrase],
    *,
    device: torch.device,
    precision: str,
    batch_size: int,
    seed: int,
) -> dict[str, Any]:
    sample_count = min(64, len(records))
    dataset = VisualPathAlignmentDataset(
        records,
        render_config=VisualSemanticDistillationRenderConfig(augment=False),
        seed=seed,
        length=sample_count,
        paraphrases=paraphrases,
    )
    loader = DataLoader(
        dataset,
        batch_size=min(batch_size, sample_count),
        shuffle=False,
        num_workers=0,
        collate_fn=visual_path_alignment_collate,
    )
    was_training = model.training
    model.eval()
    prompt_states: list[torch.Tensor] = []
    answer_states: list[torch.Tensor] = []
    for raw_batch in loader:
        batch = visual_path_alignment_pixel_batch(raw_batch)
        pixels = batch["prompt_anchor_pixels"].to(device)
        mask = batch["prompt_anchor_mask"].to(device)
        with autocast_context(device, precision):
            output = model(pixels, mask)
        prompt_states.append(output.prompt_state.float().cpu())
        answer_states.append(output.answer_state.float().cpu())
    model.train(was_training)
    prompts = torch.cat(prompt_states)
    answers = torch.cat(answer_states)
    return {
        "samples": sample_count,
        "seed": seed,
        "prompt_effective_rank": centered_effective_rank(prompts),
        "answer_effective_rank": centered_effective_rank(answers),
        "finite": bool(torch.isfinite(prompts).all() and torch.isfinite(answers).all()),
    }


def _protocol_receipt(*, evidence: bool) -> dict[str, Any]:
    path = Path(PROTOCOL_DOCUMENT)
    if not path.is_file():
        if evidence:
            raise FileNotFoundError(PROTOCOL_DOCUMENT)
        return {"path": PROTOCOL_DOCUMENT, "sha256": None, "frozen": False}
    digest = file_sha256(path)
    if evidence and digest != EXPECTED_PROTOCOL_SHA256:
        raise RuntimeError("V38 protocol changed after preregistration")
    return {
        "path": PROTOCOL_DOCUMENT,
        "sha256": digest,
        "frozen": digest == EXPECTED_PROTOCOL_SHA256,
    }


def _validate_production_hashes(args: argparse.Namespace, *, evidence: bool) -> None:
    if not evidence:
        return
    expected = {
        args.instruction_manifest: EXPECTED_INSTRUCTION_SHA256,
        args.paraphrase_manifest: EXPECTED_PARAPHRASE_SHA256,
        args.target_bank: EXPECTED_TRAIN_BANK_SHA256,
        args.development_target_bank: EXPECTED_DEVELOPMENT_BANK_SHA256,
        args.initialization: EXPECTED_INITIALIZATION_SHA256,
    }
    changed = [path for path, digest in expected.items() if file_sha256(path) != digest]
    if changed:
        raise RuntimeError(f"V38 pinned inputs changed: {changed}")


def main() -> None:
    raw_args = parse_args()
    require_preregistered_arguments(raw_args)
    args = effective_arguments(raw_args)
    stages = training_stages(args)
    validate_batch_geometry(
        stages,
        batch_size=args.batch_size,
        candidate_count=args.candidate_count,
        nearest_neighbors=args.nearest_neighbors,
    )
    evidence = not args.smoke and not args.exploratory
    if args.random_foundation != args.tiny_model:
        raise ValueError("V38 random foundations and tiny models must be used together")
    if not args.smoke and (args.tiny_model or args.random_foundation):
        raise ValueError("V38 tiny random training is restricted to smoke runs")
    if args.num_workers < 0 or min(args.log_every, args.save_every) < 1:
        raise ValueError("V38 worker, logging, and saving settings are invalid")
    if not -1 < args.negative_teacher_ceiling < 1:
        raise ValueError("V38 negative teacher ceiling is invalid")
    protocol = _protocol_receipt(evidence=evidence)
    _validate_production_hashes(args, evidence=evidence)

    device = choose_device(args.device, evidence=evidence)
    seed_everything(args.seed)
    startup_gpu_memory: dict[str, int] | None = None
    if device.type == "cuda":
        torch.cuda.set_device(device)
        free_bytes, total_bytes = torch.cuda.mem_get_info(device)
        startup_gpu_memory = {"free_bytes": free_bytes, "total_bytes": total_bytes}
        torch.cuda.reset_peak_memory_stats(device)
    output = Path(args.out)
    if output.exists() and any(output.iterdir()) and not args.resume:
        raise FileExistsError(f"V38 output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    metrics_path = output / "training_metrics.jsonl"
    checkpoint_path = output / "checkpoint_latest.pt"
    standalone_path = output / "student_ema.pt"
    source_hashes = {path: file_sha256(path) for path in SOURCE_FILES}

    manifest_sha256 = file_sha256(args.instruction_manifest)
    paraphrase_sha256 = file_sha256(args.paraphrase_manifest)
    train_bank_sha256 = file_sha256(args.target_bank)
    development_bank_sha256 = file_sha256(args.development_target_bank)
    bank = load_target_bank(args.target_bank)
    development_bank = load_target_bank(args.development_target_bank)
    validate_target_bank(
        bank,
        split="train",
        manifest_sha256=manifest_sha256,
        evidence=evidence,
    )
    validate_target_bank(
        development_bank,
        split="development",
        manifest_sha256=manifest_sha256,
        evidence=evidence,
    )
    if args.candidate_count > len(bank.identifiers):
        raise ValueError("V38 candidate count exceeds target-bank size")
    if evidence and development_bank.receipt.get("train_bank_sha256") != train_bank_sha256:
        raise ValueError("V38 development bank was built from another train bank")

    all_records = load_v37_instruction_records(args.instruction_manifest)
    selected_records = select_bank_records(all_records, bank)
    paraphrases = load_v38_paraphrases(args.paraphrase_manifest, selected_records)
    if evidence and len(paraphrases) != 1_024:
        raise ValueError("V38 evidence requires 1,024 audited paraphrases")
    if args.smoke:
        selected_records = selected_records[:8]
        paraphrases = {
            identifier: value
            for identifier, value in paraphrases.items()
            if identifier in {record.identifier for record in selected_records}
        }
    identifier_hash = selected_identifier_sha256(
        [record.identifier for record in selected_records]
    )
    nearest_started = time.monotonic()
    nearest_indices = bank.nearest_answer_indices(
        neighbors=args.nearest_neighbors,
        teacher_ceiling=args.negative_teacher_ceiling,
    )
    nearest_receipt = {
        "neighbors_per_record": args.nearest_neighbors,
        "teacher_cosine_ceiling": args.negative_teacher_ceiling,
        "shape": list(nearest_indices.shape),
        "sha256": tensor_sha256(nearest_indices),
        "elapsed_seconds": time.monotonic() - nearest_started,
        "role": "train-only detached candidate injection",
        "stored_in_checkpoint": False,
        "deployed": False,
    }
    data_receipt = {
        "instruction_manifest": str(Path(args.instruction_manifest).resolve()),
        "instruction_sha256": manifest_sha256,
        "paraphrase_manifest": str(Path(args.paraphrase_manifest).resolve()),
        "paraphrase_sha256": paraphrase_sha256,
        "paraphrase_receipt": str(Path(args.paraphrase_receipt).resolve()),
        "paraphrase_receipt_sha256": file_sha256(args.paraphrase_receipt),
        "raw_records": len(all_records),
        "train_target_records": len(bank.identifiers),
        "training_stream_records": len(selected_records),
        "training_paraphrases": len(paraphrases),
        "selected_identifier_sha256": identifier_hash,
        "target_bank": str(Path(args.target_bank).resolve()),
        "target_bank_sha256": train_bank_sha256,
        "development_target_bank": str(Path(args.development_target_bank).resolve()),
        "development_target_bank_sha256": development_bank_sha256,
        "nearest_answer_negatives": nearest_receipt,
        "training_fonts": list(V38_TRAIN_FONTS),
        "training_font_sha256": {
            path: file_sha256(path) for path in V38_TRAIN_FONTS
        },
    }
    render_config = VisualSemanticDistillationRenderConfig(augment=True)
    datasets = {
        stage.name: VisualPathAlignmentDataset(
            selected_records,
            render_config=render_config,
            seed=args.seed + (index + 1) * 1_000_000,
            length=stage.updates * stage.effective_batch,
            paraphrases=paraphrases,
        )
        for index, stage in enumerate(stages)
    }
    expected_ids = tuple(record.identifier for record in selected_records)
    if any(
        tuple(record.identifier for record in dataset.records) != expected_ids
        for dataset in datasets.values()
    ):
        raise RuntimeError("V38 visual stream and selected records differ")

    config = v38_model_config(tiny=args.tiny_model)
    model = VisualPathAlignmentModel(config)
    if args.random_foundation:
        initialization: dict[str, Any] = {
            "route": "random-tiny-smoke-with-identity-answer-map",
            "evidence_eligible": False,
        }
    else:
        initialization = load_v38_v37_initialization(
            model,
            args.initialization,
            expected_sha256=EXPECTED_INITIALIZATION_SHA256 if evidence else None,
        )
        rotation, rotation_receipt = orthogonal_prompt_answer_rotation(bank)
        model.initialize_answer_rotation(rotation)
        initialization["answer_rotation"] = rotation_receipt | {
            "source_bank_sha256": train_bank_sha256,
            "stored_as_deployable_model_weight": True,
            "source_matrix_stored_in_checkpoint": False,
        }
        del rotation
    boundary = visual_path_alignment_boundary_receipt(model)
    if boundary["forbidden_parameter_names"] or not boundary["parameter_cap_pass"]:
        raise RuntimeError("V38 deployable boundary is invalid")

    ema_names = tuple(name for name, _parameter in model.named_parameters())
    set_v38_stage_trainability(model, "head-realignment")
    model.to(device)
    groups = visual_path_alignment_optimizer_groups(
        model,
        head_learning_rate=args.realignment_head_lr,
        reader_learning_rate=0.0,
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
    ema = VisualPathAlignmentEMA(model, ema_names, decay=args.ema_decay)
    global_update = 0
    elapsed_before = 0.0
    stage_summaries: dict[str, dict[str, Any]] = {}
    run_receipt = {
        "experiment": EXPERIMENT,
        "label": "smoke" if args.smoke else "exploratory" if args.exploratory else "evidence",
        "architecture": V38_ARCHITECTURE,
        "protocol": protocol,
        "arguments": vars(args) | {"resume": None},
        "stages": [asdict(stage) for stage in stages],
        "source_sha256": source_hashes,
        "data": data_receipt,
        "initialization": initialization,
        "model_boundary": boundary,
        "data_boundary": visual_path_alignment_data_boundary_receipt(),
        "optimizer": optimizer_definition,
        "ema_parameter_names_sha256": selected_identifier_sha256(ema_names),
        "detached_supervision_role": "train-only semantic alignment",
        "target_tensors_in_checkpoint": False,
        "teacher_model_in_student_process": False,
        "candidate_tensors_in_checkpoint": False,
        "nearest_negative_tensors_in_checkpoint": False,
        "source_language_strings_in_checkpoint": False,
        "effective_batch_preserved_by_gradient_accumulation": True,
        "candidate_seed_derived_from_global_update": True,
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "startup_gpu_memory": startup_gpu_memory,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "maximum_vram_bytes": MAXIMUM_VRAM_BYTES,
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
            stages=stages,
            protocol=protocol,
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
        print(f"V38 received signal {signum}; saving after this update", flush=True)

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    started = time.monotonic()

    def save_checkpoint() -> None:
        elapsed = elapsed_before + time.monotonic() - started
        data_positions = {
            stage.name: completed * stage.effective_batch
            for stage, completed in _stage_progress(global_update, stages)
        }
        payload = {
            "experiment": EXPERIMENT,
            "architecture": V38_ARCHITECTURE,
            "protocol": protocol,
            "model_config": asdict(config),
            "model": cpu_model_state(model),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "ema": ema.state_dict(),
            "global_update": global_update,
            "data_positions": data_positions,
            "candidate_sampling_position": global_update,
            "training_elapsed_seconds": elapsed,
            "stage_summaries": stage_summaries,
            "run_receipt": run_receipt,
            "rng": _capture_rng(),
            "finite": tensors_are_finite(model.state_dict())
            and tensors_are_finite(optimizer.state_dict())
            and tensors_are_finite(ema.state_dict(cpu=False)),
            "peak_allocated_vram_bytes": torch.cuda.max_memory_allocated(device)
            if device.type == "cuda"
            else 0,
            "resumable": True,
            "contains_target_tensors": False,
            "contains_teacher_model": False,
            "contains_candidate_tensors": False,
            "contains_nearest_negative_tensors": False,
            "contains_source_language_strings": False,
        }
        validate_checkpoint_boundary(payload)
        atomic_torch_save(payload, checkpoint_path)

    for stage_index, (stage, completed_updates) in enumerate(
        _stage_progress(global_update, stages)
    ):
        if completed_updates >= stage.updates:
            continue
        set_v38_stage_trainability(model, stage.name)
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
            set_v38_optimizer_learning_rates(optimizer, head=head_lr, reader=reader_lr)
            optimizer.zero_grad(set_to_none=True)
            accumulated: dict[str, float] = {}
            active_patches = 0.0
            effective_identifiers: list[str] = []
            paraphrase_views = 0
            for microbatch_index in range(microbatches):
                raw_batch = next(loader)
                pixel_batch = visual_path_alignment_pixel_batch(raw_batch)
                view_keys = (
                    ("prompt_anchor_pixels", "prompt_anchor_mask"),
                    ("prompt_view_pixels", "prompt_view_mask"),
                    ("semantic_view_pixels", "semantic_view_mask"),
                    ("answer_anchor_pixels", "answer_anchor_mask"),
                    ("answer_view_pixels", "answer_view_mask"),
                )
                pixels = torch.cat(
                    [pixel_batch[pixel] for pixel, _mask in view_keys]
                ).to(device, non_blocking=device.type == "cuda")
                masks = torch.cat(
                    [pixel_batch[mask] for _pixel, mask in view_keys]
                ).to(device, non_blocking=device.type == "cuda")
                batch_identifiers = [
                    str(metadata["identifier"]) for metadata in raw_batch["metadata"]
                ]
                paraphrase_views += sum(
                    metadata["semantic_view_kind"] == "paraphrase"
                    for metadata in raw_batch["metadata"]
                )
                targets = bank.lookup(
                    batch_identifiers,
                    device=device,
                    dtype=torch.float32,
                )
                candidates = bank.candidate_set(
                    targets.bank_indices,
                    count=args.candidate_count,
                    seed=candidate_seed(
                        args.seed,
                        global_update=global_update,
                        microbatch=microbatch_index,
                    ),
                    neighbors=args.nearest_neighbors,
                    teacher_ceiling=args.negative_teacher_ceiling,
                    device=device,
                    dtype=torch.float32,
                )
                with autocast_context(device, args.precision):
                    combined = model(pixels, masks)
                    outputs = split_path_outputs(
                        combined,
                        batch_size=len(batch_identifiers),
                    )
                    losses: VisualPathAlignmentLoss = visual_path_alignment_loss(
                        *outputs,
                        targets,
                        candidates,
                    )
                    scaled_loss = losses.loss / microbatches
                if not bool(torch.isfinite(losses.loss)):
                    raise FloatingPointError("V38 encountered a non-finite loss")
                if scaler.is_enabled():
                    scaler.scale(scaled_loss).backward()
                else:
                    scaled_loss.backward()
                for name, value in losses.detached_metrics().items():
                    accumulated[name] = accumulated.get(name, 0.0) + value / microbatches
                active_patches += float(masks.sum())
                effective_identifiers.extend(batch_identifiers)
            if len(set(effective_identifiers)) != len(effective_identifiers):
                raise RuntimeError("V38 effective batch contains duplicate positives")
            if scaler.is_enabled():
                scaler.unscale_(optimizer)
            trainable = [
                parameter for parameter in model.parameters() if parameter.requires_grad
            ]
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                trainable,
                args.gradient_clip,
            )
            if not bool(torch.isfinite(gradient_norm)):
                raise FloatingPointError("V38 encountered a non-finite gradient")
            if scaler.is_enabled():
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            if not _trainable_parameters_are_finite(model):
                raise FloatingPointError("V38 encountered a non-finite parameter")
            ema.update(model)
            global_update += 1
            peak_vram = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
            if peak_vram >= MAXIMUM_VRAM_BYTES:
                raise MemoryError("V38 reached the preregistered 20 GiB VRAM stop rule")
            latest_metrics = {
                "global_update": global_update,
                "stage": stage.name,
                "stage_update": stage_update,
                "physical_batch": args.batch_size,
                "microbatches": microbatches,
                "effective_batch": stage.effective_batch,
                "candidate_count_per_microbatch": args.candidate_count,
                "nearest_neighbors_per_positive": args.nearest_neighbors,
                "paraphrase_views": paraphrase_views,
                "head_lr": head_lr,
                "reader_lr": reader_lr,
                "gradient_norm": float(gradient_norm),
                "active_visual_patches": active_patches,
                "peak_allocated_vram_bytes": peak_vram,
                "elapsed_seconds": elapsed_before + time.monotonic() - started,
                **accumulated,
            }
            append_jsonl(metrics_path, latest_metrics)
            if stage_update == 1 or stage_update % args.log_every == 0:
                print(json.dumps(latest_metrics, sort_keys=True), flush=True)
            if global_update % args.save_every == 0 or stop_requested:
                save_checkpoint()
            if stop_requested:
                break
        stage_complete = stage_update == stage.updates and not stop_requested
        rank_probe = None
        if stage_complete:
            rank_probe = deterministic_rank_probe(
                model,
                selected_records,
                paraphrases,
                device=device,
                precision=args.precision,
                batch_size=args.batch_size,
                seed=args.seed + 90_000_000,
            )
        stage_summaries[stage.name] = {
            "updates_completed": stage_update,
            "updates_planned": stage.updates,
            "examples_consumed": stage_update * stage.effective_batch,
            "elapsed_seconds": time.monotonic() - stage_started,
            "latest_metrics": latest_metrics,
            "rank_probe": rank_probe,
        }
        save_checkpoint()
        if evidence and stage_complete and float(rank_probe["answer_effective_rank"]) < 8.0:
            raise RuntimeError(
                f"V38 {stage.name} collapsed below the effective-rank stop rule"
            )
        if stop_requested:
            break

    elapsed = elapsed_before + time.monotonic() - started
    save_checkpoint()
    complete = global_update == sum(stage.updates for stage in stages)
    if complete:
        standalone = {
            "experiment": EXPERIMENT,
            "architecture": V38_ARCHITECTURE,
            "weight_route": "all-parameter-ema",
            "model_config": asdict(config),
            "model": ema_model_state(model, ema),
            "global_update": global_update,
            "protocol": protocol,
            "boundary": visual_path_alignment_boundary_receipt(model),
            "source_sha256": source_hashes,
            "training_data_sha256": manifest_sha256,
            "training_paraphrase_sha256": paraphrase_sha256,
            "training_target_bank_sha256": train_bank_sha256,
            "selected_identifier_sha256": identifier_hash,
            "initialization_sha256": (
                file_sha256(args.initialization) if not args.random_foundation else None
            ),
            "contains_target_tensors": False,
            "contains_teacher_model": False,
            "contains_candidate_tensors": False,
            "contains_nearest_negative_tensors": False,
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
        "training_elapsed_seconds": elapsed,
        "peak_allocated_vram_bytes": torch.cuda.max_memory_allocated(device)
        if device.type == "cuda"
        else 0,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "standalone_checkpoint": str(standalone_path) if complete else None,
        "standalone_checkpoint_sha256": file_sha256(standalone_path) if complete else None,
        "stopped_by_signal": stop_requested,
    }
    atomic_write_json(summary, output / "training_summary.json")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

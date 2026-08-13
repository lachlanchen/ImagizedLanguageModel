#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
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

from ilm.visual_lm.causal_glyph_flow import (
    V35_ARCHITECTURE,
    CausalGlyphFlowConfig,
    CausalGlyphFlowLM,
    causal_glyph_flow_boundary_receipt,
    file_sha256,
    load_pixar_causal_foundation,
    load_v34_ema_codec,
)
from ilm.visual_lm.causal_glyph_flow_data import (
    CausalGlyphCopyDataset,
    CausalGlyphStageCMixture,
    causal_glyph_flow_collate,
    causal_glyph_flow_data_boundary_receipt,
    causal_glyph_flow_student_batch,
)
from ilm.visual_lm.causal_glyph_flow_evaluation import (
    evaluate_visual_interface_alignment,
    v35_stage_a_gate,
)
from ilm.visual_lm.causal_glyph_flow_training import (
    causal_glyph_flow_loss,
    causal_glyph_flow_optimizer_groups,
    set_v35_optimizer_learning_rates,
    set_v35_stage_trainability,
    v35_optimizer_receipt,
    visual_interface_alignment_loss,
)
from ilm.visual_lm.direct_visual_patch_data import (
    DirectPatchContinuationDataset,
    DirectPatchInstructionDataset,
    DirectPatchRenderConfig,
    direct_patch_partition,
)
from ilm.visual_lm.direct_visual_patch_training import (
    ExponentialMovingAverage,
    module_state_sha256,
    stage_cosine_learning_rate,
)
from ilm.visual_lm.visual_semantic_raster_data import (
    VisualRasterRecord,
    VisualTextRecord,
    load_visual_raster_instructions,
    load_visual_text_records,
)


EXPERIMENT = "causal-glyph-flow-v35"
PROTOCOL_DOCUMENT = "references/causal_glyph_flow_v35_protocol.md"
EXPECTED_PROTOCOL_SHA256 = (
    "d7a4d49270676cd82c55e22ddd73466966e0b96723970f76fe66fa2381bd3718"
)
DEFAULT_PUBLIC_MANIFEST = "data/visual_grammar/chinese_wikisource_public_domain.jsonl"
DEFAULT_INSTRUCTION_MANIFEST = "data/raw/alpaca_zh.json"
DEFAULT_CODEC_CHECKPOINT = (
    "artifacts/continuous_glyph_codec_v34_20260814/checkpoint_latest.pt"
)
DEFAULT_PIXAR_CHECKPOINT = "artifacts/upstream/pixar"
EXPECTED_PUBLIC_SHA256 = (
    "76048753b52735d14c98f0d9e4eb8d751401fe8810e82eaaaebff517f6866c03"
)
EXPECTED_INSTRUCTION_SHA256 = (
    "6fcb98c6d79691d1f9a88ef513335da9124e7fdeef5103343f5d9f9a6e8f4903"
)
SEED = 20_263_500
MAXIMUM_VRAM_BYTES = 20 * 1024**3
SOURCE_FILES = (
    "ilm/visual_lm/causal_glyph_flow.py",
    "ilm/visual_lm/causal_glyph_flow_data.py",
    "ilm/visual_lm/causal_glyph_flow_training.py",
    "ilm/visual_lm/causal_glyph_flow_evaluation.py",
    "scripts/train_causal_glyph_flow_v35.py",
)


@dataclass(frozen=True)
class TrainingStage:
    name: str
    updates: int
    adapter_learning_rate: float
    head_learning_rate: float
    core_learning_rate: float


FIXED_STAGES = (
    TrainingStage("visual-interface-alignment", 2_000, 3e-4, 0.0, 0.0),
    TrainingStage("public-causal-continuation", 8_000, 0.0, 1e-4, 1e-5),
    TrainingStage("instruction-and-copy", 12_000, 0.0, 8e-5, 8e-6),
)
FIXED_OPTIMIZATION: dict[str, Any] = {
    "batch_size": 8,
    "gradient_accumulation": 8,
    "stage_warmup": 500,
    "minimum_learning_rate_ratio": 0.10,
    "weight_decay": 0.05,
    "gradient_clip": 1.0,
    "ema_decay": 0.999,
    "seed": SEED,
    "precision": "bf16",
    "save_every": 1_000,
    "gate_minimum_patches": 2_048,
}


class DatasetWindow(Dataset[dict[str, Any]]):
    def __init__(self, dataset: Dataset[dict[str, Any]], *, start: int, count: int) -> None:
        if start < 0 or count < 1 or start + count > len(dataset):
            raise ValueError("V35 dataset window lies outside its deterministic stream")
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
        description="Train the preregistered V35 causal glyph flow language model."
    )
    parser.add_argument("--public-manifest", default=DEFAULT_PUBLIC_MANIFEST)
    parser.add_argument("--instruction-manifest", default=DEFAULT_INSTRUCTION_MANIFEST)
    parser.add_argument("--codec-checkpoint", default=DEFAULT_CODEC_CHECKPOINT)
    parser.add_argument("--pixar-checkpoint", default=DEFAULT_PIXAR_CHECKPOINT)
    parser.add_argument(
        "--out",
        default="artifacts/causal_glyph_flow_v35_20260814",
    )
    parser.add_argument("--resume", default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--precision", choices=("fp32", "fp16", "bf16"), default="bf16")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument("--alignment-updates", type=int, default=2_000)
    parser.add_argument("--continuation-updates", type=int, default=8_000)
    parser.add_argument("--instruction-updates", type=int, default=12_000)
    parser.add_argument("--alignment-adapter-lr", type=float, default=3e-4)
    parser.add_argument("--continuation-head-lr", type=float, default=1e-4)
    parser.add_argument("--continuation-core-lr", type=float, default=1e-5)
    parser.add_argument("--instruction-head-lr", type=float, default=8e-5)
    parser.add_argument("--instruction-core-lr", type=float, default=8e-6)
    parser.add_argument("--stage-warmup", type=int, default=500)
    parser.add_argument("--minimum-learning-rate-ratio", type=float, default=0.10)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--gate-minimum-patches", type=int, default=2_048)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--save-every", type=int, default=1_000)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--exploratory", action="store_true")
    parser.add_argument("--tiny-model", action="store_true")
    parser.add_argument("--random-foundations", action="store_true")
    parser.add_argument("--allow-failed-stage-a", action="store_true")
    parser.add_argument("--alignment-only", action="store_true")
    return parser.parse_args()


def effective_arguments(args: argparse.Namespace) -> argparse.Namespace:
    values = vars(args).copy()
    if args.smoke:
        values.update(
            {
                "num_workers": 0,
                "batch_size": 1,
                "gradient_accumulation": 1,
                "alignment_updates": 1,
                "continuation_updates": 1,
                "instruction_updates": 1,
                "stage_warmup": 0,
                "gate_minimum_patches": 8,
                "log_every": 1,
                "save_every": 1,
                "allow_failed_stage_a": True,
            }
        )
    return argparse.Namespace(**values)


def training_stages(args: argparse.Namespace) -> tuple[TrainingStage, ...]:
    stages = (
        TrainingStage(
            "visual-interface-alignment",
            args.alignment_updates,
            args.alignment_adapter_lr,
            0.0,
            0.0,
        ),
        TrainingStage(
            "public-causal-continuation",
            args.continuation_updates,
            0.0,
            args.continuation_head_lr,
            args.continuation_core_lr,
        ),
        TrainingStage(
            "instruction-and-copy",
            args.instruction_updates,
            0.0,
            args.instruction_head_lr,
            args.instruction_core_lr,
        ),
    )
    if any(stage.updates < 1 for stage in stages):
        raise ValueError("V35 stages must contain at least one update")
    return stages


def require_preregistered_arguments(args: argparse.Namespace) -> None:
    if args.smoke or args.exploratory:
        return
    if args.tiny_model or args.random_foundations or args.allow_failed_stage_a:
        raise ValueError("V35 evidence requires the frozen production foundations")
    expected = {
        **FIXED_OPTIMIZATION,
        "alignment_updates": FIXED_STAGES[0].updates,
        "continuation_updates": FIXED_STAGES[1].updates,
        "instruction_updates": FIXED_STAGES[2].updates,
        "alignment_adapter_lr": FIXED_STAGES[0].adapter_learning_rate,
        "continuation_head_lr": FIXED_STAGES[1].head_learning_rate,
        "continuation_core_lr": FIXED_STAGES[1].core_learning_rate,
        "instruction_head_lr": FIXED_STAGES[2].head_learning_rate,
        "instruction_core_lr": FIXED_STAGES[2].core_learning_rate,
    }
    for name, expected_value in expected.items():
        if getattr(args, name) != expected_value:
            option = name.replace("_", "-")
            raise ValueError(f"V35 evidence requires --{option}={expected_value}")


def choose_device(value: str, *, evidence: bool) -> torch.device:
    device = torch.device(value)
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda:0")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("V35 requested CUDA but CUDA is unavailable")
    if evidence:
        if device.type != "cuda":
            raise RuntimeError("V35 evidence requires one CUDA device")
        torch.cuda.set_device(device)
        name = torch.cuda.get_device_name(device)
        if "4090" not in name:
            raise RuntimeError(f"V35 evidence requires an RTX 4090, found {name!r}")
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("V35 evidence requires CUDA BF16 support")
    return device


def model_config(args: argparse.Namespace) -> CausalGlyphFlowConfig:
    if not args.tiny_model:
        return CausalGlyphFlowConfig()
    if not (args.smoke or args.exploratory):
        raise ValueError("V35 tiny model is ineligible for evidence")
    return CausalGlyphFlowConfig(
        maximum_patches=32,
        hidden_size=64,
        layers=2,
        attention_heads=4,
        key_value_heads=2,
        intermediate_size=128,
        flow_width=64,
        flow_depth=2,
        codec_channels=(8, 16, 24, 32),
        codec_group_norm_groups=8,
    )


def render_config(config: CausalGlyphFlowConfig) -> DirectPatchRenderConfig:
    if config.production_shape:
        return DirectPatchRenderConfig()
    return DirectPatchRenderConfig(
        maximum_patches=config.maximum_patches,
        maximum_prompt_patches=config.maximum_patches - 8,
        maximum_answer_patches=8,
    )


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


def _capture_rng(flow_generator: torch.Generator) -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "flow": flow_generator.get_state(),
    }


def _restore_rng(state: Mapping[str, Any], flow_generator: torch.Generator) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available() and state.get("torch_cuda"):
        torch.cuda.set_rng_state_all(state["torch_cuda"])
    flow_generator.set_state(state["flow"])


def _trim_batch(batch: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    visible = int(batch["patch_mask"].sum(dim=1).max().item())
    result = dict(batch)
    result["pixels"] = batch["pixels"][..., : visible * 32]
    for key in (
        "patch_mask",
        "next_patch_mask",
        "reconstruction_mask",
        "stop_targets",
        "stop_mask",
    ):
        result[key] = batch[key][:, :visible]
    return result


def _variants_for_examples(records: int, examples: int) -> int:
    if records < 1 or examples < 1:
        raise ValueError("V35 deterministic stream sizes must be positive")
    return max(1, math.ceil(examples / records))


def build_datasets(
    public_records: Sequence[VisualTextRecord],
    instruction_records: Sequence[VisualRasterRecord],
    *,
    config: DirectPatchRenderConfig,
    args: argparse.Namespace,
) -> tuple[dict[str, Dataset[dict[str, Any]]], Dataset[dict[str, Any]], dict[str, Any]]:
    examples_per_update = args.batch_size * args.gradient_accumulation
    alignment_examples = args.alignment_updates * examples_per_update
    continuation_examples = args.continuation_updates * examples_per_update
    instruction_examples = args.instruction_updates * examples_per_update

    public_probe = DirectPatchContinuationDataset(
        public_records,
        split="train",
        config=config,
        variants_per_record=1,
        seed=args.seed,
    )
    instruction_probe = DirectPatchInstructionDataset(
        instruction_records,
        split="train",
        config=config,
        variants_per_record=1,
        seed=args.seed,
    )
    alignment = DirectPatchContinuationDataset(
        public_records,
        split="train",
        config=config,
        variants_per_record=_variants_for_examples(
            len(public_probe),
            alignment_examples,
        ),
        seed=args.seed,
    )
    continuation = DirectPatchContinuationDataset(
        public_records,
        split="train",
        config=config,
        variants_per_record=_variants_for_examples(
            len(public_probe),
            continuation_examples,
        ),
        seed=args.seed + 1_000_000,
    )
    instruction_count = sum(
        CausalGlyphStageCMixture.stream_for_index(index) == "instruction"
        for index in range(instruction_examples)
    )
    copy_count = sum(
        CausalGlyphStageCMixture.stream_for_index(index) == "copy"
        for index in range(instruction_examples)
    )
    public_count = instruction_examples - instruction_count - copy_count
    instruction = DirectPatchInstructionDataset(
        instruction_records,
        split="train",
        config=config,
        variants_per_record=_variants_for_examples(
            len(instruction_probe),
            max(1, instruction_count),
        ),
        seed=args.seed + 2_000_000,
    )
    copy = CausalGlyphCopyDataset(
        public_records,
        split="train",
        config=config,
        length=max(1, copy_count),
        seed=args.seed + 3_000_000,
        maximum_characters=min(16, config.maximum_answer_patches),
    )
    replay = DirectPatchContinuationDataset(
        public_records,
        split="train",
        config=config,
        variants_per_record=_variants_for_examples(
            len(public_probe),
            max(1, public_count),
        ),
        seed=args.seed + 4_000_000,
    )
    mixture = CausalGlyphStageCMixture(
        instruction,
        copy,
        replay,
        length=instruction_examples,
    )
    development = DirectPatchContinuationDataset(
        public_records,
        split="development",
        config=config,
        variants_per_record=2,
        seed=args.seed + 5_000_000,
    )
    datasets: dict[str, Dataset[dict[str, Any]]] = {
        "visual-interface-alignment": alignment,
        "public-causal-continuation": continuation,
        "instruction-and-copy": mixture,
    }
    receipt = {
        "examples_per_update": examples_per_update,
        "public_train_records": len(public_probe),
        "instruction_train_records": len(instruction_probe),
        "alignment_examples": alignment_examples,
        "continuation_examples": continuation_examples,
        "stage_c_examples": instruction_examples,
        "stage_c_mixture": mixture.mixture_counts(),
        "development_examples": len(development),
    }
    return datasets, development, receipt


def _stage_progress(
    global_update: int,
    stages: Sequence[TrainingStage],
) -> list[tuple[TrainingStage, int]]:
    if global_update < 0 or global_update > sum(stage.updates for stage in stages):
        raise ValueError("V35 global update lies outside the training plan")
    output: list[tuple[TrainingStage, int]] = []
    prior = 0
    for stage in stages:
        completed = min(stage.updates, max(0, global_update - prior))
        output.append((stage, completed))
        prior += stage.updates
    return output


def _stage_loader(
    dataset: Dataset[dict[str, Any]],
    *,
    stage: TrainingStage,
    completed_updates: int,
    batch_size: int,
    gradient_accumulation: int,
    num_workers: int,
    pin_memory: bool,
    seed: int,
) -> Iterator[dict[str, Any]]:
    consumed = completed_updates * batch_size * gradient_accumulation
    remaining = (stage.updates - completed_updates) * batch_size * gradient_accumulation
    window = DatasetWindow(dataset, start=consumed, count=remaining)
    loader_generator = torch.Generator().manual_seed(seed)
    return iter(
        DataLoader(
            window,
            batch_size=batch_size,
            shuffle=False,
            drop_last=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=num_workers > 0,
            collate_fn=causal_glyph_flow_collate,
            generator=loader_generator,
        )
    )


def _data_receipt(
    public_records: Sequence[VisualTextRecord],
    instruction_records: Sequence[VisualRasterRecord],
    args: argparse.Namespace,
) -> dict[str, Any]:
    return {
        "public_manifest": args.public_manifest,
        "public_sha256": file_sha256(args.public_manifest),
        "public_records": len(public_records),
        "public_splits": {
            split: sum(
                direct_patch_partition(record.identifier, stream="public-domain")
                == split
                for record in public_records
            )
            for split in ("train", "development", "sealed")
        },
        "instruction_manifest": args.instruction_manifest,
        "instruction_sha256": file_sha256(args.instruction_manifest),
        "instruction_records": len(instruction_records),
        "instruction_splits": {
            split: sum(
                direct_patch_partition(record.identifier, stream="instruction") == split
                for record in instruction_records
            )
            for split in ("train", "development", "sealed")
        },
    }


def _validate_resume(
    checkpoint: Mapping[str, Any],
    *,
    config: CausalGlyphFlowConfig,
    source_hashes: Mapping[str, str],
    data_receipt: Mapping[str, Any],
    arguments: Mapping[str, Any],
) -> None:
    if checkpoint.get("experiment") != EXPERIMENT:
        raise ValueError("resume checkpoint is not V35")
    if checkpoint.get("architecture") != V35_ARCHITECTURE:
        raise ValueError("V35 resume has the wrong architecture")
    if checkpoint.get("protocol", {}).get("sha256") != EXPECTED_PROTOCOL_SHA256:
        raise ValueError("V35 resume has a different protocol")
    if checkpoint.get("model_config") != asdict(config):
        raise ValueError("V35 resume has a different model configuration")
    receipt = checkpoint.get("run_receipt", {})
    if receipt.get("source_sha256") != dict(source_hashes):
        raise ValueError("V35 source changed since the interrupted run")
    if receipt.get("data") != dict(data_receipt):
        raise ValueError("V35 resume has a different data receipt")
    expected_arguments = dict(arguments) | {"resume": None}
    if receipt.get("arguments") != expected_arguments:
        raise ValueError("V35 resume has different effective arguments")
    if checkpoint.get("optimizer") is None or checkpoint.get("rng") is None:
        raise ValueError("V35 resume lacks optimizer or RNG state")
    if not checkpoint.get("resumable", False):
        raise ValueError("V35 checkpoint is not resumable")


def _trainable_parameters_are_finite(model: nn.Module) -> bool:
    return all(
        not parameter.requires_grad
        or not parameter.is_floating_point()
        or bool(torch.isfinite(parameter).all())
        for parameter in model.parameters()
    )


def main() -> None:
    raw_args = parse_args()
    require_preregistered_arguments(raw_args)
    args = effective_arguments(raw_args)
    stages = training_stages(args)
    evidence = not args.smoke and not args.exploratory
    if args.num_workers < 0 or min(args.log_every, args.save_every) < 1:
        raise ValueError("V35 worker, logging, and saving settings are invalid")
    if min(args.batch_size, args.gradient_accumulation) < 1:
        raise ValueError("V35 batch and accumulation must be positive")
    if file_sha256(PROTOCOL_DOCUMENT) != EXPECTED_PROTOCOL_SHA256:
        raise RuntimeError("V35 protocol changed after preregistration")
    data_hashes = {
        "public": file_sha256(args.public_manifest),
        "instruction": file_sha256(args.instruction_manifest),
    }
    if data_hashes != {
        "public": EXPECTED_PUBLIC_SHA256,
        "instruction": EXPECTED_INSTRUCTION_SHA256,
    }:
        raise RuntimeError("V35 data differs from preregistration")
    device = choose_device(args.device, evidence=evidence)
    seed_everything(args.seed)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)

    output = Path(args.out)
    if output.exists() and any(output.iterdir()) and not args.resume:
        raise FileExistsError(f"V35 output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    metrics_path = output / "training_metrics.jsonl"
    checkpoint_path = output / "checkpoint_latest.pt"
    protocol = {"path": PROTOCOL_DOCUMENT, "sha256": EXPECTED_PROTOCOL_SHA256}
    source_hashes = {path: file_sha256(path) for path in SOURCE_FILES}

    config = model_config(args)
    raster_config = render_config(config)
    public_records = load_visual_text_records(args.public_manifest)
    instruction_records = load_visual_raster_instructions(
        args.instruction_manifest,
        maximum_prompt_characters=64 if config.production_shape else 20,
        maximum_answer_cells=32 if config.production_shape else 8,
    )
    data_receipt = _data_receipt(public_records, instruction_records, args)
    datasets, development_alignment, stream_receipt = build_datasets(
        public_records,
        instruction_records,
        config=raster_config,
        args=args,
    )

    model = CausalGlyphFlowLM(config)
    if args.random_foundations:
        if evidence:
            raise ValueError("V35 evidence cannot use random foundations")
        teacher_projection = nn.Conv2d(
            1,
            config.hidden_size,
            kernel_size=config.patch_size,
            stride=config.patch_size,
            bias=False,
        )
        initialization: dict[str, Any] = {
            "route": "random-foundations",
            "evidence_eligible": False,
        }
    else:
        if not config.production_shape:
            raise ValueError("V35 tiny model requires --random-foundations")
        codec_receipt = load_v34_ema_codec(model.codec, args.codec_checkpoint)
        pixar_receipt, teacher_projection = load_pixar_causal_foundation(
            model,
            args.pixar_checkpoint,
        )
        initialization = {
            "route": "V34-EMA-plus-PIXAR-causal-core",
            "evidence_eligible": True,
            "codec": codec_receipt,
            "pixar": pixar_receipt,
        }
    initial_core_hash = module_state_sha256(model.backbone)
    initial_codec_hash = module_state_sha256(model.codec)
    model.backbone.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    set_v35_stage_trainability(model, "visual-interface-alignment")
    model.to(device)
    teacher_projection: nn.Conv2d | None = (
        teacher_projection.to(device).requires_grad_(False).eval()
    )
    groups = causal_glyph_flow_optimizer_groups(
        model,
        adapter_learning_rate=args.alignment_adapter_lr,
        head_learning_rate=0.0,
        core_learning_rate=0.0,
        weight_decay=args.weight_decay,
    )
    optimizer_receipt = v35_optimizer_receipt(model, groups)
    optimizer = torch.optim.AdamW(
        groups,
        betas=(0.9, 0.95),
        fused=device.type == "cuda",
    )
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=device.type == "cuda" and args.precision == "fp16",
    )
    ema = ExponentialMovingAverage(model, decay=args.ema_decay)
    flow_generator = torch.Generator(device=device)
    flow_generator.manual_seed(args.seed + 35)
    global_update = 0
    elapsed_before = 0.0
    stage_a_report: dict[str, Any] | None = None
    stage_summaries: dict[str, dict[str, Any]] = {}

    sample_batch = causal_glyph_flow_collate([datasets[stages[0].name][0]])
    run_receipt = {
        "experiment": EXPERIMENT,
        "label": "smoke" if args.smoke else "exploratory" if args.exploratory else "evidence",
        "architecture": V35_ARCHITECTURE,
        "protocol": protocol,
        "arguments": vars(args) | {"resume": None},
        "stages": [asdict(stage) for stage in stages],
        "source_sha256": source_hashes,
        "data": data_receipt,
        "streams": stream_receipt,
        "initialization": initialization,
        "initial_core_sha256": initial_core_hash,
        "initial_codec_sha256": initial_codec_hash,
        "model_boundary": causal_glyph_flow_boundary_receipt(model),
        "data_boundary": causal_glyph_flow_data_boundary_receipt(sample_batch),
        "optimizer": optimizer_receipt,
        "gradient_checkpointing": {
            "enabled": True,
            "use_reentrant": False,
        },
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "maximum_vram_bytes": MAXIMUM_VRAM_BYTES,
        "runtime_teacher_retained": False,
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
        if module_state_sha256(model.codec) != initial_codec_hash:
            raise RuntimeError("V35 resume checkpoint changed the frozen V34 codec")
        optimizer.load_state_dict(checkpoint["optimizer"])
        scaler.load_state_dict(checkpoint.get("scaler", {}))
        ema.load_state_dict(checkpoint["ema"])
        global_update = int(checkpoint["global_update"])
        elapsed_before = float(checkpoint.get("training_elapsed_seconds", 0.0))
        stage_a_report = checkpoint.get("stage_a_report")
        if stage_a_report is not None:
            aligned_adapter_hash = stage_a_report.get("aligned_adapter_sha256")
            if aligned_adapter_hash != module_state_sha256(model.input_adapter):
                raise RuntimeError("V35 resume checkpoint changed the aligned adapter")
        stage_summaries = dict(checkpoint.get("stage_summaries", {}))
        _restore_rng(checkpoint["rng"], flow_generator)
        run_receipt = dict(checkpoint["run_receipt"])
    else:
        atomic_write_json(run_receipt, output / "run_receipt.json")
        print(json.dumps(run_receipt, ensure_ascii=False, indent=2), flush=True)

    stop_requested = False

    def request_stop(signum: int, _frame: Any) -> None:
        nonlocal stop_requested
        stop_requested = True
        print(f"V35 received signal {signum}; saving after this update", flush=True)

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    started = time.monotonic()
    current_elapsed = elapsed_before

    def save_checkpoint() -> None:
        if module_state_sha256(model.codec) != initial_codec_hash:
            raise RuntimeError("V35 frozen codec changed before checkpointing")
        if stage_a_report is not None:
            aligned_adapter_hash = stage_a_report.get("aligned_adapter_sha256")
            if aligned_adapter_hash != module_state_sha256(model.input_adapter):
                raise RuntimeError("V35 aligned adapter changed before checkpointing")
        current_elapsed = elapsed_before + time.monotonic() - started
        payload = {
            "experiment": EXPERIMENT,
            "architecture": V35_ARCHITECTURE,
            "protocol": protocol,
            "model_config": asdict(model.config),
            "model": cpu_model_state(model),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "ema": ema.state_dict(),
            "global_update": global_update,
            "training_elapsed_seconds": current_elapsed,
            "stage_a_report": stage_a_report,
            "stage_summaries": stage_summaries,
            "run_receipt": run_receipt,
            "rng": _capture_rng(flow_generator),
            "finite": tensors_are_finite(model.state_dict())
            and tensors_are_finite(optimizer.state_dict()),
            "peak_allocated_vram_bytes": (
                torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
            ),
            "resumable": True,
        }
        atomic_torch_save(payload, checkpoint_path)

    def evaluate_and_close_stage_a() -> None:
        nonlocal stage_a_report, teacher_projection
        if teacher_projection is None:
            raise RuntimeError("V35 Stage A evaluator lacks its offline projection")
        alignment = evaluate_visual_interface_alignment(
            model,
            teacher_projection,
            development_alignment,
            device=device,
            precision=args.precision,
            minimum_patches=args.gate_minimum_patches,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            gallery_path=output / "stage_a_alignment.png",
        )
        stage_a_report = v35_stage_a_gate(
            alignment,
            causal_glyph_flow_boundary_receipt(model),
            initial_core_sha256=initial_core_hash,
            observed_core_sha256=module_state_sha256(model.backbone),
            initial_codec_sha256=initial_codec_hash,
            observed_codec_sha256=module_state_sha256(model.codec),
        )
        stage_a_report["aligned_adapter_sha256"] = module_state_sha256(
            model.input_adapter
        )
        stage_a_report["evidence_gate"] = evidence
        stage_a_report["requested_minimum_patches"] = args.gate_minimum_patches
        atomic_write_json(stage_a_report, output / "stage_a_report.json")
        print(json.dumps(stage_a_report, indent=2), flush=True)
        save_checkpoint()
        teacher_projection = teacher_projection.cpu()
        teacher_projection = None
        if device.type == "cuda":
            torch.cuda.empty_cache()
        if not stage_a_report["passed"] and not args.allow_failed_stage_a:
            raise RuntimeError("V35 Stage A did not pass its frozen gate")

    if stage_a_report is not None:
        teacher_projection = teacher_projection.cpu()
        teacher_projection = None
        if device.type == "cuda":
            torch.cuda.empty_cache()

    for stage_index, (stage, completed_updates) in enumerate(
        _stage_progress(global_update, stages)
    ):
        if completed_updates >= stage.updates:
            if stage_index == 0 and stage_a_report is None:
                evaluate_and_close_stage_a()
            if args.alignment_only and stage_index == 0:
                break
            continue
        if stage_index > 0 and stage_a_report is None:
            raise RuntimeError("V35 cannot enter causal training before Stage A evaluation")
        if stage_index > 0 and not stage_a_report["passed"]:
            if not args.allow_failed_stage_a:
                raise RuntimeError("V35 Stage A failed; causal training is forbidden")
        trainability = "visual-interface-alignment" if stage_index == 0 else "causal"
        set_v35_stage_trainability(model, trainability)
        model.train()
        loader = _stage_loader(
            datasets[stage.name],
            stage=stage,
            completed_updates=completed_updates,
            batch_size=args.batch_size,
            gradient_accumulation=args.gradient_accumulation,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
            seed=args.seed + stage_index * 10_000,
        )
        stage_started = time.monotonic()
        latest_metrics: dict[str, float] = {}
        for stage_update in range(completed_updates + 1, stage.updates + 1):
            warmup = min(args.stage_warmup, max(0, stage.updates - 1))
            adapter_lr = stage_cosine_learning_rate(
                stage_update,
                peak=stage.adapter_learning_rate,
                warmup=warmup,
                total=stage.updates,
                minimum_ratio=args.minimum_learning_rate_ratio,
            )
            head_lr = stage_cosine_learning_rate(
                stage_update,
                peak=stage.head_learning_rate,
                warmup=warmup,
                total=stage.updates,
                minimum_ratio=args.minimum_learning_rate_ratio,
            )
            core_lr = stage_cosine_learning_rate(
                stage_update,
                peak=stage.core_learning_rate,
                warmup=warmup,
                total=stage.updates,
                minimum_ratio=args.minimum_learning_rate_ratio,
            )
            set_v35_optimizer_learning_rates(
                optimizer,
                adapter=adapter_lr,
                head=head_lr,
                core=core_lr,
            )
            optimizer.zero_grad(set_to_none=True)
            accumulated: dict[str, float] = {}
            active_patches = 0.0
            for _ in range(args.gradient_accumulation):
                raw_batch = next(loader)
                student = {
                    key: value.to(device, non_blocking=True)
                    for key, value in causal_glyph_flow_student_batch(raw_batch).items()
                }
                student = _trim_batch(student)
                with autocast_context(device, args.precision):
                    if stage_index == 0:
                        if teacher_projection is None:
                            raise RuntimeError("V35 alignment projection was released early")
                        losses = visual_interface_alignment_loss(
                            model,
                            student["pixels"],
                            student["patch_mask"],
                            teacher_projection,
                        )
                    else:
                        model_output = model(student["pixels"], student["patch_mask"])
                        losses = causal_glyph_flow_loss(
                            model,
                            model_output,
                            student,
                            generator=flow_generator,
                        )
                    scaled_loss = losses.loss / args.gradient_accumulation
                if not bool(torch.isfinite(scaled_loss)):
                    raise FloatingPointError("V35 encountered a non-finite loss")
                if scaler.is_enabled():
                    scaler.scale(scaled_loss).backward()
                else:
                    scaled_loss.backward()
                for key, value in losses.detached_metrics().items():
                    accumulated[key] = (
                        accumulated.get(key, 0.0)
                        + value / args.gradient_accumulation
                    )
                active_patches += float(student["patch_mask"].sum())
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
                raise FloatingPointError("V35 encountered a non-finite gradient")
            if scaler.is_enabled():
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            if not _trainable_parameters_are_finite(model):
                raise FloatingPointError("V35 encountered a non-finite parameter")
            ema.update(model)
            global_update += 1
            peak_vram = (
                torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
            )
            if peak_vram >= MAXIMUM_VRAM_BYTES:
                raise MemoryError("V35 reached the preregistered 20 GiB VRAM stop rule")
            latest_metrics = {
                "global_update": global_update,
                "stage": stage.name,
                "stage_update": stage_update,
                "adapter_lr": adapter_lr,
                "head_lr": head_lr,
                "core_lr": core_lr,
                "gradient_norm": float(gradient_norm),
                "active_input_patches": active_patches,
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
        stage_summaries[stage.name] = {
            "updates_completed": stage_update,
            "updates_planned": stage.updates,
            "elapsed_seconds": time.monotonic() - stage_started,
            "latest_metrics": latest_metrics,
        }
        save_checkpoint()
        if stop_requested:
            break

        if stage_index == 0:
            evaluate_and_close_stage_a()
            if args.alignment_only:
                break

    current_elapsed = elapsed_before + time.monotonic() - started
    save_checkpoint()
    summary = {
        "experiment": EXPERIMENT,
        "global_update": global_update,
        "planned_updates": sum(stage.updates for stage in stages),
        "alignment_only": args.alignment_only,
        "stage_a_passed": bool(stage_a_report and stage_a_report["passed"]),
        "stage_summaries": stage_summaries,
        "training_elapsed_seconds": current_elapsed,
        "peak_allocated_vram_bytes": (
            torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
        ),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "stopped_by_signal": stop_requested,
    }
    atomic_write_json(summary, output / "training_summary.json")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

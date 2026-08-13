#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import random
import tempfile
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from ilm.visual_lm.visual_semantic_plan import (
    PIXEL_LINGUIST_REVISION,
    PIXEL_LINGUIST_WEIGHT_SHA256,
    VisualSemanticPlanConfig,
    VisualSentenceImageTeacher,
    file_sha256,
    load_pixel_linguist_reader,
    resolve_pixel_linguist_checkpoint,
)
from ilm.visual_lm.visual_semantic_plan_data import (
    VisualSemanticPlanAnswerDataset,
    VisualSemanticPlanRenderConfig,
    load_v36_instruction_records,
    visual_semantic_plan_answer_collate,
)
from ilm.visual_lm.visual_semantic_plan_training import (
    VisualSemanticPlanTargetBank,
    encode_visual_semantic_teacher_targets,
)


EXPERIMENT = "visual-semantic-plan-target-bank-v36"
PROTOCOL_DOCUMENT = "references/visual_semantic_plan_v36_protocol.md"
EXPECTED_PROTOCOL_SHA256 = (
    "7e637698af08803c4ef509c564160ea63e5a952398a1e50cd924ec888167d6fb"
)
DEFAULT_INSTRUCTION_MANIFEST = "data/raw/alpaca_zh.json"
EXPECTED_INSTRUCTION_SHA256 = (
    "6fcb98c6d79691d1f9a88ef513335da9124e7fdeef5103343f5d9f9a6e8f4903"
)
SEED = 20_263_600
SOURCE_FILES = (
    "ilm/visual_lm/visual_semantic_plan.py",
    "ilm/visual_lm/visual_semantic_plan_data.py",
    "ilm/visual_lm/visual_semantic_plan_training.py",
    "scripts/build_visual_semantic_plan_targets_v36.py",
)
ANSWER_KEYS = (
    "answer_pixels",
    "answer_mask",
    "answer_chunk_pixels",
    "answer_chunk_mask",
    "answer_length",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build detached answer-image plan targets for V36."
    )
    parser.add_argument("--instruction-manifest", default=DEFAULT_INSTRUCTION_MANIFEST)
    parser.add_argument("--split", choices=("train", "development"), default="train")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument(
        "--out",
        default="artifacts/visual_semantic_plan_v36_targets/train.pt",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--precision", choices=("fp32", "fp16", "bf16"), default="bf16")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--maximum-records", type=int, default=0)
    parser.add_argument("--tiny-model", action="store_true")
    parser.add_argument("--random-foundation", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def effective_arguments(args: argparse.Namespace) -> argparse.Namespace:
    values = vars(args).copy()
    if args.smoke:
        values.update(
            {
                "device": "cpu",
                "precision": "fp32",
                "batch_size": 2,
                "num_workers": 0,
                "maximum_records": args.maximum_records or 4,
                "tiny_model": True,
                "random_foundation": True,
            }
        )
    return argparse.Namespace(**values)


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


def choose_device(value: str) -> torch.device:
    device = torch.device(value)
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda:0")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("V36 requested CUDA but CUDA is unavailable")
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


@torch.no_grad()
def build_target_bank(
    teacher: VisualSentenceImageTeacher,
    batches: Iterable[Mapping[str, Any]],
    *,
    device: torch.device,
    precision: str,
    receipt: Mapping[str, Any],
) -> VisualSemanticPlanTargetBank:
    identifiers: list[str] = []
    global_plans: list[torch.Tensor] = []
    chunk_plans: list[torch.Tensor] = []
    chunk_active: list[torch.Tensor] = []
    lengths: list[torch.Tensor] = []
    teacher.requires_grad_(False).eval()
    for raw_batch in batches:
        batch = {
            key: raw_batch[key].to(device, non_blocking=device.type == "cuda")
            for key in ANSWER_KEYS
        }
        with autocast_context(device, precision):
            targets = encode_visual_semantic_teacher_targets(teacher, batch)
        identifiers.extend(
            str(metadata["identifier"]) for metadata in raw_batch["metadata"]
        )
        global_plans.append(targets.global_plan.detach().half().cpu())
        chunk_plans.append(targets.chunk_plans.detach().half().cpu())
        chunk_active.append(targets.chunk_active.detach().half().cpu())
        lengths.append(targets.length.detach().half().cpu())
    if not identifiers:
        raise ValueError("V36 target builder received no records")
    bank = VisualSemanticPlanTargetBank(
        identifiers=tuple(identifiers),
        global_plans=torch.cat(global_plans),
        chunk_plans=torch.cat(chunk_plans),
        chunk_active=torch.cat(chunk_active),
        lengths=torch.cat(lengths),
        receipt=dict(receipt),
    )
    if not all(
        bool(torch.isfinite(value).all())
        for value in (
            bank.global_plans,
            bank.chunk_plans,
            bank.chunk_active,
            bank.lengths,
        )
    ):
        raise FloatingPointError("V36 target bank contains non-finite values")
    return bank


def main() -> None:
    args = effective_arguments(parse_args())
    if min(args.batch_size, args.num_workers + 1) < 1 or args.maximum_records < 0:
        raise ValueError("V36 target-builder sizes are invalid")
    if args.random_foundation != args.tiny_model:
        raise ValueError("V36 random foundations and tiny models must be used together")
    if not args.smoke and (args.tiny_model or args.random_foundation):
        raise ValueError("V36 production target banks require Pixel-Linguist")
    if file_sha256(PROTOCOL_DOCUMENT) != EXPECTED_PROTOCOL_SHA256:
        raise RuntimeError("V36 protocol changed after preregistration")
    manifest_sha256 = file_sha256(args.instruction_manifest)
    if not args.smoke and manifest_sha256 != EXPECTED_INSTRUCTION_SHA256:
        raise RuntimeError("V36 instruction data differs from preregistration")

    output = Path(args.out)
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"V36 target bank already exists: {output}")
    device = choose_device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)
    seed_everything(args.seed)
    config = v36_model_config(tiny=args.tiny_model)
    render_config = VisualSemanticPlanRenderConfig(augment=args.split == "train")
    records = load_v36_instruction_records(args.instruction_manifest)
    dataset = VisualSemanticPlanAnswerDataset(
        records,
        split=args.split,
        render_config=render_config,
        seed=args.seed,
    )
    eligible_records = len(dataset)
    selected_records = (
        eligible_records
        if args.maximum_records <= 0
        else min(eligible_records, args.maximum_records)
    )
    selected_dataset = Subset(dataset, range(selected_records))
    loader = DataLoader(
        selected_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
        collate_fn=visual_semantic_plan_answer_collate,
    )

    teacher = VisualSentenceImageTeacher(config)
    if args.random_foundation:
        initialization: dict[str, Any] = {
            "route": "random-tiny-smoke",
            "evidence_eligible": False,
        }
    else:
        checkpoint = resolve_pixel_linguist_checkpoint(
            args.checkpoint,
            local_files_only=args.checkpoint is None,
        )
        initialization = load_pixel_linguist_reader(teacher.reader, checkpoint)
        initialization["evidence_eligible"] = True
    teacher.requires_grad_(False).eval().to(device)
    receipt: dict[str, Any] = {
        "experiment": EXPERIMENT,
        "label": "smoke" if args.smoke else "evidence-input",
        "split": args.split,
        "seed": args.seed,
        "model_config": asdict(config),
        "render_config": asdict(render_config),
        "protocol": {
            "path": PROTOCOL_DOCUMENT,
            "sha256": EXPECTED_PROTOCOL_SHA256,
        },
        "source_sha256": {path: file_sha256(path) for path in SOURCE_FILES},
        "data": {
            "manifest": str(Path(args.instruction_manifest).resolve()),
            "sha256": manifest_sha256,
            "raw_records": len(records),
            "eligible_records": eligible_records,
            "rejected_records": len(dataset.rejected_identifiers),
            "rejected_identifiers": list(dataset.rejected_identifiers),
            "selected_records": selected_records,
        },
        "initialization": initialization,
        "teacher_frozen": True,
        "teacher_receives_prompt": False,
        "student_present": False,
        "storage_dtype": "float16",
        "external_revision": PIXEL_LINGUIST_REVISION,
        "external_weight_sha256": PIXEL_LINGUIST_WEIGHT_SHA256,
    }
    bank = build_target_bank(
        teacher,
        loader,
        device=device,
        precision=args.precision,
        receipt=receipt,
    )
    atomic_torch_save(bank.state_dict(), output)
    summary = receipt | {
        "target_bank": str(output.resolve()),
        "target_bank_sha256": file_sha256(output),
        "target_count": len(bank.identifiers),
        "target_dimension": bank.global_plans.shape[-1],
        "finite": True,
        "peak_allocated_vram_bytes": (
            torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
        ),
    }
    atomic_write_json(summary, output.with_suffix(".receipt.json"))
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

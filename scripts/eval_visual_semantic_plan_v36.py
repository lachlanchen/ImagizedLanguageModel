#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ilm.visual_lm.visual_semantic_plan import (
    PIXEL_LINGUIST_REVISION,
    PIXEL_LINGUIST_WEIGHT_SHA256,
    V36_ARCHITECTURE,
    VisualSemanticPlanConfig,
    VisualSemanticPlanModel,
    VisualSentenceImageTeacher,
    file_sha256,
    load_pixel_linguist_reader,
    resolve_pixel_linguist_checkpoint,
    visual_semantic_plan_boundary_receipt,
)
from ilm.visual_lm.visual_semantic_plan_data import (
    V36_DEVELOPMENT_FONTS,
    V36_PATCHES,
    V36_PATCH_SIZE,
    V36_TRAIN_FONTS,
    V36_WIDTH,
    VisualSemanticPlanPromptDataset,
    VisualSemanticPlanRenderConfig,
    load_v36_instruction_records,
    render_visual_sentence_strip,
    select_v36_instruction_records,
    visual_semantic_plan_prompt_collate,
)
from ilm.visual_lm.visual_semantic_plan_evaluation import (
    mean_plan_cosine,
    nearest_length_counterfactual_pairs,
    v36_semantic_plan_gate,
    visual_plan_control_metrics,
    visual_plan_counterfactual_assignment,
    visual_plan_retrieval_metrics,
)
from ilm.visual_lm.visual_semantic_plan_training import (
    VisualSemanticPlanTargetBank,
)
from ilm.visual_lm.visual_semantic_raster_data import (
    VisualRasterRecord,
    load_visual_raster_paraphrases,
    visual_raster_partition,
)


EXPERIMENT = "visual-semantic-plan-v36"
PROTOCOL_DOCUMENT = "references/visual_semantic_plan_v36_protocol.md"
EXPECTED_PROTOCOL_SHA256 = (
    "7e637698af08803c4ef509c564160ea63e5a952398a1e50cd924ec888167d6fb"
)
DEFAULT_INSTRUCTION_MANIFEST = "data/raw/alpaca_zh.json"
DEFAULT_PARAPHRASE_MANIFEST = "data/teacher/folio_paraphrases_zh_holdout.jsonl"
DEFAULT_DEVELOPMENT_BANK = (
    "artifacts/visual_semantic_plan_v36_targets/development.pt"
)
EXPECTED_INSTRUCTION_SHA256 = (
    "6fcb98c6d79691d1f9a88ef513335da9124e7fdeef5103343f5d9f9a6e8f4903"
)
EXPECTED_PARAPHRASE_SHA256 = (
    "132a8a257d31be3cf607d0313bf042dafaee7f2df08440bf7e012a502ed6c02f"
)
SEED = 20_263_600
SOURCE_FILES = (
    "ilm/visual_lm/visual_semantic_plan.py",
    "ilm/visual_lm/visual_semantic_plan_data.py",
    "ilm/visual_lm/visual_semantic_plan_evaluation.py",
    "scripts/eval_visual_semantic_plan_v36.py",
)


@dataclass
class PromptRasterSet:
    identifiers: tuple[str, ...]
    pixels: torch.Tensor
    mask: torch.Tensor

    def __len__(self) -> int:
        return len(self.identifiers)


@dataclass
class PlanInference:
    plans: torch.Tensor
    lengths: torch.Tensor
    elapsed_seconds: float
    examples_per_second: float
    finite: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the preregistered V36 visual semantic planner."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--development-bank", default=DEFAULT_DEVELOPMENT_BANK)
    parser.add_argument("--external-checkpoint", default=None)
    parser.add_argument("--instruction-manifest", default=DEFAULT_INSTRUCTION_MANIFEST)
    parser.add_argument("--paraphrase-manifest", default=DEFAULT_PARAPHRASE_MANIFEST)
    parser.add_argument("--out", default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--precision", choices=("fp32", "fp16", "bf16"), default="bf16")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--raw-weights", action="store_true")
    parser.add_argument("--smoke", action="store_true")
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
            }
        )
    return argparse.Namespace(**values)


def choose_device(value: str) -> torch.device:
    device = torch.device(value)
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda:0")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("V36 requested CUDA but CUDA is unavailable")
    return device


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


def tensors_are_finite(value: Any) -> bool:
    if isinstance(value, torch.Tensor):
        return not value.is_floating_point() or bool(torch.isfinite(value).all())
    if isinstance(value, Mapping):
        return all(tensors_are_finite(item) for item in value.values())
    if isinstance(value, (tuple, list)):
        return all(tensors_are_finite(item) for item in value)
    return True


def load_target_bank(path: str | Path) -> VisualSemanticPlanTargetBank:
    state = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(state, Mapping):
        raise TypeError("V36 development bank must contain a state mapping")
    return VisualSemanticPlanTargetBank.from_state_dict(state)


def load_checkpoint_model(
    path: str | Path,
    *,
    device: torch.device,
    raw_weights: bool,
) -> tuple[VisualSemanticPlanModel, dict[str, Any], dict[str, Any]]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("architecture") != V36_ARCHITECTURE:
        raise ValueError("evaluation checkpoint is not V36")
    config = VisualSemanticPlanConfig(**checkpoint["model_config"])
    model = VisualSemanticPlanModel(config)
    model.load_state_dict(checkpoint["model"], strict=True)
    weight_route = "raw"
    if not raw_weights:
        ema = checkpoint.get("ema")
        if not isinstance(ema, Mapping) or not isinstance(ema.get("shadow"), Mapping):
            if checkpoint.get("weight_route") != "selective-ema":
                raise ValueError("V36 checkpoint has no selective EMA state")
        else:
            parameters = dict(model.named_parameters())
            for name, value in ema["shadow"].items():
                if name not in parameters or parameters[name].shape != value.shape:
                    raise ValueError(f"V36 EMA parameter mismatch: {name}")
                parameters[name].data.copy_(value.to(parameters[name]))
        weight_route = "selective-ema"
    model.requires_grad_(False).eval().to(device)
    boundary = visual_semantic_plan_boundary_receipt(model)
    receipt = {
        "checkpoint": str(Path(path).resolve()),
        "checkpoint_sha256": file_sha256(path),
        "weight_route": weight_route,
        "global_update": int(checkpoint.get("global_update", 0)),
        "finite_model_state": tensors_are_finite(model.state_dict()),
        "boundary": boundary,
    }
    return model, checkpoint, receipt


def select_bank_records(
    records: Sequence[VisualRasterRecord],
    bank: VisualSemanticPlanTargetBank,
) -> list[VisualRasterRecord]:
    by_identifier = {record.identifier: record for record in records}
    try:
        return [by_identifier[identifier] for identifier in bank.identifiers]
    except KeyError as error:
        raise KeyError(f"V36 manifest lacks bank record {error.args[0]!r}") from error


def collect_prompt_rasters(
    records: Sequence[VisualRasterRecord],
    *,
    render_config: VisualSemanticPlanRenderConfig,
    batch_size: int,
    num_workers: int,
) -> PromptRasterSet:
    dataset = VisualSemanticPlanPromptDataset(
        records,
        split="development",
        render_config=render_config,
        seed=SEED,
        length=len(records),
        include_all_records=True,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
        collate_fn=visual_semantic_plan_prompt_collate,
    )
    identifiers: list[str] = []
    pixels: list[torch.Tensor] = []
    masks: list[torch.Tensor] = []
    for batch in loader:
        pixels.append(batch["prompt_pixels"])
        masks.append(batch["prompt_mask"])
        identifiers.extend(
            str(metadata["identifier"]) for metadata in batch["metadata"]
        )
    if len(set(identifiers)) != len(records):
        raise RuntimeError("V36 evaluation prompt stream is not one-to-one")
    return PromptRasterSet(
        identifiers=tuple(identifiers),
        pixels=torch.cat(pixels),
        mask=torch.cat(masks),
    )


def render_records_with_font(
    records: Sequence[VisualRasterRecord],
    *,
    field: str,
    font_path: str,
    render_config: VisualSemanticPlanRenderConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    if field not in {"prompt", "answer"}:
        raise ValueError("V36 can only render prompt or answer fields")
    pixels: list[torch.Tensor] = []
    masks: list[torch.Tensor] = []
    for index, record in enumerate(records):
        image, mask, _ = render_visual_sentence_strip(
            getattr(record, field),
            config=render_config,
            font_path=font_path,
            variant=SEED + index,
            force_origin=0,
        )
        pixels.append(image)
        masks.append(mask)
    return torch.stack(pixels), torch.stack(masks)


def controlled_prompt_rasters(
    prompts: PromptRasterSet,
    condition: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    if condition == "correct":
        return prompts.pixels, prompts.mask
    if condition == "shuffled":
        permutation = torch.roll(torch.arange(len(prompts)), shifts=1)
        return prompts.pixels[permutation], prompts.mask[permutation]
    if condition == "blank":
        return torch.ones_like(prompts.pixels), torch.zeros_like(prompts.mask)
    if condition == "final-quarter":
        pixels = torch.ones_like(prompts.pixels)
        pixels[..., -V36_WIDTH // 4 :] = prompts.pixels[..., -V36_WIDTH // 4 :]
        mask = torch.zeros_like(prompts.mask)
        mask[:, -V36_PATCHES // 4 :] = prompts.mask[:, -V36_PATCHES // 4 :]
        return pixels, mask
    raise ValueError(f"unknown V36 prompt condition: {condition}")


@torch.no_grad()
def infer_plans(
    model: VisualSemanticPlanModel,
    pixels: torch.Tensor,
    mask: torch.Tensor,
    *,
    device: torch.device,
    precision: str,
    batch_size: int,
) -> PlanInference:
    if pixels.shape != (len(mask), 3, V36_PATCH_SIZE, V36_WIDTH):
        raise ValueError("V36 inference rasters do not align")
    plans: list[torch.Tensor] = []
    lengths: list[torch.Tensor] = []
    started = time.perf_counter()
    for start in range(0, len(pixels), batch_size):
        batch_pixels = pixels[start : start + batch_size].to(device)
        batch_mask = mask[start : start + batch_size].to(device)
        with autocast_context(device, precision):
            output = model.generate_plan(batch_pixels, batch_mask)
        plans.append(output.plans[:, 0].float().cpu())
        lengths.append(output.length.float().cpu())
    elapsed = time.perf_counter() - started
    plan_tensor = torch.cat(plans)
    length_tensor = torch.cat(lengths)
    return PlanInference(
        plans=plan_tensor,
        lengths=length_tensor,
        elapsed_seconds=elapsed,
        examples_per_second=len(pixels) / max(elapsed, 1e-9),
        finite=bool(torch.isfinite(plan_tensor).all() and torch.isfinite(length_tensor).all()),
    )


@torch.no_grad()
def encode_teacher_rasters(
    teacher: VisualSentenceImageTeacher,
    pixels: torch.Tensor,
    mask: torch.Tensor,
    *,
    device: torch.device,
    precision: str,
    batch_size: int,
) -> torch.Tensor:
    output: list[torch.Tensor] = []
    for start in range(0, len(pixels), batch_size):
        with autocast_context(device, precision):
            output.append(
                teacher(
                    pixels[start : start + batch_size].to(device),
                    mask[start : start + batch_size].to(device),
                )
                .float()
                .cpu()
            )
    return torch.cat(output)


def indexed_plan_retrieval_metrics(
    plans: torch.Tensor,
    candidates: torch.Tensor,
    correct_indices: torch.Tensor,
) -> dict[str, float | int]:
    if plans.ndim != 2 or candidates.ndim != 2 or plans.shape[1] != candidates.shape[1]:
        raise ValueError("V36 indexed retrieval matrices are invalid")
    if correct_indices.shape != (len(plans),):
        raise ValueError("V36 indexed retrieval labels do not align")
    if not bool(((correct_indices >= 0) & (correct_indices < len(candidates))).all()):
        raise ValueError("V36 indexed retrieval label is outside the candidate bank")
    plans = F.normalize(plans.float(), dim=-1)
    candidates = F.normalize(candidates.float(), dim=-1)
    similarities = plans @ candidates.T
    ranking = similarities.argsort(dim=1, descending=True)
    positions = (ranking == correct_indices[:, None]).nonzero(as_tuple=False)[:, 1]
    correct_cosine = similarities[
        torch.arange(len(plans)),
        correct_indices,
    ]
    return {
        "samples": len(plans),
        "candidates": len(candidates),
        "top1": float((positions == 0).float().mean()),
        "top5": float((positions < min(5, len(candidates))).float().mean()),
        "mrr": float((1.0 / (positions.float() + 1.0)).mean()),
        "correct_cosine": float(correct_cosine.mean()),
    }


def build_visual_teacher(
    config: VisualSemanticPlanConfig,
    *,
    checkpoint_path: str | Path | None,
    smoke: bool,
    device: torch.device,
) -> tuple[VisualSentenceImageTeacher, dict[str, Any]]:
    with torch.random.fork_rng(
        devices=[device.index] if device.type == "cuda" else [],
    ):
        torch.manual_seed(SEED)
        teacher = VisualSentenceImageTeacher(config)
    if smoke:
        receipt: dict[str, Any] = {
            "route": "random-tiny-smoke",
            "evidence_eligible": False,
        }
    else:
        checkpoint = resolve_pixel_linguist_checkpoint(
            checkpoint_path,
            local_files_only=checkpoint_path is None,
        )
        receipt = load_pixel_linguist_reader(teacher.reader, checkpoint)
        receipt["evidence_eligible"] = True
    teacher.requires_grad_(False).eval().to(device)
    return teacher, receipt


def build_untrained_head_baseline(
    config: VisualSemanticPlanConfig,
    *,
    checkpoint_path: str | Path | None,
    smoke: bool,
    device: torch.device,
) -> VisualSemanticPlanModel:
    with torch.random.fork_rng(
        devices=[device.index] if device.type == "cuda" else [],
    ):
        torch.manual_seed(SEED)
        model = VisualSemanticPlanModel(config)
    if not smoke:
        checkpoint = resolve_pixel_linguist_checkpoint(
            checkpoint_path,
            local_files_only=checkpoint_path is None,
        )
        load_pixel_linguist_reader(model.reader, checkpoint)
    return model.requires_grad_(False).eval().to(device)


def _integrity_report(
    model: VisualSemanticPlanModel,
    checkpoint: Mapping[str, Any],
    target_bank: VisualSemanticPlanTargetBank,
    *,
    instruction_sha256: str,
    paraphrase_sha256: str,
    smoke: bool,
) -> dict[str, Any]:
    run_receipt = checkpoint.get("run_receipt", {})
    source_hashes = run_receipt.get("source_sha256", {})
    source_ok = bool(source_hashes) and all(
        Path(path).is_file() and file_sha256(path) == digest
        for path, digest in source_hashes.items()
    )
    target_source_hashes = target_bank.receipt.get("source_sha256", {})
    source_ok = source_ok and bool(target_source_hashes) and all(
        Path(path).is_file() and file_sha256(path) == digest
        for path, digest in target_source_hashes.items()
    )
    initialization = run_receipt.get("initialization", {})
    target_initialization = target_bank.receipt.get("initialization", {})
    external_ok = smoke or (
        initialization.get("sha256") == PIXEL_LINGUIST_WEIGHT_SHA256
        and initialization.get("revision") == PIXEL_LINGUIST_REVISION
        and target_initialization.get("sha256") == PIXEL_LINGUIST_WEIGHT_SHA256
        and target_initialization.get("revision") == PIXEL_LINGUIST_REVISION
    )
    data_ok = (
        instruction_sha256 == EXPECTED_INSTRUCTION_SHA256
        and paraphrase_sha256 == EXPECTED_PARAPHRASE_SHA256
        and target_bank.receipt.get("data", {}).get("sha256")
        == EXPECTED_INSTRUCTION_SHA256
    )
    strict_mapping = smoke or (
        initialization.get("selected_tensors") == 198
        and initialization.get("missing_keys") == []
        and initialization.get("unexpected_keys") == []
        and target_initialization.get("selected_tensors") == 198
    )
    boundary = visual_semantic_plan_boundary_receipt(model)
    boundary_ok = (
        not boundary["forbidden_parameter_names"]
        and boundary["parameter_cap_pass"]
        and not boundary["uses_strings"]
        and not boundary["uses_token_ids"]
        and not boundary["uses_ocr"]
        and not boundary["candidate_bank_deployed"]
        and not checkpoint.get("contains_target_tensors", False)
        and not checkpoint.get("contains_answer_teacher", False)
    )
    return {
        "source_hashes": source_ok,
        "external_hashes": external_ok,
        "data_hashes": data_ok,
        "strict_mapping": strict_mapping,
        "boundary": boundary_ok,
        "total_parameters": boundary["total_parameters"],
        "finite_checkpoint": bool(checkpoint.get("finite", True)),
        "finite_target_bank": tensors_are_finite(target_bank.state_dict()),
        "model_boundary": boundary,
        "evaluation_source_sha256": {
            path: file_sha256(path) for path in SOURCE_FILES
        },
    }


def main() -> None:
    args = effective_arguments(parse_args())
    if args.batch_size < 1 or args.num_workers < 0:
        raise ValueError("V36 evaluation batch settings are invalid")
    if file_sha256(PROTOCOL_DOCUMENT) != EXPECTED_PROTOCOL_SHA256:
        raise RuntimeError("V36 protocol changed after preregistration")
    instruction_sha256 = file_sha256(args.instruction_manifest)
    paraphrase_sha256 = file_sha256(args.paraphrase_manifest)
    if not args.smoke and (
        instruction_sha256 != EXPECTED_INSTRUCTION_SHA256
        or paraphrase_sha256 != EXPECTED_PARAPHRASE_SHA256
    ):
        raise RuntimeError("V36 development data differs from preregistration")
    device = choose_device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)

    model, checkpoint, checkpoint_receipt = load_checkpoint_model(
        args.checkpoint,
        device=device,
        raw_weights=args.raw_weights,
    )
    if checkpoint.get("protocol", {}).get("sha256") != EXPECTED_PROTOCOL_SHA256:
        raise ValueError("V36 checkpoint has a different protocol")
    config = model.config
    all_records = load_v36_instruction_records(args.instruction_manifest)
    render_config = VisualSemanticPlanRenderConfig(augment=False)
    selected_records, _ = select_v36_instruction_records(
        all_records,
        split="development",
        render_config=render_config,
    )
    prompts = collect_prompt_rasters(
        selected_records,
        render_config=render_config,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    record_by_identifier = {record.identifier: record for record in selected_records}
    ordered_records = [record_by_identifier[identifier] for identifier in prompts.identifiers]

    correct_pixels, correct_mask = controlled_prompt_rasters(prompts, "correct")
    correct_output = infer_plans(
        model,
        correct_pixels,
        correct_mask,
        device=device,
        precision=args.precision,
        batch_size=args.batch_size,
    )
    if not correct_output.finite:
        raise FloatingPointError("V36 correct-prompt inference is non-finite")

    # Candidate targets are intentionally loaded only after autonomous student inference.
    bank = load_target_bank(args.development_bank)
    if bank.receipt.get("split") != "development":
        raise ValueError("V36 evaluator requires a development target bank")
    if bank.receipt.get("protocol", {}).get("sha256") != EXPECTED_PROTOCOL_SHA256:
        raise ValueError("V36 development bank has a different protocol")
    if bank.receipt.get("model_config") != asdict(config):
        raise ValueError("V36 development bank has a different model configuration")
    bank_data = bank.receipt.get("data", {})
    if bank_data.get("selected_records") != bank_data.get("eligible_records"):
        raise ValueError("V36 evaluation requires the complete development bank")
    if set(bank.identifiers) != set(prompts.identifiers):
        raise ValueError("V36 development prompts and answer candidates differ")
    targets = bank.lookup(
        prompts.identifiers,
        device="cpu",
        dtype=torch.float32,
    )
    target_plans = targets.global_plan
    target_lengths = targets.length
    correct_metrics = visual_plan_retrieval_metrics(
        correct_output.plans,
        target_plans,
        predicted_lengths=correct_output.lengths,
        target_lengths=target_lengths,
    )
    controls: dict[str, Any] = {}
    for name in ("shuffled", "blank", "final-quarter"):
        pixels, mask = controlled_prompt_rasters(prompts, name)
        output = infer_plans(
            model,
            pixels,
            mask,
            device=device,
            precision=args.precision,
            batch_size=args.batch_size,
        )
        controls[name.replace("-", "_")] = visual_plan_control_metrics(
            output.plans,
            correct_output.plans,
            target_plans,
            predicted_lengths=output.lengths,
            target_lengths=target_lengths,
        ) | {
            "finite": output.finite,
            "elapsed_seconds": output.elapsed_seconds,
        }

    held_font = next(path for path in V36_TRAIN_FONTS if Path(path).is_file())
    held_prompt_pixels, held_prompt_mask = render_records_with_font(
        ordered_records,
        field="prompt",
        font_path=held_font,
        render_config=render_config,
    )
    held_output = infer_plans(
        model,
        held_prompt_pixels,
        held_prompt_mask,
        device=device,
        precision=args.precision,
        batch_size=args.batch_size,
    )

    nonsealed_records = [
        record
        for record in all_records
        if visual_raster_partition(record.identifier, stream="instruction") != "sealed"
    ]
    paraphrase_records = load_visual_raster_paraphrases(
        args.paraphrase_manifest,
        nonsealed_records,
    )
    if len(paraphrase_records) != 30:
        raise RuntimeError(
            f"V36 expected 30 nonsealed paraphrases, found {len(paraphrase_records)}"
        )
    nonsealed_by_identifier = {
        record.identifier: record for record in nonsealed_records
    }
    paraphrase_source_ids = [
        record.identifier.removeprefix("paraphrase:")
        for record in paraphrase_records
    ]
    paraphrase_source_records = [
        nonsealed_by_identifier[identifier] for identifier in paraphrase_source_ids
    ]
    development_font = next(
        path for path in V36_DEVELOPMENT_FONTS if Path(path).is_file()
    )
    paraphrase_pixels, paraphrase_mask = render_records_with_font(
        paraphrase_records,
        field="prompt",
        font_path=development_font,
        render_config=render_config,
    )
    original_paraphrase_pixels, original_paraphrase_mask = render_records_with_font(
        paraphrase_source_records,
        field="prompt",
        font_path=development_font,
        render_config=render_config,
    )
    paraphrase_output = infer_plans(
        model,
        paraphrase_pixels,
        paraphrase_mask,
        device=device,
        precision=args.precision,
        batch_size=args.batch_size,
    )
    original_paraphrase_output = infer_plans(
        model,
        original_paraphrase_pixels,
        original_paraphrase_mask,
        device=device,
        precision=args.precision,
        batch_size=args.batch_size,
    )

    teacher, teacher_receipt = build_visual_teacher(
        config,
        checkpoint_path=args.external_checkpoint,
        smoke=args.smoke,
        device=device,
    )
    direct_prompt_plans = encode_teacher_rasters(
        teacher,
        correct_pixels,
        correct_mask,
        device=device,
        precision=args.precision,
        batch_size=args.batch_size,
    )
    alternate_answer_pixels, alternate_answer_mask = render_records_with_font(
        ordered_records,
        field="answer",
        font_path=held_font,
        render_config=render_config,
    )
    alternate_answer_plans = encode_teacher_rasters(
        teacher,
        alternate_answer_pixels,
        alternate_answer_mask,
        device=device,
        precision=args.precision,
        batch_size=args.batch_size,
    )
    paraphrase_answer_pixels, paraphrase_answer_mask = render_records_with_font(
        paraphrase_source_records,
        field="answer",
        font_path=development_font,
        render_config=render_config,
    )
    paraphrase_answer_plans = encode_teacher_rasters(
        teacher,
        paraphrase_answer_pixels,
        paraphrase_answer_mask,
        device=device,
        precision=args.precision,
        batch_size=args.batch_size,
    )
    font_metrics = {
        "prompt_plan_cosine": mean_plan_cosine(
            held_output.plans,
            correct_output.plans,
        ),
        "answer_teacher_cosine": mean_plan_cosine(
            alternate_answer_plans,
            target_plans,
        ),
        "held_train_font": visual_plan_control_metrics(
            held_output.plans,
            correct_output.plans,
            target_plans,
            predicted_lengths=held_output.lengths,
            target_lengths=target_lengths,
        ),
        "held_font_path": held_font,
    }
    direct_baseline = visual_plan_retrieval_metrics(
        direct_prompt_plans,
        target_plans,
    )
    paraphrase_metrics = indexed_plan_retrieval_metrics(
        paraphrase_output.plans,
        paraphrase_answer_plans,
        torch.arange(len(paraphrase_records)),
    )
    paraphrase_metrics["original_plan_cosine"] = mean_plan_cosine(
        paraphrase_output.plans,
        original_paraphrase_output.plans,
    )
    paraphrase_metrics["source_split_counts"] = {
        split: sum(
            visual_raster_partition(identifier, stream="instruction") == split
            for identifier in paraphrase_source_ids
        )
        for split in ("train", "development")
    }
    paraphrase_metrics["sealed_source_rows_excluded"] = 1
    teacher = teacher.cpu()
    del teacher
    if device.type == "cuda":
        torch.cuda.empty_cache()

    untrained_model = build_untrained_head_baseline(
        config,
        checkpoint_path=args.external_checkpoint,
        smoke=args.smoke,
        device=device,
    )
    untrained_output = infer_plans(
        untrained_model,
        correct_pixels,
        correct_mask,
        device=device,
        precision=args.precision,
        batch_size=args.batch_size,
    )
    untrained_baseline = visual_plan_retrieval_metrics(
        untrained_output.plans,
        target_plans,
        predicted_lengths=untrained_output.lengths,
        target_lengths=target_lengths,
    )
    untrained_model = untrained_model.cpu()
    del untrained_model

    answers = [record.answer for record in ordered_records]
    pairs = nearest_length_counterfactual_pairs(
        target_lengths,
        target_plans,
        labels=answers,
    )
    counterfactual = visual_plan_counterfactual_assignment(
        correct_output.plans,
        target_plans,
        pairs,
    )
    most_frequent = min(
        range(len(answers)),
        key=lambda index: (-answers.count(answers[index]), index),
    )
    frequency_plans = target_plans[most_frequent].expand_as(target_plans)
    frequency_baseline = visual_plan_retrieval_metrics(
        frequency_plans,
        target_plans,
    )

    output_path = Path(args.out) if args.out else Path(args.checkpoint).with_name(
        "development_report_raw.json"
        if args.raw_weights
        else "development_report_ema.json"
    )
    matrix_path = output_path.with_name(output_path.stem + "_similarity.pt")
    similarity = F.normalize(correct_output.plans, dim=-1) @ F.normalize(
        target_plans,
        dim=-1,
    ).T
    atomic_torch_save(
        {
            "identifiers": list(prompts.identifiers),
            "similarity": similarity.half(),
        },
        matrix_path,
    )
    report: dict[str, Any] = {
        "experiment": EXPERIMENT,
        "split": "development",
        "weight_route": checkpoint_receipt["weight_route"],
        "checkpoint": checkpoint_receipt,
        "integrity": _integrity_report(
            model,
            checkpoint,
            bank,
            instruction_sha256=instruction_sha256,
            paraphrase_sha256=paraphrase_sha256,
            smoke=args.smoke,
        ),
        "training": {
            "global_update": int(checkpoint.get("global_update", 0)),
            "finite": bool(checkpoint.get("finite", True)),
        },
        "resources": {
            "peak_vram_bytes": int(
                checkpoint.get("peak_allocated_vram_bytes", 0)
            ),
            "evaluation_peak_vram_bytes": (
                torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
            ),
            "correct_inference_seconds": correct_output.elapsed_seconds,
            "correct_examples_per_second": correct_output.examples_per_second,
        },
        "data": {
            "instruction_sha256": instruction_sha256,
            "paraphrase_sha256": paraphrase_sha256,
            "development_records": len(prompts),
            "paraphrase_records": len(paraphrase_records),
            "target_bank_sha256": file_sha256(args.development_bank),
            "primary_candidate_bank_created_after_student_inference": True,
            "paraphrase_candidate_bank_created_after_student_inference": True,
        },
        "correct": correct_metrics,
        "controls": controls,
        "counterfactual": counterfactual,
        "font": font_metrics,
        "paraphrase": paraphrase_metrics,
        "baselines": {
            "untrained_head": untrained_baseline,
            "direct_pixel_linguist": direct_baseline,
            "answer_frequency": frequency_baseline,
            "cyclic_prompt": controls["shuffled"],
            "blank_prompt": controls["blank"],
        },
        "teacher": teacher_receipt,
        "similarity_matrix": {
            "path": str(matrix_path.resolve()),
            "sha256": file_sha256(matrix_path),
            "shape": list(similarity.shape),
        },
        "finite": all(
            (
                correct_output.finite,
                held_output.finite,
                paraphrase_output.finite,
                original_paraphrase_output.finite,
                untrained_output.finite,
                tensors_are_finite(direct_prompt_plans),
                tensors_are_finite(alternate_answer_plans),
                tensors_are_finite(paraphrase_answer_plans),
                tensors_are_finite(similarity),
            )
        ),
        "sealed_opened": False,
    }
    report["gate"] = v36_semantic_plan_gate(report)
    atomic_write_json(report, output_path)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

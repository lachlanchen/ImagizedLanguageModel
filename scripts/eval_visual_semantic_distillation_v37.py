#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F

from ilm.visual_lm.visual_semantic_distillation import (
    PIXEL_LINGUIST_REVISION,
    PIXEL_LINGUIST_WEIGHT_SHA256,
    V37_ARCHITECTURE,
    VisualSemanticDistillationConfig,
    VisualSemanticDistillationModel,
    file_sha256,
    load_v37_pixel_linguist_initialization,
    visual_semantic_distillation_boundary_receipt,
)
from ilm.visual_lm.visual_semantic_distillation_data import (
    V37_DEVELOPMENT_FONT,
    V37_HELD_FONT,
    V37_PATCHES,
    V37_PATCH_SIZE,
    V37_WIDTH,
    VisualSemanticDistillationRenderConfig,
    load_v37_instruction_records,
    render_visual_semantic_distillation_strip,
    select_v37_instruction_records,
)
from ilm.visual_lm.visual_semantic_distillation_evaluation import (
    counterfactual_assignment,
    indexed_semantic_retrieval_metrics,
    mean_semantic_cosine,
    nearest_length_counterfactual_pairs,
    semantic_control_metrics,
    semantic_retrieval_metrics,
    v37_semantic_distillation_gate,
)
from ilm.visual_lm.visual_semantic_distillation_training import (
    VisualSemanticDistillationTargetBank,
)
from ilm.visual_lm.visual_semantic_raster_data import (
    VisualRasterRecord,
    load_visual_raster_paraphrases,
    visual_raster_partition,
)


EXPERIMENT = V37_ARCHITECTURE
PROTOCOL_DOCUMENT = "references/visual_semantic_distillation_v37_protocol.md"
EXPECTED_PROTOCOL_SHA256 = (
    "e3cca1c8eedb387f80a88cf17a93466f59532ea666d6dcbfe57e5d7d5e91f6d7"
)
DEFAULT_INSTRUCTION_MANIFEST = "data/raw/alpaca_zh.json"
DEFAULT_PARAPHRASE_MANIFEST = "data/teacher/folio_paraphrases_zh_holdout.jsonl"
DEFAULT_TRAIN_BANK = "artifacts/visual_semantic_distillation_v37_targets/train.pt"
DEFAULT_DEVELOPMENT_BANK = (
    "artifacts/visual_semantic_distillation_v37_targets/development.pt"
)
EXPECTED_INSTRUCTION_SHA256 = (
    "6fcb98c6d79691d1f9a88ef513335da9124e7fdeef5103343f5d9f9a6e8f4903"
)
EXPECTED_PARAPHRASE_SHA256 = (
    "132a8a257d31be3cf607d0313bf042dafaee7f2df08440bf7e012a502ed6c02f"
)
BGE_MANIFEST_SHA256 = "7907646426070047a77226ac3e684fbbe8410524f7b4a74d02837e43f2146bab"
BGE_MODEL_SHA256 = "daec91ffb5dd0c27411bd71f29932917c49cf529a641d0168496c3a501e3062c"
SEED = 20_263_700
SOURCE_FILES = (
    "ilm/visual_lm/visual_semantic_distillation.py",
    "ilm/visual_lm/visual_semantic_distillation_data.py",
    "ilm/visual_lm/visual_semantic_distillation_evaluation.py",
    "ilm/visual_lm/visual_semantic_distillation_training.py",
    "scripts/eval_visual_semantic_distillation_v37.py",
)


@dataclass
class RasterSet:
    identifiers: tuple[str, ...]
    pixels: torch.Tensor
    mask: torch.Tensor

    def __len__(self) -> int:
        return len(self.identifiers)


@dataclass
class SemanticInference:
    semantic_states: torch.Tensor
    answer_plans: torch.Tensor
    lengths: torch.Tensor
    pooled_visual_states: torch.Tensor
    elapsed_seconds: float
    examples_per_second: float
    finite: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the preregistered V37 image-native semantic student."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--train-bank", default=DEFAULT_TRAIN_BANK)
    parser.add_argument("--development-bank", default=DEFAULT_DEVELOPMENT_BANK)
    parser.add_argument("--external-checkpoint", default=None)
    parser.add_argument("--instruction-manifest", default=DEFAULT_INSTRUCTION_MANIFEST)
    parser.add_argument("--paraphrase-manifest", default=DEFAULT_PARAPHRASE_MANIFEST)
    parser.add_argument("--out", default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--precision", choices=("fp32", "fp16", "bf16"), default="bf16")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--raw-weights", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def effective_arguments(args: argparse.Namespace) -> argparse.Namespace:
    values = vars(args).copy()
    if args.smoke:
        values.update({"device": "cpu", "precision": "fp32", "batch_size": 8})
    return argparse.Namespace(**values)


def choose_device(value: str) -> torch.device:
    device = torch.device(value)
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda:0")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("V37 requested CUDA but CUDA is unavailable")
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


def _iter_tensor_paths(value: Any, path: tuple[str, ...] = ()):
    if isinstance(value, torch.Tensor):
        yield path, value
    elif isinstance(value, Mapping):
        for key, item in value.items():
            yield from _iter_tensor_paths(item, path + (str(key),))
    elif isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            yield from _iter_tensor_paths(item, path + (str(index),))


def checkpoint_tensor_boundary(checkpoint: Mapping[str, Any]) -> bool:
    forbidden = ("target", "teacher", "candidate", "bge")
    return not any(
        any(fragment in ".".join(path).lower() for fragment in forbidden)
        for path, _tensor in _iter_tensor_paths(checkpoint)
    )


def load_target_bank(path: str | Path) -> VisualSemanticDistillationTargetBank:
    state = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(state, Mapping):
        raise TypeError("V37 target bank must contain a state mapping")
    return VisualSemanticDistillationTargetBank.from_state_dict(state)


def load_checkpoint_model(
    path: str | Path,
    *,
    device: torch.device,
    raw_weights: bool,
) -> tuple[VisualSemanticDistillationModel, dict[str, Any], dict[str, Any]]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, Mapping):
        raise TypeError("V37 checkpoint must contain a state mapping")
    if checkpoint.get("architecture") != V37_ARCHITECTURE:
        raise ValueError("evaluation checkpoint is not V37")
    config = VisualSemanticDistillationConfig(**checkpoint["model_config"])
    model = VisualSemanticDistillationModel(config)
    model.load_state_dict(checkpoint["model"], strict=True)
    weight_route = "raw"
    if not raw_weights:
        ema = checkpoint.get("ema")
        if isinstance(ema, Mapping) and isinstance(ema.get("shadow"), Mapping):
            parameters = dict(model.named_parameters())
            names = tuple(str(name) for name in ema.get("names", ()))
            if set(names) != set(parameters) or set(ema["shadow"]) != set(parameters):
                raise ValueError("V37 checkpoint EMA is not all-parameter")
            for name, value in ema["shadow"].items():
                if parameters[name].shape != value.shape:
                    raise ValueError(f"V37 EMA parameter mismatch: {name}")
                parameters[name].data.copy_(value.to(parameters[name]))
        elif checkpoint.get("weight_route") != "all-parameter-ema":
            raise ValueError("V37 checkpoint has no all-parameter EMA state")
        weight_route = "all-parameter-ema"
    model.requires_grad_(False).eval().to(device)
    boundary = visual_semantic_distillation_boundary_receipt(model)
    receipt = {
        "checkpoint": str(Path(path).resolve()),
        "checkpoint_sha256": file_sha256(path),
        "weight_route": weight_route,
        "global_update": int(checkpoint.get("global_update", 0)),
        "finite_model_state": tensors_are_finite(model.state_dict()),
        "tensor_boundary": checkpoint_tensor_boundary(checkpoint),
        "boundary": boundary,
    }
    return model, dict(checkpoint), receipt


def render_records(
    records: Sequence[VisualRasterRecord],
    *,
    field: str,
    font_path: str,
    render_config: VisualSemanticDistillationRenderConfig,
) -> RasterSet:
    if field not in {"prompt", "answer"}:
        raise ValueError("V37 can only render prompt or answer fields")
    pixels: list[torch.Tensor] = []
    masks: list[torch.Tensor] = []
    for index, record in enumerate(records):
        image, mask, _metadata = render_visual_semantic_distillation_strip(
            getattr(record, field),
            config=render_config,
            font_path=font_path,
            font_size=render_config.evaluation_font_size,
            variant=SEED + index,
            force_origin=0,
        )
        pixels.append(image)
        masks.append(mask)
    return RasterSet(
        identifiers=tuple(record.identifier for record in records),
        pixels=torch.stack(pixels),
        mask=torch.stack(masks),
    )


def controlled_rasters(rasters: RasterSet, condition: str) -> RasterSet:
    if condition == "correct":
        return rasters
    if condition == "shuffled":
        permutation = torch.roll(torch.arange(len(rasters)), shifts=1)
        return RasterSet(
            identifiers=rasters.identifiers,
            pixels=rasters.pixels[permutation],
            mask=rasters.mask[permutation],
        )
    if condition == "blank":
        return RasterSet(
            identifiers=rasters.identifiers,
            pixels=torch.ones_like(rasters.pixels),
            mask=torch.zeros_like(rasters.mask),
        )
    if condition == "final-quarter":
        pixels = torch.ones_like(rasters.pixels)
        pixels[..., -V37_WIDTH // 4 :] = rasters.pixels[..., -V37_WIDTH // 4 :]
        mask = torch.zeros_like(rasters.mask)
        mask[:, -V37_PATCHES // 4 :] = rasters.mask[:, -V37_PATCHES // 4 :]
        return RasterSet(
            identifiers=rasters.identifiers,
            pixels=pixels,
            mask=mask,
        )
    raise ValueError(f"unknown V37 prompt condition: {condition}")


@torch.no_grad()
def infer_semantics(
    model: VisualSemanticDistillationModel,
    rasters: RasterSet,
    *,
    device: torch.device,
    precision: str,
    batch_size: int,
) -> SemanticInference:
    if rasters.pixels.shape != (
        len(rasters),
        3,
        V37_PATCH_SIZE,
        V37_WIDTH,
    ) or rasters.mask.shape != (len(rasters), V37_PATCHES):
        raise ValueError("V37 inference rasters do not align")
    semantic: list[torch.Tensor] = []
    plans: list[torch.Tensor] = []
    lengths: list[torch.Tensor] = []
    pooled: list[torch.Tensor] = []
    started = time.perf_counter()
    for start in range(0, len(rasters), batch_size):
        pixels = rasters.pixels[start : start + batch_size].to(device)
        mask = rasters.mask[start : start + batch_size].to(device)
        with autocast_context(device, precision):
            output = model.generate_plan(pixels, mask)
        semantic.append(output.semantic_state.float().cpu())
        plans.append(output.answer_plan.float().cpu())
        lengths.append(output.length.float().cpu())
        pooled.append(output.pooled_visual_state.float().cpu())
    elapsed = time.perf_counter() - started
    semantic_tensor = torch.cat(semantic)
    plan_tensor = torch.cat(plans)
    length_tensor = torch.cat(lengths)
    pooled_tensor = torch.cat(pooled)
    return SemanticInference(
        semantic_states=semantic_tensor,
        answer_plans=plan_tensor,
        lengths=length_tensor,
        pooled_visual_states=pooled_tensor,
        elapsed_seconds=elapsed,
        examples_per_second=len(rasters) / max(elapsed, 1e-9),
        finite=all(
            bool(torch.isfinite(value).all())
            for value in (
                semantic_tensor,
                plan_tensor,
                length_tensor,
                pooled_tensor,
            )
        ),
    )


def build_untrained_baseline(
    config: VisualSemanticDistillationConfig,
    *,
    checkpoint_path: str | Path | None,
    smoke: bool,
    device: torch.device,
) -> tuple[VisualSemanticDistillationModel, dict[str, Any]]:
    devices = [device.index] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(SEED)
        model = VisualSemanticDistillationModel(config)
    if smoke:
        receipt: dict[str, Any] = {
            "route": "random-smoke-baseline",
            "evidence_eligible": False,
        }
    else:
        receipt = load_v37_pixel_linguist_initialization(
            model,
            checkpoint_path,
            local_files_only=checkpoint_path is None,
        )
    return model.requires_grad_(False).eval().to(device), receipt


def _targets_for_identifiers(
    identifiers: Sequence[str],
    banks: Sequence[VisualSemanticDistillationTargetBank],
) -> torch.Tensor:
    rows: list[torch.Tensor] = []
    for identifier in identifiers:
        found = [bank for bank in banks if identifier in bank._index]
        if len(found) != 1:
            raise ValueError(f"V37 target lookup for {identifier!r} is not unique")
        index = found[0]._index[identifier]
        rows.append(found[0].answer_targets[index].float())
    return torch.stack(rows)


def _target_sanity(receipt: Mapping[str, Any]) -> bool:
    metrics = receipt.get("target_metrics", {})
    return all(
        (
            float(metrics.get("top1", -1)) >= 0.80,
            float(metrics.get("top5", -1)) >= 0.90,
            float(metrics.get("mrr", -1)) >= 0.85,
            float(metrics.get("cyclic_margin", -1)) >= 0.50,
            float(metrics.get("answer_effective_rank", -1)) >= 70.0,
        )
    )


def _integrity_report(
    model: VisualSemanticDistillationModel,
    checkpoint: Mapping[str, Any],
    checkpoint_receipt: Mapping[str, Any],
    train_bank: VisualSemanticDistillationTargetBank,
    development_bank: VisualSemanticDistillationTargetBank,
    *,
    train_bank_path: str | Path,
    instruction_sha256: str,
    paraphrase_sha256: str,
    smoke: bool,
) -> dict[str, Any]:
    run_receipt = checkpoint.get("run_receipt", {})
    source_sets = (
        run_receipt.get("source_sha256", {}),
        train_bank.receipt.get("source_sha256", {}),
        development_bank.receipt.get("source_sha256", {}),
    )
    source_ok = all(
        bool(source_hashes)
        and all(
            Path(path).is_file() and file_sha256(path) == digest
            for path, digest in source_hashes.items()
        )
        for source_hashes in source_sets
    )
    initialization = run_receipt.get("initialization", {})
    pixel_linguist_ok = smoke or (
        initialization.get("sha256") == PIXEL_LINGUIST_WEIGHT_SHA256
        and initialization.get("revision") == PIXEL_LINGUIST_REVISION
    )
    strict_mapping = smoke or (
        initialization.get("selected_tensors") == 198
        and initialization.get("missing_keys") == []
        and initialization.get("unexpected_keys") == []
    )
    teachers = (
        train_bank.receipt.get("teacher", {}),
        development_bank.receipt.get("teacher", {}),
    )
    bge_ok = smoke or all(
        teacher.get("manifest_sha256") == BGE_MANIFEST_SHA256
        and teacher.get("model_layer_sha256") == BGE_MODEL_SHA256
        and teacher.get("evidence_eligible", False)
        and not teacher.get("student_runtime_dependency", True)
        for teacher in teachers
    )
    train_hash = file_sha256(train_bank_path)
    data_receipt = run_receipt.get("data", {})
    data_ok = smoke or (
        instruction_sha256 == EXPECTED_INSTRUCTION_SHA256
        and paraphrase_sha256 == EXPECTED_PARAPHRASE_SHA256
        and train_bank.receipt.get("data", {}).get("sha256")
        == EXPECTED_INSTRUCTION_SHA256
        and development_bank.receipt.get("data", {}).get("sha256")
        == EXPECTED_INSTRUCTION_SHA256
        and data_receipt.get("target_bank_sha256") == train_hash
        and development_bank.receipt.get("train_bank_sha256") == train_hash
        and torch.equal(
            train_bank.teacher_mean.float(),
            development_bank.teacher_mean.float(),
        )
    )
    protocol_ok = (
        file_sha256(PROTOCOL_DOCUMENT) == EXPECTED_PROTOCOL_SHA256
        and checkpoint.get("protocol", {}).get("sha256") == EXPECTED_PROTOCOL_SHA256
        and train_bank.receipt.get("protocol", {}).get("sha256")
        == EXPECTED_PROTOCOL_SHA256
        and development_bank.receipt.get("protocol", {}).get("sha256")
        == EXPECTED_PROTOCOL_SHA256
    )
    boundary = visual_semantic_distillation_boundary_receipt(model)
    boundary_ok = (
        not boundary["forbidden_parameter_names"]
        and boundary["parameter_cap_pass"]
        and not boundary["uses_strings"]
        and not boundary["uses_token_ids"]
        and not boundary["uses_ocr"]
        and not boundary["candidate_bank_deployed"]
        and not boundary["uses_bge_at_runtime"]
        and bool(checkpoint_receipt["tensor_boundary"])
        and not checkpoint.get("contains_target_tensors", False)
        and not checkpoint.get("contains_teacher_mean", False)
        and not checkpoint.get("contains_teacher_model", False)
        and not checkpoint.get("contains_candidate_tensors", False)
        and not checkpoint.get("contains_source_language_strings", False)
    )
    return {
        "protocol_hash": protocol_ok,
        "source_hashes": source_ok,
        "data_hashes": data_ok,
        "pixel_linguist_hashes": pixel_linguist_ok,
        "bge_hashes": bge_ok,
        "strict_mapping": strict_mapping,
        "target_sanity": _target_sanity(development_bank.receipt),
        "boundary": boundary_ok,
        "total_parameters": boundary["total_parameters"],
        "finite_targets": tensors_are_finite(train_bank.state_dict())
        and tensors_are_finite(development_bank.state_dict()),
        "finite_model": bool(checkpoint_receipt["finite_model_state"]),
        "finite_optimizer": isinstance(checkpoint.get("optimizer"), Mapping)
        and tensors_are_finite(checkpoint["optimizer"]),
        "finite_ema": isinstance(checkpoint.get("ema"), Mapping)
        and tensors_are_finite(checkpoint["ema"]),
        "model_boundary": boundary,
        "evaluation_source_sha256": {path: file_sha256(path) for path in SOURCE_FILES},
    }


def main() -> None:
    args = effective_arguments(parse_args())
    if args.batch_size < 1:
        raise ValueError("V37 evaluation batch size must be positive")
    if file_sha256(PROTOCOL_DOCUMENT) != EXPECTED_PROTOCOL_SHA256:
        raise RuntimeError("V37 protocol changed after preregistration")
    instruction_sha256 = file_sha256(args.instruction_manifest)
    paraphrase_sha256 = file_sha256(args.paraphrase_manifest)
    if not args.smoke and (
        instruction_sha256 != EXPECTED_INSTRUCTION_SHA256
        or paraphrase_sha256 != EXPECTED_PARAPHRASE_SHA256
    ):
        raise RuntimeError("V37 development data differs from preregistration")
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
        raise ValueError("V37 checkpoint has a different protocol")
    config = model.config
    all_records = load_v37_instruction_records(args.instruction_manifest)
    render_config = VisualSemanticDistillationRenderConfig(augment=False)
    selected_records, rejected = select_v37_instruction_records(
        all_records,
        split="development",
        render_config=render_config,
    )
    if not args.smoke and (len(selected_records) != 196 or len(rejected) != 1):
        raise RuntimeError("V37 development selection changed")
    canonical_prompts = render_records(
        selected_records,
        field="prompt",
        font_path=V37_DEVELOPMENT_FONT,
        render_config=render_config,
    )
    canonical_answers = render_records(
        selected_records,
        field="answer",
        font_path=V37_DEVELOPMENT_FONT,
        render_config=render_config,
    )
    held_prompts = render_records(
        selected_records,
        field="prompt",
        font_path=V37_HELD_FONT,
        render_config=render_config,
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
            f"V37 expected 30 nonsealed paraphrases, found {len(paraphrase_records)}"
        )
    nonsealed_by_identifier = {
        record.identifier: record for record in nonsealed_records
    }
    paraphrase_source_ids = [
        record.identifier.removeprefix("paraphrase:") for record in paraphrase_records
    ]
    paraphrase_source_records = [
        nonsealed_by_identifier[identifier] for identifier in paraphrase_source_ids
    ]
    paraphrase_prompts = render_records(
        paraphrase_records,
        field="prompt",
        font_path=V37_DEVELOPMENT_FONT,
        render_config=render_config,
    )
    original_paraphrase_prompts = render_records(
        paraphrase_source_records,
        field="prompt",
        font_path=V37_DEVELOPMENT_FONT,
        render_config=render_config,
    )

    # Every student condition is inferred before evaluator candidate matrices load.
    correct_output = infer_semantics(
        model,
        canonical_prompts,
        device=device,
        precision=args.precision,
        batch_size=args.batch_size,
    )
    control_outputs = {
        name: infer_semantics(
            model,
            controlled_rasters(canonical_prompts, name),
            device=device,
            precision=args.precision,
            batch_size=args.batch_size,
        )
        for name in ("shuffled", "blank", "final-quarter")
    }
    held_output = infer_semantics(
        model,
        held_prompts,
        device=device,
        precision=args.precision,
        batch_size=args.batch_size,
    )
    paraphrase_output = infer_semantics(
        model,
        paraphrase_prompts,
        device=device,
        precision=args.precision,
        batch_size=args.batch_size,
    )
    original_paraphrase_output = infer_semantics(
        model,
        original_paraphrase_prompts,
        device=device,
        precision=args.precision,
        batch_size=args.batch_size,
    )
    model = model.cpu()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    untrained_model, untrained_receipt = build_untrained_baseline(
        config,
        checkpoint_path=args.external_checkpoint,
        smoke=args.smoke,
        device=device,
    )
    untrained_prompt_output = infer_semantics(
        untrained_model,
        canonical_prompts,
        device=device,
        precision=args.precision,
        batch_size=args.batch_size,
    )
    untrained_answer_output = infer_semantics(
        untrained_model,
        canonical_answers,
        device=device,
        precision=args.precision,
        batch_size=args.batch_size,
    )
    untrained_model = untrained_model.cpu()
    del untrained_model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # Candidate banks become visible only after every image-conditioned forward pass.
    train_bank = load_target_bank(args.train_bank)
    development_bank = load_target_bank(args.development_bank)
    if train_bank.receipt.get("split") != "train":
        raise ValueError("V37 evaluator requires a train target bank")
    if development_bank.receipt.get("split") != "development":
        raise ValueError("V37 evaluator requires a development target bank")
    if tuple(development_bank.identifiers) != canonical_prompts.identifiers:
        raise ValueError("V37 canonical prompts and development targets differ")
    prompt_targets = development_bank.prompt_targets.float()
    answer_targets = development_bank.answer_targets.float()
    target_lengths = development_bank.lengths.float()

    prompt_metrics = semantic_retrieval_metrics(
        correct_output.semantic_states,
        prompt_targets,
    )
    plan_metrics = semantic_retrieval_metrics(
        correct_output.answer_plans,
        answer_targets,
        predicted_lengths=correct_output.lengths,
        target_lengths=target_lengths,
    )
    controls: dict[str, Any] = {}
    for name, output in control_outputs.items():
        controls[name.replace("-", "_")] = semantic_control_metrics(
            output.answer_plans,
            correct_output.answer_plans,
            answer_targets,
            predicted_lengths=output.lengths,
            target_lengths=target_lengths,
        ) | {
            "prompt_state": semantic_control_metrics(
                output.semantic_states,
                correct_output.semantic_states,
                prompt_targets,
            ),
            "finite": output.finite,
            "elapsed_seconds": output.elapsed_seconds,
        }

    held_metrics = semantic_control_metrics(
        held_output.answer_plans,
        correct_output.answer_plans,
        answer_targets,
        predicted_lengths=held_output.lengths,
        target_lengths=target_lengths,
    )
    font_metrics = {
        "prompt_plan_cosine": mean_semantic_cosine(
            held_output.answer_plans,
            correct_output.answer_plans,
        ),
        "prompt_state_cosine": mean_semantic_cosine(
            held_output.semantic_states,
            correct_output.semantic_states,
        ),
        "held_prompt_state": semantic_retrieval_metrics(
            held_output.semantic_states,
            prompt_targets,
        ),
        "held_answer_plan": held_metrics,
        "held_font_path": V37_HELD_FONT,
    }

    paraphrase_answer_targets = _targets_for_identifiers(
        paraphrase_source_ids,
        (train_bank, development_bank),
    )
    paraphrase_metrics = indexed_semantic_retrieval_metrics(
        paraphrase_output.answer_plans,
        paraphrase_answer_targets,
        torch.arange(len(paraphrase_records), dtype=torch.long),
    )
    paraphrase_metrics["original_plan_cosine"] = mean_semantic_cosine(
        paraphrase_output.answer_plans,
        original_paraphrase_output.answer_plans,
    )
    paraphrase_metrics["original_state_cosine"] = mean_semantic_cosine(
        paraphrase_output.semantic_states,
        original_paraphrase_output.semantic_states,
    )
    paraphrase_metrics["source_split_counts"] = {
        split: sum(
            visual_raster_partition(identifier, stream="instruction") == split
            for identifier in paraphrase_source_ids
        )
        for split in ("train", "development")
    }
    paraphrase_metrics["sealed_source_rows_excluded"] = 1

    answers = [record.answer for record in selected_records]
    pairs = nearest_length_counterfactual_pairs(
        target_lengths,
        answer_targets,
        labels=answers,
    )
    counterfactual = counterfactual_assignment(
        correct_output.answer_plans,
        answer_targets,
        pairs,
    )
    untrained_plan = semantic_retrieval_metrics(
        untrained_prompt_output.answer_plans,
        answer_targets,
        predicted_lengths=untrained_prompt_output.lengths,
        target_lengths=target_lengths,
    )
    untrained_reading = semantic_retrieval_metrics(
        untrained_prompt_output.semantic_states,
        prompt_targets,
    )
    direct_pixel_linguist = semantic_retrieval_metrics(
        untrained_prompt_output.pooled_visual_states,
        untrained_answer_output.pooled_visual_states,
    )
    most_frequent = min(
        range(len(answers)),
        key=lambda index: (-answers.count(answers[index]), index),
    )
    frequency_states = answer_targets[most_frequent].expand_as(answer_targets)
    frequency_baseline = semantic_retrieval_metrics(
        frequency_states,
        answer_targets,
    )

    output_path = (
        Path(args.out)
        if args.out
        else Path(args.checkpoint).with_name(
            "development_report_raw_v37.json"
            if args.raw_weights
            else "development_report_ema_v37.json"
        )
    )
    matrix_path = output_path.with_name(output_path.stem + "_similarity.pt")
    similarity = (
        F.normalize(correct_output.answer_plans, dim=-1)
        @ F.normalize(
            answer_targets,
            dim=-1,
        ).T
    )
    atomic_torch_save(
        {
            "identifiers": list(canonical_prompts.identifiers),
            "similarity": similarity.half(),
        },
        matrix_path,
    )
    inference_outputs = (
        correct_output,
        *control_outputs.values(),
        held_output,
        paraphrase_output,
        original_paraphrase_output,
        untrained_prompt_output,
        untrained_answer_output,
    )
    report: dict[str, Any] = {
        "experiment": EXPERIMENT,
        "split": "development",
        "weight_route": checkpoint_receipt["weight_route"],
        "checkpoint": checkpoint_receipt,
        "integrity": _integrity_report(
            model,
            checkpoint,
            checkpoint_receipt,
            train_bank,
            development_bank,
            train_bank_path=args.train_bank,
            instruction_sha256=instruction_sha256,
            paraphrase_sha256=paraphrase_sha256,
            smoke=args.smoke,
        ),
        "training": {
            "global_update": int(checkpoint.get("global_update", 0)),
            "finite": bool(checkpoint.get("finite", True)),
            "stage_summaries": checkpoint.get("stage_summaries", {}),
        },
        "resources": {
            "peak_vram_bytes": int(checkpoint.get("peak_allocated_vram_bytes", 0)),
            "evaluation_peak_vram_bytes": (
                torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
            ),
            "correct_inference_seconds": correct_output.elapsed_seconds,
            "correct_examples_per_second": correct_output.examples_per_second,
        },
        "data": {
            "instruction_sha256": instruction_sha256,
            "paraphrase_sha256": paraphrase_sha256,
            "development_records": len(selected_records),
            "development_rejected": list(rejected),
            "paraphrase_records": len(paraphrase_records),
            "train_bank_sha256": file_sha256(args.train_bank),
            "development_bank_sha256": file_sha256(args.development_bank),
            "candidate_banks_loaded_after_all_student_inference": True,
            "sealed_records_rendered": 0,
            "sealed_paraphrase_rows_rendered": 0,
        },
        "correct": {
            "prompt_state": prompt_metrics,
            "answer_plan": plan_metrics,
        },
        "controls": controls,
        "counterfactual": counterfactual,
        "font": font_metrics,
        "paraphrase": paraphrase_metrics,
        "baselines": {
            "centered_bge_prompt_to_answer": development_bank.receipt.get(
                "target_metrics", {}
            ),
            "untrained_head": untrained_plan,
            "untrained_reading": untrained_reading,
            "direct_pixel_linguist_masked_mean": direct_pixel_linguist,
            "answer_frequency": frequency_baseline,
            "cyclic_prompt": controls["shuffled"],
            "blank_prompt": controls["blank"],
        },
        "untrained_initialization": untrained_receipt,
        "similarity_matrix": {
            "path": str(matrix_path.resolve()),
            "sha256": file_sha256(matrix_path),
            "shape": list(similarity.shape),
        },
        "finite": all(output.finite for output in inference_outputs)
        and tensors_are_finite(similarity),
        "sealed_opened": False,
        "renderer_authorized": False,
    }
    report["gate"] = v37_semantic_distillation_gate(report)
    report["sealed_evaluation_permitted"] = bool(report["gate"]["passed"])
    atomic_write_json(report, output_path)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

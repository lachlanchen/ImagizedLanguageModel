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

from ilm.visual_lm.visual_path_alignment import (
    V38_ARCHITECTURE,
    VisualPathAlignmentConfig,
    VisualPathAlignmentModel,
    file_sha256,
    visual_path_alignment_boundary_receipt,
)
from ilm.visual_lm.visual_path_alignment_data import (
    V38_DEVELOPMENT_FONT,
    V38_HELD_FONT,
    load_v37_instruction_records,
    select_v38_instruction_records,
)
from ilm.visual_lm.visual_path_alignment_evaluation import (
    counterfactual_assignment,
    indexed_semantic_retrieval_metrics,
    mean_semantic_cosine,
    nearest_length_counterfactual_pairs,
    semantic_control_metrics,
    semantic_retrieval_metrics,
    semantic_transition_metrics,
    v38_path_alignment_gate,
)
from ilm.visual_lm.visual_path_alignment_training import (
    VisualPathAlignmentTargetBank,
)
from ilm.visual_lm.visual_semantic_distillation_data import (
    V37_PATCHES,
    V37_PATCH_SIZE,
    V37_WIDTH,
    VisualSemanticDistillationRenderConfig,
    render_visual_semantic_distillation_strip,
)
from ilm.visual_lm.visual_semantic_raster_data import (
    VisualRasterRecord,
    load_visual_raster_paraphrases,
    visual_raster_partition,
)


EXPERIMENT = V38_ARCHITECTURE
PROTOCOL_DOCUMENT = "references/visual_path_alignment_v38_protocol.md"
# Set with the trainer after smoke validation and before evidence evaluation.
EXPECTED_PROTOCOL_SHA256: str | None = None
DEFAULT_INSTRUCTION_MANIFEST = "data/raw/alpaca_zh.json"
DEFAULT_PARAPHRASE_MANIFEST = "data/teacher/folio_paraphrases_zh_holdout.jsonl"
DEFAULT_TRAIN_PARAPHRASE_MANIFEST = (
    "data/teacher/visual_path_alignment_paraphrases_v38.jsonl"
)
DEFAULT_TRAIN_BANK = "artifacts/visual_semantic_distillation_v37_targets/train.pt"
DEFAULT_DEVELOPMENT_BANK = (
    "artifacts/visual_semantic_distillation_v37_targets/development.pt"
)
DEFAULT_V37_REPORT = (
    "artifacts/visual_semantic_distillation_v37_20260814/"
    "development_report_ema_v37.json"
)
EXPECTED_INSTRUCTION_SHA256 = (
    "6fcb98c6d79691d1f9a88ef513335da9124e7fdeef5103343f5d9f9a6e8f4903"
)
EXPECTED_PARAPHRASE_SHA256 = (
    "132a8a257d31be3cf607d0313bf042dafaee7f2df08440bf7e012a502ed6c02f"
)
EXPECTED_TRAIN_PARAPHRASE_SHA256 = (
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
SOURCE_FILES = (
    "ilm/visual_lm/visual_path_alignment.py",
    "ilm/visual_lm/visual_path_alignment_data.py",
    "ilm/visual_lm/visual_path_alignment_evaluation.py",
    "ilm/visual_lm/visual_path_alignment_training.py",
    "scripts/eval_visual_path_alignment_v38.py",
)


@dataclass
class RasterSet:
    identifiers: tuple[str, ...]
    pixels: torch.Tensor
    mask: torch.Tensor

    def __len__(self) -> int:
        return len(self.identifiers)


@dataclass
class PathInference:
    prompt_states: torch.Tensor
    answer_states: torch.Tensor
    lengths: torch.Tensor
    pooled_visual_states: torch.Tensor
    elapsed_seconds: float
    examples_per_second: float
    finite: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the V38 image-native visual-path alignment student."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--train-bank", default=DEFAULT_TRAIN_BANK)
    parser.add_argument("--development-bank", default=DEFAULT_DEVELOPMENT_BANK)
    parser.add_argument("--instruction-manifest", default=DEFAULT_INSTRUCTION_MANIFEST)
    parser.add_argument("--paraphrase-manifest", default=DEFAULT_PARAPHRASE_MANIFEST)
    parser.add_argument(
        "--train-paraphrase-manifest",
        default=DEFAULT_TRAIN_PARAPHRASE_MANIFEST,
    )
    parser.add_argument("--v37-report", default=DEFAULT_V37_REPORT)
    parser.add_argument("--out", default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--precision", choices=("fp32", "fp16", "bf16"), default="bf16")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--raw-weights", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--exploratory", action="store_true")
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
        raise RuntimeError("V38 requested CUDA but CUDA is unavailable")
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
    forbidden = ("target", "teacher", "candidate", "nearest", "bge", "rotation")
    return not any(
        any(fragment in ".".join(path).lower() for fragment in forbidden)
        for path, _tensor in _iter_tensor_paths(checkpoint)
    )


def load_target_bank(path: str | Path) -> VisualPathAlignmentTargetBank:
    state = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(state, Mapping):
        raise TypeError("V38 target bank must contain a state mapping")
    return VisualPathAlignmentTargetBank.from_v37_state_dict(state)


def load_checkpoint_model(
    path: str | Path,
    *,
    device: torch.device,
    raw_weights: bool,
) -> tuple[VisualPathAlignmentModel, dict[str, Any], dict[str, Any]]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, Mapping):
        raise TypeError("V38 checkpoint must contain a state mapping")
    if checkpoint.get("architecture") != V38_ARCHITECTURE:
        raise ValueError("evaluation checkpoint is not V38")
    config = VisualPathAlignmentConfig(**checkpoint["model_config"])
    model = VisualPathAlignmentModel(config)
    model.load_state_dict(checkpoint["model"], strict=True)
    weight_route = "raw"
    if not raw_weights:
        ema = checkpoint.get("ema")
        if isinstance(ema, Mapping) and isinstance(ema.get("shadow"), Mapping):
            parameters = dict(model.named_parameters())
            names = tuple(str(name) for name in ema.get("names", ()))
            if set(names) != set(parameters) or set(ema["shadow"]) != set(parameters):
                raise ValueError("V38 checkpoint EMA is not all-parameter")
            for name, value in ema["shadow"].items():
                if parameters[name].shape != value.shape:
                    raise ValueError(f"V38 EMA parameter mismatch: {name}")
                parameters[name].data.copy_(value.to(parameters[name]))
        elif checkpoint.get("weight_route") != "all-parameter-ema":
            raise ValueError("V38 checkpoint has no all-parameter EMA state")
        weight_route = "all-parameter-ema"
    model.requires_grad_(False).eval().to(device)
    boundary = visual_path_alignment_boundary_receipt(model)
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
        raise ValueError("V38 can only render prompt or answer fields")
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
        return RasterSet(identifiers=rasters.identifiers, pixels=pixels, mask=mask)
    raise ValueError(f"unknown V38 prompt condition: {condition}")


@torch.no_grad()
def infer_paths(
    model: VisualPathAlignmentModel,
    rasters: RasterSet,
    *,
    device: torch.device,
    precision: str,
    batch_size: int,
) -> PathInference:
    if rasters.pixels.shape != (
        len(rasters),
        3,
        V37_PATCH_SIZE,
        V37_WIDTH,
    ) or rasters.mask.shape != (len(rasters), V37_PATCHES):
        raise ValueError("V38 inference rasters do not align")
    prompt_states: list[torch.Tensor] = []
    answer_states: list[torch.Tensor] = []
    lengths: list[torch.Tensor] = []
    pooled: list[torch.Tensor] = []
    started = time.perf_counter()
    for start in range(0, len(rasters), batch_size):
        pixels = rasters.pixels[start : start + batch_size].to(device)
        mask = rasters.mask[start : start + batch_size].to(device)
        with autocast_context(device, precision):
            output = model.generate_plan(pixels, mask)
        prompt_states.append(output.prompt_state.float().cpu())
        answer_states.append(output.answer_state.float().cpu())
        lengths.append(output.length.float().cpu())
        pooled.append(output.pooled_visual_state.float().cpu())
    elapsed = time.perf_counter() - started
    prompt_tensor = torch.cat(prompt_states)
    answer_tensor = torch.cat(answer_states)
    length_tensor = torch.cat(lengths)
    pooled_tensor = torch.cat(pooled)
    return PathInference(
        prompt_states=prompt_tensor,
        answer_states=answer_tensor,
        lengths=length_tensor,
        pooled_visual_states=pooled_tensor,
        elapsed_seconds=elapsed,
        examples_per_second=len(rasters) / max(elapsed, 1e-9),
        finite=all(
            bool(torch.isfinite(value).all())
            for value in (prompt_tensor, answer_tensor, length_tensor, pooled_tensor)
        ),
    )


def _targets_for_identifiers(
    identifiers: Sequence[str],
    banks: Sequence[VisualPathAlignmentTargetBank],
    *,
    field: str,
) -> torch.Tensor:
    if field not in {"prompt", "answer", "length"}:
        raise ValueError("V38 target field is invalid")
    rows: list[torch.Tensor] = []
    for identifier in identifiers:
        found = [bank for bank in banks if identifier in bank._index]
        if len(found) != 1:
            raise ValueError(f"V38 target lookup for {identifier!r} is not unique")
        bank = found[0]
        index = bank._index[identifier]
        value = {
            "prompt": bank.prompt_targets,
            "answer": bank.answer_targets,
            "length": bank.lengths,
        }[field][index]
        rows.append(value.float())
    return torch.stack(rows)


def _load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"expected a JSON object in {path}")
    return value


def _training_paraphrase_ids(path: str | Path) -> set[str]:
    result: set[str] = set()
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                result.add(str(json.loads(line)["identifier"]))
    return result


def _integrity_report(
    model: VisualPathAlignmentModel,
    checkpoint: Mapping[str, Any],
    checkpoint_receipt: Mapping[str, Any],
    train_bank: VisualPathAlignmentTargetBank,
    development_bank: VisualPathAlignmentTargetBank,
    *,
    train_bank_path: str | Path,
    development_bank_path: str | Path,
    instruction_sha256: str,
    paraphrase_sha256: str,
    train_paraphrase_sha256: str,
    holdout_source_ids: set[str],
    train_paraphrase_ids: set[str],
    smoke: bool,
    exploratory: bool,
) -> dict[str, Any]:
    permissive = smoke or exploratory
    run_receipt = checkpoint.get("run_receipt", {})
    source_hashes = run_receipt.get("source_sha256", checkpoint.get("source_sha256", {}))
    source_ok = bool(source_hashes) and all(
        Path(path).is_file() and file_sha256(path) == digest
        for path, digest in source_hashes.items()
    )
    initialization = run_receipt.get("initialization", {})
    initialization_ok = permissive or (
        initialization.get("checkpoint_sha256") == EXPECTED_INITIALIZATION_SHA256
        and initialization.get("evidence_eligible", False)
        and initialization.get("route")
        == "v37-ema-reader-and-prompt-head-plus-train-only-rotation"
    )
    rotation = initialization.get("answer_rotation", {})
    strict_mapping = permissive or (
        int(initialization.get("copied_tensors", 0)) > 190
        and rotation.get("source_bank_sha256") == file_sha256(train_bank_path)
        and float(rotation.get("orthogonality_max_error", 1.0)) < 2e-3
        and rotation.get("stored_as_deployable_model_weight", False)
        and not rotation.get("source_matrix_stored_in_checkpoint", True)
    )
    teachers = (train_bank.receipt.get("teacher", {}), development_bank.receipt.get("teacher", {}))
    bge_ok = permissive or all(
        teacher.get("manifest_sha256") == EXPECTED_BGE_MANIFEST_SHA256
        and teacher.get("model_layer_sha256") == EXPECTED_BGE_MODEL_SHA256
        and teacher.get("evidence_eligible", False)
        and not teacher.get("student_runtime_dependency", True)
        for teacher in teachers
    )
    data_receipt = run_receipt.get("data", {})
    train_hash = file_sha256(train_bank_path)
    development_hash = file_sha256(development_bank_path)
    data_ok = permissive or (
        instruction_sha256 == EXPECTED_INSTRUCTION_SHA256
        and paraphrase_sha256 == EXPECTED_PARAPHRASE_SHA256
        and train_paraphrase_sha256 == EXPECTED_TRAIN_PARAPHRASE_SHA256
        and train_hash == EXPECTED_TRAIN_BANK_SHA256
        and development_hash == EXPECTED_DEVELOPMENT_BANK_SHA256
        and data_receipt.get("instruction_sha256") == instruction_sha256
        and data_receipt.get("paraphrase_sha256") == train_paraphrase_sha256
        and data_receipt.get("target_bank_sha256") == train_hash
        and data_receipt.get("development_target_bank_sha256") == development_hash
        and development_bank.receipt.get("train_bank_sha256") == train_hash
        and torch.equal(train_bank.teacher_mean.float(), development_bank.teacher_mean.float())
    )
    protocol_ok = permissive or (
        EXPECTED_PROTOCOL_SHA256 is not None
        and file_sha256(PROTOCOL_DOCUMENT) == EXPECTED_PROTOCOL_SHA256
        and checkpoint.get("protocol", {}).get("sha256") == EXPECTED_PROTOCOL_SHA256
    )
    boundary = visual_path_alignment_boundary_receipt(model)
    boundary_ok = (
        not boundary["forbidden_parameter_names"]
        and boundary["parameter_cap_pass"]
        and not boundary["uses_strings"]
        and not boundary["uses_token_ids"]
        and not boundary["uses_unicode_ids"]
        and not boundary["uses_ocr"]
        and not boundary["candidate_bank_deployed"]
        and not boundary["uses_bge_at_runtime"]
        and not boundary["uses_qwen_at_runtime"]
        and bool(checkpoint_receipt["tensor_boundary"])
        and not checkpoint.get("contains_target_tensors", False)
        and not checkpoint.get("contains_teacher_model", False)
        and not checkpoint.get("contains_candidate_tensors", False)
        and not checkpoint.get("contains_nearest_negative_tensors", False)
        and not checkpoint.get("contains_source_language_strings", False)
    )
    optimizer = checkpoint.get("optimizer")
    ema = checkpoint.get("ema")
    return {
        "protocol_hash": protocol_ok,
        "source_hashes": source_ok,
        "data_hashes": data_ok,
        "initialization_hash": initialization_ok,
        "bge_hashes": bge_ok,
        "strict_mapping": strict_mapping,
        "boundary": boundary_ok,
        "student_inference_before_banks": True,
        "holdout_exclusion": not holdout_source_ids.intersection(train_paraphrase_ids),
        "total_parameters": boundary["total_parameters"],
        "finite_targets": tensors_are_finite(
            (
                train_bank.prompt_targets,
                train_bank.answer_targets,
                train_bank.lengths,
                development_bank.prompt_targets,
                development_bank.answer_targets,
                development_bank.lengths,
            )
        ),
        "finite_model": bool(checkpoint_receipt["finite_model_state"]),
        "finite_optimizer": isinstance(optimizer, Mapping) and tensors_are_finite(optimizer),
        "finite_ema": isinstance(ema, Mapping) and tensors_are_finite(ema),
        "model_boundary": boundary,
        "evaluation_source_sha256": {path: file_sha256(path) for path in SOURCE_FILES},
    }


def _v37_baseline(path: str | Path) -> dict[str, Any]:
    report = _load_json(path)
    return {
        "path": str(Path(path).resolve()),
        "sha256": file_sha256(path),
        "weight_route": report.get("weight_route"),
        "decision": report.get("gate", {}).get("decision"),
        "prompt": report.get("correct", {}).get("prompt_state", {}),
        "answer": report.get("correct", {}).get("answer_plan", {}),
        "held_prompt_cosine": report.get("font", {}).get("prompt_state_cosine"),
        "held_answer_cosine": report.get("font", {}).get("prompt_plan_cosine"),
        "paraphrase_prompt_cosine": report.get("paraphrase", {}).get("original_state_cosine"),
        "paraphrase_answer_cosine": report.get("paraphrase", {}).get("original_plan_cosine"),
    }


def main() -> None:
    args = effective_arguments(parse_args())
    if args.batch_size < 1:
        raise ValueError("V38 evaluation batch size must be positive")
    if args.smoke and args.exploratory:
        raise ValueError("V38 smoke and exploratory modes are mutually exclusive")
    evidence = not args.smoke and not args.exploratory
    if evidence:
        if EXPECTED_PROTOCOL_SHA256 is None:
            raise RuntimeError("V38 evidence protocol has not been frozen")
        if file_sha256(PROTOCOL_DOCUMENT) != EXPECTED_PROTOCOL_SHA256:
            raise RuntimeError("V38 protocol changed after preregistration")
    instruction_sha256 = file_sha256(args.instruction_manifest)
    paraphrase_sha256 = file_sha256(args.paraphrase_manifest)
    train_paraphrase_sha256 = file_sha256(args.train_paraphrase_manifest)
    if evidence and (
        instruction_sha256 != EXPECTED_INSTRUCTION_SHA256
        or paraphrase_sha256 != EXPECTED_PARAPHRASE_SHA256
        or train_paraphrase_sha256 != EXPECTED_TRAIN_PARAPHRASE_SHA256
    ):
        raise RuntimeError("V38 development data differs from preregistration")
    device = choose_device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)

    model, checkpoint, checkpoint_receipt = load_checkpoint_model(
        args.checkpoint,
        device=device,
        raw_weights=args.raw_weights,
    )
    if evidence and checkpoint.get("protocol", {}).get("sha256") != EXPECTED_PROTOCOL_SHA256:
        raise ValueError("V38 checkpoint has a different protocol")
    all_records = load_v37_instruction_records(args.instruction_manifest)
    render_config = VisualSemanticDistillationRenderConfig(augment=False)
    selected_records, rejected = select_v38_instruction_records(
        all_records,
        split="development",
        render_config=render_config,
    )
    if evidence and (len(selected_records) != 196 or len(rejected) != 1):
        raise RuntimeError("V38 development selection changed")
    if args.smoke:
        selected_records = selected_records[:8]
    canonical_prompts = render_records(
        selected_records,
        field="prompt",
        font_path=V38_DEVELOPMENT_FONT,
        render_config=render_config,
    )
    canonical_answers = render_records(
        selected_records,
        field="answer",
        font_path=V38_DEVELOPMENT_FONT,
        render_config=render_config,
    )
    held_prompts = render_records(
        selected_records,
        field="prompt",
        font_path=V38_HELD_FONT,
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
    if evidence and len(paraphrase_records) != 30:
        raise RuntimeError(
            f"V38 expected 30 nonsealed paraphrases, found {len(paraphrase_records)}"
        )
    if args.smoke:
        paraphrase_records = paraphrase_records[:8]
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
        font_path=V38_DEVELOPMENT_FONT,
        render_config=render_config,
    )
    original_paraphrase_prompts = render_records(
        paraphrase_source_records,
        field="prompt",
        font_path=V38_DEVELOPMENT_FONT,
        render_config=render_config,
    )

    # Candidate banks remain unavailable until every image-conditioned pass completes.
    correct_output = infer_paths(
        model,
        canonical_prompts,
        device=device,
        precision=args.precision,
        batch_size=args.batch_size,
    )
    answer_read_output = infer_paths(
        model,
        canonical_answers,
        device=device,
        precision=args.precision,
        batch_size=args.batch_size,
    )
    control_outputs = {
        name: infer_paths(
            model,
            controlled_rasters(canonical_prompts, name),
            device=device,
            precision=args.precision,
            batch_size=args.batch_size,
        )
        for name in ("shuffled", "blank", "final-quarter")
    }
    held_output = infer_paths(
        model,
        held_prompts,
        device=device,
        precision=args.precision,
        batch_size=args.batch_size,
    )
    paraphrase_output = infer_paths(
        model,
        paraphrase_prompts,
        device=device,
        precision=args.precision,
        batch_size=args.batch_size,
    )
    original_paraphrase_output = infer_paths(
        model,
        original_paraphrase_prompts,
        device=device,
        precision=args.precision,
        batch_size=args.batch_size,
    )
    model = model.cpu()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # Detached candidates become visible only after all student inference above.
    train_bank = load_target_bank(args.train_bank)
    development_bank = load_target_bank(args.development_bank)
    banks = (train_bank, development_bank)
    prompt_targets = _targets_for_identifiers(
        canonical_prompts.identifiers,
        banks,
        field="prompt",
    )
    answer_targets = _targets_for_identifiers(
        canonical_prompts.identifiers,
        banks,
        field="answer",
    )
    target_lengths = _targets_for_identifiers(
        canonical_prompts.identifiers,
        banks,
        field="length",
    )

    prompt_metrics = semantic_retrieval_metrics(
        correct_output.prompt_states,
        prompt_targets,
    )
    answer_read_metrics = semantic_retrieval_metrics(
        answer_read_output.prompt_states,
        answer_targets,
    )
    answer_metrics = semantic_retrieval_metrics(
        correct_output.answer_states,
        answer_targets,
        predicted_lengths=correct_output.lengths,
        target_lengths=target_lengths,
    )
    controls: dict[str, Any] = {}
    for name, output in control_outputs.items():
        controls[name.replace("-", "_")] = semantic_control_metrics(
            output.answer_states,
            correct_output.answer_states,
            answer_targets,
            predicted_lengths=output.lengths,
            target_lengths=target_lengths,
        ) | {
            "prompt_state": semantic_control_metrics(
                output.prompt_states,
                correct_output.prompt_states,
                prompt_targets,
            ),
            "finite": output.finite,
            "elapsed_seconds": output.elapsed_seconds,
        }
    held_answer_metrics = semantic_control_metrics(
        held_output.answer_states,
        correct_output.answer_states,
        answer_targets,
        predicted_lengths=held_output.lengths,
        target_lengths=target_lengths,
    )
    font_metrics = {
        "prompt_state_cosine": mean_semantic_cosine(
            held_output.prompt_states,
            correct_output.prompt_states,
        ),
        "answer_state_cosine": mean_semantic_cosine(
            held_output.answer_states,
            correct_output.answer_states,
        ),
        "held_prompt_state": semantic_retrieval_metrics(
            held_output.prompt_states,
            prompt_targets,
        ),
        "held_answer_state": held_answer_metrics,
        "canonical_font_path": V38_DEVELOPMENT_FONT,
        "held_font_path": V38_HELD_FONT,
    }
    paraphrase_answer_targets = _targets_for_identifiers(
        paraphrase_source_ids,
        banks,
        field="answer",
    )
    paraphrase_metrics = indexed_semantic_retrieval_metrics(
        paraphrase_output.answer_states,
        paraphrase_answer_targets,
        torch.arange(len(paraphrase_records), dtype=torch.long),
    )
    paraphrase_metrics["original_prompt_cosine"] = mean_semantic_cosine(
        paraphrase_output.prompt_states,
        original_paraphrase_output.prompt_states,
    )
    paraphrase_metrics["original_answer_cosine"] = mean_semantic_cosine(
        paraphrase_output.answer_states,
        original_paraphrase_output.answer_states,
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
        correct_output.answer_states,
        answer_targets,
        pairs,
        bootstrap_seed=SEED + 1,
    )
    transition = semantic_transition_metrics(
        correct_output.prompt_states,
        correct_output.answer_states,
        prompt_targets,
        answer_targets,
    )

    holdout_source_ids = {
        record.identifier.removeprefix("paraphrase:")
        for record in load_visual_raster_paraphrases(
            args.paraphrase_manifest,
            all_records,
        )
    }
    train_paraphrase_ids = _training_paraphrase_ids(args.train_paraphrase_manifest)
    integrity = _integrity_report(
        model,
        checkpoint,
        checkpoint_receipt,
        train_bank,
        development_bank,
        train_bank_path=args.train_bank,
        development_bank_path=args.development_bank,
        instruction_sha256=instruction_sha256,
        paraphrase_sha256=paraphrase_sha256,
        train_paraphrase_sha256=train_paraphrase_sha256,
        holdout_source_ids=holdout_source_ids,
        train_paraphrase_ids=train_paraphrase_ids,
        smoke=args.smoke,
        exploratory=args.exploratory,
    )
    all_inference = (
        correct_output,
        answer_read_output,
        held_output,
        paraphrase_output,
        original_paraphrase_output,
        *control_outputs.values(),
    )
    finite = all(output.finite for output in all_inference)
    report = {
        "experiment": EXPERIMENT,
        "label": "smoke" if args.smoke else "exploratory" if args.exploratory else "evidence",
        "split": "development",
        "weight_route": checkpoint_receipt["weight_route"],
        "checkpoint": checkpoint_receipt,
        "training": {
            "global_update": int(checkpoint.get("global_update", 0)),
            "finite": bool(checkpoint.get("finite", True)),
            "stage_summaries": checkpoint.get("stage_summaries", {}),
        },
        "resources": {
            "peak_vram_bytes": int(checkpoint.get("peak_allocated_vram_bytes", 0)),
            "evaluation_peak_vram_bytes": torch.cuda.max_memory_allocated(device)
            if device.type == "cuda"
            else 0,
            "correct_inference_seconds": correct_output.elapsed_seconds,
            "correct_examples_per_second": correct_output.examples_per_second,
        },
        "data": {
            "instruction_sha256": instruction_sha256,
            "paraphrase_sha256": paraphrase_sha256,
            "train_paraphrase_sha256": train_paraphrase_sha256,
            "development_records": len(selected_records),
            "development_rejected": list(rejected),
            "paraphrase_records": len(paraphrase_records),
            "train_bank_sha256": file_sha256(args.train_bank),
            "development_bank_sha256": file_sha256(args.development_bank),
            "candidate_banks_loaded_after_all_student_inference": True,
            "sealed_rows_rendered": 0,
        },
        "integrity": integrity,
        "correct": {
            "prompt_state": prompt_metrics,
            "answer_reading": answer_read_metrics,
            "answer_state": answer_metrics,
        },
        "controls": controls,
        "font": font_metrics,
        "paraphrase": paraphrase_metrics,
        "transition": transition,
        "counterfactual": counterfactual,
        "baselines": {"v37_ema": _v37_baseline(args.v37_report)},
        "finite": finite,
    }
    report["gate"] = v38_path_alignment_gate(report)
    output_path = Path(args.out) if args.out else Path(args.checkpoint).parent / (
        "development_report_raw_v38.json"
        if args.raw_weights
        else "development_report_ema_v38.json"
    )
    similarity_path = output_path.with_name(output_path.stem + "_similarity.pt")
    atomic_torch_save(
        {
            "architecture": V38_ARCHITECTURE,
            "weight_route": report["weight_route"],
            "prompt_states": F.normalize(correct_output.prompt_states, dim=-1),
            "answer_states": F.normalize(correct_output.answer_states, dim=-1),
            "prompt_targets": F.normalize(prompt_targets, dim=-1),
            "answer_targets": F.normalize(answer_targets, dim=-1),
            "held_prompt_states": F.normalize(held_output.prompt_states, dim=-1),
            "held_answer_states": F.normalize(held_output.answer_states, dim=-1),
            "contains_source_language_strings": False,
        },
        similarity_path,
    )
    report["similarity_artifact"] = {
        "path": str(similarity_path),
        "sha256": file_sha256(similarity_path),
        "contains_source_language_strings": False,
    }
    atomic_write_json(report, output_path)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

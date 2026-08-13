#!/usr/bin/env python3
from __future__ import annotations

import argparse
import inspect
import json
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import DataLoader, Dataset, Subset

from ilm.visual_lm.visual_semantic_raster_data import (
    V32_DEVELOPMENT_FONTS,
    V32_TRAIN_FONTS,
    VisualRasterInstructionDataset,
    VisualRasterRecord,
    VisualRasterRenderConfig,
    load_visual_raster_instructions,
    load_visual_raster_paraphrases,
    visual_raster_partition,
    visual_semantic_raster_collate,
    visual_semantic_raster_student_batch,
)
from ilm.visual_lm.visual_semantic_raster_evaluation import (
    RasterCharacterBank,
    build_raster_character_bank,
    decode_raster_cells,
    raster_quality_evaluation,
    sequence_evaluation,
)
from ilm.visual_lm.visual_semantic_raster_transducer import (
    VisualSemanticRasterConfig,
    VisualSemanticRasterGeneration,
    VisualSemanticRasterTransducer,
    file_sha256,
    visual_semantic_raster_boundary_receipt,
)


DEFAULT_INSTRUCTION_MANIFEST = "data/raw/alpaca_zh.json"
DEFAULT_PARAPHRASE_MANIFEST = "data/teacher/folio_paraphrases_zh_holdout.jsonl"
EXPECTED_INSTRUCTION_SHA256 = (
    "6fcb98c6d79691d1f9a88ef513335da9124e7fdeef5103343f5d9f9a6e8f4903"
)
EXPECTED_PARAPHRASE_SHA256 = (
    "132a8a257d31be3cf607d0313bf042dafaee7f2df08440bf7e012a502ed6c02f"
)


@dataclass
class RasterEvaluationData:
    identifiers: tuple[str, ...]
    targets: tuple[str, ...]
    prompt_pixels: torch.Tensor
    prompt_mask: torch.Tensor
    answer_cells: torch.Tensor
    answer_mask: torch.Tensor
    stop_targets: torch.Tensor
    stop_mask: torch.Tensor

    def __len__(self) -> int:
        return len(self.identifiers)


@dataclass
class GeneratedRasterSet:
    cells: torch.Tensor
    lengths: torch.Tensor
    stop_probabilities: torch.Tensor
    finite: bool
    elapsed_seconds: float
    examples_per_second: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit autonomous raster-language behavior of a V32 checkpoint."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--instruction-manifest", default=DEFAULT_INSTRUCTION_MANIFEST)
    parser.add_argument("--paraphrase-manifest", default=DEFAULT_PARAPHRASE_MANIFEST)
    parser.add_argument("--out", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--precision", choices=("fp32", "fp16", "bf16"), default="bf16")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--maximum-original", type=int, default=0)
    parser.add_argument("--maximum-paraphrases", type=int, default=0)
    parser.add_argument("--raw-weights", action="store_true")
    parser.add_argument("--skip-controls", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def choose_device(value: str) -> torch.device:
    if value == "auto":
        value = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = torch.device(value)
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda:0")
    return device


def autocast_context(device: torch.device, precision: str):
    if device.type != "cuda" or precision == "fp32":
        return nullcontext()
    dtype = torch.float16 if precision == "fp16" else torch.bfloat16
    return torch.autocast(device_type="cuda", dtype=dtype)


def _maximum_prompt_characters(config: VisualSemanticRasterConfig) -> int:
    return 160 if config.maximum_prompt_patches == 192 else min(
        160, config.prompt_width // 16
    )


def _load_checkpoint_model(
    path: str | Path,
    *,
    device: torch.device,
    raw_weights: bool,
) -> tuple[VisualSemanticRasterTransducer, dict[str, Any], dict[str, Any]]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("architecture") != "visual-semantic-raster-transducer-v32":
        raise ValueError("evaluation checkpoint is not V32")
    config = VisualSemanticRasterConfig(**checkpoint["model_config"])
    model = VisualSemanticRasterTransducer(config)
    model.load_state_dict(checkpoint["model"])
    weight_route = "raw"
    if not raw_weights:
        ema = checkpoint.get("ema")
        if not isinstance(ema, Mapping) or not isinstance(ema.get("shadow"), Mapping):
            raise ValueError("V32 checkpoint has no selective EMA state")
        parameters = dict(model.named_parameters())
        for name, value in ema["shadow"].items():
            if name not in parameters or parameters[name].shape != value.shape:
                raise ValueError(f"V32 EMA parameter mismatch: {name}")
            parameters[name].data.copy_(value.to(parameters[name]))
        weight_route = "selective-ema"
    model.requires_grad_(False).eval().to(device)
    finite = all(torch.isfinite(value).all() for value in model.state_dict().values())
    receipt = {
        "checkpoint": str(Path(path).resolve()),
        "checkpoint_sha256": file_sha256(path),
        "weight_route": weight_route,
        "global_update": int(checkpoint.get("global_update", 0)),
        "finite_model_state": bool(finite),
        "boundary": visual_semantic_raster_boundary_receipt(model),
    }
    return model, checkpoint, receipt


def collect_evaluation_data(
    dataset: Dataset,
    target_by_identifier: Mapping[str, str],
    *,
    batch_size: int,
    num_workers: int,
    maximum: int = 0,
) -> RasterEvaluationData:
    count = len(dataset) if maximum <= 0 else min(len(dataset), maximum)
    if count < 1:
        raise ValueError("V32 evaluation dataset is empty")
    loader = DataLoader(
        Subset(dataset, range(count)),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
        collate_fn=visual_semantic_raster_collate,
    )
    identifiers: list[str] = []
    targets: list[str] = []
    tensors: dict[str, list[torch.Tensor]] = {
        name: []
        for name in (
            "prompt_pixels",
            "prompt_mask",
            "answer_cells",
            "answer_mask",
            "stop_targets",
            "stop_mask",
        )
    }
    for raw in loader:
        student = visual_semantic_raster_student_batch(raw)
        for name, value in student.items():
            tensors[name].append(value)
        for metadata in raw["metadata"]:
            identifier = str(metadata["identifier"])
            identifiers.append(identifier)
            targets.append(target_by_identifier[identifier])
    return RasterEvaluationData(
        identifiers=tuple(identifiers),
        targets=tuple(targets),
        **{name: torch.cat(values) for name, values in tensors.items()},
    )


def controlled_prompts(
    data: RasterEvaluationData,
    condition: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    pixels = data.prompt_pixels
    mask = data.prompt_mask
    if condition == "correct":
        return pixels, mask
    if condition == "shuffled":
        permutation = torch.roll(torch.arange(len(data)), shifts=1)
        return pixels[permutation], mask[permutation]
    if condition == "blank":
        return torch.ones_like(pixels), mask.clone()
    if condition == "suffix":
        cutoff = pixels.shape[-1] * 3 // 4
        suffix_pixels = pixels.clone()
        suffix_pixels[..., :cutoff] = 1.0
        suffix_mask = mask.clone()
        suffix_mask[:, : mask.shape[1] * 3 // 4] = 0.0
        return suffix_pixels, suffix_mask
    raise ValueError(f"unknown V32 prompt condition: {condition}")


@torch.no_grad()
def autonomous_generate_batch(
    model: VisualSemanticRasterTransducer,
    prompt_pixels: torch.Tensor,
    prompt_mask: torch.Tensor,
) -> VisualSemanticRasterGeneration:
    """Strict student inference boundary: prompts in, generated rasters out."""

    return model.generate(prompt_pixels, prompt_mask)


def generate_condition(
    model: VisualSemanticRasterTransducer,
    data: RasterEvaluationData,
    *,
    condition: str,
    device: torch.device,
    precision: str,
    batch_size: int,
) -> GeneratedRasterSet:
    pixels, mask = controlled_prompts(data, condition)
    cells = []
    lengths = []
    probabilities = []
    started = time.perf_counter()
    for start in range(0, len(data), batch_size):
        prompt_pixels = pixels[start : start + batch_size].to(device)
        prompt_mask = mask[start : start + batch_size].to(device)
        with autocast_context(device, precision):
            output = autonomous_generate_batch(model, prompt_pixels, prompt_mask)
        cells.append(output.cells.float().cpu())
        lengths.append(output.lengths.cpu())
        probabilities.append(output.stop_probabilities.float().cpu())
    elapsed = time.perf_counter() - started
    generated_cells = torch.cat(cells)
    stop_probabilities = torch.cat(probabilities)
    finite = bool(
        torch.isfinite(generated_cells).all()
        and torch.isfinite(stop_probabilities).all()
    )
    return GeneratedRasterSet(
        cells=generated_cells,
        lengths=torch.cat(lengths),
        stop_probabilities=stop_probabilities,
        finite=finite,
        elapsed_seconds=elapsed,
        examples_per_second=len(data) / max(elapsed, 1e-9),
    )


def _lengths_from_stop_probabilities(
    probabilities: torch.Tensor,
    *,
    maximum: int,
    threshold: float = 0.5,
) -> torch.Tensor:
    lengths = torch.full((len(probabilities),), maximum, dtype=torch.long)
    for index in range(len(probabilities)):
        stopping = torch.nonzero(probabilities[index, 1 : maximum + 1] > threshold)
        if len(stopping):
            lengths[index] = int(stopping[0, 0]) + 1
    return lengths


@torch.no_grad()
def teacher_forced_generation(
    model: VisualSemanticRasterTransducer,
    data: RasterEvaluationData,
    *,
    device: torch.device,
    precision: str,
    batch_size: int,
) -> GeneratedRasterSet:
    cells = []
    lengths = []
    probabilities = []
    started = time.perf_counter()
    for start in range(0, len(data), batch_size):
        prompt_pixels = data.prompt_pixels[start : start + batch_size].to(device)
        prompt_mask = data.prompt_mask[start : start + batch_size].to(device)
        answer_cells = data.answer_cells[start : start + batch_size].to(device)
        answer_mask = data.answer_mask[start : start + batch_size].to(device)
        with autocast_context(device, precision):
            output = model(
                prompt_pixels,
                prompt_mask,
                answer_cells,
                answer_mask,
                feedback_mode="clean",
            )
            predicted_states = F.layer_norm(
                output.state_mean,
                (model.config.latent_dim,),
            )
            predicted_cells = model.raster_decoder(predicted_states).sigmoid()
        stop = output.stop_logits.sigmoid().float().cpu()
        cells.append(predicted_cells.float().cpu())
        probabilities.append(stop)
        lengths.append(
            _lengths_from_stop_probabilities(
                stop,
                maximum=model.config.maximum_answer_cells,
            )
        )
    elapsed = time.perf_counter() - started
    generated_cells = torch.cat(cells)
    stop_probabilities = torch.cat(probabilities)
    return GeneratedRasterSet(
        cells=generated_cells,
        lengths=torch.cat(lengths),
        stop_probabilities=stop_probabilities,
        finite=bool(
            torch.isfinite(generated_cells).all()
            and torch.isfinite(stop_probabilities).all()
        ),
        elapsed_seconds=elapsed,
        examples_per_second=len(data) / max(elapsed, 1e-9),
    )


def score_generated_set(
    model: VisualSemanticRasterTransducer,
    generated: GeneratedRasterSet,
    data: RasterEvaluationData,
    bank: RasterCharacterBank,
    *,
    device: torch.device,
    precision: str,
    batch_size: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    def evaluator_autocast():
        return autocast_context(device, precision)

    indices, log_probabilities = decode_raster_cells(
        model,
        generated.cells,
        bank,
        device=device,
        batch_size=max(64, batch_size * model.config.maximum_answer_cells),
        autocast=evaluator_autocast,
    )
    sequence_metrics, rows = sequence_evaluation(
        indices,
        generated.lengths,
        data.targets,
        bank,
        log_probabilities=log_probabilities,
    )
    maximum = model.config.maximum_answer_cells
    overflow = (generated.lengths >= maximum) & (
        generated.stop_probabilities[:, maximum] <= 0.5
    )
    raster_metrics, raster_rows = raster_quality_evaluation(
        generated.cells,
        generated.lengths,
        data.answer_cells,
        data.answer_mask,
        maximum_cells=maximum,
        overflow_flags=overflow,
    )
    for index, row in enumerate(rows):
        row.update(raster_rows[index])
        row["identifier"] = data.identifiers[index]
    return {
        "samples": len(data),
        "finite_generation": generated.finite,
        "inference_seconds": generated.elapsed_seconds,
        "examples_per_second": generated.examples_per_second,
        "sequence": sequence_metrics,
        "raster": raster_metrics,
    }, rows


def _strip_from_cells(cells: torch.Tensor, length: int) -> Image.Image:
    length = max(1, int(length))
    strip = cells[:length].permute(1, 2, 0, 3).reshape(1, 24, length * 24)[0]
    array = ((1.0 - strip.clamp(0, 1)) * 255.0).byte().numpy()
    return Image.fromarray(array, mode="L").convert("RGB")


def save_autonomous_gallery(
    data: RasterEvaluationData,
    generated: GeneratedRasterSet,
    path: str | Path,
    *,
    maximum_rows: int = 16,
) -> None:
    rows = min(len(data), maximum_rows)
    canvas = Image.new("RGB", (1_000, rows * 92 + 32), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    draw.text((8, 7), "PROMPT RASTER / TARGET RASTER / AUTONOMOUS OUTPUT", fill="black", font=font)
    for index in range(rows):
        top = 32 + index * 92
        prompt = (data.prompt_pixels[index].mean(dim=0).clamp(0, 1) * 255).byte().numpy()
        occupied = torch.nonzero(data.prompt_mask[index] > 0)
        right_patch = int(occupied[-1]) + 2 if len(occupied) else 8
        prompt_image = Image.fromarray(prompt[:, : right_patch * 16], mode="L").convert("RGB")
        prompt_image.thumbnail((600, 24))
        target_length = int(data.answer_mask[index].sum())
        target_image = _strip_from_cells(data.answer_cells[index], target_length)
        generated_image = _strip_from_cells(
            generated.cells[index], int(generated.lengths[index])
        )
        draw.text((8, top), f"{index:03d} prompt", fill="#555555", font=font)
        canvas.paste(prompt_image, (92, top))
        draw.text((8, top + 29), "target", fill="#555555", font=font)
        canvas.paste(target_image, (92, top + 26))
        draw.text((8, top + 58), "output", fill="#555555", font=font)
        canvas.paste(generated_image, (92, top + 55))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def v32_gate_report(
    *,
    checkpoint_receipt: Mapping[str, Any],
    correct: Mapping[str, Any],
    shuffled: Mapping[str, Any] | None,
    blank: Mapping[str, Any] | None,
    teacher: Mapping[str, Any],
    paraphrase: Mapping[str, Any] | None,
    training_font: Mapping[str, Any] | None,
    frequency_baseline_exact: float,
    training_complete: bool,
    peak_vram_bytes: int | None,
) -> dict[str, Any]:
    def metric(result: Mapping[str, Any], section: str, name: str) -> float:
        return float(result[section][name]["mean"])

    boundary = checkpoint_receipt["boundary"]
    integrity = {
        "parameter_cap": bool(boundary["parameter_cap_pass"]),
        "peak_vram_below_20_gib": (
            peak_vram_bytes is not None
            and peak_vram_bytes > 0
            and peak_vram_bytes < 20 * 1024**3
        ),
        "twelve_thousand_updates": bool(training_complete),
        "finite_checkpoint": bool(checkpoint_receipt["finite_model_state"]),
        "finite_generation": bool(correct["finite_generation"]),
        "forbidden_boundary_empty": not boundary["forbidden_parameter_names"],
        "strict_generate_signature": list(
            inspect.signature(autonomous_generate_batch).parameters
        )
        == ["model", "prompt_pixels", "prompt_mask"],
        "candidate_bank_absent_from_generate": True,
    }
    autonomous_accuracy = metric(correct, "sequence", "character_accuracy")
    teacher_accuracy = metric(teacher, "sequence", "character_accuracy")
    direct = {
        "nonblank_answer_rate": metric(correct, "raster", "nonblank_answer") >= 0.95,
        "character_accuracy": autonomous_accuracy >= 0.60,
        "character_error_rate": metric(correct, "sequence", "character_error_rate") <= 0.45,
        "exact_sequence_accuracy": metric(correct, "sequence", "exact") >= 0.15,
        "length_exact_accuracy": metric(correct, "sequence", "length_exact") >= 0.60,
        "autonomous_teacher_gap": teacher_accuracy - autonomous_accuracy <= 0.15,
    }
    correct_log = metric(correct, "sequence", "target_log_similarity")
    shuffled_log = (
        metric(shuffled, "sequence", "target_log_similarity")
        if shuffled is not None
        else None
    )
    correct_exact = metric(correct, "sequence", "exact")
    blank_exact = metric(blank, "sequence", "exact") if blank is not None else None
    development_accuracy = autonomous_accuracy
    training_font_accuracy = (
        metric(training_font, "sequence", "character_accuracy")
        if training_font is not None
        else None
    )
    visual_language: dict[str, Any] = {
        "original_exact_accuracy": correct_exact >= 0.20,
        "paraphrase_character_accuracy": (
            paraphrase is not None
            and paraphrase["samples"] >= 24
            and metric(paraphrase, "sequence", "character_accuracy") >= 0.45
        ),
        "correct_over_shuffled_log_margin": (
            shuffled_log is not None and correct_log - shuffled_log >= 0.15
        ),
        "correct_over_blank_exact_margin": (
            blank_exact is not None and correct_exact - blank_exact >= 0.10
        ),
        "heldout_font_degradation": (
            training_font_accuracy is not None
            and training_font_accuracy - development_accuracy <= 0.15
        ),
        "frequency_baseline": correct_exact > frequency_baseline_exact,
        "counterfactual_pair_assignment": "unavailable",
        "decoded_feedback_ablation": "unavailable",
        "random_reader_probe": "unavailable",
    }
    integrity_pass = all(integrity.values())
    direct_pass = all(direct.values())
    available_language = [value for value in visual_language.values() if isinstance(value, bool)]
    language_pass = bool(available_language) and all(available_language)
    if not integrity_pass:
        decision = "invalid-run"
    elif not direct_pass:
        decision = "rejected-writer"
    elif language_pass:
        decision = "provisional-visual-semantic-raster-proof-pending-ablations"
    else:
        decision = "accepted-raster-transducer-only"
    return {
        "integrity": integrity,
        "integrity_pass": integrity_pass,
        "direct_raster": direct,
        "direct_raster_pass": direct_pass,
        "visual_language": visual_language,
        "available_visual_language_pass": language_pass,
        "decision": decision,
    }


def _frequency_baseline(
    records: Sequence[VisualRasterRecord],
    targets: Sequence[str],
) -> tuple[str, float]:
    frequencies: dict[str, int] = {}
    for record in records:
        if visual_raster_partition(record.identifier, stream="instruction") == "train":
            frequencies[record.answer] = frequencies.get(record.answer, 0) + 1
    answer = max(frequencies, key=frequencies.get)
    exact = sum(target == answer for target in targets) / max(1, len(targets))
    return answer, exact


def _training_summary(checkpoint_path: Path) -> dict[str, Any]:
    path = checkpoint_path.parent / "training_summary.json"
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    args = parse_args()
    if args.smoke:
        args.maximum_original = args.maximum_original or 2
        args.maximum_paraphrases = args.maximum_paraphrases or 2
        args.num_workers = 0
        args.batch_size = min(args.batch_size, 2)
    device = choose_device(args.device)
    checkpoint_path = Path(args.checkpoint)
    output_dir = Path(args.out or checkpoint_path.parent / "evaluation")
    output_dir.mkdir(parents=True, exist_ok=True)
    if file_sha256(args.instruction_manifest) != EXPECTED_INSTRUCTION_SHA256:
        raise ValueError("V32 evaluator instruction manifest hash mismatch")
    if file_sha256(args.paraphrase_manifest) != EXPECTED_PARAPHRASE_SHA256:
        raise ValueError("V32 evaluator paraphrase manifest hash mismatch")
    model, checkpoint, checkpoint_receipt = _load_checkpoint_model(
        checkpoint_path,
        device=device,
        raw_weights=args.raw_weights,
    )
    config = model.config
    render_config = VisualRasterRenderConfig(
        maximum_prompt_patches=config.maximum_prompt_patches,
        maximum_answer_cells=config.maximum_answer_cells,
        augment=False,
    )
    records = load_visual_raster_instructions(
        args.instruction_manifest,
        maximum_prompt_characters=_maximum_prompt_characters(config),
        maximum_answer_cells=config.maximum_answer_cells,
    )
    development_records = [
        record
        for record in records
        if visual_raster_partition(record.identifier, stream="instruction")
        == "development"
    ]
    target_by_identifier = {record.identifier: record.answer for record in records}
    original_dataset = VisualRasterInstructionDataset(
        development_records,
        split="development",
        render_config=render_config,
        seed=20_263_251,
        include_all_records=True,
    )
    original = collect_evaluation_data(
        original_dataset,
        target_by_identifier,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        maximum=args.maximum_original,
    )
    paraphrase_records = [
        record
        for record in load_visual_raster_paraphrases(args.paraphrase_manifest, records)
        if len(record.prompt) <= _maximum_prompt_characters(config)
        and len(record.answer) <= config.maximum_answer_cells
    ]
    paraphrase_by_identifier = {
        record.identifier: record.answer for record in paraphrase_records
    }
    paraphrase = None
    if paraphrase_records:
        paraphrase_dataset = VisualRasterInstructionDataset(
            paraphrase_records,
            split="development",
            render_config=render_config,
            seed=20_263_257,
            include_all_records=True,
        )
        paraphrase = collect_evaluation_data(
            paraphrase_dataset,
            paraphrase_by_identifier,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            maximum=args.maximum_paraphrases,
        )

    candidate_characters = set("".join(original.targets))
    if paraphrase is not None:
        candidate_characters.update("".join(paraphrase.targets))
    def evaluator_autocast():
        return autocast_context(device, args.precision)

    bank = build_raster_character_bank(
        model,
        sorted(candidate_characters),
        render_config=render_config,
        font_paths=(V32_DEVELOPMENT_FONTS[0], *V32_TRAIN_FONTS),
        device=device,
        variants_per_font=1,
        batch_size=max(64, args.batch_size * config.maximum_answer_cells),
        autocast=evaluator_autocast,
    )

    correct_generated = generate_condition(
        model,
        original,
        condition="correct",
        device=device,
        precision=args.precision,
        batch_size=args.batch_size,
    )
    correct_metrics, correct_rows = score_generated_set(
        model,
        correct_generated,
        original,
        bank,
        device=device,
        precision=args.precision,
        batch_size=args.batch_size,
    )
    save_autonomous_gallery(
        original,
        correct_generated,
        output_dir / "autonomous_original.png",
    )
    teacher_generated = teacher_forced_generation(
        model,
        original,
        device=device,
        precision=args.precision,
        batch_size=args.batch_size,
    )
    teacher_metrics, _ = score_generated_set(
        model,
        teacher_generated,
        original,
        bank,
        device=device,
        precision=args.precision,
        batch_size=args.batch_size,
    )

    shuffled_metrics = None
    blank_metrics = None
    training_font_metrics = None
    if not args.skip_controls:
        for condition in ("shuffled", "blank"):
            generated = generate_condition(
                model,
                original,
                condition=condition,
                device=device,
                precision=args.precision,
                batch_size=args.batch_size,
            )
            metrics, _ = score_generated_set(
                model,
                generated,
                original,
                bank,
                device=device,
                precision=args.precision,
                batch_size=args.batch_size,
            )
            if condition == "shuffled":
                shuffled_metrics = metrics
            else:
                blank_metrics = metrics
        training_font_dataset = VisualRasterInstructionDataset(
            development_records,
            split="train",
            render_config=render_config,
            seed=20_263_263,
            include_all_records=True,
        )
        training_font_data = collect_evaluation_data(
            training_font_dataset,
            target_by_identifier,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            maximum=args.maximum_original,
        )
        training_font_generated = generate_condition(
            model,
            training_font_data,
            condition="correct",
            device=device,
            precision=args.precision,
            batch_size=args.batch_size,
        )
        training_font_metrics, _ = score_generated_set(
            model,
            training_font_generated,
            training_font_data,
            bank,
            device=device,
            precision=args.precision,
            batch_size=args.batch_size,
        )

    paraphrase_metrics = None
    paraphrase_rows = None
    if paraphrase is not None:
        paraphrase_generated = generate_condition(
            model,
            paraphrase,
            condition="correct",
            device=device,
            precision=args.precision,
            batch_size=args.batch_size,
        )
        paraphrase_metrics, paraphrase_rows = score_generated_set(
            model,
            paraphrase_generated,
            paraphrase,
            bank,
            device=device,
            precision=args.precision,
            batch_size=args.batch_size,
        )
        save_autonomous_gallery(
            paraphrase,
            paraphrase_generated,
            output_dir / "autonomous_paraphrases.png",
        )

    frequency_answer, frequency_exact = _frequency_baseline(records, original.targets)
    training_summary = _training_summary(checkpoint_path)
    gate_report = v32_gate_report(
        checkpoint_receipt=checkpoint_receipt,
        correct=correct_metrics,
        shuffled=shuffled_metrics,
        blank=blank_metrics,
        teacher=teacher_metrics,
        paraphrase=paraphrase_metrics,
        training_font=training_font_metrics,
        frequency_baseline_exact=frequency_exact,
        training_complete=(
            int(checkpoint.get("global_update", 0)) == 12_000
            and bool(training_summary.get("complete"))
        ),
        peak_vram_bytes=training_summary.get("peak_vram_bytes"),
    )
    report = {
        "architecture": "visual-semantic-raster-transducer-v32",
        "checkpoint": checkpoint_receipt,
        "device": str(device),
        "precision": args.precision,
        "candidate_bank": bank.receipt(),
        "autonomous_inference_inputs": ["prompt_pixels", "prompt_mask"],
        "answers_enter_after_generation": True,
        "correct_original": correct_metrics,
        "teacher_forced": teacher_metrics,
        "shuffled_prompt": shuffled_metrics,
        "blank_prompt": blank_metrics,
        "training_font": training_font_metrics,
        "paraphrase": paraphrase_metrics,
        "frequency_baseline": {
            "most_frequent_training_answer": frequency_answer,
            "exact_accuracy": frequency_exact,
        },
        "controls": {
            "counterfactual_pair_assignment": "unavailable: constructor not yet implemented",
            "decoded_feedback_ablation": "unavailable: separate checkpoint required",
            "random_reader_probe": "unavailable: separate checkpoint required",
            "suffix_only": "not run by this audit revision",
            "nonaligned_origin": "covered stochastically by fixed development renderer",
        },
        "gates": gate_report,
        "training_summary": training_summary,
        "files": {
            "report": str(output_dir / "evaluation_report.json"),
            "original_rows": str(output_dir / "original_rows.jsonl"),
            "paraphrase_rows": (
                str(output_dir / "paraphrase_rows.jsonl")
                if paraphrase_rows is not None
                else None
            ),
            "autonomous_original_png": str(output_dir / "autonomous_original.png"),
        },
    }
    (output_dir / "evaluation_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    with (output_dir / "original_rows.jsonl").open("w", encoding="utf-8") as handle:
        for row in correct_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    if paraphrase_rows is not None:
        with (output_dir / "paraphrase_rows.jsonl").open("w", encoding="utf-8") as handle:
            for row in paraphrase_rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

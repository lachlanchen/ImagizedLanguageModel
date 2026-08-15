#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any, Mapping

import torch
from torch.utils.data import DataLoader

from ilm.visual_lm.canonical_glyph_language import (
    CanonicalGlyphLanguageModel,
    canonical_glyph_language_config_from_payload,
)
from ilm.visual_lm.canonical_glyph_language_data import (
    V42_ARCHITECTURE,
    CanonicalGlyphAuditDataset,
    CanonicalGlyphPairAuditDataset,
    CanonicalGlyphRenderConfig,
    canonical_glyph_audit_collate,
    canonical_glyph_pair_audit_collate,
    render_canonical_character_bank,
)
from ilm.visual_lm.canonical_glyph_language_evaluation import (
    evaluate_canonical_language,
    evaluate_counterfactual_pairs,
)
from ilm.visual_lm.factorized_visual_context_data import (
    build_factorized_suffix_pairs,
)
from ilm.visual_lm.predictive_state_diagnostic import (
    audit_window_digest,
    build_partition_audit_windows,
    collect_direct_actuator_predictions,
    evaluate_context_length_curve,
    evaluate_direct_actuator_predictions,
)
from ilm.visual_lm.visual_cell_data import (
    load_v25_records,
    pack_visual_cells,
    verify_v25_manifest,
    visual_cell_partition_receipt,
)
from ilm.visual_lm.visual_cell_eval_data import (
    build_visual_cell_audit_windows,
    build_visual_character_statistics,
    visual_character_statistics_receipt,
)
from ilm.visual_lm.visual_future_block_language_v48 import (
    VisualFutureBlockLanguageModelV48,
    visual_future_block_language_boundary_receipt_v48,
    visual_future_block_language_config_from_payload_v48,
)
from ilm.visual_lm.visual_future_block_language_v48_data import (
    V48_ARCHITECTURE,
    VisualFutureBlockAuditDataset,
    build_four_future_audit_windows_v48,
    visual_future_block_audit_collate,
    visual_future_block_data_boundary_receipt,
    visual_pair_digest_v48,
)
from ilm.visual_lm.visual_future_block_language_v48_evaluation import (
    V48_EXPECTED_OFFSET_TOP1,
    V48_FROZEN_V42_FULL_TOP1,
    build_offset_conditional_counts_v48,
    evaluate_closed_loop_generation_v48,
    evaluate_four_future_fields_v48,
    evaluate_offset_conditional_control_v48,
    finite_metric_tree_v48,
    visual_future_block_gate_report_v48,
    visual_future_block_language_boundary_is_clean_v48,
)
from scripts.train_visual_future_block_language_v48 import (
    DEFAULT_MANIFEST,
    DEFAULT_OUTPUT,
    FIXED_OPTIMIZATION,
    PROTOCOL_DOCUMENT,
    SOURCE_FILES,
    _atomic_json,
)
from scripts.train_visual_state_actuator import (
    autocast_context,
    choose_device,
    file_sha256,
    seed_everything,
)


DEFAULT_CHECKPOINT = f"{DEFAULT_OUTPUT}/checkpoint_final.pt"
DEFAULT_V42_CHECKPOINT = (
    "artifacts/canonical_glyph_language_v42_20260814/checkpoint_final.pt"
)
DEFAULT_AUDIT_OUTPUT = f"{DEFAULT_OUTPUT}/development"
AUDIT_ERRATUM_DOCUMENT = (
    "references/visual_future_block_language_v48_audit_erratum.md"
)
EXPECTED_AUDIT_ERRATUM_SHA256 = (
    "1cc9bdd777da6079b0cf0dbdaf88645114c5c25c77c283bdf92300730aa99f20"
)
EVALUATOR_SOURCE = "scripts/eval_visual_future_block_language_v48.py"
ORIGINAL_EVALUATOR_SHA256 = (
    "d7c2e865362715eb5eda7f23b2726031a648acafdba2a03b9b739e3d2e7e446d"
)
MATCHED_WINDOW_SEED = 20264220
FUTURE_WINDOW_SEED = 20264820
PAIR_SEED = 20264221
TERMINAL_WINDOW_SEED = 20264811
RASTER_SHEET_SEED = 20264877
TERMINAL_LENGTHS = tuple(range(56, 65))
EXPECTED_V42_CHECKPOINT_SHA256 = (
    "a5ce2ff20d0fc6d336f489b1e2d29bb96a1d0666399c2e2b95e7ac33a9abe870"
)
EXPECTED_BANK_SHA256 = (
    "a88509b3c1d3093e63dd1ceb77dcd86c7ef282c80927284b93c4ef09cd9456ad"
)
EXPECTED_MATCHED_WINDOW_SHA256 = (
    "4665be2d5cf1714a21d6523ca697749456036d7719302171762bc30291f451f3"
)
EXPECTED_FUTURE_ELIGIBLE = 31_555
EXPECTED_FUTURE_WINDOW_SHA256 = (
    "9ba79f8f7e196192cd07daa7438daa8a60665b78650dc1b3f4c3d441db561d03"
)
EXPECTED_PAIR_SHA256 = (
    "b1d62615aebeda8b48bc9ee526ad5362320bc95dd0a7be612a789a37b6e16e1a"
)
EXPECTED_TERMINAL_WINDOW_SHA256 = (
    "e68db6c27b75a4d502a0dac7fe08487f1c5e3850d46ea1e561c7f18a5757e630"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the frozen V48 image-only future-block model."
    )
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--v42-checkpoint", default=DEFAULT_V42_CHECKPOINT)
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--out", default=DEFAULT_AUDIT_OUTPUT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--precision", choices=("fp32", "fp16", "bf16"), default="bf16"
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--windows", type=int, default=2_048)
    parser.add_argument("--future-windows", type=int, default=2_048)
    parser.add_argument("--bank-size", type=int, default=1_024)
    parser.add_argument("--pairs", type=int, default=512)
    parser.add_argument("--closed-loop-examples", type=int, default=256)
    parser.add_argument("--allow-smoke", action="store_true")
    return parser.parse_args()


def _resolve_device(value: str) -> torch.device:
    device = choose_device(value)
    if device.type != "cuda":
        return device
    index = device.index
    if index is None:
        index = torch.cuda.current_device()
    torch.cuda.set_device(index)
    return torch.device("cuda", index)


def _loader(
    dataset: Any,
    *,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    collate_fn: Any,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=False,
        collate_fn=collate_fn,
    )


def _bank_digest(characters: tuple[str, ...]) -> str:
    return hashlib.sha256("".join(characters).encode("utf-8")).hexdigest()


def _current_source_hashes() -> dict[str, str]:
    missing = [path for path in SOURCE_FILES if not Path(path).is_file()]
    if missing:
        raise FileNotFoundError(f"V48 audit source files are missing: {missing}")
    return {path: file_sha256(path) for path in SOURCE_FILES}


def _source_receipt(checkpoint: Mapping[str, Any]) -> dict[str, Any]:
    registered = checkpoint.get("protocol", {}).get("source_files_sha256", {})
    current = _current_source_hashes()
    mismatches = sorted(
        path for path in SOURCE_FILES if registered.get(path) != current[path]
    )
    exact_match = not mismatches and set(registered) == set(SOURCE_FILES)
    documented_amendment = (
        mismatches == [EVALUATOR_SOURCE]
        and set(registered) == set(SOURCE_FILES)
        and registered.get(EVALUATOR_SOURCE) == ORIGINAL_EVALUATOR_SHA256
        and file_sha256(AUDIT_ERRATUM_DOCUMENT)
        == EXPECTED_AUDIT_ERRATUM_SHA256
    )
    return {
        "valid": exact_match or documented_amendment,
        "exact_checkpoint_source_match": exact_match,
        "documented_evaluator_amendment": documented_amendment,
        "mismatched_registered_sources": mismatches,
        "unchanged_registered_source_count": len(SOURCE_FILES) - len(mismatches),
        "registered_evaluator_sha256": registered.get(EVALUATOR_SOURCE),
        "current_evaluator_sha256": current[EVALUATOR_SOURCE],
        "erratum_document": AUDIT_ERRATUM_DOCUMENT,
        "erratum_sha256": file_sha256(AUDIT_ERRATUM_DOCUMENT),
    }


def _fixed_training_arguments(checkpoint: Mapping[str, Any]) -> bool:
    effective = checkpoint.get("protocol", {}).get("effective_arguments", {})
    return all(effective.get(key) == value for key, value in FIXED_OPTIMIZATION.items())


def _fixed_evaluation_arguments(args: argparse.Namespace) -> bool:
    return (
        args.precision == "bf16"
        and args.windows == 2_048
        and args.future_windows == 2_048
        and args.bank_size == 1_024
        and args.pairs == 512
        and args.closed_loop_examples == 256
    )


def _lightweight_offset_batches(
    windows: tuple[Any, ...],
    character_index: Mapping[str, int],
    *,
    batch_size: int,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for start in range(0, len(windows), batch_size):
        batch = windows[start : start + batch_size]
        output.append(
            {
                "last_character": [window.last_context for window in batch],
                "target_indices": torch.tensor(
                    [
                        [character_index[value] for value in window.continuation]
                        for window in batch
                    ],
                    dtype=torch.long,
                ),
            }
        )
    return output


@torch.no_grad()
def _save_raster_sheet(
    model: VisualFutureBlockLanguageModelV48,
    dataset: VisualFutureBlockAuditDataset,
    path: Path,
    *,
    device: torch.device,
    precision: str,
) -> None:
    seed_everything(RASTER_SHEET_SEED)
    count = min(4, len(dataset))
    context = torch.stack([dataset[index]["context"] for index in range(count)]).to(
        device
    )
    target = torch.stack(
        [dataset[index]["future_pixels"] for index in range(count)]
    ).to(device)
    with autocast_context(device, precision):
        _forecast_pixels, forecast_trace = model.forecast(context)
        _sequence, rollout_trace = model.generate(context, new_cells=4)
    proposal = model.field.probabilities(forecast_trace["future_fields"])
    visible = model.field.binary(forecast_trace["future_fields"])
    rollout = rollout_trace["generated_cells"]
    rows = [
        torch.cat((target[index], proposal[index], visible[index], rollout[index]))
        for index in range(count)
    ]
    cells = torch.stack(rows).flatten(0, 1)
    path.parent.mkdir(parents=True, exist_ok=True)
    pack_visual_cells(cells.cpu(), columns=16, gutter=2).save(path)


def main() -> None:
    args = parse_args()
    positive = (
        args.batch_size,
        args.windows,
        args.future_windows,
        args.bank_size,
        args.pairs,
        args.closed_loop_examples,
    )
    if min(positive) < 1 or args.num_workers < 0:
        raise ValueError("V48 audit sizes are invalid")
    started = time.perf_counter()
    seed_everything(FUTURE_WINDOW_SEED)
    device = _resolve_device(args.device)
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    checkpoint = torch.load(
        args.checkpoint,
        map_location="cpu",
        weights_only=False,
        mmap=True,
    )
    if checkpoint.get("architecture") != V48_ARCHITECTURE:
        raise ValueError("audit checkpoint is not V48")
    checkpoint_is_smoke = bool(checkpoint.get("smoke_only"))
    checkpoint_is_exploratory = bool(checkpoint.get("exploratory"))
    if (checkpoint_is_smoke or checkpoint_is_exploratory) and not args.allow_smoke:
        raise PermissionError("smoke or exploratory checkpoints require --allow-smoke")
    strict_evidence = not checkpoint_is_smoke and not checkpoint_is_exploratory
    source_receipt = _source_receipt(checkpoint)
    if strict_evidence:
        if checkpoint.get("update") != FIXED_OPTIMIZATION["steps"]:
            raise ValueError("V48 evidence requires the update-10000 checkpoint")
        if not _fixed_training_arguments(checkpoint):
            raise ValueError("V48 checkpoint changed the frozen optimization")
        if not _fixed_evaluation_arguments(args):
            raise ValueError("V48 production audit sizes are frozen")
        if device.type != "cuda" or device.index != 0:
            raise ValueError("V48 production audit is frozen to CUDA device 0")
        if checkpoint.get("protocol", {}).get("sha256") != file_sha256(
            PROTOCOL_DOCUMENT
        ):
            raise ValueError("V48 checkpoint protocol digest changed")
        if not source_receipt["valid"]:
            raise ValueError("V48 checkpoint source receipt is invalid")

    render_config = CanonicalGlyphRenderConfig(**checkpoint["render_config"])

    v42_checkpoint = torch.load(
        args.v42_checkpoint,
        map_location="cpu",
        weights_only=False,
        mmap=True,
    )
    if v42_checkpoint.get("architecture") != V42_ARCHITECTURE:
        raise ValueError("matched checkpoint is not V42")
    if strict_evidence and file_sha256(args.v42_checkpoint) != (
        EXPECTED_V42_CHECKPOINT_SHA256
    ):
        raise ValueError("matched V42 checkpoint digest changed")
    if v42_checkpoint.get("smoke_only") or v42_checkpoint.get("exploratory"):
        raise PermissionError("matched V42 checkpoint must be production evidence")
    v42_render = CanonicalGlyphRenderConfig(**v42_checkpoint["render_config"])
    if v42_render != render_config:
        raise ValueError("V42 and V48 render configurations differ")

    manifest = verify_v25_manifest(args.manifest, strict=strict_evidence)
    if not (
        manifest["sha256"]
        == checkpoint["manifest"]["sha256"]
        == v42_checkpoint["manifest"]["sha256"]
    ):
        raise ValueError("V42, V48, and audit corpora differ")
    records = load_v25_records(args.manifest, strict_manifest=strict_evidence)
    partition = visual_cell_partition_receipt(records)
    if strict_evidence and partition != checkpoint["partition"]:
        raise ValueError("V48 audit partition receipt changed")
    statistics = build_visual_character_statistics(
        records,
        bank_size=args.bank_size,
        script_views_mode=render_config.script_views,
    )
    bank_sha256 = _bank_digest(statistics.characters)
    if strict_evidence and bank_sha256 != EXPECTED_BANK_SHA256:
        raise ValueError("V48 evaluator bank digest changed")
    bank_images = render_canonical_character_bank(
        statistics,
        render_config=render_config,
    )

    matched_windows = build_visual_cell_audit_windows(
        records,
        statistics,
        count=args.windows,
        continuation_cells=16,
        seed=MATCHED_WINDOW_SEED,
        script_views_mode=render_config.script_views,
    )
    matched_digest = audit_window_digest(matched_windows)
    if strict_evidence and matched_digest != EXPECTED_MATCHED_WINDOW_SHA256:
        raise ValueError("V48 matched next-cell window digest changed")
    matched_dataset = CanonicalGlyphAuditDataset(
        matched_windows,
        statistics,
        render_config=render_config,
    )
    matched_loader = _loader(
        matched_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=device,
        collate_fn=canonical_glyph_audit_collate,
    )

    v42 = CanonicalGlyphLanguageModel(
        canonical_glyph_language_config_from_payload(
            v42_checkpoint["model_config"]
        )
    )
    v42.load_state_dict(v42_checkpoint["model"], strict=True)
    v42.to(device).eval()
    matched_v42 = evaluate_canonical_language(
        v42,
        matched_loader,
        statistics,
        bank_images,
        device=device,
        precision=args.precision,
    )
    if strict_evidence and not math.isclose(
        matched_v42["full_top1"],
        V48_FROZEN_V42_FULL_TOP1,
        rel_tol=0.0,
        abs_tol=1e-6,
    ):
        raise ValueError("same-process V42 matched top-1 changed")
    del v42
    del v42_checkpoint
    if device.type == "cuda":
        torch.cuda.empty_cache()

    model = VisualFutureBlockLanguageModelV48(
        visual_future_block_language_config_from_payload_v48(
            checkpoint["model_config"]
        )
    )
    model.load_state_dict(checkpoint["model"], strict=True)
    model.to(device).eval()

    language = evaluate_canonical_language(
        model,
        matched_loader,
        statistics,
        bank_images,
        device=device,
        precision=args.precision,
    )

    future_windows, future_eligible = build_four_future_audit_windows_v48(
        records,
        statistics,
        count=args.future_windows,
        seed=FUTURE_WINDOW_SEED,
        script_views_mode=render_config.script_views,
    )
    future_digest = audit_window_digest(future_windows)
    if strict_evidence and (
        future_eligible != EXPECTED_FUTURE_ELIGIBLE
        or future_digest != EXPECTED_FUTURE_WINDOW_SHA256
    ):
        raise ValueError("V48 four-future window receipt changed")
    future_dataset = VisualFutureBlockAuditDataset(
        future_windows,
        statistics,
        render_config=render_config,
    )
    future_loader = _loader(
        future_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=device,
        collate_fn=visual_future_block_audit_collate,
    )
    offset_counts = build_offset_conditional_counts_v48(
        records,
        statistics,
        script_views_mode=render_config.script_views,
    )
    offset_control = evaluate_offset_conditional_control_v48(
        offset_counts,
        statistics,
        _lightweight_offset_batches(
            future_windows,
            statistics.index,
            batch_size=args.batch_size,
        ),
    )
    if strict_evidence:
        measured_offset = tuple(
            offset_control[str(index)]["top1"] for index in range(1, 5)
        )
        if measured_offset != V48_EXPECTED_OFFSET_TOP1:
            raise ValueError("V48 frozen offset controls changed")
    future = evaluate_four_future_fields_v48(
        model,
        future_loader,
        bank_images,
        device=device,
        precision=args.precision,
    )
    future["offset_conditional_control"] = offset_control

    pairs = build_factorized_suffix_pairs(
        records,
        split="development",
        suffix_cells=4,
        count=args.pairs,
        seed=PAIR_SEED,
        require_different_identifiers=True,
        allowed_targets=set(statistics.characters),
        script_views_mode=render_config.script_views,
    )
    pair_digest = visual_pair_digest_v48(pairs)
    if strict_evidence and pair_digest != EXPECTED_PAIR_SHA256:
        raise ValueError("V48 counterfactual pair digest changed")
    pair_dataset = CanonicalGlyphPairAuditDataset(
        pairs,
        render_config=render_config,
    )
    pair_loader = _loader(
        pair_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=device,
        collate_fn=canonical_glyph_pair_audit_collate,
    )
    counterfactual = evaluate_counterfactual_pairs(
        model,
        pair_loader,
        device=device,
        precision=args.precision,
    )

    terminal_windows = build_partition_audit_windows(
        records,
        statistics,
        split="development",
        count=args.windows,
        seed=TERMINAL_WINDOW_SEED,
        script_views_mode=render_config.script_views,
    )
    terminal_digest = audit_window_digest(terminal_windows)
    if strict_evidence and terminal_digest != EXPECTED_TERMINAL_WINDOW_SHA256:
        raise ValueError("V48 terminal-position window digest changed")
    terminal_dataset = CanonicalGlyphAuditDataset(
        terminal_windows,
        statistics,
        render_config=render_config,
    )
    terminal_loader = _loader(
        terminal_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=device,
        collate_fn=canonical_glyph_audit_collate,
    )
    bank_fields = model.field.encode_unit(bank_images.to(device))
    terminal = evaluate_context_length_curve(
        model,
        terminal_loader,
        bank_fields,
        lengths=TERMINAL_LENGTHS,
        device=device,
        precision=args.precision,
    )

    direct_predictions = collect_direct_actuator_predictions(
        model,
        matched_loader,
        device=device,
        precision=args.precision,
    )
    direct = evaluate_direct_actuator_predictions(
        model,
        direct_predictions,
        bank_fields,
        threshold=0.0,
    )
    closed_loop = evaluate_closed_loop_generation_v48(
        model,
        future_loader,
        bank_images,
        device=device,
        precision=args.precision,
        maximum_examples=args.closed_loop_examples,
    )

    output = Path(args.out)
    raster_sheet = output / "target_proposal_visible_rollout.png"
    _save_raster_sheet(
        model,
        future_dataset,
        raster_sheet,
        device=device,
        precision=args.precision,
    )
    raster_sheet_sha256 = file_sha256(raster_sheet)

    evaluator_peak = (
        torch.cuda.max_memory_allocated(device) / 1024**3
        if device.type == "cuda"
        else 0.0
    )
    checkpoint_peak = float(checkpoint.get("peak_allocated_vram_gib", 0.0))
    peak = max(checkpoint_peak, evaluator_peak)
    boundary_clean = visual_future_block_language_boundary_is_clean_v48(model)
    model_boundary = visual_future_block_language_boundary_receipt_v48(model)
    gates = visual_future_block_gate_report_v48(
        language,
        future,
        counterfactual,
        terminal,
        direct,
        closed_loop,
        boundary_clean=boundary_clean,
        trainable_parameters=int(model_boundary["trainable_parameters"]),
        peak_allocated_vram_gib=peak,
        training_elapsed_seconds=float(checkpoint["training_elapsed_seconds"]),
        matched_v42_full_top1=float(matched_v42["full_top1"]),
    )
    elapsed = time.perf_counter() - started
    metrics = {
        "matched_v42": matched_v42,
        "language": language,
        "future": future,
        "counterfactual_pairs": counterfactual,
        "terminal": terminal,
        "direct_raster": direct,
        "closed_loop": closed_loop,
    }
    integrity = {
        "strict_update_10000_checkpoint": (
            checkpoint.get("update") == FIXED_OPTIMIZATION["steps"]
        ),
        "checkpoint_is_non_smoke": not checkpoint_is_smoke,
        "checkpoint_is_non_exploratory": not checkpoint_is_exploratory,
        "fixed_training_arguments": _fixed_training_arguments(checkpoint),
        "fixed_evaluation_arguments": _fixed_evaluation_arguments(args),
        "protocol_digest_matches": checkpoint.get("protocol", {}).get("sha256")
        == file_sha256(PROTOCOL_DOCUMENT),
        "source_receipt_valid": source_receipt["valid"],
        "manifest_matches_checkpoint": manifest["sha256"]
        == checkpoint["manifest"]["sha256"],
        "partition_matches_checkpoint": partition == checkpoint["partition"],
        "bank_digest_matches": bank_sha256 == EXPECTED_BANK_SHA256,
        "matched_window_digest_matches": matched_digest
        == EXPECTED_MATCHED_WINDOW_SHA256,
        "future_window_receipt_matches": (
            future_eligible == EXPECTED_FUTURE_ELIGIBLE
            and future_digest == EXPECTED_FUTURE_WINDOW_SHA256
        ),
        "pair_digest_matches": pair_digest == EXPECTED_PAIR_SHA256,
        "terminal_window_digest_matches": terminal_digest
        == EXPECTED_TERMINAL_WINDOW_SHA256,
        "offset_controls_match": tuple(
            offset_control[str(index)]["top1"] for index in range(1, 5)
        )
        == V48_EXPECTED_OFFSET_TOP1,
        "same_process_v42_matches": math.isclose(
            matched_v42["full_top1"],
            V48_FROZEN_V42_FULL_TOP1,
            rel_tol=0.0,
            abs_tol=1e-6,
        ),
        "student_boundary_clean": boundary_clean,
        "all_metrics_finite": finite_metric_tree_v48(metrics),
        "predictions_precede_candidate_scoring": closed_loop[
            "generated_before_candidate_bank_scoring"
        ],
        "recurrent_cells_are_visible_rasters": closed_loop[
            "recurrent_cells_are_visible_rasters"
        ],
        "evaluator_labels_excluded_from_student": True,
        "optimizer_inactive_during_evaluation": True,
        "frozen_partition_remained_closed": True,
    }
    report = {
        "experiment": "visual-future-block-language-v48-development-audit",
        "architecture": V48_ARCHITECTURE,
        "checkpoint": args.checkpoint,
        "checkpoint_sha256": file_sha256(args.checkpoint),
        "checkpoint_update": checkpoint["update"],
        "checkpoint_smoke_only": checkpoint_is_smoke,
        "checkpoint_exploratory": checkpoint_is_exploratory,
        "matched_v42_checkpoint": args.v42_checkpoint,
        "matched_v42_checkpoint_sha256": file_sha256(args.v42_checkpoint),
        "manifest": manifest,
        "partition": partition,
        "statistics": visual_character_statistics_receipt(statistics),
        "bank_sha256": bank_sha256,
        "audit_receipts": {
            "matched_window_sha256": matched_digest,
            "future_window_sha256": future_digest,
            "future_eligible_windows": future_eligible,
            "pair_sha256": pair_digest,
            "terminal_window_sha256": terminal_digest,
            "raster_sheet_seed": RASTER_SHEET_SEED,
            "raster_sheet_sha256": raster_sheet_sha256,
        },
        "data_boundary": visual_future_block_data_boundary_receipt(),
        "model_boundary": model_boundary,
        "source_receipt": source_receipt,
        **metrics,
        "peak_allocated_vram_gib": peak,
        "training_elapsed_seconds": checkpoint["training_elapsed_seconds"],
        "audit_elapsed_seconds": elapsed,
        "integrity": integrity,
        "all_integrity_checks_pass": all(integrity.values()),
        "gates": gates,
        "all_gates_pass": all(gates.values()),
        "frozen_partition_opened": False,
        "strict_evidence": strict_evidence,
        "effective_arguments": vars(args),
    }
    if strict_evidence and not report["all_integrity_checks_pass"]:
        raise RuntimeError("V48 strict development report failed integrity")

    _atomic_json(report, output / "development_report.json")
    _atomic_json(
        {
            "strict_evidence": strict_evidence,
            "all_integrity_checks_pass": report["all_integrity_checks_pass"],
            "all_gates_pass": report["all_gates_pass"],
            "checkpoint_update": checkpoint["update"],
            "matched_v42_full_top1": matched_v42["full_top1"],
            "v48_full_top1": language["full_top1"],
            "v48_full_minus_v42_top1": (
                language["full_top1"] - matched_v42["full_top1"]
            ),
            "future_top1": {
                key: value["top1"] for key, value in future["horizons"].items()
            },
            "counterfactual_arm_accuracy": counterfactual["full_arm_accuracy"],
            "direct_visible_identity_top1": direct["visible_identity_top1"],
            "direct_visible_pixel_f1": direct["visible_pixel_f1"],
            "closed_loop_mean_identity_top1": closed_loop[
                "mean_identity_top1"
            ],
            "peak_allocated_vram_gib": peak,
            "training_elapsed_seconds": checkpoint["training_elapsed_seconds"],
        },
        output / "development_summary.json",
    )
    print(json.dumps(report["integrity"], sort_keys=True), flush=True)
    print(json.dumps(report["gates"], sort_keys=True), flush=True)
    print(json.dumps(report["language"], sort_keys=True), flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from ilm.visual_lm.causal_glyph_flow import causal_glyph_flow_boundary_receipt, file_sha256
from ilm.visual_lm.causal_glyph_flow_data import CausalGlyphCopyDataset
from ilm.visual_lm.causal_glyph_flow_development import (
    V35_DEVELOPMENT_ARCHITECTURE,
    V35_EVALUATION_SEED,
    V35RasterCase,
    TesseractStripOCR,
    autonomous_case_audit,
    cases_from_dataset,
    copy_counterfactual_audit,
    instruction_case,
    load_v35_checkpoint_model,
    report_sha256,
    select_v35_writer,
    teacher_forced_diagnostics,
    v35_checkpoint_audit,
    v35_development_gate,
)
from ilm.visual_lm.direct_visual_patch_data import (
    DirectPatchContinuationDataset,
    DirectPatchInstructionDataset,
    DirectPatchRenderConfig,
    direct_patch_partition,
    render_direct_patch_instruction,
)
from ilm.visual_lm.visual_semantic_raster_data import (
    VisualRasterRecord,
    VisualTextRecord,
    load_visual_raster_instructions,
    load_visual_raster_paraphrases,
    load_visual_text_records,
)


DEFAULT_CHECKPOINT = "artifacts/causal_glyph_flow_v35_20260814/checkpoint_latest.pt"
DEFAULT_PUBLIC_MANIFEST = "data/visual_grammar/chinese_wikisource_public_domain.jsonl"
DEFAULT_INSTRUCTION_MANIFEST = "data/raw/alpaca_zh.json"
DEFAULT_PARAPHRASE_MANIFEST = "data/teacher/folio_paraphrases_zh_holdout.jsonl"
DEFAULT_OUTPUT = "artifacts/causal_glyph_flow_v35_20260814/development"
EXPECTED_PUBLIC_SHA256 = (
    "76048753b52735d14c98f0d9e4eb8d751401fe8810e82eaaaebff517f6866c03"
)
EXPECTED_INSTRUCTION_SHA256 = (
    "6fcb98c6d79691d1f9a88ef513335da9124e7fdeef5103343f5d9f9a6e8f4903"
)
EXPECTED_PARAPHRASE_SHA256 = (
    "132a8a257d31be3cf607d0313bf042dafaee7f2df08440bf7e012a502ed6c02f"
)
PRODUCTION_EVALUATION = {
    "teacher_batch_size": 4,
    "teacher_maximum_examples": 0,
    "public_cases": 32,
    "copy_cases": 32,
    "instruction_cases": 32,
    "paraphrase_cases": 31,
    "writer_selection_copy_cases": 16,
    "writer_selection_instruction_cases": 16,
    "counterfactual_pairs": 16,
    "raw_diagnostic_cases": 8,
    "conditions": ("correct", "shuffled", "blank", "final-quarter"),
    "flow_steps": 8,
    "raster_threshold": 0.5,
    "stop_threshold": 0.5,
    "seed": V35_EVALUATION_SEED,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the fixed V35 development and prompt-control audit."
    )
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--public-manifest", default=DEFAULT_PUBLIC_MANIFEST)
    parser.add_argument("--instruction-manifest", default=DEFAULT_INSTRUCTION_MANIFEST)
    parser.add_argument("--paraphrase-manifest", default=DEFAULT_PARAPHRASE_MANIFEST)
    parser.add_argument("--out", default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--precision", choices=("fp32", "fp16", "bf16"), default="bf16")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def choose_device(value: str) -> torch.device:
    if value == "auto":
        value = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = torch.device(value)
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda:0")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("V35 evaluation requested CUDA but CUDA is unavailable")
    return device


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


def _evaluation_config(smoke: bool) -> dict[str, Any]:
    if not smoke:
        return dict(PRODUCTION_EVALUATION)
    return dict(PRODUCTION_EVALUATION) | {
        "teacher_batch_size": 1,
        "teacher_maximum_examples": 4,
        "public_cases": 2,
        "copy_cases": 2,
        "instruction_cases": 2,
        "paraphrase_cases": 2,
        "writer_selection_copy_cases": 1,
        "writer_selection_instruction_cases": 1,
        "counterfactual_pairs": 1,
        "raw_diagnostic_cases": 1,
    }


def _verify_inputs(args: argparse.Namespace, *, smoke: bool) -> dict[str, Any]:
    hashes = {
        "public": file_sha256(args.public_manifest),
        "instruction": file_sha256(args.instruction_manifest),
        "paraphrase": file_sha256(args.paraphrase_manifest),
    }
    expected = {
        "public": EXPECTED_PUBLIC_SHA256,
        "instruction": EXPECTED_INSTRUCTION_SHA256,
        "paraphrase": EXPECTED_PARAPHRASE_SHA256,
    }
    if not smoke and hashes != expected:
        raise RuntimeError("V35 development data differs from the frozen protocol")
    return {
        "paths": {
            "public": args.public_manifest,
            "instruction": args.instruction_manifest,
            "paraphrase": args.paraphrase_manifest,
        },
        "sha256": hashes,
        "expected_sha256": expected,
        "hashes_match": hashes == expected,
    }


def _split_counts(
    public: Sequence[VisualTextRecord],
    instructions: Sequence[VisualRasterRecord],
) -> dict[str, Any]:
    return {
        "public": {
            split: sum(
                direct_patch_partition(record.identifier, stream="public-domain")
                == split
                for record in public
            )
            for split in ("train", "development", "sealed")
        },
        "instruction": {
            split: sum(
                direct_patch_partition(record.identifier, stream="instruction")
                == split
                for record in instructions
            )
            for split in ("train", "development", "sealed")
        },
    }


def _paraphrase_cases(
    records: Sequence[VisualRasterRecord],
    *,
    config: DirectPatchRenderConfig,
    limit: int,
) -> list[V35RasterCase]:
    cases = []
    for index, record in enumerate(records[:limit]):
        sample = render_direct_patch_instruction(
            record,
            split="development",
            config=config,
            variant=V35_EVALUATION_SEED + 6_000_000 + index * 104_729,
        )
        cases.append(instruction_case(sample, stream="paraphrase"))
    if not cases:
        raise ValueError("V35 selected no wording-shift cases")
    return cases


def _copy_spans(
    dataset: CausalGlyphCopyDataset,
    *,
    required: int,
) -> list[tuple[str, Mapping[str, Any]]]:
    spans: list[tuple[str, Mapping[str, Any]]] = []
    seen: set[str] = set()
    for index in range(len(dataset)):
        sample = dataset[index]
        metadata = sample["metadata"]
        span = str(metadata["copy_span"])
        if span not in seen:
            spans.append((span, metadata))
            seen.add(span)
        if len(spans) >= required:
            return spans
    raise ValueError("V35 could not collect enough distinct copy spans")


def build_copy_counterfactual_pairs(
    public_records: Sequence[VisualTextRecord],
    *,
    config: DirectPatchRenderConfig,
    count: int,
    split: str = "development",
) -> list[tuple[V35RasterCase, V35RasterCase]]:
    if count < 1:
        raise ValueError("V35 counterfactual pair count must be positive")
    probe = CausalGlyphCopyDataset(
        public_records,
        split=split,
        config=config,
        length=max(2_000, count * 200),
        seed=V35_EVALUATION_SEED + 7_000_000,
    )
    spans = _copy_spans(probe, required=max(64, count * 8))
    by_length: dict[int, list[tuple[str, Mapping[str, Any]]]] = {}
    for span, metadata in spans:
        by_length.setdefault(len(span), []).append((span, metadata))
    pairs: list[tuple[V35RasterCase, V35RasterCase]] = []
    attempted: set[tuple[str, str]] = set()
    for length in sorted(by_length):
        candidates = by_length[length]
        for first_index, (first_span, first_meta) in enumerate(candidates):
            for second_span, second_meta in candidates[first_index + 1 :]:
                key = (first_span, second_span)
                if key in attempted or first_span == second_span:
                    continue
                attempted.add(key)
                for attempt in range(32):
                    variant = (
                        V35_EVALUATION_SEED
                        + 8_000_000
                        + len(pairs) * 1_000_003
                        + attempt
                    )
                    rendered = []
                    for side, span, metadata in (
                        ("a", first_span, first_meta),
                        ("b", second_span, second_meta),
                    ):
                        record = VisualRasterRecord(
                            identifier=f"counterfactual:{len(pairs)}:{side}:{span}",
                            prompt=f"照写：{span}",
                            answer=span,
                            language=str(metadata.get("language", "zh")),
                            source=str(metadata.get("source", "public-domain copy")),
                            rights=str(metadata.get("rights", "public domain")),
                        )
                        sample = render_direct_patch_instruction(
                            record,
                            split=split,
                            config=config,
                            variant=variant,
                        )
                        rendered.append(instruction_case(sample, stream="copy"))
                    first, second = rendered
                    same_shape = (
                        first.prompt_length == second.prompt_length
                        and first.target_length == second.target_length
                    )
                    distinct_pixels = not torch.equal(
                        first.target_patches,
                        second.target_patches,
                    )
                    if same_shape and distinct_pixels:
                        pairs.append((first, second))
                        break
                if len(pairs) >= count:
                    return pairs
    raise ValueError(f"V35 built only {len(pairs)} of {count} counterfactual pairs")


def _build_development_data(
    public_records: Sequence[VisualTextRecord],
    instruction_records: Sequence[VisualRasterRecord],
    paraphrase_records: Sequence[VisualRasterRecord],
    *,
    evaluation: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, list[V35RasterCase]], list[Any]]:
    config = DirectPatchRenderConfig()
    public_dataset = DirectPatchContinuationDataset(
        public_records,
        split="development",
        config=config,
        variants_per_record=1,
        seed=V35_EVALUATION_SEED + 1_000_000,
    )
    instruction_dataset = DirectPatchInstructionDataset(
        instruction_records,
        split="development",
        config=config,
        variants_per_record=1,
        seed=V35_EVALUATION_SEED + 2_000_000,
    )
    copy_dataset = CausalGlyphCopyDataset(
        public_records,
        split="development",
        config=config,
        length=max(128, int(evaluation["copy_cases"])),
        seed=V35_EVALUATION_SEED + 3_000_000,
    )
    cases = {
        "public": cases_from_dataset(
            public_dataset,
            stream="public",
            limit=int(evaluation["public_cases"]),
        ),
        "copy": cases_from_dataset(
            copy_dataset,
            stream="copy",
            limit=int(evaluation["copy_cases"]),
        ),
        "instruction": cases_from_dataset(
            instruction_dataset,
            stream="instruction",
            limit=int(evaluation["instruction_cases"]),
        ),
        "paraphrase": _paraphrase_cases(
            paraphrase_records,
            config=config,
            limit=int(evaluation["paraphrase_cases"]),
        ),
    }
    counterfactual = build_copy_counterfactual_pairs(
        public_records,
        config=config,
        count=int(evaluation["counterfactual_pairs"]),
    )
    datasets = {
        "public": public_dataset,
        "copy": copy_dataset,
        "instruction": instruction_dataset,
    }
    return datasets, cases, counterfactual


def _selection_cases(
    cases: Mapping[str, Sequence[V35RasterCase]],
    evaluation: Mapping[str, Any],
) -> list[V35RasterCase]:
    return list(cases["copy"][: int(evaluation["writer_selection_copy_cases"])]) + list(
        cases["instruction"][
            : int(evaluation["writer_selection_instruction_cases"])
        ]
    )


def _evaluate_state(
    model: torch.nn.Module,
    *,
    state: str,
    datasets: Mapping[str, Any],
    cases: Mapping[str, Sequence[V35RasterCase]],
    counterfactual: Sequence[tuple[V35RasterCase, V35RasterCase]],
    evaluation: Mapping[str, Any],
    device: torch.device,
    precision: str,
    ocr: TesseractStripOCR,
    output: Path,
) -> dict[str, Any]:
    def progress(message: str) -> None:
        print(f"[{state}] {message}", flush=True)

    teacher = {}
    for index, (stream, dataset) in enumerate(datasets.items()):
        progress(f"teacher-forced/{stream}: starting {len(dataset)} examples")
        teacher[stream] = teacher_forced_diagnostics(
            model,
            dataset,
            device=device,
            precision=precision,
            batch_size=int(evaluation["teacher_batch_size"]),
            maximum_examples=int(evaluation["teacher_maximum_examples"]),
            flow_seed=V35_EVALUATION_SEED + index * 100_000,
        )
        progress(
            f"teacher-forced/{stream}: complete "
            f"({teacher[stream]['examples']} examples)"
        )
    raw_limit = int(evaluation["raw_diagnostic_cases"])
    selected_cases = _selection_cases(cases, evaluation)
    if state == "raw":
        selected_cases = list(cases["copy"][:raw_limit]) + list(
            cases["instruction"][:raw_limit]
        )
    writer_reports = {}
    for writer in ("anchor", "flow"):
        progress(f"writer-selection/{writer}: starting {len(selected_cases)} cases")
        writer_reports[writer] = autonomous_case_audit(
            model,
            selected_cases,
            writer=writer,
            conditions=("correct",),
            device=device,
            precision=precision,
            ocr=ocr,
            seed=V35_EVALUATION_SEED + 10_000_000,
            progress=progress,
        )
    selection = select_v35_writer(writer_reports["anchor"], writer_reports["flow"])
    selected_writer = str(selection["selected"])
    other_writer = "flow" if selected_writer == "anchor" else "anchor"
    autonomous: dict[str, Any] = {}
    for stream_index, stream in enumerate(("copy", "public", "instruction", "paraphrase")):
        stream_cases = list(cases[stream])
        if state == "raw":
            stream_cases = stream_cases[:raw_limit]
        progress(
            f"autonomous/{stream}: selected={selected_writer}, "
            f"cases={len(stream_cases)}"
        )
        autonomous[stream] = {
            selected_writer: autonomous_case_audit(
                model,
                stream_cases,
                writer=selected_writer,
                conditions=tuple(evaluation["conditions"]),
                device=device,
                precision=precision,
                ocr=ocr,
                seed=V35_EVALUATION_SEED + 20_000_000 + stream_index * 100_000,
                gallery_path=(
                    output / "galleries" / state / f"{stream}_{selected_writer}.png"
                ),
                progress=progress,
            ),
            other_writer: autonomous_case_audit(
                model,
                stream_cases,
                writer=other_writer,
                conditions=("correct",),
                device=device,
                precision=precision,
                ocr=ocr,
                seed=V35_EVALUATION_SEED + 20_000_000 + stream_index * 100_000,
                progress=progress,
            ),
        }
    selected_pairs = list(counterfactual)
    if state == "raw":
        selected_pairs = selected_pairs[:raw_limit]
    autonomous["copy_counterfactual"] = {
        selected_writer: copy_counterfactual_audit(
            model,
            selected_pairs,
            writer=selected_writer,
            device=device,
            precision=precision,
            ocr=ocr,
            progress=progress,
        )
    }
    return {
        "teacher_forced": teacher,
        "writer_selection": selection | {"reports": writer_reports},
        "autonomous": autonomous,
    }


def _closed_loop_receipt(
    model: torch.nn.Module,
    state_report: Mapping[str, Any],
) -> dict[str, Any]:
    boundary = causal_glyph_flow_boundary_receipt(model)
    selected = state_report["writer_selection"]["selected"]
    streams = tuple(
        stream
        for stream in ("copy", "public", "instruction", "paraphrase")
        if stream in state_report["autonomous"]
    )
    finite = all(
        state_report["autonomous"][stream][selected]["conditions"]["correct"][
            "finite"
        ]
        for stream in streams
    )
    checks = {
        "forward_pixels_and_mask_only": boundary["forward_parameters"]
        == ["pixels", "patch_mask"],
        "generated_primary_output_is_raster": boundary["primary_output"]
        == "generated binary writing raster patches",
        "visible_feedback": boundary["feedback_boundary"]
        == "decode-threshold-reencode-visible-raster",
        "no_forbidden_parameter_names": not boundary[
            "parameter_names_with_forbidden_fragments"
        ],
        "no_runtime_teacher": boundary["uses_runtime_teacher"] is False,
        "no_symbolic_runtime": all(
            boundary[name] is False
            for name in (
                "uses_strings",
                "uses_token_ids",
                "uses_unicode_ids",
                "uses_character_ids",
                "uses_embedding_table",
                "uses_vocabulary_logits",
                "uses_ocr",
                "uses_visual_codebook",
                "uses_quantization",
                "uses_retrieval",
            )
        ),
        "finite_autonomous_generation": finite,
    }
    return {
        "passed": bool(streams) and all(checks.values()),
        "checks": checks,
        "streams": list(streams),
        "boundary": boundary,
    }


def main() -> None:
    args = parse_args()
    evaluation = _evaluation_config(args.smoke)
    device = choose_device(args.device)
    output = Path(args.out)
    report_path = output / "development_report.json"
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"V35 development output is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)
    input_receipt = _verify_inputs(args, smoke=args.smoke)
    public_records = load_visual_text_records(args.public_manifest)
    instruction_records = load_visual_raster_instructions(
        args.instruction_manifest,
        maximum_prompt_characters=64,
        maximum_answer_cells=32,
    )
    paraphrase_records = load_visual_raster_paraphrases(
        args.paraphrase_manifest,
        instruction_records,
    )
    datasets, cases, counterfactual = _build_development_data(
        public_records,
        instruction_records,
        paraphrase_records,
        evaluation=evaluation,
    )
    ocr = TesseractStripOCR()
    started = time.perf_counter()
    states: dict[str, Any] = {}
    checkpoint_audit = None
    closed_loop = None
    receipts: dict[str, Any] = {}
    for state in ("raw", "ema"):
        print(f"[{state}] loading checkpoint state", flush=True)
        model, checkpoint, receipt = load_v35_checkpoint_model(
            args.checkpoint,
            device=device,
            state=state,
        )
        receipts[state] = receipt
        states[state] = _evaluate_state(
            model,
            state=state,
            datasets=datasets,
            cases=cases,
            counterfactual=counterfactual,
            evaluation=evaluation,
            device=device,
            precision=args.precision,
            ocr=ocr,
            output=output,
        )
        if state == "ema":
            checkpoint_audit = v35_checkpoint_audit(model, checkpoint)
            closed_loop = _closed_loop_receipt(model, states[state])
        del model, checkpoint
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    if checkpoint_audit is None or closed_loop is None:
        raise RuntimeError("V35 evaluator did not complete the primary EMA state")
    report: dict[str, Any] = {
        "architecture": V35_DEVELOPMENT_ARCHITECTURE,
        "label": "smoke" if args.smoke else "evidence",
        "split": "development",
        "checkpoint": receipts,
        "checkpoint_audit": checkpoint_audit,
        "closed_loop_receipt": closed_loop,
        "inputs": input_receipt,
        "data": {
            "records": _split_counts(public_records, instruction_records),
            "paraphrases": len(paraphrase_records),
            "teacher_examples": {name: len(value) for name, value in datasets.items()},
            "autonomous_cases": {name: len(value) for name, value in cases.items()},
            "counterfactual_pairs": len(counterfactual),
        },
        "evaluation": evaluation,
        "ocr_evaluator": ocr.identity,
        "states": states,
        "elapsed_seconds": time.perf_counter() - started,
        "peak_allocated_vram_bytes": (
            torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
        ),
        "device": str(device),
        "device_name": (
            torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"
        ),
    }
    report["decision"] = v35_development_gate(report)
    report["evidence_eligible"] = not args.smoke
    report["report_sha256_without_self"] = report_sha256(report)
    atomic_write_json(report, report_path)
    summary = {
        "report": str(report_path),
        "report_sha256": file_sha256(report_path),
        "status": report["decision"]["status"],
        "selected_writer": report["decision"]["selected_writer"],
        "elapsed_seconds": report["elapsed_seconds"],
        "peak_allocated_vram_bytes": report["peak_allocated_vram_bytes"],
    }
    atomic_write_json(summary, output / "development_summary.json")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

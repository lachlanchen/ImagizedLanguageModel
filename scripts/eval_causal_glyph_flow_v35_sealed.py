#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch

from ilm.visual_lm.causal_glyph_flow import file_sha256
from ilm.visual_lm.causal_glyph_flow_data import CausalGlyphCopyDataset
from ilm.visual_lm.causal_glyph_flow_development import (
    V35_DEVELOPMENT_ARCHITECTURE,
    V35_EVALUATION_SEED,
    TesseractStripOCR,
    autonomous_case_audit,
    cases_from_dataset,
    copy_counterfactual_audit,
    load_v35_checkpoint_model,
    report_sha256,
    teacher_forced_diagnostics,
    v35_checkpoint_audit,
    v35_sealed_transfer_gate,
)
from ilm.visual_lm.direct_visual_patch_data import (
    DirectPatchContinuationDataset,
    DirectPatchInstructionDataset,
    DirectPatchRenderConfig,
)
from ilm.visual_lm.visual_semantic_raster_data import (
    load_visual_raster_instructions,
    load_visual_text_records,
)
from scripts.eval_causal_glyph_flow_v35 import (
    DEFAULT_CHECKPOINT,
    DEFAULT_INSTRUCTION_MANIFEST,
    DEFAULT_PARAPHRASE_MANIFEST,
    DEFAULT_PUBLIC_MANIFEST,
    PRODUCTION_EVALUATION,
    _closed_loop_receipt,
    _split_counts,
    _verify_inputs,
    atomic_write_json,
    build_copy_counterfactual_pairs,
    choose_device,
    evaluator_source_receipt,
)


DEFAULT_DEVELOPMENT_REPORT = (
    "artifacts/causal_glyph_flow_v35_20260814/development/development_report.json"
)
DEFAULT_OUTPUT = "artifacts/causal_glyph_flow_v35_20260814/sealed"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Open and run the one-shot V35 sealed audit after qualification."
    )
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--development-report", default=DEFAULT_DEVELOPMENT_REPORT)
    parser.add_argument("--public-manifest", default=DEFAULT_PUBLIC_MANIFEST)
    parser.add_argument("--instruction-manifest", default=DEFAULT_INSTRUCTION_MANIFEST)
    parser.add_argument("--paraphrase-manifest", default=DEFAULT_PARAPHRASE_MANIFEST)
    parser.add_argument("--out", default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--precision", choices=("fp32", "fp16", "bf16"), default="bf16")
    return parser.parse_args()


def _load_development_guard(path: str | Path) -> dict[str, Any]:
    report = json.loads(Path(path).read_text(encoding="utf-8"))
    if report.get("split") != "development":
        raise ValueError("V35 sealed guard requires a development report")
    if report.get("label") != "evidence" or not report.get("evidence_eligible", False):
        raise ValueError("V35 sealed guard rejects smoke or ineligible development")
    status = report.get("decision", {}).get("status")
    if status not in {"visual-causal-qualified", "semantic-raster-qualified"}:
        raise ValueError("V35 sealed data cannot open after unqualified development")
    if report.get("report_sha256_without_self") is None:
        raise ValueError("V35 development report lacks its content receipt")
    return report


def main() -> None:
    args = parse_args()
    development = _load_development_guard(args.development_report)
    selected_writer = str(development["decision"]["selected_writer"])
    output = Path(args.out)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(
            "V35 sealed output already exists; the frozen protocol permits one run"
        )
    output.mkdir(parents=True, exist_ok=True)
    device = choose_device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)

    # The qualification guard runs before any sealed record is loaded or rendered.
    input_receipt = _verify_inputs(args, smoke=False)
    public_records = load_visual_text_records(args.public_manifest)
    instruction_records = load_visual_raster_instructions(
        args.instruction_manifest,
        maximum_prompt_characters=64,
        maximum_answer_cells=32,
    )
    config = DirectPatchRenderConfig()
    public_dataset = DirectPatchContinuationDataset(
        public_records,
        split="sealed",
        config=config,
        variants_per_record=1,
        seed=V35_EVALUATION_SEED + 1_000_000,
    )
    instruction_dataset = DirectPatchInstructionDataset(
        instruction_records,
        split="sealed",
        config=config,
        variants_per_record=1,
        seed=V35_EVALUATION_SEED + 2_000_000,
    )
    copy_dataset = CausalGlyphCopyDataset(
        public_records,
        split="sealed",
        config=config,
        length=128,
        seed=V35_EVALUATION_SEED + 3_000_000,
    )
    datasets = {
        "public": public_dataset,
        "copy": copy_dataset,
        "instruction": instruction_dataset,
    }
    cases = {
        "public": cases_from_dataset(
            public_dataset,
            stream="public",
            limit=int(PRODUCTION_EVALUATION["public_cases"]),
        ),
        "copy": cases_from_dataset(
            copy_dataset,
            stream="copy",
            limit=int(PRODUCTION_EVALUATION["copy_cases"]),
        ),
        "instruction": cases_from_dataset(
            instruction_dataset,
            stream="instruction",
            limit=int(PRODUCTION_EVALUATION["instruction_cases"]),
        ),
    }
    counterfactual = build_copy_counterfactual_pairs(
        public_records,
        config=config,
        count=int(PRODUCTION_EVALUATION["counterfactual_pairs"]),
        split="sealed",
    )
    model, checkpoint, checkpoint_receipt = load_v35_checkpoint_model(
        args.checkpoint,
        device=device,
        state="ema",
    )
    if (
        checkpoint_receipt["checkpoint_sha256"]
        != development["checkpoint"]["ema"]["checkpoint_sha256"]
    ):
        raise RuntimeError("V35 sealed audit checkpoint differs from development")
    ocr = TesseractStripOCR()
    started = time.perf_counter()
    def progress(message: str) -> None:
        print(f"[sealed/ema] {message}", flush=True)

    teacher = {}
    for index, (stream, dataset) in enumerate(datasets.items()):
        progress(f"teacher-forced/{stream}: starting {len(dataset)} examples")
        teacher[stream] = teacher_forced_diagnostics(
            model,
            dataset,
            device=device,
            precision=args.precision,
            batch_size=int(PRODUCTION_EVALUATION["teacher_batch_size"]),
            maximum_examples=0,
            flow_seed=V35_EVALUATION_SEED + index * 100_000,
        )
        progress(
            f"teacher-forced/{stream}: complete "
            f"({teacher[stream]['examples']} examples)"
        )
    autonomous = {}
    for index, (stream, stream_cases) in enumerate(cases.items()):
        progress(f"autonomous/{stream}: starting {len(stream_cases)} cases")
        autonomous[stream] = {
            selected_writer: autonomous_case_audit(
                model,
                stream_cases,
                writer=selected_writer,
                conditions=tuple(PRODUCTION_EVALUATION["conditions"]),
                device=device,
                precision=args.precision,
                ocr=ocr,
                seed=V35_EVALUATION_SEED + 20_000_000 + index * 100_000,
                gallery_path=(
                    output
                    / "galleries"
                    / "ema"
                    / f"{stream}_{selected_writer}.png"
                ),
                progress=progress,
            )
        }
    autonomous["copy_counterfactual"] = {
        selected_writer: copy_counterfactual_audit(
            model,
            counterfactual,
            writer=selected_writer,
            device=device,
            precision=args.precision,
            ocr=ocr,
            progress=progress,
        )
    }
    state_report = {
        "teacher_forced": teacher,
        "writer_selection": {
            "selected": selected_writer,
            "locked_from_development": True,
        },
        "autonomous": autonomous,
    }
    report: dict[str, Any] = {
        "architecture": V35_DEVELOPMENT_ARCHITECTURE,
        "label": "evidence",
        "split": "sealed",
        "development_report": {
            "path": str(Path(args.development_report).resolve()),
            "sha256": file_sha256(args.development_report),
            "status": development["decision"]["status"],
            "selected_writer": selected_writer,
        },
        "checkpoint": {"ema": checkpoint_receipt},
        "checkpoint_audit": v35_checkpoint_audit(model, checkpoint),
        "evaluator_source_sha256": evaluator_source_receipt(
            ("scripts/eval_causal_glyph_flow_v35_sealed.py",)
        ),
        "inputs": input_receipt,
        "data": {
            "records": _split_counts(public_records, instruction_records),
            "teacher_examples": {name: len(value) for name, value in datasets.items()},
            "autonomous_cases": {name: len(value) for name, value in cases.items()},
            "counterfactual_pairs": len(counterfactual),
        },
        "evaluation": dict(PRODUCTION_EVALUATION),
        "ocr_evaluator": ocr.identity,
        "states": {"ema": state_report},
        "elapsed_seconds": time.perf_counter() - started,
        "peak_allocated_vram_bytes": (
            torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
        ),
        "device": str(device),
        "device_name": (
            torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"
        ),
    }
    report["closed_loop_receipt"] = _closed_loop_receipt(model, state_report)
    report["decision"] = v35_sealed_transfer_gate(development, report)
    report["report_sha256_without_self"] = report_sha256(report)
    report_path = output / "sealed_report.json"
    atomic_write_json(report, report_path)
    summary = {
        "report": str(report_path),
        "report_sha256": file_sha256(report_path),
        "passed": report["decision"]["passed"],
        "status": report["decision"]["sealed_status"],
        "selected_writer": selected_writer,
        "elapsed_seconds": report["elapsed_seconds"],
    }
    atomic_write_json(summary, output / "sealed_summary.json")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

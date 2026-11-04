#!/usr/bin/env python3
"""Inspect saved answer structure without model calls or output changes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from ilm.visual_lm.visual_qa_data import VisualQARenderer, read_questions
from ilm.visual_lm.visual_qa_evaluation import image_keys
from scripts.train_canonical_glyph_binding_v43 import _atomic_json
from scripts.train_visual_state_actuator import file_sha256


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--audit", default="artifacts/visual_copy_language_v57_20261105/development"
    )
    p.add_argument("--data", default="data/visual_qa/cmrc2018_short/development.jsonl")
    p.add_argument(
        "--out", default="artifacts/visual_copy_language_v57_answer_diagnostic"
    )
    a = p.parse_args()
    torch.set_num_threads(4)
    audit = Path(a.audit)
    out = Path(a.out)
    if (out / "report.json").exists():
        raise FileExistsError("completed diagnostic exists")
    out.mkdir(parents=True, exist_ok=True)
    records = {r["id"]: r for r in read_questions(a.data)}
    renderer = VisualQARenderer()
    rows = []
    raw = json.loads((audit / "copy_answers.json").read_text())
    for item in raw:
        saved = audit / "predictions/copy" / (item["id"] + ".pt")
        result = torch.load(saved, map_location="cpu", weights_only=False)
        output = image_keys(result["images"])
        refs = [
            image_keys(renderer(r["text"])) for r in records[item["id"]]["references"]
        ]
        first = bool(output) and any(output[0] == ref[0] for ref in refs)
        prefixes = []
        for ref in refs:
            correct = 0
            for x, y in zip(output, ref):
                if x != y:
                    break
                correct += 1
            prefixes.append(correct)
        extra = any(
            len(output) > len(ref) and output[: len(ref)] == ref for ref in refs
        )
        repeats = sum(x == y for x, y in zip(output, output[1:]))
        rows.append(
            {
                "id": item["id"],
                "first_image_correct": float(first),
                "longest_correct_prefix": max(prefixes),
                "extra_after_complete_reference": float(extra),
                "reference_length_match": float(
                    any(len(output) == len(ref) for ref in refs)
                ),
                "output_cells": len(output),
                "stopped": result["stopped"],
                "capped": result["capped"],
                "exact_answer": item["exact_answer"],
                "raster_f1": item["raster_f1"],
                "adjacent_repeats": repeats,
                "transitions": max(len(output) - 1, 0),
            }
        )
    summary = {
        key: float(np.mean([r[key] for r in rows]))
        for key in [
            "first_image_correct",
            "longest_correct_prefix",
            "extra_after_complete_reference",
            "reference_length_match",
            "output_cells",
            "stopped",
            "capped",
            "exact_answer",
            "raster_f1",
        ]
    }
    first = [r for r in rows if r["first_image_correct"]]
    report = {
        "post_result": True,
        "model_calls": 0,
        "examples": len(rows),
        "summary": summary,
        "first_correct_subset": {
            "examples": len(first),
            "exact_answer": float(np.mean([r["exact_answer"] for r in first]))
            if first
            else 0.0,
            "mean_f1": float(np.mean([r["raster_f1"] for r in first]))
            if first
            else 0.0,
        },
        "adjacent_repeat_fraction": sum(r["adjacent_repeats"] for r in rows)
        / max(1, sum(r["transitions"] for r in rows)),
        "source_sha256": {
            str(path): file_sha256(path)
            for path in [
                audit / "audit_identity.json",
                audit / "copy_answers.json",
                Path(a.data),
                Path(__file__),
                Path("references/v57_answer_error_diagnostic_plan.md"),
            ]
        },
    }
    _atomic_json(rows, out / "per_example.json")
    _atomic_json(report, out / "report.json")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Locked natural-question audit of independent image-answer generation."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import time

import numpy as np
import torch

from ilm.visual_lm.canonical_glyph_language_data import CanonicalGlyphRenderConfig
from ilm.visual_lm.raster_beam import raster_beam_search
from ilm.visual_lm.visual_copy_language import model_from_checkpoint
from ilm.visual_lm.visual_qa_data import VisualQARenderer, pack_qa, read_questions
from ilm.visual_lm.visual_qa_evaluation import (
    answer_scores,
    paired_article_bootstrap,
    save_answer_image,
)
from scripts.train_canonical_glyph_binding_v43 import (
    _atomic_json,
    _atomic_save,
    _resolve_device,
)
from scripts.train_visual_copy_language_v57 import DATA_HASHES, DATA, SOURCES
from scripts.train_visual_state_actuator import file_sha256

DEFAULT_RUN = "artifacts/visual_copy_language_v57_20261105"
MOTOR = "artifacts/joint_exact_raster_v55_20261105/joint/development/motor_image_evidence.pt"


def summary(items):
    result = {
        key: float(np.mean([r[key] for r in items]))
        for key in ["exact_answer", "raster_f1", "output_cells", "stopped", "capped"]
    }
    routes = Counter(event["route"] for r in items for event in r["events"])
    result.update(examples=len(items), routing=dict(routes))
    return result


@torch.no_grad()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", default=DEFAULT_RUN)
    p.add_argument("--data", default=DATA)
    p.add_argument("--out")
    p.add_argument("--device", default="auto")
    p.add_argument("--smoke", action="store_true")
    a = p.parse_args()
    run = Path(a.run)
    out = Path(a.out or run / "development")
    if (out / "report.json").exists():
        raise FileExistsError("completed audit is immutable")
    out.mkdir(parents=True, exist_ok=True)
    checkpoints = {
        arm: run / arm / "checkpoint_final.pt" for arm in ["copy", "no_copy"]
    }
    # Complete both trained arms before reading any development performance.
    payloads = {
        arm: torch.load(path, map_location="cpu", weights_only=False)
        for arm, path in checkpoints.items()
    }
    for arm, c in payloads.items():
        if c["arm"] != arm or c["smoke_only"] != a.smoke:
            raise ValueError("training arm/smoke mismatch")
        if not a.smoke and c["update"] != 4000:
            raise ValueError("incomplete locked training")
        for name, sha in c["provenance"].items():
            if file_sha256(name) != sha:
                raise ValueError("trained source changed")
    if not all(
        torch.equal(payloads["copy"][k], payloads["no_copy"][k])
        for k in ["data_rng", "training_rng"]
    ):
        raise ValueError("matched data RNGs differ")
    for name, sha in DATA_HASHES.items():
        if file_sha256(Path(a.data) / name) != sha:
            raise ValueError("prepared dataset changed")
    provenance = {
        p: file_sha256(p)
        for p in SOURCES
        + [
            "scripts/eval_visual_copy_language_v57.py",
            "ilm/visual_lm/visual_qa_evaluation.py",
            "ilm/visual_lm/raster_beam.py",
        ]
    }
    identity = {
        "checkpoint_sha256": {k: file_sha256(v) for k, v in checkpoints.items()},
        "provenance": provenance,
        "data_sha256": DATA_HASHES,
        "smoke_only": a.smoke,
        "motor_evidence_sha256": file_sha256(MOTOR),
    }
    if (out / "audit_identity.json").exists():
        if json.loads((out / "audit_identity.json").read_text()) != identity:
            raise ValueError("partial audit identity differs")
    else:
        _atomic_json(identity, out / "audit_identity.json")
    device = _resolve_device(a.device)
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("highest")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    renderer = VisualQARenderer(
        CanonicalGlyphRenderConfig(**payloads["copy"]["render_config"])
    )
    questions = sorted(
        read_questions(Path(a.data) / "development.jsonl"), key=lambda r: r["id"]
    )
    primary_indices = np.random.default_rng(20265720).choice(
        len(questions), 256, replace=False
    )
    primary = [questions[i] for i in primary_indices[: 4 if a.smoke else 256]]
    by_id = {r["id"]: r for r in questions}
    pairs = [
        json.loads(s)
        for s in (Path(a.data) / "counterfactual_pairs.jsonl").read_text().splitlines()
    ]
    if a.smoke:
        pairs = pairs[:2]
        questions = questions[:8]
    likelihood = {}
    answers = {}
    motors = {}
    started = time.perf_counter()

    def predict(model, record, route):
        folder = out / "predictions" / route
        folder.mkdir(parents=True, exist_ok=True)
        saved = folder / (record["id"] + ".pt")
        if saved.exists():
            return torch.load(saved, map_location="cpu", weights_only=False)
        prompt = renderer(record["prompt"]).to(device)
        if route == "copy_blank_question":
            start = 3 + len(record["context"]) + 3
            assert (
                record["prompt"][start : start + len(record["question"])]
                == record["question"]
            )
            prompt[start : start + len(record["question"])] = 0
        result = model.generate(prompt, maximum_output_cells=17, beam_width=8)
        images = result.pop("images").cpu()
        references = [renderer(ref["text"]) for ref in record["references"]]
        row = {
            "id": record["id"],
            "article": record["article_title"],
            **result,
            **answer_scores(images, references, stopped=result["stopped"]),
            "images": images,
        }
        _atomic_save(row, saved)
        save_answer_image(images, saved.with_suffix(".png"))
        return row

    for arm in ["copy", "no_copy"]:
        model = model_from_checkpoint(payloads[arm], device)
        # The likelihood pass never enters model.generate and supplies no test labels there.
        ll = []
        for offset in range(0, len(questions), 4):
            records = questions[offset : offset + 4]
            batch = pack_qa(records, renderer, device)
            cond = model.visual_condition(
                batch["sequence"], batch["prompt_lengths"], batch["query_positions"]
            )
            nll = model.event_nll(
                cond, batch["targets"], batch["continued"], batch["stopped"]
            )
            valid = batch["continued"] | batch["stopped"]
            for b, r in enumerate(records):
                ll.append(
                    {
                        "id": r["id"],
                        "article": r["article_title"],
                        "sequence_nll": float(nll[b].sum()),
                        "event_nll": float(nll[b].sum() / valid[b].sum()),
                    }
                )
        likelihood[arm] = ll
        _atomic_json(ll, out / f"{arm}_likelihood.json")
        routes = ["copy", "copy_blank_question"] if arm == "copy" else ["no_copy"]
        for route in routes:
            rows = []
            for index, r in enumerate(primary):
                rows.append(predict(model, r, route))
                if index % 32 == 0:
                    print(
                        json.dumps(
                            {"phase": "primary", "route": route, "completed": index + 1}
                        ),
                        flush=True,
                    )
            answers[route] = rows
            _atomic_json(
                [{k: v for k, v in r.items() if k != "images"} for r in rows],
                out / f"{route}_answers.json",
            )
        if arm == "copy":
            pair_rows = []
            for index, pair in enumerate(pairs):
                rows = [predict(model, by_id[id], "copy") for id in pair["record_ids"]]
                pair_rows.append(
                    {
                        "context_hash": pair["context_hash"],
                        "record_ids": pair["record_ids"],
                        "article": rows[0]["article"],
                        "both_exact": float(all(r["exact_answer"] for r in rows)),
                        "mean_f1": float(np.mean([r["raster_f1"] for r in rows])),
                    }
                )
                if index % 32 == 0:
                    print(
                        json.dumps(
                            {"phase": "paired_questions", "completed": index + 1}
                        ),
                        flush=True,
                    )
            _atomic_json(pair_rows, out / "paired_question_answers.json")
            # Fixed V55 audit images, not images selected by the new model.
            supplied = torch.load(MOTOR, map_location="cpu", weights_only=False)[
                "targets"
            ][:, 0]
            if a.smoke:
                supplied = supplied[:8]
            exact = []
            for offset in range(0, len(supplied), 16):
                y = supplied[offset : offset + 16].to(device)
                fields = model.reader.field.encode_unit(y)
                pred = raster_beam_search(
                    model, torch.zeros(len(y), 384, device=device), fields, beam_width=8
                ).images
                exact.extend((pred == y).flatten(1).all(-1).cpu().tolist())
            motors = {"examples": len(exact), "exact_fraction": float(np.mean(exact))}
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    articles = [r["article"] for r in answers["copy"]]
    comparisons = {}
    for other in ["copy_blank_question", "no_copy"]:
        for metric in ["exact_answer", "raster_f1"]:
            comparisons[f"copy_minus_{other}_{metric}"] = paired_article_bootstrap(
                [
                    x[metric] - y[metric]
                    for x, y in zip(answers["copy"], answers[other])
                ],
                articles,
            )
    comparisons["no_copy_minus_copy_event_nll"] = paired_article_bootstrap(
        [
            x["event_nll"] - y["event_nll"]
            for x, y in zip(likelihood["no_copy"], likelihood["copy"])
        ],
        [r["article"] for r in likelihood["copy"]],
    )
    measured = {route: summary(rows) for route, rows in answers.items()}
    pair_accuracy = float(np.mean([r["both_exact"] for r in pair_rows]))
    peak = (
        torch.cuda.max_memory_allocated(device) / 1024**3
        if device.type == "cuda"
        else 0.0
    )
    gates = {
        "image_only_boundary": True,
        "clean_split_finite_resource": peak < 16
        and all(
            np.isfinite(r["event_nll"]) for rows in likelihood.values() for r in rows
        ),
        "exact_supplied_motor": motors["exact_fraction"] == 1,
        "exact_primary_answers": measured["copy"]["exact_answer"] >= 0.05,
        "primary_raster_f1": measured["copy"]["raster_f1"] >= 0.25,
        "question_dependent_f1": comparisons[
            "copy_minus_copy_blank_question_raster_f1"
        ]["ci95"][0]
        > 0,
        "both_correct_paired_questions": pair_accuracy >= 0.02,
    }
    report = {
        **identity,
        "measurements": measured,
        "comparisons": comparisons,
        "likelihood": {
            arm: {
                key: float(np.mean([r[key] for r in rows]))
                for key in ["event_nll", "sequence_nll"]
            }
            for arm, rows in likelihood.items()
        },
        "paired_questions": {
            "pairs": len(pair_rows),
            "both_exact_fraction": pair_accuracy,
        },
        "motor": motors,
        "gates": gates,
        "gates_passed": sum(gates.values()),
        "decision": "bounded_reading_pass"
        if all(gates.values())
        else "reject_reading_model",
        "elapsed_seconds": time.perf_counter() - started,
        "peak_allocated_vram_gib": peak,
        "float32_matmul_precision": "highest",
        "primary_ids": [r["id"] for r in primary],
        "qualification": "general ILM remains unqualified; grounded-reading gates only",
    }
    _atomic_json(report, out / "report.json")
    # Save untouched first eight primary answers in tensor form for publication.
    _atomic_save(
        {route: [r["images"] for r in rows[:8]] for route, rows in answers.items()},
        out / "first_primary_image_evidence.pt",
    )
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()

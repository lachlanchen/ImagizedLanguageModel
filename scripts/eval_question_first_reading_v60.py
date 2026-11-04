#!/usr/bin/env python3
"""Locked matched input-order audit of question-conditioned visual reading."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from ilm.visual_lm.visual_answer_blocks import model_from_checkpoint
from ilm.visual_lm.question_first_reading import ordered_question, blank_question_images
from ilm.visual_lm.visual_qa_data import VisualQARenderer, pack_qa, read_questions
from ilm.visual_lm.raster_beam import raster_beam_search
from scripts.train_visual_state_actuator import file_sha256 as sha256
from ilm.visual_lm.visual_qa_evaluation import answer_scores, save_answer_image
from scripts.train_canonical_glyph_binding_v43 import (
    _atomic_json, _atomic_save, _resolve_device,
)
from scripts.train_question_first_reading_v60 import SOURCES
from scripts.train_visual_answer_blocks_v58 import DATA_HASHES


def bootstrap(values, articles):
    values, articles = np.asarray(values, float), np.asarray(articles)
    groups = [np.flatnonzero(articles == a) for a in sorted(set(articles))]
    totals = np.asarray([values[g].sum() for g in groups])
    counts = np.asarray([len(g) for g in groups])
    draws = np.random.default_rng(20266119).integers(
        0, len(groups), (2000, len(groups))
    )
    means = totals[draws].sum(1) / counts[draws].sum(1)
    return dict(mean=float(values.mean()), ci95=np.quantile(means, [.025, .975]).tolist(),
                articles=len(groups), examples=len(values), replicates=2000, seed=20266119)


@torch.no_grad()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", default="artifacts/question_first_reading_v60_20261105")
    p.add_argument("--data", default="data/visual_qa/cmrc2018_short")
    p.add_argument("--out")
    p.add_argument("--device", default="auto")
    a = p.parse_args()
    run, data = Path(a.run), Path(a.data)
    out = Path(a.out or run / "development")
    if (out / "report.json").exists():
        raise FileExistsError("completed audit is immutable")
    out.mkdir(parents=True, exist_ok=True)
    paths = {arm: run / arm / "checkpoint_final.pt" for arm in ("context", "question")}
    payloads = {arm: torch.load(path, map_location="cpu", weights_only=False)
                for arm, path in paths.items()}
    for arm, c in payloads.items():
        if c["order"] != arm or c["smoke_only"] or c["update"] != 4000:
            raise ValueError("both locked production arms must finish before audit")
        for name, digest in c["provenance"].items():
            if sha256(name) != digest:
                raise ValueError("trained source changed")
    if not torch.equal(payloads["context"]["data_rng"], payloads["question"]["data_rng"]):
        raise ValueError("matched training data RNG differs")
    for name, digest in DATA_HASHES.items():
        if sha256(data / name) != digest:
            raise ValueError("prepared question dataset changed")
    identity = dict(
        checkpoint_sha256={arm: sha256(path) for arm, path in paths.items()},
        data_sha256=DATA_HASHES,
        provenance={name: sha256(name) for name in SOURCES + [__file__]},
    )
    identity_path = out / "audit_identity.json"
    if identity_path.exists() and json.loads(identity_path.read_text()) != identity:
        raise ValueError("partial audit identity differs")
    _atomic_json(identity, identity_path)
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("highest")
    device = _resolve_device(a.device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    renderer = VisualQARenderer()
    questions = sorted(read_questions(data / "development.jsonl"), key=lambda r: r["id"])
    train = read_questions(data / "train.jsonl")
    overlap = {key: len({r[key] for r in train} & {r[key] for r in questions})
               for key in ("article_title", "context_hash")}
    if any(overlap.values()):
        raise ValueError("training and development sources overlap")
    indices = np.random.default_rng(20265720).choice(len(questions), 256, replace=False)
    primary = [questions[i] for i in indices]
    by_id = {r["id"]: r for r in questions}
    pairs = [json.loads(s) for s in (data / "counterfactual_pairs.jsonl").read_text().splitlines()]
    old = json.loads(Path("artifacts/visual_answer_blocks_v58_20261105/development/report.json").read_text())
    if [r["id"] for r in primary] != old["primary_ids"] or len(pairs) != 393:
        raise ValueError("locked primary IDs or pairs differ")
    started = time.perf_counter()
    answers, paired, likelihood, motors = {}, {}, {}, {}

    def predict(model, record, arm, blank=False):
        route = arm + ("_blank" if blank else "")
        folder = out / "predictions" / route
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / (record["id"] + ".pt")
        if path.exists():
            return torch.load(path, map_location="cpu", weights_only=False)
        pixels = renderer(ordered_question(record, arm)["prompt"]).to(device)
        if blank:
            pixels = blank_question_images(pixels, record, arm)
        result = model.generate_block(pixels)
        images = result.pop("images").cpu()
        # Labels enter scoring only after the source-image-only decode completes.
        references = [renderer(ref["text"]) for ref in record["references"]]
        row = {"id": record["id"], "article": record["article_title"], "images": images,
               **result, **answer_scores(images, references, stopped=result["stopped"])}
        _atomic_save(row, path)
        save_answer_image(images, path.with_suffix(".png"))
        return row

    for arm in ("context", "question"):
        model = model_from_checkpoint(payloads[arm], device)
        ll = []
        for offset in range(0, len(questions), 8):
            records = questions[offset:offset + 8]
            batch = pack_qa([ordered_question(r, arm) for r in records], renderer, device)
            x, n, y = batch["sequence"], batch["prompt_lengths"], batch["targets"]
            m = batch["continued"].sum(1)
            nll = model.block_nll(model.block_distribution(x, n), x, n, y, m)
            for b, record in enumerate(records):
                ll.append(dict(id=record["id"], sequence_nll=float(nll[b]),
                               event_nll=float(nll[b] / (m[b] + 1))))
        likelihood[arm] = ll
        _atomic_json(ll, out / (arm + "_likelihood.json"))
        for blank in (False, True):
            route = arm + ("_blank" if blank else "")
            rows = []
            for i, record in enumerate(primary):
                rows.append(predict(model, record, arm, blank))
                if i % 64 == 0:
                    print(json.dumps(dict(route=route, primary_completed=i + 1)), flush=True)
            answers[route] = rows
            _atomic_json([{k: v for k, v in r.items() if k != "images"} for r in rows],
                         out / (route + "_answers.json"))
        pair_rows = []
        for i, pair in enumerate(pairs):
            rows = [predict(model, by_id[key], arm) for key in pair["record_ids"]]
            pair_rows.append(dict(context_hash=pair["context_hash"], record_ids=pair["record_ids"],
                                  both_exact=float(all(r["exact_answer"] for r in rows))))
            if i % 64 == 0:
                print(json.dumps(dict(arm=arm, pairs_completed=i + 1)), flush=True)
        paired[arm] = dict(pairs=len(pair_rows), both_exact_fraction=float(np.mean(
            [r["both_exact"] for r in pair_rows])))
        _atomic_json(pair_rows, out / (arm + "_paired_answers.json"))
        supplied = torch.load(
            "artifacts/joint_exact_raster_v55_20261105/joint/development/motor_image_evidence.pt",
            map_location="cpu", weights_only=False)["targets"][:, 0]
        exact = []
        for offset in range(0, len(supplied), 16):
            images = supplied[offset:offset + 16].to(device)
            fields = model.reader.field.encode_unit(images)
            pred = raster_beam_search(model, torch.zeros(len(images), 384, device=device),
                                      fields, beam_width=8).images
            exact.extend((pred == images).flatten(1).all(-1).cpu().tolist())
        motors[arm] = dict(examples=len(exact), exact_fraction=float(np.mean(exact)))
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    measurements = {
        route: {key: float(np.mean([r[key] for r in rows])) for key in
                ("exact_answer", "raster_f1", "output_cells", "stopped", "capped")}
        for route, rows in answers.items()
    }
    comparisons = {}
    for left, right in (("question", "context"), ("question", "question_blank"),
                        ("context", "context_blank")):
        for metric in ("exact_answer", "raster_f1"):
            comparisons[f"{left}_minus_{right}_{metric}"] = bootstrap(
                [x[metric] - y[metric] for x, y in zip(answers[left], answers[right])],
                [r["article"] for r in answers[left]])
    elapsed = time.perf_counter() - started
    peak = torch.cuda.max_memory_allocated(device) / 1024**3 if device.type == "cuda" else 0.
    gates = dict(
        pixel_source_boundary=all(c["boundary"][key] is False for c in payloads.values()
                                 for key in ("text_tokenizer", "symbol_ids", "glyph_inventory",
                                             "codebook", "ocr", "external_runtime_model")),
        clean_split_finite_resource=not any(overlap.values()) and peak < 16 and elapsed < 14400
            and all(np.isfinite(r["sequence_nll"]) for ll in likelihood.values() for r in ll),
        question_first_f1_gain=comparisons["question_minus_context_raster_f1"]["ci95"][0] > 0,
        exact_answers=measurements["question"]["exact_answer"] >= .15,
        raster_f1=measurements["question"]["raster_f1"] >= .35,
        question_dependent_f1=comparisons["question_minus_question_blank_raster_f1"]["ci95"][0] > 0,
        both_correct_paired_questions=paired["question"]["both_exact_fraction"] >= .02,
        retained_motor=all(m["exact_fraction"] == 1 for m in motors.values()),
    )
    report = dict(**identity, measurements=measurements, comparisons=comparisons,
                  paired_questions=paired, motor=motors, split_overlap=overlap, gates=gates,
                  gates_passed=sum(gates.values()),
                  decision="bounded_reading_pass" if all(gates.values()) else "reject_question_first_intervention",
                  likelihood={arm: {key: float(np.mean([r[key] for r in rows]))
                                     for key in ("event_nll", "sequence_nll")}
                              for arm, rows in likelihood.items()},
                  primary_ids=[r["id"] for r in primary], primary_examples=256,
                  elapsed_seconds=elapsed, peak_allocated_vram_gib=peak,
                  inference_cost_scope="complete local reader and block decoder, plus retained motor audit",
                  float32_matmul_precision="highest", development_reused=True,
                  qualification="observed-block reading only; general image-language generation remains unqualified",
                  likelihood_units="conditional observed-image-block NLL; event denominator reference cells plus one")
    _atomic_save({route: [r["images"] for r in rows[:8]] for route, rows in answers.items()},
                 out / "first_primary_image_evidence.pt")
    _atomic_json(report, out / "report.json")
    print(json.dumps({k: report[k] for k in ("measurements", "comparisons", "paired_questions", "gates", "decision")}, indent=2), flush=True)


if __name__ == "__main__":
    main()


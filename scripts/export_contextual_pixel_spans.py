#!/usr/bin/env python3
"""Independently check source-pixel answers and retain every scored raster."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import random
import shutil

import numpy as np
from PIL import Image
import torch

from ilm.visual_lm.clean_image_actor import tensor_digest
from scripts.cache_semantic_images_v72 import atomic_json, sha


def read(p): return json.loads(Path(p).read_text())
def pixels(p): return np.array(Image.open(p).convert("L"), copy=True)


def trim(a):
    occupied = np.flatnonzero((a == 0).any(0))
    if not len(occupied): return a[:, :0]
    return a[:, occupied[0]:occupied[-1] + 1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args(); run = args.run
    if read(run / "status.json")["state"] != "completed":
        raise ValueError("terminal complete pipeline required")
    original = read(run / "result.json"); prep = run / "prepared"
    prep_report = read(prep / "report.json"); index = read(prep / "inputs.json")
    if sha(prep / "inputs.json") != prep_report["inputs_sha256"] or sha(prep / "references.json") != prep_report["references_sha256"]:
        raise ValueError("original data inventories changed")
    for name, digest in prep_report["identity"]["sources"].items():
        if sha(name) != digest: raise ValueError("executed implementation changed: " + name)
    training = read(run / "training/report.json")
    if sha(run / "training/checkpoint.pt") != training["checkpoint_sha256"] or sha(run / "training/updates.jsonl") != training["updates_sha256"]:
        raise ValueError("numeric model or optimizer journal changed")
    journals = [json.loads(line) for line in (run / "training/updates.jsonl").read_text().splitlines()]
    ids = [r["id"] for r in index if r["split"] == "train"]
    rng = random.Random(2026110704); expected = []
    for _ in range(80):
        order = list(range(320)); rng.shuffle(order); expected.extend(ids[i] for i in order)
    draws = [name for r in journals for name in r["ids"]]
    if (len(journals) != 3200 or [r["update"] for r in journals] != list(range(1, 3201))
            or draws != expected or draws != read(run / "training/schedule.json")
            or set(Counter(draws).values()) != {80}
            or not all(math.isfinite(r["loss"]) and math.isfinite(r["gradient_norm"]) for r in journals)):
        raise ValueError("complete finite80-pass optimizer coverage required")
    features = read(run / "features/report.json")
    for row in features["records"]:
        if sha(run / "features" / row["file"]) != row["sha256"]:
            raise ValueError("source features changed")
    if len(features["source_replay"]) != 4 or not all(r["exact"] for r in features["source_replay"]):
        raise ValueError("four source memory replays required")
    refs = {r["id"]: r for r in read(prep / "references.json")}
    raw = read(run / "assessment/outputs.json")
    if len(raw) != 256 or sha(run / "assessment/outputs.json") != original["assessment"]["outputs_sha256"]:
        raise ValueError("all unmodified assessment calls required")
    args.output.mkdir(parents=True, exist_ok=False)
    files = {}

    def retain(source, name, digest=None):
        if digest is not None and sha(source) != digest: raise ValueError("source artifact changed")
        dest = args.output / name; dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists() and sha(dest) != sha(source): raise ValueError("retained path collision")
        shutil.copy2(source, dest); files[name] = sha(dest)
        return name

    cases = []
    for i, row in enumerate(raw):
        source = prep / row["input"]; answer = run / "assessment" / row["output"]
        a, b = pixels(source), pixels(answer)
        # Reconstruct every answer directly from its own original input columns.
        ink = (a == 0).any(0); edge = np.diff(np.r_[False, ink, False].astype(np.int8))
        left, right = np.flatnonzero(edge == 1), np.flatnonzero(edge == -1)
        start, length = row["event"]["start"], row["event"]["length"]
        if not 0 <= start < start + length <= len(left): raise ValueError("invalid original source boundary")
        reconstructed = np.pad(a[:, left[start]:right[start + length - 1]], ((0, 0), (2, 2)), constant_values=255)
        if not np.array_equal(b, reconstructed): raise ValueError("answer is not its declared source pixels")
        key = row["id"].split("/question_erased")[0].split("/passage_erased")[0]
        targets, paths = [], []
        for t in refs[key]["targets"]:
            target = prep / t["file"]; targets.append(pixels(target))
            paths.append(retain(target, "references/" + t["sha256"] + ".png", t["sha256"]))
        exact = any(np.array_equal(trim(b), trim(t)) for t in targets)
        prediction = Counter(bytes(c) for c in trim(b).T)
        f1 = max(2 * sum((prediction & Counter(bytes(c) for c in trim(t).T)).values()) /
                 (len(trim(b).T) + len(trim(t).T)) for t in targets)
        scored = original["cases"][i]
        if scored["id"] != row["id"] or exact != scored["exact"] or abs(f1 - scored["unordered_column_f1"]) > 1e-12:
            raise ValueError("independent PNG score differs")
        cases.append(dict(**{k: row[k] for k in ("id", "split", "task", "language", "article", "pair", "condition", "event")},
            input=retain(source, f"inputs/{i:04d}.png", row["input_sha256"]),
            output=retain(answer, f"answers/{i:04d}.png", row["output_sha256"]), references=paths,
            exact=exact, unordered_column_f1=f1))
    # Full native-image inference must reproduce both fresh source features and answers.
    native = read(run / "native_replay/report.json")
    cached = {r["id"]: r for r in features["records"]}; checks = []
    for i, n in enumerate(native["outputs"]):
        candidates = [r for r in raw if r["input_sha256"] == n["input_sha256"]]
        if len(candidates) != 1: raise ValueError("fixed native replay input must be unambiguous")
        row = candidates[0]; saved = torch.load(run / "features" / cached[row["id"]]["file"], map_location="cpu", weights_only=True)
        hashes = {k: tensor_digest(saved[k]) for k in ("visual", "memory", "geometry", "source_units")}
        if hashes != n["feature_hashes"] or not np.array_equal(pixels(run / "native_replay" / n["output"]), pixels(run / "assessment" / row["output"])):
            raise ValueError("actual native image forward does not replay cached inference")
        retain(run / "native_replay" / n["output"], f"native_replay/{i:03d}.png", n["output_sha256"])
        checks.append(dict(id=row["id"], feature_hashes=hashes, source_features_bit_exact=True, answer_pixels_exact=True))
    if len(checks) != 4: raise ValueError("all four native calls required")
    for source, name in ((run / "prepared/report.json", "reports/preparation.json"),
        (run / "features/report.json", "reports/source_features.json"), (run / "training/report.json", "reports/training.json"),
        (run / "training/updates.jsonl", "reports/updates.jsonl"), (run / "assessment/report.json", "reports/assessment.json"),
        (run / "assessment/outputs.json", "reports/raw_outputs.json"), (run / "native_replay/report.json", "reports/native_replay.json"),
        (run / "data_review/review.json", "reports/data_review.json")):
        retain(source, name)
    groups = defaultdict(list); pairs = defaultdict(list)
    for row in cases:
        groups[f"{row['split']}/{row['task']}/{row['condition']}/{row['language']}"].append(row)
        if row["split"] == "development" and row["task"] == "qa" and row["condition"] == "normal": pairs[row["pair"]].append(row)
    summary = {k: dict(cases=len(v), exact=sum(r["exact"] for r in v), unordered_column_f1=sum(r["unordered_column_f1"] for r in v) / len(v)) for k, v in groups.items()}
    both = sum(all(r["exact"] for r in v) for v in pairs.values())
    if summary != original["groups"] or both != original["both_correct_pairs"]: raise ValueError("complete independent summary differs")
    atomic_json(args.output / "result.json", dict(groups=summary, both_correct_pairs=both, gates=original["gates"],
        component_qualified=original["component_qualified"], cases=cases, files_sha256=files, source_run=str(run),
        source_result_sha256=sha(run / "result.json"), training=training, native_replay_checks=checks,
        independent_png_scores_exact=True, all256_answers_reconstructed_from_actual_source=True,
        all320_training_ids_seen80times=True, source_feature_files_authenticated=544,
        export_script_sha256=sha(__file__), image_answers_are_source_pixel_copies=True,
        reused_public_development=True, free_generation_demonstrated=False, general_ilm_goal_complete=False,
        copyright="CMRC2018 Cui et al. (2019), CC-BY-SA-4.0; Qwen3-VL Alibaba, Apache-2.0; synthetic copies."))
    print("Verified and exported all256 physical answers and four native image forwards.", flush=True)


if __name__ == "__main__": main()

"""Refit paired visual evidence objectives and measure variation across stories and seeds."""

import argparse
from pathlib import Path
import gzip
import json
from collections import defaultdict
import numpy as np
from scripts.answer_dialogue_surface import load
from scripts.dialogue_ink_data import read_data
from scripts.dialogue_teacher_stories import new_stories, absent
from scripts.evaluate_dialogue_evidence import (
    predict,
    summarize,
    difference,
    turn_metrics,
)
from scripts.qualify_dialogue_teacher import PUB, dump, sha
from scripts.train_dialogue_evidence import fit, fitting_data, RELEASE, SEED
from scripts.train_dialogue_surface import ROOT_MODEL, corpus, representation
from scripts.train_span_choice import gpu_owner

SEEDS = list(range(SEED, SEED + 4))


def joint_interval(deltas, counts):
    rng = np.random.default_rng(SEED)
    nseed, nstory = deltas.shape
    values = []
    for _ in range(100):
        seeds = rng.integers(nseed, size=(1000, nseed, 1))
        stories = rng.integers(nstory, size=(1000, 1, nstory))
        values.extend(
            deltas[seeds, stories].sum((1, 2)) / (nseed * counts[stories[:, 0]].sum(1))
        )
    return np.quantile(values, [0.025, 0.975]).tolist()


def main(destination):
    destination.mkdir(parents=True, exist_ok=True)
    if (destination / "replication.json").exists():
        raise FileExistsError("Completed replication in this output directory")
    registry = read_data()
    data, targets = fitting_data(registry)
    if targets is not None:
        raise ValueError("Registered human-only comparison requires rejected teacher.")
    parent = sha(ROOT_MODEL / "ink/writer.safetensors")
    cohorts = {group: new_stories(registry, group) for group in ("long", "longer")}
    face, unit, sep = representation("ink")
    decode = {face(chr(i)): chr(i) for i in range(32, 127)}
    result = {"parent": parent, "seeds": SEEDS, "by_seed": {}, "groups": {}}
    totals = defaultdict(list)
    for seed in SEEDS:
        root = RELEASE if seed == SEED else destination / str(seed)
        entry = result["by_seed"][seed] = {"arms": {}, "groups": {}}
        for arm in ("plain", "human"):
            folder = root / "ink" / arm
            if not folder.exists() and seed != SEED:
                fit("ink", arm, data, None, root, seed)
            meta = json.loads((folder / "weights.json").read_text())
            if (
                meta["parent"],
                meta["seed"],
                meta["steps"],
                meta["labels"],
                meta["sha256"],
            ) != (parent, seed, 1500, arm, sha(folder / "writer.safetensors")):
                raise ValueError(f"Inconsistent fit at {folder}")
            entry["arms"][arm] = meta
        for group, (held, markers, evaluator) in cohorts.items():
            rows = held["rows"]
            hard = [
                i
                for i, r in enumerate(rows)
                if absent(
                    evaluator.gold_data[markers[i]], (held["pages"][r[1]], r[3], r[4])
                )
            ]
            scores = entry["groups"][group] = {}
            outputs = []
            for arm in ("plain", "human"):
                folder = root / "ink" / arm
                net = load(folder, "cuda")
                ds = corpus(held, "fresh", net, face, unit)
                records = predict(folder, ds, net, sep, unit, decode, rows)
                if seed == SEED:
                    old = PUB / "responses" / str(seed) / f"{group}-{arm}.json.gz"
                    if records != json.loads(gzip.decompress(old.read_bytes())):
                        raise ValueError("Released first seed no longer reproduces")
                target = destination / str(seed) / f"{group}-{arm}.json.gz"
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(
                    gzip.compress(
                        json.dumps(records, separators=(",", ":")).encode(), mtime=0
                    )
                )
                scores[arm] = summarize(records, markers, evaluator, hard)
                scores[arm]["outputs_sha256"] = sha(target)
                outputs.append(records)
            scores["comparison"] = difference(
                *outputs, rows, markers, evaluator, set(hard)
            )
            by_story = defaultdict(list)
            for i, r in enumerate(rows):
                before, after = (
                    turn_metrics(evaluator, markers[i], a[i]["original"])["f1"]
                    for a in outputs
                )
                by_story[r[1]].append(after - before)
            totals[group].append([sum(by_story[k]) for k in sorted(by_story)])
            counts = np.array([len(by_story[k]) for k in sorted(by_story)])
            if seed == SEEDS[-1]:
                sums = np.array(totals[group])
                delta = float(sums.mean(0).sum() / counts.sum())
                ci = joint_interval(sums, counts)
                result["groups"][group] = dict(
                    turns=len(rows),
                    stories=len(counts),
                    absent=len(hard),
                    delta=delta,
                    interval95=ci,
                    passes=delta > 0 and ci[0] > 0,
                    draws=100000,
                )
            print("measured", seed, group, scores["comparison"], flush=True)
    result["passes"] = all(g["passes"] for g in result["groups"].values())
    dump(destination / "replication.json", result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with gpu_owner():
        main(args.output)

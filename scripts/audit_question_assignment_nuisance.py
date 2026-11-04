#!/usr/bin/env python3
"""Cross-validate coarse visual width probes on frozen V63 TRAIN pairs only."""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import torch


PAIR_ROOT = Path("artifacts/question_assignment_v63_20261105/data")
TRAIN_SOURCE = Path("data/visual_qa/cmrc2018_short/train.jsonl")
PLAN = "references/question_assignment_nuisance_audit_plan.md"
PAIR_REPORT_SHA = "55d7e641fe4a51547d7819359cf5a233dbe894d73a7b7ee5feae21a51a17aec6"
FEATURES = ["question_width_cells", "question_ink_fraction", "ink_mean_x",
            "ink_mean_y", "ink_variance_x", "ink_variance_y"]
TOLERANCE = 1e-10


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def article_fold(article):
    return int(hashlib.sha256(article.encode("utf-8")).hexdigest(), 16) % 5


def ridge_predictions(features, targets, folds):
    """Every prediction uses only other articles; normalization is fold-local."""
    predictions = np.full(len(targets), np.nan)
    for fold in range(5):
        train, held = folds != fold, folds == fold
        if not train.any() or not held.any():
            raise ValueError("empty cross-validation fold")
        x, y = features[train], targets[train]
        mean, scale = x.mean(0), x.std(0)
        scale = np.where(scale > 1e-12, scale, 1.)
        z, y_mean = (x-mean)/scale, y.mean()
        coefficients = np.linalg.solve(z.T@z + np.eye(z.shape[1]), z.T@(y-y_mean))
        predictions[held] = (features[held]-mean)/scale@coefficients + y_mean
    if not np.isfinite(predictions).all():
        raise ValueError("nonfinite out-of-fold predictions")
    return predictions


def assignment_margin(predictions, measured):
    correct = (predictions[0]-measured[0])**2 + (predictions[1]-measured[1])**2
    swapped = (predictions[0]-measured[1])**2 + (predictions[1]-measured[0])**2
    margin = float(swapped-correct)
    # Independent expansion guards pair orientation and squared-loss assembly.
    if not np.isclose(margin, 2*(predictions[0]-predictions[1])*(measured[0]-measured[1]),
                      atol=1e-9, rtol=1e-9):
        raise AssertionError("assignment orientation differs")
    return margin


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="artifacts/question_assignment_nuisance_20261105")
    out = Path(parser.parse_args().out)
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("explicit CPU-only CUDA_VISIBLE_DEVICES='' required")
    if (out/"report.json").exists():
        raise FileExistsError("completed diagnostic is immutable")
    torch.set_num_threads(1)
    begun = time.perf_counter()
    if digest(PAIR_ROOT/"report.json") != PAIR_REPORT_SHA:
        raise ValueError("frozen pair report differs")
    pair_report = json.loads((PAIR_ROOT/"report.json").read_text())
    if not pair_report["training_only"] or pair_report["development_examples"] != 0:
        raise ValueError("only training pairs allowed")
    if digest(TRAIN_SOURCE) != pair_report["source_sha256"]:
        raise ValueError("training source differs")
    if digest(PAIR_ROOT/"pairs.jsonl") != pair_report["manifest_sha256"]:
        raise ValueError("pair manifest differs")
    cache = Path(pair_report["image_cache"])
    if digest(cache/"report.json") != pair_report["image_report_sha256"]:
        raise ValueError("native image report differs")
    cache_report = json.loads((cache/"report.json").read_text())
    for name, expected in pair_report["provenance"].items():
        if digest(name) != expected:
            raise ValueError("pair preparation sources differ")
    for name, expected in cache_report["sources"].items():
        if digest(name) != expected:
            raise ValueError("image preparation sources differ")
    pairs = [json.loads(line) for line in (PAIR_ROOT/"pairs.jsonl").read_text().splitlines()]
    source = [json.loads(line) for line in TRAIN_SOURCE.read_text().splitlines()]
    if len(source) != pair_report["records"] or len(pairs) != pair_report["pairs"]:
        raise ValueError("source or pair count differs")
    records = {row["id"]: row for row in source}
    ids = sorted({identifier for pair in pairs for identifier in pair["record_ids"]})
    if len(records) != len(source) or len(ids) != pair_report["paired_records"]:
        raise ValueError("duplicate or missing training questions")
    rows, accessed = [], {}
    for identifier in ids:
        row = records[identifier]
        if row["split"] != "train":
            raise ValueError("non-training row")
        expected_prompt = "资料："+row["context"]+"问题："+row["question"]+"回答："
        if row["prompt"] != expected_prompt:
            raise ValueError("unexpected question image layout")
        relative = "train/"+identifier+".pt"
        expected = cache_report["files_sha256"][relative]
        if digest(cache/relative) != expected:
            raise ValueError("training pixels differ")
        accessed[relative] = expected
        packed = torch.load(cache/relative, map_location="cpu", weights_only=True)
        width = int(packed["width"])
        whites = np.unpackbits(packed["bits"].numpy(), count=32*width).reshape(32, width)
        boundary = int(packed["prompt_blocks"])
        if (packed["id"] != identifier or int(packed["prompt_cells"]) != len(expected_prompt)
                or width % 64 or not whites[:, -128:].all()):
            raise ValueError("invalid physical image metadata or termination")
        start = 32*(len("资料：")+len(row["context"])+len("问题："))
        stop = start+32*len(row["question"])
        ink = 1-whites[:, start:stop]
        if stop > boundary*64 or not ink.size or not ink.any():
            raise ValueError("invalid question pixels")
        y, x = np.nonzero(ink)
        x, y = (x+.5)/ink.shape[1], (y+.5)/ink.shape[0]
        answer_cells = int(packed["answer_cells"])
        if not 1 <= answer_cells <= 16 or boundary*64+answer_cells*32 > width-128:
            raise ValueError("invalid physical answer bounds")
        # Width is a physical target from the packed cache, not an answer ID.
        rows.append(dict(id=identifier, context_hash=row["context_hash"],
            article_hash=hashlib.sha256(row["article_title"].encode()).hexdigest(),
            fold=article_fold(row["article_title"]),
            features=[ink.shape[1]/32, float(ink.mean()), float(x.mean()), float(y.mean()),
                      float(x.var()), float(y.var())],
            answer_width_cells=answer_cells, answer_suffix_blocks=width//64-boundary))
    matrix = np.array([row["features"] for row in rows], dtype=np.float64)
    targets = np.array([row["answer_width_cells"] for row in rows], dtype=np.float64)
    folds = np.array([row["fold"] for row in rows])
    lookup = {row["id"]: i for i, row in enumerate(rows)}
    probes = dict(width_only=ridge_predictions(matrix[:, :1], targets, folds),
                  width_and_ink_moments=ridge_predictions(matrix, targets, folds),
                  constant=np.zeros(len(rows)))
    summaries, scored, strata = {}, [], Counter()
    groups = defaultdict(set)
    for row in rows:
        groups[row["fold"]].add(row["article_hash"])
    for fold in range(5):
        other = set().union(*(groups[f] for f in range(5) if f != fold))
        if groups[fold] & other:
            raise AssertionError("article leaks across folds")
    for pair in pairs:
        indices = [lookup[i] for i in pair["record_ids"]]
        a, b = [rows[i] for i in indices]
        if any(row["context_hash"] != pair["context_hash"] for row in (a, b)):
            raise ValueError("pair paragraph identity differs")
        if a["fold"] != b["fold"] or a["article_hash"] != b["article_hash"]:
            raise ValueError("same-paragraph pair crosses articles/folds")
        equal_answer = a["answer_width_cells"] == b["answer_width_cells"]
        equal_suffix = a["answer_suffix_blocks"] == b["answer_suffix_blocks"]
        strata["equal_answer_width"] += int(equal_answer)
        strata["equal_suffix_width"] += int(equal_suffix)
        strata["equal_question_width"] += int(a["features"][0] == b["features"][0])
        strata["both_equal_answer_and_suffix_width"] += int(equal_answer and equal_suffix)
        result = dict(context_hash=pair["context_hash"], record_ids=pair["record_ids"],
                      fold=a["fold"], equal_answer_width=equal_answer, probes={})
        for name, predictions in probes.items():
            margin = assignment_margin(predictions[indices], targets[indices])
            label = "correct" if margin > TOLERANCE else "wrong" if margin < -TOLERANCE else "tie"
            result["probes"][name] = dict(margin=margin, label=label)
        scored.append(result)
    for name in probes:
        counts = Counter(row["probes"][name]["label"] for row in scored)
        by_context = defaultdict(list)
        for row in scored:
            label = row["probes"][name]["label"]
            by_context[row["context_hash"]].append({"correct": 1., "tie": .5, "wrong": 0.}[label])
        summaries[name] = dict(counts=dict(counts),
            pair_weighted_tie_half_accuracy=(counts["correct"]+.5*counts["tie"])/len(scored),
            context_weighted_tie_half_accuracy=float(np.mean([np.mean(v) for v in by_context.values()])))
    if summaries["constant"]["counts"] != {"tie": len(pairs)}:
        raise AssertionError("constant predictor must tie every assignment")
    out.mkdir(parents=True, exist_ok=True)
    for index, row in enumerate(rows):
        row["out_of_fold_predictions"] = {name: float(p[index]) for name, p in probes.items()}
    for filename, entries in (("questions.jsonl", rows), ("pair_scores.jsonl", scored)):
        (out/filename).write_text("".join(json.dumps(row)+"\n" for row in entries))
    report = dict(training_only=True, development_examples=0, sealed_test_examples=0,
        model_calls=0, gpu_allocations=0, actor_quality_claim=False, posterior_susceptibility_estimated=False,
        active_V63_modified=False, pairs=len(pairs), questions=len(rows),
        contexts=len({row["context_hash"] for row in scored}),
        articles=len({row["article_hash"] for row in rows}),
        feature_names=FEATURES, target="answer physical width in 32-pixel cell units",
        regularization=1., folds=5, fold_assignment="SHA256(article_title) modulo 5",
        fold_question_counts=dict(Counter(int(f) for f in folds)),
        all_articles_isolated=True, tie_tolerance=TOLERANCE,
        strata=dict(strata), probes=summaries,
        pair_weighted_width_only_assignment_ceiling=1-.5*strata["equal_answer_width"]/len(pairs),
        source_sha256={str(TRAIN_SOURCE): digest(TRAIN_SOURCE),
            str(PAIR_ROOT/"report.json"): PAIR_REPORT_SHA,
            str(PAIR_ROOT/"pairs.jsonl"): digest(PAIR_ROOT/"pairs.jsonl"),
            str(cache/"report.json"): digest(cache/"report.json")},
        provenance={name: digest(name) for name in [__file__, PLAN]},
        accessed_files_sha256=accessed,
        output_sha256={name: digest(out/name) for name in ("questions.jsonl", "pair_scores.jsonl")},
        elapsed_seconds=time.perf_counter()-begun,
        limitations="Training-only cross-validation of fixed coarse cues. Not an estimate of model behavior, semantic understanding, Bayesian susceptibility, causal data value or generalization on a language benchmark.")
    (out/"report.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps({k: v for k, v in report.items() if k not in ("accessed_files_sha256",)}, indent=2))


if __name__ == "__main__":
    main()

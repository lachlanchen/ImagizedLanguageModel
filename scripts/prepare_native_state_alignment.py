#!/usr/bin/env python3
"""Prepare bilingual whole-page supervision, excluding assessment articles."""
import argparse
from collections import Counter
import hashlib
import itertools
import json
import os
from pathlib import Path

from ilm.image_edit.baseline import save_json, sha
from ilm.image_edit.data import font_and_coverage, page
from prepare_native_english_transfer import acquire, valid_questions, distinct

SEED = 20269821


def read(path):
    return json.loads(Path(path).read_text())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, default=Path("data/raw/squad_v1.1/dev-v1.1.json"))
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    original = Path("artifacts/image_edit_learning/pages_v2")
    alignment = Path("artifacts/native_visual_alignment/prepared_v1")
    manifest = [json.loads(line) for line in (original / "pairs.jsonl").read_text().splitlines()]
    previous = {r["id"]: r for r in read(alignment / "training.json")}
    if sha(alignment / "training.json") != read(alignment / "preparation.json")["training_sha256"]:
        raise ValueError("complete original training labels changed")
    english_validation = read("artifacts/native_english_transfer/prepared_v2/references.json")
    chinese_validation = read("artifacts/native_chinese_interface/prepared_v1/references.json")
    excluded_titles = {r["article"] for r in english_validation}
    excluded_ids = {r["id"] for r in chinese_validation}
    source = acquire(args.source)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "english").mkdir()
    training, copy_inputs, copy_refs = [], [], []
    for original_row in manifest:
        path = original / original_row["input"]
        if sha(path) != original_row["input_sha256"]:
            raise ValueError("original whole-page pixels changed")
        if original_row["split"] == "train":
            prior = previous[original_row["id"]]
            if prior["id"] in excluded_ids or prior["transcript_sha256"] != original_row["input_geometry"]["text_sha256"]:
                raise ValueError("validation overlap or original transcript mismatch")
            training.append(dict(id=prior["id"], task=prior["task"], language=prior["language"],
                transcript=prior["transcript"], transcript_sha256=prior["transcript_sha256"], expected=prior["expected"],
                article=original_row.get("article"), pair=original_row.get("pair"),
                input=os.path.relpath(path.resolve(), args.output.resolve()), input_sha256=sha(path)))
        elif original_row["task"] == "copy":
            i = len(copy_inputs)
            copy_inputs.append(dict(id=original_row["id"], task="copy", language=original_row["language"],
                input=os.path.relpath(path.resolve(), args.output.resolve()), input_sha256=sha(path),
                seed=20269921 + i, readout_seed=20270021 + i))
            copy_refs.append(dict(id=original_row["id"], task="copy", language=original_row["language"], expected=original_row["reference_text"]))
    if len(training) != 384 or len(copy_inputs) != 32 or set(previous) != {r["id"] for r in training}:
        raise ValueError("complete original train/copy cohorts required")
    candidates = []
    for article in source["data"]:
        if article["title"] in excluded_titles:
            continue
        for paragraph in article["paragraphs"]:
            digest = hashlib.sha256(paragraph["context"].encode()).hexdigest()
            order = hashlib.sha256(f"{SEED}:{article['title']}:{digest}".encode()).hexdigest()
            candidates.append((order, article["title"], digest, paragraph))
    selected, rejects, counts = [], Counter(), Counter()
    english_rows = []
    for _, title, digest, paragraph in sorted(candidates):
        if counts[title] >= 8:
            continue
        chosen = None
        for a, b in itertools.combinations(valid_questions(paragraph), 2):
            if not distinct(a, b):
                continue
            rows = [dict(id="squad_train/" + q["id"], task="qa", language="en", article=title, pair="squad_train/" + digest,
                transcript=f'Passage:\n{paragraph["context"]}\nQuestion: {q["question"]}\nAnswer:',
                expected=q["references"][0]["text"], annotations=q["references"]) for q in (a, b)]
            try:
                font, provenance = font_and_coverage([dict(prompt=r["transcript"], reference=r["expected"]) for r in rows])
                images = [page(r["transcript"], (1024, 1024), font) for r in rows]
                for r in rows:
                    page(r["expected"], (512, 256), font)
            except ValueError as error:
                rejects[str(error)] += 1
                continue
            chosen = rows, images, provenance
            break
        if chosen is None:
            rejects["no_fitting_distinct_pair"] += 1
            continue
        rows, images, provenance = chosen
        i = len(selected)
        for j, (row, (image, geometry)) in enumerate(zip(rows, images)):
            name = f"english/{i:03d}_{j}.png"
            image.save(args.output / name)
            row.update(input=name, input_sha256=sha(args.output / name), transcript_sha256=hashlib.sha256(row["transcript"].encode()).hexdigest(), input_geometry=geometry)
            training.append(row)
            english_rows.append(dict(prompt=row["transcript"], reference=row["expected"]))
        selected.append(dict(article=title, context_sha256=digest, ids=[r["id"] for r in rows]))
        counts[title] += 1
        if len(selected) == 128:
            break
    if len(selected) != 128 or len(training) != 640 or set(counts) & excluded_titles:
        raise ValueError("complete article-disjoint bilingual training cohort unavailable")
    _, global_font = font_and_coverage(english_rows)
    for i, row in enumerate(training):
        row["seed"] = 20270121 + i
    if len({r["id"] for r in training}) != 640:
        raise ValueError("duplicate training IDs")
    for name, value in (("training.json", training), ("copy_inputs.json", copy_inputs), ("copy_references.json", copy_refs), ("english_selected.json", selected)):
        save_json(args.output / name, value)
    save_json(args.output / "preparation.json", dict(examples=640, copies=128, chinese_qa=256, english_qa=256,
        training_sha256=sha(args.output / "training.json"), copy_inputs_sha256=sha(args.output / "copy_inputs.json"),
        copy_references_sha256=sha(args.output / "copy_references.json"), english_selected_sha256=sha(args.output / "english_selected.json"),
        source_sha256=sha(args.source), original_manifest_sha256=sha(original / "pairs.jsonl"),
        original_training_sha256=sha(alignment / "training.json"), english_font=global_font,
        english_article_counts=dict(counts), excluded_english_articles=sorted(excluded_titles), rejection_counts=dict(rejects),
        english_paragraph_pairs=128, seed=SEED, chinese_validation_ids_overlap=0, english_validation_articles_overlap=0,
        source_pixels_resized=False, all_source_pages_complete=True, teacher_outcomes_used_for_selection=False,
        implementation_sha256=sha(__file__), model_calls=0, optimizer_updates=0,
        copyright="CMRC 2018 and SQuAD 1.1: CC-BY-SA-4.0, Cui et al. 2019 and Rajpurkar et al. 2016.",
        general_ilm_goal_complete=False))
    print(json.dumps(dict(examples=len(training), english_pairs=len(selected), english_articles=len(counts), rejected=dict(rejects)), indent=2))


if __name__ == "__main__":
    main()

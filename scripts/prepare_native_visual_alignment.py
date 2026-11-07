#!/usr/bin/env python3
"""Bind complete training-page transcripts to existing authenticated rasters."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path

from ilm.image_edit.baseline import save_json, sha
from ilm.image_edit.paired_data import copy_cases, qa_candidates


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, default=Path("artifacts/image_edit_learning/pages_v2/pairs.jsonl"))
    p.add_argument("--copy-manifest", type=Path, default=Path("artifacts/image_edit_learning/strip_pages/pairs.jsonl"))
    p.add_argument("--cache", type=Path, default=Path("artifacts/native_visual_access/cache_v1"))
    p.add_argument("--source", type=Path, default=Path("data/visual_qa/cmrc2018_short/train.jsonl"))
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    rows = [json.loads(s) for s in args.manifest.read_text().splitlines()]
    strips = {r["id"]: r for r in (json.loads(s) for s in args.copy_manifest.read_text().splitlines())}
    cache_path = args.cache / "manifest.json"
    cache = json.loads(cache_path.read_text())
    if cache["source_manifest_sha256"] != sha(args.manifest) or cache["copy_manifest_sha256"] != sha(args.copy_manifest):
        raise ValueError("original image/cache manifests differ")
    cached = {r["id"]: r for r in cache["records"]}
    source = {r["id"]: r for r in copy_cases()}
    source.update({r["id"]: r for pair in qa_candidates(args.source) for r in pair})
    training, held_out = [], []
    for row in rows:
        original = source[row["id"]]
        text = original["prompt"]
        if hashlib.sha256(text.encode()).hexdigest() != row["input_geometry"]["text_sha256"]:
            raise ValueError("complete transcription differs from original rendered text")
        if original["reference"] != row["reference_text"] or original["split"] != row["split"]:
            raise ValueError("original answer or split changed")
        image_row = strips[row["id"]] if row["task"] == "copy" else row
        folder = args.copy_manifest.parent if row["task"] == "copy" else args.manifest.parent
        image_path = folder / image_row["input"]
        c = cached[row["id"]]
        if sha(image_path) != c["input_sha256"] or c["input_sha256"] != image_row["input_sha256"]:
            raise ValueError("source-image/cache binding differs")
        if sha(args.cache / c["latent"]) != c["latent_sha256"]:
            raise ValueError("cached latent bytes changed")
        record = dict(id=row["id"], task=row["task"], language=row["language"],
            transcript=text, transcript_sha256=hashlib.sha256(text.encode()).hexdigest(),
            expected=row["reference_text"], input_sha256=c["input_sha256"],
            latent=c["latent"], latent_sha256=c["latent_sha256"], seed=c["seed"])
        if row["split"] == "train":
            training.append(record)
        else:
            held_out.append((record, image_path))
    if len(training) != 384 or len(held_out) != 64:
        raise ValueError("complete original train/validation cohorts required")
    if Counter(r["task"] for r in training) != {"copy": 128, "qa": 256}:
        raise ValueError("training task counts differ")
    args.output.mkdir(parents=True, exist_ok=False)
    save_json(args.output / "training.json", training)
    screen = [pair for task in ("copy", "qa") for pair in [x for x in held_out if x[0]["task"] == task][:4]]
    inputs, references = [], []
    for row, path in screen:
        inputs.append(dict(id=row["id"], task=row["task"], input=os.path.relpath(path.resolve(), args.output.resolve()),
            input_sha256=row["input_sha256"], seed=row["seed"]))
        references.append(dict(id=row["id"], task=row["task"], language=row["language"], expected=row["transcript"]))
    save_json(args.output / "transcription_screen_inputs.json", inputs)
    save_json(args.output / "transcription_screen_references.json", references)
    han = lambda c: "\u3400" <= c <= "\u4dbf" or "\u4e00" <= c <= "\u9fff" or "\U00020000" <= c <= "\U0003134f"
    qa = [r for r in training if r["task"] == "qa"]
    counts = {key: Counter(c for r in qa for c in r[key] if han(c)) for key in ("transcript", "expected")}
    save_json(args.output / "preparation.json", dict(training_examples=384, held_out_examples=64,
        cache_manifest_sha256=sha(cache_path), source_manifest_sha256=sha(args.manifest), source_sha256=sha(args.source),
        training_sha256=sha(args.output / "training.json"),
        screen_inputs_sha256=sha(args.output / "transcription_screen_inputs.json"),
        screen_references_sha256=sha(args.output / "transcription_screen_references.json"),
        implementation_sha256=sha(__file__),
        qa_label_coverage={k: dict(unique_han=len(v), occurrences=sum(v.values())) for k, v in counts.items()},
        all_448_image_and_latent_bindings_verified=True, all_448_full_transcript_hashes_match=True,
        new_images=0, new_codec_calls=0, validation_used_for_training=False,
        full_transcription_supervision_is_auxiliary=True, general_ilm_goal_complete=False))
    print("Prepared 384 complete training transcripts and eight input-only diagnostic pages; no model calls.")


if __name__ == "__main__":
    main()

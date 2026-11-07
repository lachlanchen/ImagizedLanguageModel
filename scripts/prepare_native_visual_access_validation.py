#!/usr/bin/env python3
"""Prepare the complete held-out input-only index without loading a model."""
import argparse
import json
import os
from pathlib import Path

from ilm.image_edit.baseline import save_json, sha


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, default=Path("artifacts/image_edit_learning/pages_v2/pairs.jsonl"))
    p.add_argument("--copy-manifest", type=Path, default=Path("artifacts/image_edit_learning/strip_pages/pairs.jsonl"))
    p.add_argument("--output", type=Path, default=Path("artifacts/native_visual_access"))
    args = p.parse_args()
    rows = [json.loads(s) for s in args.manifest.read_text().splitlines()]
    strips = {r["id"]: r for r in (json.loads(s) for s in args.copy_manifest.read_text().splitlines())}
    diagnostic = [r for split in ("train", "validation") for r in
        [x for x in rows if x["task"] == "copy" and x["split"] == split][:4]]
    diagnostic += [r for r in rows if r["task"] == "qa" and r["split"] == "validation"][:4]
    fixed = {r["id"]: 20268521 + i for i, r in enumerate(diagnostic)}
    inputs, references = [], []
    for index, original in enumerate(rows):
        if original["split"] != "validation":
            continue
        is_copy = original["task"] == "copy"
        row = strips[original["id"]] if is_copy else original
        folder = args.copy_manifest.parent if is_copy else args.manifest.parent
        path = folder / row["input"]
        if sha(path) != row["input_sha256"]:
            raise ValueError("prepared source input changed")
        inputs.append(dict(id=row["id"], task=row["task"],
            input=os.path.relpath(path.resolve(), args.output.resolve()),
            input_sha256=row["input_sha256"], seed=fixed.get(row["id"], 20268921 + index)))
        references.append(dict(id=original["id"], task=original["task"], language=original["language"],
            expected=original["reference_text"], pair=original.get("pair"), article=original.get("article")))
    if len(inputs) != 64 or sum(r["task"] == "copy" for r in inputs) != 32:
        raise ValueError("complete 32-copy/32-QA held-out cohort required")
    names = ("full_validation_inputs.json", "full_validation_references.json", "full_validation_preparation.json")
    if any((args.output / name).exists() for name in names):
        raise FileExistsError("preserve the original prepared cohort; use a fresh output directory")
    args.output.mkdir(parents=True, exist_ok=True)
    save_json(args.output / names[0], inputs)
    save_json(args.output / names[1], references)
    save_json(args.output / names[2], dict(cases=64, training_overlap=0, input_only_index=True,
        index_sha256=sha(args.output / names[0]), references_sha256=sha(args.output / names[1]),
        source_manifest_sha256=sha(args.manifest), copy_manifest_sha256=sha(args.copy_manifest),
        implementation_sha256=sha(__file__),
        rule="Use only after the twelve-case screen improves; retain screen cases separately and never pool repeats as independent observations.",
        readout_seed_rule="20268621+i in complete-cohort order; VAE seeds follow the original declared cache rule.",
        model_calls=0, general_ilm_goal_complete=False))
    print("Prepared all 64 held-out inputs and separate references; no model calls.")


if __name__ == "__main__":
    main()

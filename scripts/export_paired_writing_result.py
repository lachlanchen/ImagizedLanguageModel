#!/usr/bin/env python3
"""Retain every fixed paired-writing example and authenticate its raw pixels."""
import argparse
import json
from pathlib import Path
import shutil

import numpy as np
from PIL import Image
import torch

from ilm.image_edit.baseline import save_json, sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root, out = args.run, args.output
    load = lambda name: json.loads((root / name).read_text())
    trained = load("writing_v1/report.json")
    if trained["updates"] != 512 or trained["training_examples"] != 32:
        raise ValueError("the declared completed training run is required")
    journal = [json.loads(s) for s in (root / "writing_v1/updates.jsonl").read_text().splitlines()]
    if ([r["update"] for r in journal] != list(range(1, 513))
            or any(s not in trained["metadata"]["training_ids"] for r in journal for s in r["ids"])
            or not all(g["finite"] for r in journal for g in r["gradients"].values())):
        raise ValueError("training journal or optimization boundary differs")
    before, after = load("before_v2/report.json"), load("after_v1/report.json")
    for key in ("source", "runner_sha256", "inputs_sha256", "condition_sha256", "steps", "guidance", "output_canvas"):
        if before[key] != after[key]:
            raise ValueError("the before/after inference boundary differs: " + key)
    if after["checkpoint_sha256"] != trained["checkpoint_sha256"] or sha(root / "writing_v1/checkpoint.pt") != trained["checkpoint_sha256"]:
        raise ValueError("actual learned state differs")
    out.mkdir(parents=True, exist_ok=False)
    files = {}

    def retain(source, name, expected=None):
        source = Path(source); digest = sha(source)
        if expected is not None and digest != expected:
            raise ValueError("source bytes changed: " + str(source))
        target = out / name; target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        if sha(target) != digest:
            raise ValueError("retained source differs")
        files[name] = digest
        return name

    data = {r["id"]: r for r in (json.loads(s) for s in (root / "pages_v2/pairs.jsonl").read_text().splitlines())}
    cache = load("cache_v1/pairs/manifest.json")["records"]
    for row in cache:
        original = data[row["id"]]
        retain(root / "pages_v2" / original["target"], "references/" + row["file"].replace(".pt", ".png"), row["target_sha256"])
        retain(root / "cache_v1/pairs" / row["target_codec_png"], "codec/" + row["target_codec_png"], row["target_codec_sha256"])
    left, right = load("before_v2/outputs.json"), load("after_v1/outputs.json")
    if len(left) != 8 or [r["id"] for r in left] != [r["id"] for r in right]:
        raise ValueError("complete original fixed eight-case comparison required")
    examples = []
    for i, (a, b) in enumerate(zip(left, right)):
        row = data[a["id"]]
        if a["seed"] != b["seed"] or a["input_sha256"] != b["input_sha256"]:
            raise ValueError("fixed input or sampling seed differs")
        entry = dict(id=row["id"], split=row["split"], language=row["language"], expected=row["reference_text"], seed=a["seed"])
        for key in ("input", "target"):
            entry[key] = retain(root / "pages_v2" / row[key], f"raw/{i:02}_{key}.png", row[key + "_sha256"])
        for route, record, folder in (("before", a, "before_v2"), ("after", b, "after_v1")):
            path = root / folder / record["path"]
            tensor_path = path.parent / "pixels.pt"
            if sha(tensor_path) != record["result"]["pixels_sha256"]:
                raise ValueError("retained postprocessed tensor changed")
            pixels = torch.load(tensor_path, map_location="cpu", weights_only=True)
            array = (pixels[0].permute(1, 2, 0).float().numpy() * 255).round().astype(np.uint8)
            if not np.array_equal(array, np.asarray(Image.open(path).convert("RGB"))):
                raise ValueError("PNG does not match the actual postprocessed decoder pixels")
            entry[route] = retain(path, f"raw/{i:02}_{route}.png", record["sha256"])
        examples.append(entry)
    reports = ("writing_v1/report.json", "writing_v1/updates.jsonl", "before_v2/report.json",
        "after_v1/report.json", "reference_ocr_v1/report.json", "before_ocr_v1/report.json",
        "after_ocr_v1/report.json", "cache_v1/pairs/manifest.json", "copy_coverage.json")
    for name in reports:
        retain(root / name, name.replace("/", "_"))
    for i in range(2):
        retain(root / f"comparison_{i}.png", f"comparison_{i}.png")
    save_json(out / "result.json", dict(examples=examples, files_sha256=files,
        exporter_source_sha256=sha(__file__), complete_fixed_eight_retained=True,
        all_16_answer_pngs_match_raw_postprocessed_tensors=True,
        reference_pages=64, supplied_codec_reconstructions=64, autonomous_answers=16,
        full_64_case_generation_performed=False, language_quality_qualified=False,
        general_ilm_goal_complete=False))
    print("Retained all fixed before/after examples, reference controls and reports; no model weights exported.")


if __name__ == "__main__":
    main()

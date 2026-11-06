#!/usr/bin/env python3
"""Authenticate the complete padding experiment and retain every raw answer."""
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
    load = lambda p: json.loads((root / p).read_text())
    lines = lambda p: [json.loads(s) for s in (root / p).read_text().splitlines() if s]
    square, strip = [load(p) for p in ("writing_v1/report.json", "padding_trial_v1/writing/report.json")]
    journals = [lines(p) for p in ("writing_v1/updates.jsonl", "padding_trial_v1/writing/updates.jsonl")]
    for report, journal in zip((square, strip), journals):
        if (report["updates"] != 512 or report["training_examples"] != 32
                or [r["update"] for r in journal] != list(range(1, 513))
                or any(not g["finite"] for r in journal for g in r["gradients"].values())):
            raise ValueError("completed declared optimization required")
    for key in ("source", "rank", "learning_rate", "updates", "accumulation", "seed", "condition_sha256", "training_ids"):
        if square["metadata"][key] != strip["metadata"][key]:
            raise ValueError("training boundary differs: " + key)
    if square["implementation"]["files"] != strip["implementation"]["files"]:
        raise ValueError("core training code changed")
    for a, b in zip(*journals):
        if a["ids"] != b["ids"] or [o["sigma"] for o in a["observations"]] != [o["sigma"] for o in b["observations"]]:
            raise ValueError("training ID or sigma coupling changed")
    if sha(root / "padding_trial_v1/writing/checkpoint.pt") != strip["checkpoint_sha256"]:
        raise ValueError("learned state changed")
    outputs = load("padding_full_observation/outputs.json")
    expected_ids = [f"copy/{split}/{i:03}" for split in ("train", "validation") for i in range(32)]
    if [r["id"] for r in outputs] != expected_ids:
        raise ValueError("complete ordered cohort required")
    fixed = {r["id"]: r for r in load("padding_trial_v1/after/outputs.json")}
    score = load("padding_full_ocr/report.json")
    if [r["id"] for r in score["rows"]] != expected_ids:
        raise ValueError("score cohort differs")
    summaries = {s: {"cases": 32, "ocr_exact": sum(r["ocr_exact"] for r in score["rows"] if r["split"] == s)} for s in ("train", "validation")}
    if score["summary"] != summaries:
        raise ValueError("OCR summary not reproduced")
    for folder in ("padding_trial_v1/after", "padding_full_remaining"):
        report = load(folder + "/report.json")
        if report["checkpoint_sha256"] != strip["checkpoint_sha256"]:
            raise ValueError("inference state differs")
        if report["condition_sha256"] != strip["metadata"]["condition_sha256"] or report["steps"] != 50 or report["guidance"] != 1 or report["output_canvas"] != [512, 256]:
            raise ValueError("inference boundary differs")
    padding = load("strip_pages/padding.json")
    original = {r["id"]: r for r in lines("pages_v2/pairs.jsonl")}
    data = {r["id"]: r for r in lines("strip_pages/pairs.jsonl")}
    if len(padding["rows"]) != 160:
        raise ValueError("complete padding audit required")
    # Recheck all derivative pixels, independently of the preparation report.
    for record in padding["rows"]:
        a, b = original[record["id"]], data[record["id"]]
        with Image.open(root / "pages_v2" / a["input"]) as source, Image.open(root / "strip_pages" / b["input"]) as cropped:
            restored = Image.new("RGB", source.size, "white"); restored.paste(cropped, (0, 0))
            if cropped.size != (1024, 256) or not np.array_equal(np.asarray(source.convert("RGB")), np.asarray(restored)):
                raise ValueError("source ink lost or moved")
        if a["target_sha256"] != b["target_sha256"]:
            raise ValueError("target changed")
    out.mkdir(parents=True, exist_ok=False)
    files = {}

    def retain(path, name, digest=None):
        actual = sha(path)
        if digest is not None and actual != digest:
            raise ValueError("source bytes changed: " + str(path))
        destination = out / name; destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
        if sha(destination) != actual:
            raise ValueError("copied bytes differ")
        files[name] = actual
        return name

    def answer(record, folder, name):
        path = root / folder / record["path"]
        pixels = path.parent / "pixels.pt"
        if sha(pixels) != record["result"]["pixels_sha256"]:
            raise ValueError("raw postprocessed tensor changed")
        value = torch.load(pixels, map_location="cpu", weights_only=True)
        array = (value[0].permute(1, 2, 0).float().numpy() * 255).round().astype(np.uint8)
        if not np.array_equal(array, np.asarray(Image.open(path).convert("RGB"))):
            raise ValueError("PNG not equal to original decoder pixels")
        return retain(path, name, record["sha256"])

    cases = []
    for i, (output, scored) in enumerate(zip(outputs, score["rows"])):
        row = data[output["id"]]
        seed = fixed[row["id"]]["seed"] if row["id"] in fixed else 20268421 + i
        if output["seed"] != seed or output["input_sha256"] != row["input_sha256"] or output["sha256"] != scored["png_sha256"]:
            raise ValueError("declared seed/input/score differs")
        entry = dict(id=row["id"], split=row["split"], language=row["language"], expected=row["reference_text"], seed=seed, observation=output["observation"])
        for key in ("input", "target"):
            entry[key] = retain(root / "strip_pages" / row[key], f"raw/{i:02}_{key}.png", row[key + "_sha256"])
        entry["answer"] = answer(output, "padding_full_observation", f"raw/{i:02}_answer.png")
        cases.append(entry)
    for i, output in enumerate(load("padding_trial_v1/before/outputs.json")):
        answer(output, "padding_trial_v1/before", f"before/{i:02}_answer.png")
    reports = ("padding_trial_v1/writing/report.json", "padding_trial_v1/writing/updates.jsonl",
        "padding_trial_v1/before/report.json", "padding_trial_v1/after/report.json",
        "padding_trial_v1/before_ocr/report.json", "padding_trial_v1/after_ocr/report.json",
        "padding_trial_v1/feature_comparison.json", "padding_full_remaining/report.json",
        "padding_full_observation/aggregation.json", "padding_full_ocr/report.json",
        "reference_ocr_v1/report.json", "strip_pages/padding.json", "padding_audit.json")
    for name in reports:
        retain(root / name, "reports/" + name.replace("/", "_"))
    for n in range(4):
        name = f"all_cases_{16*n:02}_{16*n+15:02}.png"
        retain(root / "padding_full_observation" / name, name)
    result = dict(cases=cases, summary=summaries, reference_calibration=score["reference_calibration_summary"],
        logical_cases=64, retained_calls=8, additional_calls=56, before_calls=8,
        all_72_answer_pngs_match_postprocessed_tensors=True, all_160_source_pixel_restorations_exact=True,
        ids_and_sigma_match_all_512_updates=True, checkpoint_sha256=strip["checkpoint_sha256"],
        cost={name: {k: report[k] for k in ("elapsed_seconds", "peak_allocated_vram_gib")} for name, report in (("square_train", square), ("strip_train", strip))},
        files_sha256=files, exporter_sha256=sha(__file__), output_repair=False,
        complete_copy_reliability_pass=False, language_quality_qualified=False, general_ilm_goal_complete=False)
    save_json(out / "result.json", result)
    print(json.dumps({k: result[k] for k in ("summary", "logical_cases", "retained_calls", "additional_calls")}, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Authenticate every pilot output and compare to the unchanged starting state."""
import argparse
import json
from pathlib import Path
import shutil

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch

from ilm.image_edit.baseline import save_json, sha


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("run", "manifest", "inputs", "starting", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    args = p.parse_args()
    load = lambda path: json.loads(path.read_text())
    lines = lambda path: [json.loads(s) for s in path.read_text().splitlines() if s]
    run, out = args.run, args.output
    trained = load(run / "pilot/report.json")
    if trained["stage"] != "train" or trained["updates"] != 32:
        raise ValueError("completed original 32-update pilot required")
    metadata = trained["metadata"]
    starting = load(args.starting / "result.json")
    if metadata["initial_sha256"] != starting["recognition_checkpoint_sha256"]:
        raise ValueError("declared unchanged starting state required")
    journal = lines(run / "pilot/updates.jsonl")
    sampler = torch.Generator().manual_seed(20268402)
    seeds = torch.Generator().manual_seed(20268403)
    if len(journal) != 32:
        raise ValueError("complete pilot journal required")
    for i, row in enumerate(journal):
        expected_id = metadata["training_ids"][int(torch.randint(32, (1,), generator=sampler))]
        expected_seed = int(torch.randint(2**31 - 1, (1,), generator=seeds))
        if row["update"] != i + 1 or row["id"] != expected_id or row["noise_seed"] != expected_seed:
            raise ValueError("registered example/noise trajectory differs")
    inference = load(run / "inference/report.json")
    if (inference["checkpoint_sha256"] != trained["checkpoint_sha256"]
            or inference["inputs_sha256"] != sha(args.inputs)
            or inference["condition_sha256"] != metadata["condition_sha256"]
            or inference["steps"] != 50 or inference["guidance"] != 1.
            or inference["output_canvas"] != [512, 256]
            or any(inference[k] for k in ("training_manifest_loaded", "reference_answer_used",
                "question_specific_text_condition", "output_repair"))):
        raise ValueError("autonomous native inference boundary differs")
    outputs = load(run / "inference/outputs.json")
    inputs = load(args.inputs)
    expected = [f"copy/{s}/{i:03}" for s in ("train", "validation") for i in range(32)]
    if [r["id"] for r in outputs] != expected or len(inputs) != 64 or inference["complete_generations"] != 64:
        raise ValueError("complete original ordered 64-case cohort required")
    score = load(run / "output_ocr/report.json")
    calibration_path = run / "reference_ocr/report.json"
    calibration = load(calibration_path)
    if ([r["id"] for r in score["rows"]] != expected or score["manifest_sha256"] != sha(args.manifest)
            or score["reference_calibration_sha256"] != sha(calibration_path)
            or not score["complete_cohort"]):
        raise ValueError("complete unchanged independent evaluator required")
    source_rows = {r["id"]: r for r in lines(args.manifest)}
    if [r["id"] for r in starting["rows"]] != expected:
        raise ValueError("all original starting outputs required")
    out.mkdir(parents=True, exist_ok=False)
    rows = []
    for index, (generated, scored, original, input_row) in enumerate(zip(outputs, score["rows"], starting["rows"], inputs)):
        if (input_row["id"] != generated["id"] or input_row["seed"] != generated["seed"]
                or input_row["input_sha256"] != generated["input_sha256"]
                or original["seed"] != generated["seed"] or original["input_sha256"] != generated["input_sha256"]):
            raise ValueError("starting and final inference seeds/input images differ")
        source = run / "inference" / generated["path"]
        pixels = source.parent / "pixels.pt"
        if (sha(source) != generated["sha256"] or sha(source) != scored["png_sha256"]
                or sha(pixels) != generated["result"]["pixels_sha256"]):
            raise ValueError("actual final-image provenance differs")
        rgb = torch.load(pixels, map_location="cpu", weights_only=True)
        quantized = (rgb[0].permute(1, 2, 0).float().numpy() * 255).round().astype(np.uint8)
        if not np.array_equal(quantized, np.array(Image.open(source).convert("RGB"))):
            raise ValueError("final PNG differs from actual neural output")
        item = source_rows[generated["id"]]
        if original["expected"] != scored["expected"] or scored["expected"] != item["reference_text"]:
            raise ValueError("expected target strings changed")
        for suffix, key, digest in (("reference", "target", item["target_sha256"]), ("input", "input", item["input_sha256"])):
            file = args.manifest.parent / item[key]
            if sha(file) != digest:
                raise ValueError("original page bytes changed")
            shutil.copyfile(file, out / f"{index:03}_{suffix}.png")
        old = args.starting / original["files"]["answer"]
        if sha(old) != original["png_sha256"]:
            raise ValueError("retained starting PNG changed")
        shutil.copyfile(old, out / f"{index:03}_starting.png")
        shutil.copyfile(source, out / f"{index:03}_answer.png")
        rows.append(dict(scored, seed=generated["seed"], starting_ocr_exact=original["ocr_exact"],
            starting_transcript=original["raw_transcript"], starting_png_sha256=sha(old),
            input_sha256=item["input_sha256"], reference_png_sha256=item["target_sha256"],
            neural_tensor_sha256=sha(pixels), png_equals_quantized_neural_tensor=True,
            files={k: f"{index:03}_{k}.png" for k in ("input", "reference", "starting", "answer")}))
    summary = {s: dict(cases=32, ocr_exact=sum(r["ocr_exact"] for r in rows if r["split"] == s))
        for s in ("train", "validation")}
    if summary != score["summary"]:
        raise ValueError("independent OCR summary does not reproduce")
    before = starting["summary"]
    progress = summary["validation"]["ocr_exact"] >= before["validation"]["ocr_exact"] + 3 and summary["train"]["ocr_exact"] >= before["train"]["ocr_exact"]
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 11)
    galleries = []
    for first in range(0, 64, 16):
        canvas = Image.new("RGB", (2048, 4 * 792), "white"); draw = ImageDraw.Draw(canvas)
        for offset in range(16):
            index = first + offset; x, y = (offset % 4) * 512, (offset // 4) * 792
            draw.rectangle((x, y, x + 511, y + 23), fill="#eeeeee")
            draw.text((x + 4, y + 5), f"{expected[index]} | target / starting / after pilot", font=font, fill="black")
            for position, suffix in enumerate(("reference", "starting", "answer")):
                with Image.open(out / f"{index:03}_{suffix}.png") as page:
                    canvas.paste(page.convert("RGB"), (x, y + 24 + position * 256))
        name = f"all_cases_{first:02}_{first + 15:02}.png"; canvas.save(out / name); galleries.append(name)
    reports = {}
    for name, path in (("training_report.json", run / "pilot/report.json"), ("training_updates.jsonl", run / "pilot/updates.jsonl"),
            ("inference_report.json", run / "inference/report.json"), ("output_ocr_report.json", run / "output_ocr/report.json"),
            ("reference_ocr_report.json", calibration_path)):
        shutil.copyfile(path, out / name); reports[name] = sha(path)
    save_json(out / "result.json", dict(rows=rows, summary=summary, starting_summary=before,
        reference_summary=calibration["summary"], complete_outputs=64, galleries=galleries,
        reports=reports, registered_sampling_trajectory_reproduced=True,
        starting_checkpoint_sha256=metadata["initial_sha256"], final_checkpoint_sha256=trained["checkpoint_sha256"],
        exploratory_progress_rule_met=progress, independent_neural_replay=False,
        all_final_pngs_equal_actual_neural_tensors=True, output_repair=False,
        writing_qualification=False, general_ilm_goal_complete=False))
    print(json.dumps(dict(summary=summary, starting=before, progress_rule_met=progress), indent=2))


if __name__ == "__main__":
    main()

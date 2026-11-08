#!/usr/bin/env python3
"""Export the complete controlled writer result, including every raw answer."""
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
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--inputs", type=Path, required=True)
    p.add_argument("--control", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    root, out = args.run, args.output
    load = lambda path: json.loads(path.read_text())
    lines = lambda path: [json.loads(s) for s in path.read_text().splitlines() if s]
    training = load(root / "training/report.json")
    control = load(args.control / "writing/report.json")
    journals = [lines(p) for p in (root / "training/updates.jsonl", args.control / "writing/updates.jsonl")]
    if any(len(j) != 512 for j in journals):
        raise ValueError("both complete 512-update journals required")
    for a, b in zip(*journals):
        if a["update"] != b["update"] or a["ids"] != b["ids"] or a["observations"][0]["sigma"] != b["observations"][0]["sigma"]:
            raise ValueError("control example/sigma trajectory differs")
    if training["metadata"]["source"] != control["source"] or training["metadata"]["training_ids"] != control["metadata"]["training_ids"]:
        raise ValueError("control source or training cohort differs")
    if training["metadata"]["control_journal_sha256"] != sha(args.control / "writing/updates.jsonl"):
        raise ValueError("control journal binding differs")
    admission = load(root / "calibration_r2/report.json")
    if not admission["teacher_admitted"] or training["metadata"]["admission_sha256"] != sha(root / "calibration_r2/report.json"):
        raise ValueError("original admitted reader required")
    generated = load(root / "inference/report.json")
    if generated["checkpoint_sha256"] != training["checkpoint_sha256"] or generated["complete_generations"] != 64:
        raise ValueError("complete fresh inference with the trained checkpoint required")
    if (generated["inputs_sha256"] != sha(args.inputs)
            or generated["condition_sha256"] != training["metadata"]["condition_sha256"]
            or generated["steps"] != 50 or generated["guidance"] != 1.
            or generated["output_canvas"] != [512, 256]
            or any(generated[k] for k in ("training_manifest_loaded", "reference_answer_used",
                "question_specific_text_condition", "output_repair"))):
        raise ValueError("registered autonomous inference boundary differs")
    outputs = load(root / "inference/outputs.json")
    inputs = load(args.inputs)
    if [(r["id"], r["seed"], r["input_sha256"]) for r in outputs] != [
            (r["id"], r["seed"], r["input_sha256"]) for r in inputs]:
        raise ValueError("inference IDs, seeds or input pages differ")
    score = load(root / "output_ocr/report.json")
    expected = [f"copy/{split}/{i:03}" for split in ("train", "validation") for i in range(32)]
    if [r["id"] for r in outputs] != expected or [r["id"] for r in score["rows"]] != expected:
        raise ValueError("complete ordered 64-case observation required")
    if score["manifest_sha256"] != sha(args.manifest):
        raise ValueError("scoring manifest differs")
    manifest = {r["id"]: r for r in lines(args.manifest)}
    references = load(root / "reference_ocr/report.json")
    if score["reference_calibration_sha256"] != sha(root / "reference_ocr/report.json"):
        raise ValueError("reference calibration differs")
    out.mkdir(parents=True, exist_ok=False)
    rows = []
    for index, (r, observed) in enumerate(zip(outputs, score["rows"])):
        row = manifest[r["id"]]
        source = root / "inference" / r["path"]
        tensor = source.parent / "pixels.pt"
        if (sha(source) != r["sha256"] or sha(tensor) != r["result"]["pixels_sha256"]
                or observed["png_sha256"] != sha(source) or r["input_sha256"] != row["input_sha256"]):
            raise ValueError("raw neural answer provenance differs")
        value = torch.load(tensor, map_location="cpu", weights_only=True)
        quantized = (value[0].permute(1, 2, 0).float().numpy() * 255).round().astype(np.uint8)
        if not np.array_equal(quantized, np.array(Image.open(source).convert("RGB"))):
            raise ValueError("PNG differs from the retained actual generated tensor")
        target = args.manifest.parent / row["target"]
        input_path = args.manifest.parent / row["input"]
        if sha(target) != row["target_sha256"] or sha(input_path) != row["input_sha256"]:
            raise ValueError("original page bytes differ")
        for suffix, file in (("input", input_path), ("reference", target), ("answer", source)):
            shutil.copyfile(file, out / f"{index:03}_{suffix}.png")
        rows.append(dict(observed, seed=r["seed"], input_sha256=row["input_sha256"],
            generated_tensor_sha256=sha(tensor), png_equals_quantized_neural_tensor=True,
            files={suffix: f"{index:03}_{suffix}.png" for suffix in ("input", "reference", "answer")}))
    galleries = []
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 11)
    for first in range(0, 64, 16):
        # Four rows of four cases; reference above actual unrepaired output.
        canvas = Image.new("RGB", (512 * 4, 536 * 4), "white")
        draw = ImageDraw.Draw(canvas)
        for position in range(16):
            i = first + position; x, y = (position % 4) * 512, (position // 4) * 536
            draw.rectangle((x, y, x + 511, y + 23), fill="#eeeeee")
            draw.text((x + 4, y + 5), f"{expected[i]} | reference above, generated below", font=font, fill="black")
            for offset, suffix in ((24, "reference"), (280, "answer")):
                with Image.open(out / f"{i:03}_{suffix}.png") as page:
                    canvas.paste(page.convert("RGB"), (x, y + offset))
        name = f"all_cases_{first:02}_{first + 15:02}.png"; canvas.save(out / name); galleries.append(name)
    reports = {}
    for name in ("calibration/report.json", "calibration_r2/report.json", "preflight/report.json",
            "training/report.json", "training/updates.jsonl", "inference/report.json",
            "reference_ocr/report.json", "output_ocr/report.json"):
        file = root / name
        dest = name.replace("/", "_")
        shutil.copyfile(file, out / dest); reports[dest] = sha(file)
    summaries = {split: dict(cases=32, ocr_exact=sum(r["ocr_exact"] for r in rows if r["split"] == split))
        for split in ("train", "validation")}
    if score["summary"] != summaries:
        raise ValueError("score summary does not independently reproduce")
    save_json(out / "result.json", dict(rows=rows, summary=summaries,
        reference_summary=references["summary"], galleries=galleries, reports=reports,
        trajectory_matched=True, unchanged_reference_and_input_pngs=True,
        all_generated_pngs_match_actual_tensors=True, independent_neural_replay=False,
        control_checkpoint_sha256=control["checkpoint_sha256"],
        recognition_checkpoint_sha256=training["checkpoint_sha256"],
        complete_outputs=64, output_repair=False, language_quality_qualified=False,
        general_ilm_goal_complete=False))
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()

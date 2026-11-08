#!/usr/bin/env python3
"""Authenticate the complete old-objective control and compare every raw page."""
import argparse
import json
from pathlib import Path
import shutil

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch

from ilm.image_edit.baseline import ROOT, save_json, sha


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("run", "manifest", "inputs", "pilot", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    args = p.parse_args()
    load = lambda file: json.loads(file.read_text())
    lines = lambda file: [json.loads(line) for line in file.read_text().splitlines() if line]
    run, out = args.run, args.output
    trained = load(run / "training/report.json")
    meta = trained["metadata"]
    pilot = load(args.pilot / "result.json")
    prior_train = load(args.pilot / "training_report.json")
    if (trained["stage"] != "completed" or trained["updates"] != 32
            or meta["pilot_report_sha256"] != sha(args.pilot / "training_report.json")
            or meta["pilot_journal_sha256"] != sha(args.pilot / "training_updates.jsonl")
            or meta["initial_sha256"] != pilot["starting_checkpoint_sha256"]
            or pilot["final_checkpoint_sha256"] != prior_train["checkpoint_sha256"]
            or pilot["complete_outputs"] != 64
            or any(sha(ROOT / path) != digest for path, digest in meta["implementation"].items())):
        raise ValueError("unchanged completed control and pilot required")
    for name, digest in pilot["reports"].items():
        if sha(args.pilot / name) != digest:
            raise ValueError("pilot's retained source report changed: " + name)
    if (meta["matched_compute"] or meta["matched_realized_noise"]
            or not meta["matched_update_exposure"] or not meta["native_autocast_input_encoding"]
            or not meta["retrospectively_specified_control"]):
        raise ValueError("retain the registered comparison limits")
    journal, prior_journal = lines(run / "training/updates.jsonl"), lines(args.pilot / "training_updates.jsonl")
    sampler, seeds = torch.Generator().manual_seed(20268402), torch.Generator().manual_seed(20268403)
    if len(journal) != 32 or len(prior_journal) != 32:
        raise ValueError("complete update journals required")
    for i, (row, prior) in enumerate(zip(journal, prior_journal)):
        expected = dict(update=i + 1,
            id=meta["training_ids"][int(torch.randint(32, (1,), generator=sampler))],
            noise_seed=int(torch.randint(2**31 - 1, (1,), generator=seeds)))
        if any(row[k] != value or prior[k] != value for k, value in expected.items()):
            raise ValueError("the complete exposure/seed-key stream differs")
        if not np.isfinite(row["loss"]) or not np.isfinite(row["gradient_norm"]) or row["gradient_norm"] <= 0:
            raise ValueError("invalid student update")
    if trained["recognition_updates"] != sum(r["recognition_loss"] is not None for r in journal):
        raise ValueError("recognition-active update count differs")
    if len(trained["initial_replays"]) != 2 or not all(r["native_pixels_bit_exact"] for r in trained["initial_replays"]):
        raise ValueError("two original native starting checks required")
    pilot_by_id = {r["id"]: r for r in pilot["rows"]}
    for row in trained["initial_replays"]:
        prior = pilot_by_id[row["id"]]
        folder = run / "training" / ("starting_" + prior["language"])
        if sha(folder / "pixels.pt") != row["pixels_sha256"] or sha(folder / "answer.png") != prior["starting_png_sha256"]:
            raise ValueError("initial replay artifacts differ from retained starting pixels")
    inference = load(run / "inference/report.json")
    prior_inference = load(args.pilot / "inference_report.json")
    if (inference["checkpoint_sha256"] != trained["checkpoint_sha256"]
            or inference["source"] != meta["source"]
            or inference["condition_sha256"] != meta["condition_sha256"]
            or inference["inputs_sha256"] != sha(args.inputs)
            or meta["manifest_sha256"] != sha(args.manifest)
            or inference["runner_sha256"] != prior_inference["runner_sha256"]
            or inference["steps"] != 50 or inference["guidance"] != 1.
            or inference["output_canvas"] != [512, 256]
            or any(inference[k] for k in ("training_manifest_loaded", "reference_answer_used",
                "question_specific_text_condition", "output_repair"))):
        raise ValueError("same native input-only inference required")
    calibration = run / "reference_ocr/report.json"
    scores = load(run / "output_ocr/report.json")
    old_scores = load(args.pilot / "output_ocr_report.json")
    if (sha(calibration) != sha(args.pilot / "reference_ocr_report.json")
            or scores["reference_calibration_sha256"] != sha(calibration)
            or not scores["complete_cohort"]
            or any(scores[k] != old_scores[k] for k in ("engine", "implementation", "manifest_sha256", "cohort"))):
        raise ValueError("same complete independent evaluator and clean calibration required")
    outputs, inputs = load(run / "inference/outputs.json"), load(args.inputs)
    expected = [f"copy/{s}/{i:03}" for s in ("train", "validation") for i in range(32)]
    if (inference["complete_generations"] != 64 or len(inputs) != 64 or any(
            [r["id"] for r in rows] != expected for rows in (outputs, inputs, scores["rows"], pilot["rows"]))):
        raise ValueError("all original ordered cases required")
    interrupted = None
    interruption = run / "identity_interruption.json"
    if interruption.exists():
        interrupted = load(interruption)
        original = run / "inference_interrupted_identity_r1"
        replayed = []
        for name, digest in interrupted["partial_outputs_sha256"].items():
            a, b = original / name, run / "inference" / name
            if sha(a) != digest or a.read_bytes() != b.read_bytes() or not torch.equal(
                    torch.load(a.parent / "pixels.pt", map_location="cpu", weights_only=True),
                    torch.load(b.parent / "pixels.pt", map_location="cpu", weights_only=True)):
                raise ValueError("same-seed restart changed an overlapping actual output")
            replayed.append(name)
        interrupted = dict(reason=interrupted["reason"], source_sha256=sha(interruption),
            same_seed_overlapping_png_and_pt_pixels_exact=True, overlapping_outputs=len(replayed))
    source = {r["id"]: r for r in lines(args.manifest)}
    out.mkdir(parents=True, exist_ok=False)
    rows = []
    for i, (actual, observed, before, input_row) in enumerate(zip(outputs, scores["rows"], pilot["rows"], inputs)):
        item = source[actual["id"]]
        if (actual["seed"] != input_row["seed"] or actual["seed"] != before["seed"]
                or actual["input_sha256"] != input_row["input_sha256"]
                or actual["input_sha256"] != before["input_sha256"]
                or observed["expected"] != item["reference_text"] or observed["expected"] != before["expected"]):
            raise ValueError("input, seed or target identity differs")
        for key in ("input", "target"):
            if sha(args.manifest.parent / item[key]) != item[key + "_sha256"]:
                raise ValueError("original image changed")
        png = run / "inference" / actual["path"]; tensor = png.parent / "pixels.pt"
        if (sha(png) != actual["sha256"] or sha(png) != observed["png_sha256"]
                or sha(tensor) != actual["result"]["pixels_sha256"]):
            raise ValueError("control output provenance differs")
        raw = torch.load(tensor, map_location="cpu", weights_only=True)
        if raw.shape != (1, 3, 256, 512) or not bool(torch.isfinite(raw).all()):
            raise ValueError("actual neural field differs")
        quantized = (raw[0].permute(1, 2, 0).float().numpy() * 255).round().astype(np.uint8)
        if not np.array_equal(quantized, np.array(Image.open(png).convert("RGB"))):
            raise ValueError("control PNG differs from actual neural tensor")
        for suffix, key in (("pilot", "answer"), ("reference", "reference"), ("input", "input"), ("starting", "starting")):
            file = args.pilot / before["files"][key]
            digest = before[{"answer": "png_sha256", "reference": "reference_png_sha256",
                "input": "input_sha256", "starting": "starting_png_sha256"}[key]]
            if sha(file) != digest:
                raise ValueError("retained comparison PNG changed")
        target = out / f"{i:03}_control.png"; shutil.copyfile(png, target)
        rows.append(dict(observed, seed=actual["seed"], input_sha256=item["input_sha256"],
            control_neural_tensor_sha256=sha(tensor), control_png_equals_quantized_tensor=True,
            pilot_ocr_exact=before["ocr_exact"], pilot_transcript=before["raw_transcript"],
            starting_ocr_exact=before["starting_ocr_exact"],
            files={"control": target.name, **{suffix: str(Path("..") / args.pilot.name / before["files"][key])
                for suffix, key in (("pilot", "answer"), ("reference", "reference"), ("starting", "starting"), ("input", "input"))}}))
    summary, comparison, language_summary = {}, {}, {}
    for split in ("train", "validation"):
        selected = [r for r in rows if r["split"] == split]
        summary[split] = dict(cases=len(selected), ocr_exact=sum(r["ocr_exact"] for r in selected))
        comparison[split] = dict(control_exact=sum(r["ocr_exact"] for r in selected),
            pilot_exact=sum(r["pilot_ocr_exact"] for r in selected),
            pilot_only=[r["id"] for r in selected if r["pilot_ocr_exact"] and not r["ocr_exact"]],
            control_only=[r["id"] for r in selected if r["ocr_exact"] and not r["pilot_ocr_exact"]])
        for language in ("zh", "en"):
            group = [r for r in selected if r["language"] == language]
            if len(group) != 16:
                raise ValueError("same balanced language split required")
            language_summary[split + "/" + language] = dict(cases=16,
                control_exact=sum(r["ocr_exact"] for r in group), pilot_exact=sum(r["pilot_ocr_exact"] for r in group))
    if summary != scores["summary"]:
        raise ValueError("independent counts do not reproduce")
    support = comparison["validation"]["pilot_exact"] >= comparison["validation"]["control_exact"] + 3 and comparison["train"]["pilot_exact"] >= comparison["train"]["control_exact"]
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 11)
    galleries = []
    for first in range(0, 64, 16):
        canvas = Image.new("RGB", (2048, 4 * 792), "white"); draw = ImageDraw.Draw(canvas)
        for offset in range(16):
            row = rows[first + offset]; x, y = offset % 4 * 512, offset // 4 * 792
            draw.rectangle((x, y, x + 511, y + 23), fill="#eeeeee")
            draw.text((x + 4, y + 5), f"{row['id']} | target / old-loss control / final-image pilot", font=font, fill="black")
            for position, key in enumerate(("reference", "control", "pilot")):
                with Image.open(out / row["files"][key]) as page:
                    canvas.paste(page.convert("RGB"), (x, y + 24 + position * 256))
        name = f"all_cases_{first:02}_{first + 15:02}.png"; canvas.save(out / name); galleries.append(name)
    reports = {}
    for name, file in (("training_report.json", run / "training/report.json"), ("training_updates.jsonl", run / "training/updates.jsonl"),
            ("inference_report.json", run / "inference/report.json"), ("output_ocr_report.json", run / "output_ocr/report.json"),
            ("reference_ocr_report.json", calibration)):
        shutil.copyfile(file, out / name); reports[name] = sha(file)
    if interrupted is not None:
        shutil.copyfile(interruption, out / "identity_interruption.json")
        reports["identity_interruption.json"] = sha(interruption)
    save_json(out / "result.json", dict(rows=rows, summary=summary, comparison=comparison,
        language_summary=language_summary, complete_outputs=64, reports=reports, galleries=galleries,
        initial_checkpoint_sha256=meta["initial_sha256"], control_checkpoint_sha256=trained["checkpoint_sha256"],
        pilot_checkpoint_sha256=pilot["final_checkpoint_sha256"], pilot_result_sha256=sha(args.pilot / "result.json"),
        same_complete_exposure_and_seed_keys=True, all_control_pngs_equal_actual_neural_tensors=True,
        independent_neural_replay=False, matched_compute=False, matched_realized_noise=False,
        retrospective_pilot_with_new_control=True, objective_bundle_exploratory_support=support,
        interrupted_invocation_and_exact_overlap=interrupted,
        exporter_sha256=sha(__file__),
        output_repair=False, writing_qualification=False, general_ilm_goal_complete=False))
    print(json.dumps(dict(summary=summary, language_summary=language_summary,
        objective_bundle_exploratory_support=support, comparison=comparison), indent=2))


if __name__ == "__main__":
    main()

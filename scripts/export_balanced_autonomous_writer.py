#!/usr/bin/env python3
"""Verify all balanced updates and final neural images before raw quality review."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch

from ilm.image_edit.baseline import ROOT, save_json, sha


def tensor_png(tensor, png, digest, png_digest):
    if sha(tensor) != digest or sha(png) != png_digest:
        raise ValueError("saved actual output changed: " + str(png))
    raw = torch.load(tensor, map_location="cpu", weights_only=True)
    if raw.shape != (1, 3, 256, 512) or not bool(torch.isfinite(raw).all()):
        raise ValueError("invalid actual neural image")
    values = (raw[0].permute(1, 2, 0).float().numpy() * 255).round().astype(np.uint8)
    if not np.array_equal(values, np.array(Image.open(png).convert("RGB"))):
        raise ValueError("PNG differs from actual quantized neural image")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("run", "manifest", "inputs", "pilot", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    args = p.parse_args()
    load = lambda path: json.loads(path.read_text())
    lines = lambda path: [json.loads(s) for s in path.read_text().splitlines() if s]
    run = args.run
    trained = load(run / "training/report.json"); meta = trained["metadata"]
    pilot = load(args.pilot / "result.json")
    if (trained["stage"] != "completed" or trained["updates"] != 128
            or meta["initial_sha256"] != pilot["final_checkpoint_sha256"]
            or not meta["optimizer_restored"] or meta["starting_optimizer_step"] != 32
            or meta["manifest_sha256"] != sha(args.manifest) or meta["inputs_sha256"] != sha(args.inputs)
            or any(sha(ROOT / path) != digest for path, digest in meta["implementation"].items())):
        raise ValueError("unchanged complete balanced continuation required")
    for name, digest in pilot["reports"].items():
        if sha(args.pilot / name) != digest:
            raise ValueError("retained pilot report changed")
    checkpoint = run / "training/checkpoint.pt"
    if sha(checkpoint) != trained["checkpoint_sha256"]:
        raise ValueError("final checkpoint changed")
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if (saved["step"] != 128 or saved["metadata"] != meta
            or not saved["optimizer"]["state"]
            or any(int(v["step"]) != 160 for v in saved["optimizer"]["state"].values())):
        raise ValueError("128 continued optimizer steps and complete final metadata required")
    final_adapter_count = len(saved["adapters"])
    if len(saved["optimizer"]["state"]) != final_adapter_count:
        raise ValueError("optimizer states do not cover every adapter")
    last_entry = saved["last_entry"]; del saved
    order = torch.Generator().manual_seed(20268502); seeds = torch.Generator().manual_seed(20268503)
    expected_schedule = [dict(update=epoch * 32 + j + 1, epoch=epoch + 1, id=meta["training_ids"][index],
        noise_seed=int(torch.randint(2**31 - 1, (1,), generator=seeds)))
        for epoch in range(4) for j, index in enumerate(torch.randperm(32, generator=order).tolist())]
    schedule = load(run / "training/schedule.json"); journal = lines(run / "training/updates.jsonl")
    if (schedule != expected_schedule or len(journal) != 128 or journal[-1] != last_entry
            or set(Counter(r["id"] for r in schedule).values()) != {4}
            or hashlib.sha256(json.dumps(schedule, sort_keys=True).encode()).hexdigest() != meta["schedule_sha256"]):
        raise ValueError("complete four-permutation schedule does not reproduce")
    for row, scheduled in zip(journal, schedule):
        if (any(row[k] != value for k, value in scheduled.items()) or not np.isfinite(row["loss"])
                or not np.isfinite(row["gradient_norm"]) or row["gradient_norm"] <= 0):
            raise ValueError("actual update differs from fixed schedule")
        folder = run / "training" / f"update_{row['update']:03}_before"
        tensor_png(folder / "pixels.pt", folder / "answer.png", row["autonomous_page"]["pixels_sha256"],
            row["autonomous_page"]["png_sha256"])
    first = journal[0]
    if (not first["actual_differentiable_forward_native_pixel_exact"]
            or set(first["trajectory_prediction_gradient_norms"]) != {"0", "24", "49"}
            or not all(v > 0 and np.isfinite(v) for v in first["trajectory_prediction_gradient_norms"].values())
            or not trained["changed_adapter_tensors"]):
        raise ValueError("native equivalence and complete trajectory checks required")
    before_by_id = {r["id"]: r for r in pilot["rows"]}
    replays = load(run / "training/initial_replays.json")
    if [r["id"] for r in replays] != meta["training_ids"][:2] or not all(r["native_pixels_bit_exact"] for r in replays):
        raise ValueError("two actual pilot replays required")
    for row in replays:
        original = before_by_id[row["id"]]
        folder = run / "training" / ("starting_" + original["language"])
        tensor_png(folder / "pixels.pt", folder / "answer.png", row["output"]["pixels_sha256"], original["png_sha256"])
    inference = load(run / "inference/report.json")
    if (inference["checkpoint_sha256"] != trained["checkpoint_sha256"] or inference["source"] != meta["source"]
            or inference["inputs_sha256"] != sha(args.inputs) or inference["condition_sha256"] != meta["condition_sha256"]
            or inference["steps"] != 50 or inference["guidance"] != 1. or inference["output_canvas"] != [512, 256]
            or inference["runner_sha256"] != sha(ROOT / "scripts/infer_paired_image_cohort_v2.py")
            or not inference["inference_source_files_unchanged"]
            or any(inference[k] for k in ("training_manifest_loaded", "reference_answer_used", "question_specific_text_condition", "output_repair"))):
        raise ValueError("complete unassisted native deployment required")
    scores = load(run / "output_ocr/report.json")
    prior_scores = load(args.pilot / "output_ocr_report.json")
    calibration = run / "reference_ocr/report.json"
    if (sha(calibration) != sha(args.pilot / "reference_ocr_report.json")
            or scores["reference_calibration_sha256"] != sha(calibration)
            or not scores["complete_cohort"] or any(scores[k] != prior_scores[k] for k in
                ("engine", "implementation", "manifest_sha256", "cohort"))):
        raise ValueError("unchanged independent evaluator/calibration required")
    expected_ids = [f"copy/{split}/{i:03}" for split in ("train", "validation") for i in range(32)]
    outputs, inputs = load(run / "inference/outputs.json"), load(args.inputs)
    if inference["complete_generations"] != 64 or any([r["id"] for r in a] != expected_ids
            for a in (outputs, inputs, scores["rows"], pilot["rows"])):
        raise ValueError("all 64 original ordered cases required")
    original = {r["id"]: r for r in lines(args.manifest)}
    args.output.mkdir(parents=True, exist_ok=False); rows = []
    for i, (actual, score, before, given) in enumerate(zip(outputs, scores["rows"], pilot["rows"], inputs)):
        item = original[actual["id"]]
        if (actual["seed"] != before["seed"] or actual["seed"] != given["seed"]
                or actual["input_sha256"] != given["input_sha256"] or actual["input_sha256"] != item["input_sha256"]
                or actual["input_sha256"] != before["input_sha256"]
                or score["expected"] != item["reference_text"] or score["expected"] != before["expected"]):
            raise ValueError("example, target or seed differs")
        page = run / "inference" / actual["path"]
        tensor_png(page.parent / "pixels.pt", page, actual["result"]["pixels_sha256"], score["png_sha256"])
        if sha(page) != actual["sha256"]:
            raise ValueError("output index differs from actual neural page")
        files = {}; hashes = {}
        for key, source, digest in (("input", args.manifest.parent / item["input"], item["input_sha256"]),
                ("reference", args.manifest.parent / item["target"], item["target_sha256"]),
                ("starting", args.pilot / before["files"]["answer"], before["png_sha256"]),
                ("answer", page, actual["sha256"])):
            if sha(source) != digest: raise ValueError("raw source page changed")
            target = args.output / f"{i:03}_{key}.png"; shutil.copyfile(source, target)
            files[key] = target.name; hashes[key] = digest
        rows.append(dict(score, seed=actual["seed"], starting_ocr_exact=before["ocr_exact"],
            starting_transcript=before["raw_transcript"], input_sha256=item["input_sha256"],
            neural_tensor_sha256=actual["result"]["pixels_sha256"], png_equals_quantized_neural_tensor=True,
            files=files, files_sha256=hashes))
    summary = {s: dict(cases=32, ocr_exact=sum(r["ocr_exact"] for r in rows if r["split"] == s)) for s in ("train", "validation")}
    if summary != scores["summary"]: raise ValueError("complete score does not reproduce")
    languages = {s + "/" + lang: dict(cases=sum(r["split"] == s and r["language"] == lang for r in rows),
        ocr_exact=sum(r["split"] == s and r["language"] == lang and r["ocr_exact"] for r in rows))
        for s in ("train", "validation") for lang in ("zh", "en")}
    if any(r["cases"] != 16 for r in languages.values()): raise ValueError("balanced cohorts changed")
    gates = dict(held_out_Chinese=languages["validation/zh"]["ocr_exact"] >= 6,
        seen_retained=summary["train"]["ocr_exact"] >= 15,
        held_out_English_retained=languages["validation/en"]["ocr_exact"] >= 10)
    changes = {s: dict(gained=[r["id"] for r in rows if r["split"] == s and r["ocr_exact"] and not r["starting_ocr_exact"]],
        lost=[r["id"] for r in rows if r["split"] == s and not r["ocr_exact"] and r["starting_ocr_exact"]]) for s in ("train", "validation")}
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 11); galleries = []
    for first_index in range(0, 64, 16):
        canvas = Image.new("RGB", (2048, 3168), "white"); draw = ImageDraw.Draw(canvas)
        for offset in range(16):
            row = rows[first_index + offset]; x, y = offset % 4 * 512, offset // 4 * 792
            draw.rectangle((x, y, x + 511, y + 23), fill="#eeeeee")
            draw.text((x + 4, y + 5), row["id"] + " | target / starting pilot / after 128 updates", font=font, fill="black")
            for position, key in enumerate(("reference", "starting", "answer")):
                with Image.open(args.output / row["files"][key]) as image: canvas.paste(image.convert("RGB"), (x, y + 24 + position * 256))
        name = f"all_cases_{first_index:02}_{first_index + 15:02}.png"; canvas.save(args.output / name); galleries.append(name)
    reports = {}
    for name, source in (("training_report.json", run / "training/report.json"), ("training_updates.jsonl", run / "training/updates.jsonl"),
            ("training_schedule.json", run / "training/schedule.json"), ("initial_replays.json", run / "training/initial_replays.json"),
            ("inference_report.json", run / "inference/report.json"), ("output_ocr_report.json", run / "output_ocr/report.json"),
            ("reference_ocr_report.json", calibration)):
        shutil.copyfile(source, args.output / name); reports[name] = sha(source)
    save_json(args.output / "result.json", dict(rows=rows, summary=summary, language_summary=languages,
        starting_summary=pilot["summary"], changes=changes, complete_outputs=64, reports=reports, galleries=galleries,
        registered_schedule_reproduced=True, optimizer_steps_all_160=True, complete_training_pngs_verified=128,
        all_final_pngs_equal_actual_neural_tensors=True, initial_checkpoint_sha256=meta["initial_sha256"],
        final_checkpoint_sha256=trained["checkpoint_sha256"], pilot_result_sha256=sha(args.pilot / "result.json"),
        independent_neural_replay=False, exploratory_gates=gates, exploratory_progress_rule_met=all(gates.values()),
        ocr_writing_floor_met=all(r["ocr_exact"] >= 30 for r in summary.values()),
        raw_review_completed=False, writing_qualification=False, semantic_QA_evaluated=False,
        exporter_sha256=sha(__file__), output_repair=False, general_ilm_goal_complete=False))
    print(json.dumps(dict(summary=summary, language_summary=languages, exploratory_gates=gates), indent=2))


if __name__ == "__main__":
    main()

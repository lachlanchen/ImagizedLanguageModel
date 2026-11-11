#!/usr/bin/env python3
"""Measure historical codec fidelity or evaluate complete historical answers."""

import argparse
from collections import defaultdict
import json
import os
from pathlib import Path

from PIL import Image, ImageDraw
import torch

from ilm.image_edit.baseline import ROOT, save_json, sha
from ilm.visual_lm.visual_cell_data import pack_visual_cells
from scripts.infer_compact_visual_span import read_image
from scripts.review_compact_visual_span import verified_answer


def load(path):
    return json.loads(path.read_text())


def registration(evaluation):
    prep = load(evaluation / "preparation.json")
    manifest = evaluation / "inputs/manifest.json"
    refs = evaluation / "references.json"
    if sha(manifest) != prep["manifest_sha256"] or sha(refs) != prep["references_sha256"]:
        raise ValueError("Historical evaluation input or target registration changed")
    return prep, load(manifest)["inputs"], load(refs)


def authenticated_ink(path, digest, count):
    if sha(path) != digest:
        raise ValueError(f"Historical image changed: {path.name}")
    return read_image(path, count)


def foreground_f1(actual, reference):
    positive = int(actual.sum()) + int(reference.sum())
    return 2 * int((actual & reference).sum()) / positive if positive else 1.0


def summarize(rows, grouping):
    groups = defaultdict(list)
    for row in rows:
        groups[grouping(row)].append(row)
    return {name: {"cases": len(group), "exact": sum(row["exact"] for row in group),
                   "mean_foreground_f1": sum(row["foreground_f1"] for row in group) / len(group),
                   "minimum_foreground_f1": min(row["foreground_f1"] for row in group)}
            for name, group in sorted(groups.items())}


def board(images, output, *, height):
    columns = min(6, len(images))
    rows = (len(images) + columns - 1) // columns
    cell_width, cell_height = 148, height * 2 + 32
    canvas = Image.new("L", (columns * cell_width, rows * cell_height), 255)
    draw = ImageDraw.Draw(canvas)
    for i, (record, source, reconstructed) in enumerate(images):
        row, column = divmod(i, columns)
        x, y = column * cell_width, row * cell_height
        draw.text((x + 4, y + 1), record["label"], fill=0)
        draw.text((x + 4, y + 14), "Source      Round trip", fill=0)
        for j, tiles in enumerate((source, reconstructed)):
            preview = pack_visual_cells(tiles, columns=1, gutter=0)
            canvas.paste(preview.resize((64, height * 2), Image.Resampling.NEAREST), (x + 4 + j * 70, y + 27))
    canvas.save(output)


@torch.inference_mode()
def measure_codec(evaluation, addendum, checkpoint, output):
    from ilm.visual_lm.causal_glyph_flow import load_v34_ema_codec
    from ilm.visual_lm.continuous_glyph_codec import ContinuousGlyphCodec, ContinuousGlyphCodecConfig

    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("Hide CUDA for the bounded codec measurement")
    torch.set_num_threads(2)
    prep, _, refs = registration(evaluation)
    codec = ContinuousGlyphCodec(ContinuousGlyphCodecConfig())
    weights = load_v34_ema_codec(codec, checkpoint)
    weights.pop("path")
    legacy = []
    for row in refs["unique_historical_rasters"]:
        image = authenticated_ink(evaluation / "targets" / row["file"], row["image_sha256"], 1)
        legacy.append((row, image))
    verification = load(addendum / "verification.json")
    if sha(addendum / "report.json") != verification["recovery_report_sha256"]:
        raise ValueError("High-resolution source addendum changed")
    tall = []
    for row in verification["compact_128_pixel_high_rasters"]:
        image = authenticated_ink(addendum / row["file"], row["sha256"], row["tiles_at_32"])
        if image.shape != (4, 1, 32, 32) or int(image.sum()) != row["ink_pixels"]:
            raise ValueError("Tall historical glyph does not retain its original crop")
        tall.append(({"label": row["label"], "stage": "bronze", "source_image_sha256": row["sha256"]}, image))
    actual = torch.cat([image for _, image in legacy + tall])
    restored = (codec.decode(codec.encode(1 - actual.float())) < 0).to(torch.uint8)
    output.mkdir(parents=True, exist_ok=False)
    measured, boards = [], []
    index = 0
    for label, rows, height in (("historical_32", legacy, 32), ("bronze_tall_128", tall, 128)):
        displays = []
        for provenance, source in rows:
            decoded = restored[index:index + len(source)]
            index += len(source)
            measured.append({**provenance, "representation": label, "tiles": len(source),
                             "exact": torch.equal(source, decoded), "foreground_f1": foreground_f1(decoded, source),
                             "source_ink_pixels": int(source.sum()), "decoded_ink_pixels": int(decoded.sum())})
            displays.append((provenance, source, decoded))
        path = output / f"{label}.png"
        board(displays, path, height=height)
        boards.append({"file": path.name, "sha256": sha(path)})
    receipt = {
        "weights": weights, "evaluation_preparation_sha256": sha(evaluation / "preparation.json"),
        "references_sha256": prep["references_sha256"], "addendum_verification_sha256": sha(addendum / "verification.json"),
        "summary": summarize(measured, lambda row: row["representation"] + "/" + row["stage"]),
        "records": measured, "boards": boards, "source_tiles_decoded": len(actual),
        "source_sha256": sha(Path(__file__)), "cuda_initialized": torch.cuda.is_initialized(),
        "language_generation_executed": False,
    }
    save_json(output / "report.json", receipt)
    print(json.dumps(receipt["summary"], indent=2))


def score_answers(evaluation, run, output):
    prep, manifest, refs = registration(evaluation)
    inputs = {row["id"]: row for row in manifest}
    references = {row["id"]: row for row in refs["cases"]}
    request, report = load(run / "request.json"), load(run / "report.json")
    expected = {key: (row["glyphs"], row["image_sha256"]) for key, row in inputs.items()}
    bound = {row["id"]: (row["glyphs"], row["sha256"]) for row in request["inputs"]}
    if (bound != expected or len(request["inputs"]) != len(expected)
            or sha(run / "request.json") != report["request_sha256"]
            or report["reference_inputs"] or report["output_repaired"]):
        raise ValueError("Use the complete registered historical prompts and controls")
    records = report["records"]
    if len(records) != len(expected) or {row["id"] for row in records} != set(expected):
        raise ValueError("A historical response is missing")
    results = []
    pairs = defaultdict(list)
    for row in records:
        key = row["id"]
        case = inputs[key]
        condition = case["condition"]
        if (row["input_glyphs"], row["input_sha256"]) != expected[key]:
            raise ValueError("Historical answer has a different image input")
        ref = references[key.removesuffix("_" + condition)]
        png = verified_answer(run / key, row)
        target = authenticated_ink(evaluation / "targets" / ref["target_file"], ref["target_image_sha256"], 1)
        complete = row["stopped"] and row["emitted_glyphs"] == 1
        answer = read_image(png, 1) if complete else torch.zeros_like(target)
        fidelity = foreground_f1(answer, target)
        result = {"id": key, "kind": ref["kind"], "condition": condition, "stage": ref["source"]["stage"],
                  "complete_single_glyph": complete, "exact": complete and torch.equal(answer, target),
                  "foreground_f1": fidelity, "output_png_sha256": row["image_sha256"]}
        if ref["kind"] == "selection":
            other, = [value for value in ref["stage_choices"] if value["stage"] != ref["source"]["stage"]]
            alternative = authenticated_ink(evaluation / "targets" / other["file"], other["image_sha256"], 1)
            result["target_uniquely_closer"] = complete and fidelity > foreground_f1(answer, alternative)
            pairs[condition, ref["pair"]].append(result)
        results.append(result)
    if any(len(rows) != 2 for rows in pairs.values()):
        raise ValueError("An attested stage question pair is incomplete")
    paired = {condition: sum(all(row["target_uniquely_closer"] for row in group)
                             for (route, _), group in pairs.items() if route == condition)
              for condition in ("normal", "question_blank")}
    output.mkdir(parents=True, exist_ok=False)
    receipt = {"inference_report_sha256": sha(run / "report.json"), "references_sha256": prep["references_sha256"],
               "checkpoint_sha256": request["checkpoint_sha256"], "optimizer_updates": report["optimizer_updates"],
               "summary": summarize(results, lambda row: "/".join((row["kind"], row["condition"], row["stage"]))),
               "paired_both_targets_uniquely_closer": paired, "results": results,
               "modern_readings_semantically_adjudicated": False}
    save_json(output / "report.json", receipt)
    print(json.dumps({"summary": receipt["summary"], "paired": paired}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", type=Path, default=ROOT / "artifacts/historical_visual_evaluation_v1")
    sub = parser.add_subparsers(dest="stage", required=True)
    codec = sub.add_parser("codec")
    codec.add_argument("--addendum", type=Path, default=ROOT / "artifacts/historical_glyph_addendum_v1")
    codec.add_argument("--checkpoint", type=Path, default=ROOT / "artifacts/continuous_glyph_codec_v34_20260814/checkpoint_latest.pt")
    codec.add_argument("--output", type=Path, required=True)
    answers = sub.add_parser("answers")
    answers.add_argument("--run", type=Path, required=True)
    answers.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.stage == "codec":
        measure_codec(args.evaluation, args.addendum, args.checkpoint, args.output)
    else:
        score_answers(args.evaluation, args.run, args.output)

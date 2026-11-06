#!/usr/bin/env python3
"""Export the complete fixed image-generator baseline, never selected samples."""
import argparse
import json
import os
from pathlib import Path
import shutil

import numpy as np
from PIL import Image
import torch

from ilm.image_edit.baseline import ROOT, sha, save_json
from scripts.image_edit_language_baseline import prepared_pages


def decoded_pixels(path, expected_shape):
    raw = torch.load(path, map_location="cpu", weights_only=True)
    if raw.shape != expected_shape or raw.dtype != torch.float32 or not bool(torch.isfinite(raw).all()):
        raise ValueError("owning raw decoded RGB tensor differs")
    # The source decoder emits BF16. Its real postprocess performs arithmetic
    # in that dtype before conversion to float/NumPy; FP32 arithmetic differs.
    rgb = (raw.to(torch.bfloat16) / 2 + .5).clamp(0, 1)[0].permute(1, 2, 0).float().numpy()
    return (rgb * 255).round().astype(np.uint8)


def export(root, destination):
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "" or torch.cuda.is_initialized():
        raise ValueError("CPU-only export required")
    torch.set_num_threads(2)
    data, generated, codec = root / "pages_ready", root / "generations", root / "codec"
    prepared = prepared_pages(data)
    generation = json.loads((generated / "report.json").read_text())
    reconstructions = json.loads((codec / "report.json").read_text())
    calibration = json.loads((root / "ocr_reference_calibration/report.json").read_text())
    observation = json.loads((root / "ocr_output_observation/report.json").read_text())
    if (generation["stage"] != "cohort" or generation["result"]["complete_generations"] != 40
            or generation["local_optimizer_updates"] != 0 or generation["language_quality_claim"]
            or generation["result"]["reference_images_used_by_generator"]
            or generation["result"]["prepared_pages_sha256"] != sha(data / "pages.json")
            or observation["result"]["generation_report_sha256"] != sha(generated / "report.json")
            or observation["result"]["reference_calibration_sha256"] != sha(root / "ocr_reference_calibration/report.json")
            or observation["evaluator_source_sha256"] != sha(ROOT / "scripts/evaluate_image_edit_baseline.py")):
        raise ValueError("complete original autonomous cohort and owning offline evaluation required")
    for report in (generation, reconstructions):
        for name, digest in report["identity"]["sources"].items():
            if sha(ROOT / name) != digest:
                raise ValueError("executed generator/codec source changed")
    temporary = destination.with_name(destination.name + ".partial")
    if destination.exists() or temporary.exists():
        raise FileExistsError("preserve completed and partial exports")
    (temporary / "raw").mkdir(parents=True)
    files = {}

    def copy(source, name, digest=None):
        observed = sha(source)
        if digest is not None and observed != digest:
            raise ValueError("source image identity differs: " + name)
        target = temporary / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        if sha(target) != observed:
            raise ValueError("export changed raw bytes")
        files[name] = observed

    for name, digest in prepared["files_sha256"].items():
        if Path(name).name != name:
            raise ValueError("invalid prepared page filename")
        copy(data / name, "raw/" + name, digest)
    observed = set()
    for row in generation["result"]["generations"]:
        i, route = row["position"], row["route"]
        if (i, route) in observed or not 0 <= i < 20 or route not in ("normal", "blank"):
            raise ValueError("duplicate or invalid generation case")
        observed.add((i, route))
        folder = generated / f"{i:02}_{route}"
        if sha(folder / "generation.json") != row["result_sha256"]:
            raise ValueError("generation receipt changed")
        record = json.loads((folder / "generation.json").read_text())
        source_name = prepared["cases"][i]["files"][route]
        if (record["input_original_sha256"] != prepared["files_sha256"][source_name]
                or record["seed"] != prepared["cases"][i]["seed"] or record["sampling_steps"] != 4
                or record["guidance_scale"] != 1. or record["reference_answer_used"]
                or record["per_question_text_condition"] or record["output_repair"]
                or record["text_encoder_loaded"] or record["tokenizer_loaded"]
                or record["fixed_condition_sha256"] != generation["result"]["fixed_condition_sha256"]):
            raise ValueError("owning input or image-only generation condition differs")
        for name, key in (("input.png", "input_rgb_sha256"), ("answer_rgb.pt", "raw_rgb_sha256"),
                          ("answer_latents.pt", "answer_latents_sha256")):
            if sha(folder / name) != record[key]:
                raise ValueError("retained generator payload changed")
        if not np.array_equal(np.asarray(Image.open(folder / "input.png")), np.asarray(Image.open(data / source_name))):
            raise ValueError("generator received different prompt pixels")
        png = np.asarray(Image.open(folder / "answer.png"))
        if not np.array_equal(png, decoded_pixels(folder / "answer_rgb.pt", (1, 3, 512, 512))):
            raise ValueError("PNG differs from the actual raw BF16 decoder output")
        copy(folder / "answer.png", f"raw/{i:02}_{route}_answer.png", record["answer_png_sha256"])
    if observed != {(i, route) for i in range(20) for route in ("normal", "blank")}:
        raise ValueError("full cohort coverage required")
    codec_rows = []
    for row in reconstructions["result"]["reconstructions"]:
        folder = codec / row["directory"]
        if sha(folder / "reconstruction.json") != row["result_sha256"]:
            raise ValueError("codec receipt changed")
        record = json.loads((folder / "reconstruction.json").read_text())
        shape = (1, 3, record["height"], record["width"])
        if (sha(folder / "decoded_rgb.pt") != record["raw_rgb_sha256"]
                or not np.array_equal(np.asarray(Image.open(folder / "decoded.png")), decoded_pixels(folder / "decoded_rgb.pt", shape))):
            raise ValueError("codec PNG differs from retained raw output")
        name = row["source_file"]
        if record["input_original_sha256"] != prepared["files_sha256"][name]:
            raise ValueError("codec owning page differs")
        original = np.asarray(Image.open(data / name).convert("L")) < 128
        predicted = np.asarray(Image.open(folder / "decoded.png").convert("L")) < 128
        total = int(original.sum()) + int(predicted.sum())
        f1 = 2 * int((original & predicted).sum()) / total if total else 1.
        if f1 != row["foreground_f1"]:
            raise ValueError("independent codec F1 differs")
        copy(folder / "decoded.png", "codec/" + name, record["decoded_png_sha256"])
        codec_rows.append(dict(source_file=name, foreground_f1=f1))
    for start in range(0, 20, 5):
        name = f"all_cases_{start:02}_{start+4:02}.png"
        copy(root / "generation_review" / name, name)
    receipts = {"generation": generated / "report.json", "codec": codec / "report.json",
                "condition": root / "condition/report.json", "reference_ocr": root / "ocr_reference_calibration/report.json",
                "output_ocr": root / "ocr_output_observation/report.json"}
    for name, path in receipts.items():
        copy(path, name + "_report.json")
    result = dict(exporter_source_sha256=sha(__file__), source_run=str(root.relative_to(ROOT)),
                  files_sha256=files, generation_report_sha256=sha(receipts["generation"]),
                  raw_png_bytes_unchanged=True, all_40_generation_pngs_match_raw_rgb=True,
                  all_60_codec_pngs_match_raw_rgb=True, independently_scored_codec_rows=codec_rows,
                  reference_ocr_exact=calibration["result"]["exact"], reference_ocr_cases=20,
                  ocr_summaries=observation["result"]["summaries"], external_pretraining=True,
                  local_optimizer_updates=0, independent_neural_forward_replay=False,
                  language_quality_claim=False, general_ilm_goal_complete=False)
    save_json(temporary / "result.json", result)
    temporary.rename(destination)
    print("Exported every prompt, reference, codec reconstruction and answer; all raw RGB/PNG checks pass.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=ROOT / "artifacts/image_edit_baseline")
    parser.add_argument("--output", type=Path, default=ROOT / "references/evidence/image_edit_baseline_result")
    args = parser.parse_args()
    export(args.run.resolve(), args.output.resolve())

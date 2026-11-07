#!/usr/bin/env python3
"""Prepare and calibrate a fixed, whole-page 432-square reader intervention."""
import argparse
import json
import os
from pathlib import Path
import subprocess

import numpy as np
from PIL import Image

from ilm.image_edit.baseline import save_json, sha
from score_native_visual_access import canonical
from score_native_visual_alignment import edit_distance


def read(path):
    return json.loads(Path(path).read_text())


def recognize(path, expected):
    output = subprocess.run(["tesseract", str(path), "stdout", "-l", "eng", "--psm", "6"],
        env=dict(os.environ, OMP_THREAD_LIMIT="1"), capture_output=True, text=True, check=True)
    actual, target = canonical(output.stdout, "en"), canonical(expected, "en")
    return dict(text=output.stdout, expected=expected, source_sha256=sha(path),
        canonical_exact=actual == target, edit_errors=edit_distance(actual, target), reference_characters=len(target))


def aggregate(records):
    return dict(cases=len(records), exact=sum(r["canonical_exact"] for r in records),
        character_edit_error=sum(r["edit_errors"] for r in records) / sum(r["reference_characters"] for r in records))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", choices=("prepare", "codec"), required=True)
    p.add_argument("--source", type=Path, default=Path("artifacts/native_english_transfer/prepared_v2"))
    p.add_argument("--calibration", type=Path, default=Path("artifacts/native_english_transfer/calibration_v1"))
    p.add_argument("--prepared", type=Path, default=Path("artifacts/native_english_transfer/native432_prepared_v1"))
    p.add_argument("--assets", type=Path, default=Path("artifacts/external/native_unified_source"))
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    source = read(args.source / "preparation.json")
    for name, digest in source["files_sha256"].items():
        if sha(args.source / name) != digest:
            raise ValueError("original source raster changed")
    if sha(args.source / "inputs.json") != source["inputs_sha256"]:
        raise ValueError("original input cohort changed")
    calibration = read(args.calibration / "report.json")
    if calibration["inputs_sha256"] != source["inputs_sha256"] or calibration["records_sha256"] != sha(args.calibration / "records.json"):
        raise ValueError("original transcription calibration changed")
    text = {r["id"]: r["input"]["expected"] for r in read(args.calibration / "records.json")}
    args.output.mkdir(parents=True, exist_ok=False)
    if args.stage == "prepare":
        rows, records, hashes = [], [], {}
        for original in read(args.source / "inputs.json"):
            row = dict(original)
            path = args.output / row["input"]
            path.parent.mkdir(parents=True, exist_ok=True)
            with Image.open(args.source / row["input"]) as image:
                if image.size != (1024, 1024):
                    raise ValueError("whole original square required; no cropping allowed")
                image.convert("RGB").resize((432, 432), Image.Resampling.BICUBIC).save(path)
            row["original_input_sha256"] = row["input_sha256"]
            row["input_sha256"] = sha(path)
            hashes[row["input"]] = sha(path)
            rows.append(row)
            if row["condition"] == "normal":
                records.append(dict(id=row["id"], **recognize(path, text[row["id"]])))
        save_json(args.output / "inputs.json", rows)
        save_json(args.output / "recognition.json", records)
        save_json(args.output / "preparation.json", dict(calls=len(rows), source_inputs_sha256=source["inputs_sha256"],
            inputs_sha256=sha(args.output / "inputs.json"), references_sha256=source["references_sha256"],
            files_sha256=hashes, recognition_sha256=sha(args.output / "recognition.json"), source_ocr=aggregate(records),
            operation="Whole square RGB PIL bicubic resize 1024 to 432; centre crop is identity for square inputs.",
            source_pixels_changed=True, cases_filtered_using_ocr=False, answers_in_model_index=False,
            implementation_sha256=sha(__file__), optimizer_updates=0, goal_complete=False))
        print(json.dumps(aggregate(records), indent=2))
        return
    import torch
    from ilm.native_unified.showo2_source import source_modules, assets, image_latent
    from probe_native_unified_source import gpu_slot
    from types import SimpleNamespace
    prepared = read(args.prepared / "preparation.json")
    if sha(args.prepared / "inputs.json") != prepared["inputs_sha256"]:
        raise ValueError("transformed input index changed")
    for name, digest in prepared["files_sha256"].items():
        if sha(args.prepared / name) != digest:
            raise ValueError("transformed source pixels changed")
    torch.set_num_threads(8)
    rows = [r for r in read(args.prepared / "inputs.json") if r["condition"] == "normal"]
    records = []
    with gpu_slot(), torch.inference_mode():
        native, files = source_modules(args.assets), assets(args.assets)
        vae = native.vae.WanVAE(vae_pth=str(files.vae), dtype=torch.float32, device="cuda:0")
        bundle = SimpleNamespace(device=torch.device("cuda:0"), vae=vae)
        for i, row in enumerate(rows):
            latent, _ = image_latent(bundle, args.prepared / row["input"], row["seed"])
            value = vae.batch_decode(latent[:, :, None]).squeeze(2)[0].float().permute(1, 2, 0).cpu().numpy()
            path = args.output / f"{i:02d}.png"
            Image.fromarray(np.rint((value + 1) * 127.5).clip(0, 255).astype(np.uint8)).save(path)
            records.append(dict(id=row["id"], image=path.name, **recognize(path, text[row["id"]])))
            save_json(args.output / "recognition.json", records)
    save_json(args.output / "report.json", dict(inputs_sha256=prepared["inputs_sha256"], codec_ocr=aggregate(records),
        codec_sha256=sha(files.vae), codec_dtype="float32", native_sampled=True,
        records_sha256=sha(args.output / "recognition.json"), implementation_sha256=sha(__file__),
        model_conditioning_uses_ocr=False, optimizer_updates=0, goal_complete=False))
    print(json.dumps(aggregate(records), indent=2))


if __name__ == "__main__":
    main()

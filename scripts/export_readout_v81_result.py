#!/usr/bin/env python3
"""Publish verified matched results and every unaltered answer image."""
import argparse
import json
import math
from pathlib import Path
import shutil

from PIL import Image, ImageDraw

from scripts.cache_semantic_images_v72 import ROOT, sha, committed_identity
from scripts.audit_native_neural_transcripts_v80 import write


def export(run, trained, destination):
    report = json.loads((run/"report.json").read_text())
    verified = json.loads((run/"verification.json").read_text())
    if (verified["report_sha256"] != sha(run/"report.json")
            or not verified["all_64_outputs_independently_scored"]
            or verified["identity"] != report["identity"]):
        raise ValueError("completed authenticated CPU verification required")
    identity = report["identity"]
    for name, digest in identity["sources"].items():
        if sha(ROOT/name) != digest:
            raise ValueError("evaluation source changed")
    for relative, digest in identity["retained_training_files_sha256"].items():
        if sha(trained/relative) != digest:
            raise ValueError("original failed execution changed")
    if destination.exists():
        raise FileExistsError("preserve previously exported evidence")
    destination.mkdir(parents=True)
    raw = destination/"raw";raw.mkdir()
    files = {}; headers = {}
    for arm in ("control", "adapted"):
        headers[arm] = []
        for i in range(32):
            folder = run/arm/"predictions"/f"{i:02}"
            header = json.loads((folder/"header.json").read_text())
            for name, digest in header["files_sha256"].items():
                if sha(folder/name) != digest:
                    raise ValueError("verified output changed")
                if name.endswith(".png"):
                    target = raw/f"{arm}_{i:02}_{name}"
                    shutil.copyfile(folder/name, target)
                    if sha(target) != digest:
                        raise ValueError("export is not byte-identical")
                    files[str(target.relative_to(destination))] = digest
            headers[arm].append(header)
    # Include every case in its fixed order, without interpolation or image repair.
    plates = []
    for start in range(0, 32, 8):
        panels = []
        for i in range(start, start+8):
            rows = [("REFERENCE", "control", "reference"),
                ("CONTROL: FREE", "control", "answer"),
                ("ADAPTED: FREE", "adapted", "answer"),
                ("CONTROL: CORRECT HISTORY", "control", "assisted"),
                ("ADAPTED: CORRECT HISTORY", "adapted", "assisted"),
                ("SUPPLIED CODE RECONSTRUCTION", "adapted", "supplied")]
            for label, arm, name in rows:
                path = raw/f"{arm}_{i:02}_{name}.png"
                image = Image.open(path).convert("RGB") if path.exists() else Image.new("RGB", (0, 32))
                lines = max(1, math.ceil(image.width/1024))
                panel = Image.new("RGB", (1056, 32+40*lines), "white")
                ImageDraw.Draw(panel).text((12, 8), f"{i:02} {label} | {headers[arm][i]['source']} | {image.width//64} blocks", fill="black")
                for line in range(lines):
                    if image.width:
                        panel.paste(image.crop((line*1024, 0, min(image.width, (line+1)*1024), 32)), (12, 32+line*40))
                panels.append(panel)
        plate = Image.new("RGB", (1056, sum(p.height for p in panels)), "#e8e8e8")
        y = 0
        for panel in panels:
            plate.paste(panel, (0, y));y += panel.height
        path = destination/f"all_cases_{start:02}_{start+7:02}.png";plate.save(path)
        plates.append(dict(path=str(path.relative_to(destination)), sha256=sha(path),
            cases=list(range(start, start+8)), scale=1, image_pixels_repaired=False))
    result = dict(**committed_identity(("scripts/export_readout_v81_result.py",)),
        report_sha256=sha(run/"report.json"), verification_sha256=sha(run/"verification.json"),
        original_failure=json.loads((trained/"failure.json").read_text()), report=report,
        verification=verified, raw_png_sha256=files, all_case_plates=plates,
        source_prediction_headers=headers, raw_png_bytes_unchanged=True,
        training_only=True, language_quality_claim=False, general_ilm_goal_complete=False)
    write(destination/"result.json", result)
    print(json.dumps(dict(summaries=verified["summaries"], gates=verified["gates"],
        training_fit_repair_pass=verified["training_fit_repair_pass"], raw_pngs=len(files),
        all_case_plates=plates), indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=ROOT/"artifacts/adaptive_readout_v81eval_20261107")
    parser.add_argument("--trained", type=Path, default=ROOT/"artifacts/adaptive_readout_v81r2_20261106")
    parser.add_argument("--destination", type=Path, default=ROOT/"references/evidence/adaptive_readout_v81_result")
    args = parser.parse_args();export(args.run.resolve(), args.trained.resolve(), args.destination.resolve())

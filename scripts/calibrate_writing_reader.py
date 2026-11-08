#!/usr/bin/env python3
"""Admit or reject a frozen OCR training loss on the entire existing cohort."""
import argparse
import json
import os
from pathlib import Path
import time

from ilm.image_edit.baseline import ROOT, gpu_zero, save_json, sha


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", choices=("acquire", "calibrate"), required=True)
    p.add_argument("--snapshot", type=Path)
    p.add_argument("--cache", type=Path, default=Path("artifacts/external/writing_reader/hub"))
    p.add_argument("--manifest", type=Path)
    p.add_argument("--output", type=Path)
    args = p.parse_args()
    from ilm.image_edit.recognition import RecognitionTeacher, acquire_teacher
    if args.stage == "acquire":
        print(acquire_teacher(args.cache)); return
    if args.snapshot is None or args.manifest is None or args.output is None:
        raise ValueError("snapshot, manifest and fresh output required")
    import numpy as np
    import torch
    from ilm.image_edit.paired import read_manifest
    from scripts.evaluate_image_edit_baseline import normalize
    all_rows = read_manifest(args.manifest)
    rows = [r for r in all_rows if r["task"] == "copy" and r["split"] == "train"][:32]
    rows += [r for r in all_rows if r["task"] == "copy" and r["split"] == "validation"]
    if len(rows) != 64 or {r["language"] for r in rows} != {"zh", "en"}:
        raise ValueError("full unchanged 64-case copying cohort required")
    paths = ("ilm/image_edit/recognition.py", "scripts/calibrate_writing_reader.py",
        "references/recognition_guided_writing_protocol.md",
        "references/recognition_resize_correction.md")
    implementation = {name: sha(ROOT / name) for name in paths}
    args.output.mkdir(parents=True, exist_ok=False)
    save_json(args.output / "status.json", dict(stage="starting", pid=os.getpid()))
    tick = time.perf_counter()
    try:
        with gpu_zero() as device:
            teacher = RecognitionTeacher(args.snapshot, device)
            observations, gradients, processor_errors = [], [], []
            for i, row in enumerate(rows):
                from PIL import Image
                image = Image.open(row["target_path"]).convert("RGB")
                rgb = torch.from_numpy(np.array(image).copy()).permute(2, 0, 1)[None].to(device).float() / 255
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    clean = float(teacher.loss(rgb, row["reference_text"]))
                    blank = float(teacher.loss(torch.ones_like(rgb), row["reference_text"]))
                    raw = teacher.read(rgb)
                if i < 2:
                    native = teacher.processor(image, return_tensors="pt")["pixel_values"].to(device)
                    error = (native.float() - teacher.pixels(rgb).float()).abs()
                    processor_errors.append(dict(id=row["id"], mean=float(error.mean()), maximum=float(error.max())))
                    rgb.requires_grad_(True); teacher.model.train()
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        loss = teacher.loss(rgb, row["reference_text"])
                    loss.backward()
                    gradient = rgb.grad
                    valid = gradient is not None and bool(torch.isfinite(gradient).all()) and bool(gradient.abs().max() > 0)
                    gradients.append(dict(id=row["id"], finite_nonzero=valid,
                        norm=float(gradient.norm()) if gradient is not None else None,
                        teacher_weights_without_gradients=all(p.grad is None for p in teacher.model.parameters())))
                    teacher.model.eval(); del loss
                result = dict(id=row["id"], language=row["language"], split=row["split"],
                    expected=row["reference_text"], raw_transcript=raw,
                    exact=normalize(raw, row["language"]) == normalize(row["reference_text"], row["language"]),
                    clean_nll=clean, blank_nll=blank, target_png_sha256=row["target_sha256"])
                observations.append(result)
                with (args.output / "readings.jsonl").open("a") as f:
                    f.write(json.dumps(result, ensure_ascii=False) + "\n")
                save_json(args.output / "status.json", dict(stage="calibrating", case=i + 1, budget=64, pid=os.getpid()))
                print(json.dumps(result, ensure_ascii=False), flush=True)
            exact = sum(r["exact"] for r in observations)
            by_language = {lang: sum(r["exact"] for r in observations if r["language"] == lang) for lang in ("zh", "en")}
            gap = sum(r["blank_nll"] - r["clean_nll"] for r in observations) / len(observations)
            gates = dict(clean_reading=exact >= 60, each_language=min(by_language.values()) >= 28,
                image_dependence=gap >= 1., pixel_gradients=all(r["finite_nonzero"] and r["teacher_weights_without_gradients"] for r in gradients),
                native_preprocessing=all(r["mean"] <= .005 and r["maximum"] <= .08 for r in processor_errors),
                memory=torch.cuda.max_memory_allocated() < 22 * 1024**3)
            report = dict(stage="calibrate", source=teacher.source, implementation=implementation,
                manifest_sha256=sha(args.manifest), rows=observations, exact=exact,
                by_language=by_language, mean_blank_minus_clean_nll=gap, gradients=gradients,
                processor_errors=processor_errors, gates=gates, teacher_admitted=all(gates.values()),
                parameters=sum(p.numel() for p in teacher.model.parameters()),
                peak_allocated_vram_gib=torch.cuda.max_memory_allocated() / 1024**3,
                elapsed_seconds=time.perf_counter() - tick, general_ilm_goal_complete=False,
                student_optimized=False)
            if implementation != {name: sha(ROOT / name) for name in paths}:
                raise ValueError("calibration implementation changed while live")
            save_json(args.output / "report.json", report)
            save_json(args.output / "status.json", dict(stage="completed", pid=None))
            print(json.dumps({k: v for k, v in report.items() if k not in ("rows", "source")}, indent=2))
    except BaseException as error:
        save_json(args.output / "status.json", dict(stage="failed", pid=None, reason=str(error)))
        raise


if __name__ == "__main__":
    main()

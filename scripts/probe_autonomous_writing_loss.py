#!/usr/bin/env python3
"""Measure the frozen training reader on every actual autonomous answer.

This is an offline diagnostic, not a replacement independent evaluator or an
image repair. No writer parameters are loaded or optimized.
"""
import argparse
import json
import os
from pathlib import Path
import time

from ilm.image_edit.baseline import ROOT, gpu_zero, save_json, sha


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("teacher", "admission", "generations", "manifest", "journal", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    args = p.parse_args()
    import numpy as np
    from PIL import Image
    import torch
    from ilm.image_edit.paired import read_manifest
    from ilm.image_edit.recognition import RecognitionTeacher, verify_teacher
    from scripts.evaluate_image_edit_baseline import normalize
    admission = json.loads(args.admission.read_text())
    if (not admission["teacher_admitted"] or admission["source"] != verify_teacher(args.teacher)
            or admission["implementation"]["ilm/image_edit/recognition.py"] != sha(ROOT / "ilm/image_edit/recognition.py")
            or admission["manifest_sha256"] != sha(args.manifest)):
        raise ValueError("unchanged complete reader admission required")
    source = json.loads(args.generations.read_text())
    expected = [f"copy/{s}/{i:03}" for s in ("train", "validation") for i in range(32)]
    if [r["id"] for r in source] != expected:
        raise ValueError("all 64 autonomous generations in original order required")
    manifest = {r["id"]: r for r in read_manifest(args.manifest)}
    clean = {r["id"]: r for r in admission["rows"]}
    code = {name: sha(ROOT / name) for name in
        ("scripts/probe_autonomous_writing_loss.py", "ilm/image_edit/recognition.py")}
    args.output.mkdir(parents=True, exist_ok=False)
    save_json(args.output / "status.json", dict(stage="starting", pid=os.getpid()))
    tick = time.perf_counter()
    try:
        with gpu_zero() as device:
            reader = RecognitionTeacher(args.teacher, device)
            rows = []
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                for i, source_row in enumerate(source):
                    png = args.generations.parent / source_row["path"]
                    tensor = png.parent / "pixels.pt"
                    if sha(png) != source_row["sha256"] or sha(tensor) != source_row["result"]["pixels_sha256"]:
                        raise ValueError("saved autonomous pixels changed")
                    rgb = torch.load(tensor, map_location="cpu", weights_only=True)
                    quantized = (rgb[0].permute(1, 2, 0).float().numpy() * 255).round().astype(np.uint8)
                    if not np.array_equal(quantized, np.array(Image.open(png).convert("RGB"))):
                        raise ValueError("visible PNG differs from neural pixel tensor")
                    row = manifest[source_row["id"]]
                    rgb = rgb.to(device)
                    loss = float(reader.loss(rgb, row["reference_text"]))
                    transcript = reader.read(rgb)
                    observed = dict(id=row["id"], split=row["split"], language=row["language"],
                        expected=row["reference_text"], reader_raw=transcript,
                        reader_exact=normalize(transcript, row["language"]) == normalize(row["reference_text"], row["language"]),
                        autonomous_nll=loss, clean_reference_nll=clean[row["id"]]["clean_nll"],
                        png_sha256=source_row["sha256"], neural_tensor_sha256=sha(tensor))
                    rows.append(observed)
                    with (args.output / "readings.jsonl").open("a") as f:
                        f.write(json.dumps(observed, ensure_ascii=False) + "\n")
                    save_json(args.output / "status.json", dict(stage="reading", completed=i + 1, budget=64, pid=os.getpid()))
                    print(json.dumps(observed, ensure_ascii=False), flush=True)
            journal = [json.loads(s) for s in args.journal.read_text().splitlines() if s]
            assisted = [r["observations"][0]["recognition_loss"] for r in journal
                if r["observations"][0]["recognition_loss"] is not None]
            summary = {s: dict(cases=sum(r["split"] == s for r in rows),
                reader_exact=sum(r["split"] == s and r["reader_exact"] for r in rows),
                mean_autonomous_nll=sum(r["autonomous_nll"] for r in rows if r["split"] == s) / 32,
                mean_clean_nll=sum(r["clean_reference_nll"] for r in rows if r["split"] == s) / 32)
                for s in ("train", "validation")}
            report = dict(rows=rows, summary=summary, source=reader.source, implementation=code,
                generations_sha256=sha(args.generations), admission_sha256=sha(args.admission),
                journal_sha256=sha(args.journal), assisted_training_mean_nll=sum(assisted) / len(assisted),
                assisted_updates=len(assisted), elapsed_seconds=time.perf_counter() - tick,
                peak_allocated_vram_gib=torch.cuda.max_memory_allocated() / 1024**3,
                writer_loaded=False, writer_optimized=False, output_repair=False,
                independent_quality_evaluator=False, general_ilm_goal_complete=False)
            if code != {name: sha(ROOT / name) for name in code}:
                raise ValueError("reader diagnostic implementation changed while live")
            save_json(args.output / "report.json", report)
            save_json(args.output / "status.json", dict(stage="completed", pid=None))
            print(json.dumps(summary, indent=2))
    except BaseException as error:
        save_json(args.output / "status.json", dict(stage="failed", pid=None, reason=str(error)))
        raise


if __name__ == "__main__":
    main()

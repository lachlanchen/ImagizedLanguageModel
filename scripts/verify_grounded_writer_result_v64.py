#!/usr/bin/env python3
"""Recompute the frozen writer's complete raw-image audit without a GPU."""

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run",type=Path,default=ROOT/"artifacts/grounded_writer_learning_v64_20261105")
    args = parser.parse_args()
    run = args.run
    report = json.loads((run/"report.json").read_text())
    recovery = json.loads((run/"evaluation_directory_recovery.json").read_text())
    cases = json.loads((run/"development/cases.json").read_text())
    manifest = json.loads((run/"development/manifest.json").read_text())
    if report["update"] != 10000 or len(cases) != 128:
        raise ValueError("completed fixed writer audit required")
    frozen = {name:digest(run/name)==value for name,value in recovery["frozen_files"].items()}
    if not all(frozen.values()) or digest(run/"writer_final.pt") != report["final_ema_sha256"]:
        raise ValueError("recovery changed final weights or training journals")
    for name,value in report["sources"].items():
        if digest(ROOT/name) != value:
            raise ValueError("frozen source changed: "+name)
    metrics = [json.loads(line) for line in (run/"metrics.jsonl").read_text().splitlines()]
    samples = [json.loads(line) for line in (run/"sample_order.jsonl").read_text().splitlines()]
    for journal in (metrics,samples):
        if [row["update"] for row in journal] != list(range(1,10001)):
            raise ValueError("extra/missing optimizer update in journal")
    files = {name:digest(run/name) for name in ("report.json","development/report.json",
        "development/cases.json","development/manifest.json","resume_verification.json",
        "evaluation_directory_recovery.json","initial_evaluation_failure.json")}
    measured = {}
    for route in ("aligned","shuffled"):
        rows = []
        for i,case in enumerate(cases):
            if any(case[key] != manifest[i][key] for key in manifest[i]):
                raise ValueError("case ordering differs from fixed manifest")
            name = f"development/{route}/{i:03}.png"
            reference = f"development/reference/{i:03}.png"
            a = np.asarray(Image.open(run/name).convert("L"))
            b = np.asarray(Image.open(run/reference).convert("L"))
            if a.shape != (32,64) or b.shape != (32,64) or not set(np.unique(a)) <= {0,255}:
                raise ValueError("audit must retain physical binary 32x64 blocks")
            a,b = a<128,b<128
            denominator = int(a.sum()+b.sum())
            scores = dict(foreground_f1=2*int((a&b).sum())/denominator if denominator else 1.,
                exact=bool(np.array_equal(a,b)),blank=not bool(a.any()))
            if scores != case["scores"][route]:
                raise ValueError("raw image score differs: "+name)
            files[name],files[reference] = digest(run/name),digest(run/reference)
            rows.append(scores)
        measured[route] = {key:float(np.mean([row[key] for row in rows])) for key in rows[0]}
        for key,value in measured[route].items():
            if not math.isclose(value,report["component_result"]["measurements"][route][key],abs_tol=1e-12):
                raise ValueError("aggregate image score differs")
    gain = measured["aligned"]["foreground_f1"]-measured["shuffled"]["foreground_f1"]
    gates = dict(foreground_f1=measured["aligned"]["foreground_f1"]>=.90,conditioning_gain=gain>=.20)
    if gates != report["component_result"]["gates"] or report["component_result"]["writing_component_pass"] != all(gates.values()):
        raise ValueError("registered quality decision differs")
    receipt = dict(report_sha256=files["report.json"],files_sha256=files,
        frozen_recovery_files_unchanged=frozen,raw_image_scores_recomputed=256,
        development_examples=128,quality_pass=all(gates.values()),measurements=measured,
        paired_foreground_gain=gain,training_peak_allocated_vram_gib=max(row["peak_allocated_vram_gib"] for row in metrics),
        no_extra_optimizer_updates=True,verifier_sha256=digest(Path(__file__)))
    target = ROOT/"references/evidence/grounded_writer_v64_result.json"
    target.write_text(json.dumps(receipt,indent=2)+"\n")
    print(json.dumps({key:value for key,value in receipt.items() if key!="files_sha256"},indent=2))


if __name__ == "__main__":
    main()

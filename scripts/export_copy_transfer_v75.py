#!/usr/bin/env python3
"""Publish all fixed copy-probe rasters and independently verified scores."""

import json
import os
from pathlib import Path
import shutil

from PIL import Image
import torch

from scripts.cache_semantic_images_v72 import ROOT, sha, atomic_json, committed_identity


def main():
    if os.environ.get("CUDA_VISIBLE_DEVICES")!="" or torch.cuda.is_initialized():
        raise ValueError("export must be CPU-only")
    identity=committed_identity(("scripts/export_copy_transfer_v75.py",
        "scripts/audit_copy_transfer_v75.py","scripts/prepare_copy_transfer_v75.py"))
    run=ROOT/"artifacts/copy_transfer_v75_20261106"
    data=ROOT/"data/general_instructions/copy_transfer_v75_20261106"
    verification=json.loads((run/"verification.json").read_text())
    report=json.loads((run/"report.json").read_text())
    prepared=json.loads((data/"report.json").read_text())
    if (verification["report_sha256"]!=sha(run/"report.json")
            or not verification["all_12_outputs_independently_scored"]
            or report["model_calls"]!=12 or report["identity"]["data_sha256"]!=sha(data/"report.json")):
        raise ValueError("completed verified copy diagnostic required")
    for base,files in ((run,report["files_sha256"]),(data,prepared["files_sha256"])):
        for name,digest in files.items():
            if sha(base/name)!=digest:
                raise ValueError("source raster evidence changed")
    out=ROOT/"references/evidence/copy_transfer_v75_raw"
    out.mkdir(exist_ok=True)
    files={}
    for i,case in enumerate(prepared["cases"]):
        prompt=torch.load(data/case["input_file"],map_location="cpu",weights_only=True)
        path=out/f"{i:02}_prompt.png"
        Image.fromarray(((prompt[0,0]+1)*127.5).round().to(torch.uint8).numpy()).save(path)
        files[str(path.relative_to(ROOT))]=sha(path)
        path=out/f"{i:02}_reference.png"
        shutil.copy2(data/case["reference_file"],path)
        files[str(path.relative_to(ROOT))]=sha(path)
        source=run/"predictions"/f"{i:02}"/"answer.png"
        if source.exists():
            path=out/f"{i:02}_answer.png"
            shutil.copy2(source,path)
            if sha(path)!=sha(source):
                raise ValueError("raw answer copy changed")
            files[str(path.relative_to(ROOT))]=sha(path)
    receipt=dict(export_identity=identity,source_run=str(run.relative_to(ROOT)),
        report_sha256=sha(run/"report.json"),verification_sha256=sha(run/"verification.json"),
        prepared_report_sha256=sha(data/"report.json"),checkpoint_sha256=report["identity"]["checkpoint_sha256"],
        summary=verification["summary"],verification=verification,
        runtime={k:report[k] for k in ("model_calls","elapsed_seconds","setup_seconds","peak_allocated_vram_gib")},
        raw_files_sha256=files,descriptive_diagnostic_only=True,output_pixels_repaired=False,
        language_quality_claim=False,general_ilm_goal_complete=False)
    atomic_json(ROOT/"references/evidence/copy_transfer_v75_result.json",receipt)
    summary=verification["summary"]
    all_cases=summary["groups"]["all"]
    table="\n".join(f"| {name} | {r['exact']}/{r['cases']} | {r['foreground_f1']*100:.3f}% |"
        for name,r in summary["groups"].items())
    text=f'''# Literal writing transfer remains unsuccessful

The completed full-corpus clean checkpoint reproduces
**{all_cases['exact']}/{all_cases['cases']} requested answer images exactly**
on the fixed Chinese, English and six-digit copying requests. Overall
foreground F1 is {all_cases['foreground_f1']*100:.3f}%. Learned stopping
occurs on {all_cases['learned_stopping']}/12 outputs. Both answers are exact
in {summary['both_exact_pairs']}/6 pairs.

| Requested writing | Exact images | Foreground F1 |
| --- | ---: | ---: |
{table}

These are 12 descriptive transfer cases with exact-prompt-disjoint training
images, not a general benchmark or a proof that visual language learning is
impossible. They show that this checkpoint does not reliably preserve writing
requested in new prompts. Mean owning-target minus partner-target F1 is
{all_cases['own_minus_partner_f1']*100:.3f} points. This comparison is offline
cross scoring, not an additional set of model calls.

Inference receives only the actual rendered prompt image. References are
loaded for scoring after all model calls; no target length, code, OCR,
character identifier or teacher service enters generation. Independent CPU
verification authenticates all 12 raw outputs, reconstructs source layout,
checks stopping events and recomputes physical scores. It does not replay
the entire neural generation or every feedback encoding.

Every prepared input, reference and nonempty raw output PNG is copied under
evidence/copy_transfer_v75_raw/. No output is repaired, cropped or selected
for appearance. The [receipt](evidence/copy_transfer_v75_result.json) binds
every exported file and the [protocol](copy_transfer_v75_protocol.md).
The full original 192-question audit is reported separately. The general
usable ILM goal remains unachieved.
'''
    (ROOT/"references/copy_transfer_v75_result.md").write_text(text)
    print(json.dumps(summary,indent=2))


if __name__=="__main__":
    main()

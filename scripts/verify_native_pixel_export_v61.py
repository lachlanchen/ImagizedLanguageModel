#!/usr/bin/env python3
"""Replay a completed native model from a local bundle and empty HF caches."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import torch

from ilm.visual_lm.visual_cell_data import pack_visual_cells
from ilm.visual_lm.visual_qa_data import VisualQARenderer, read_questions
from scripts.prepare_native_pixel_qa_v61 import sha256
from scripts.train_canonical_glyph_binding_v43 import _atomic_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run",default="artifacts/native_pixel_instruction_v61_20261105")
    a=p.parse_args()
    run=Path(a.run).resolve();out=run/"standalone"
    if (out/"verification.json").exists():
        raise FileExistsError("completed native replay receipt is immutable")
    checkpoint=run/"training/model_final.pt"
    report=json.loads((run/"training/training_report.json").read_text())
    if report["smoke_only"] or report["update"]!=1000:
        raise ValueError("only the completed production model qualifies")
    out.mkdir(parents=True,exist_ok=True)
    fixed=json.loads(Path("artifacts/question_first_reading_v60_20261105/development/report.json").read_text())["primary_ids"][0]
    record=next(r for r in read_questions("data/visual_qa/cmrc2018_short/development.jsonl") if r["id"]==fixed)
    prompt=VisualQARenderer()(record["prompt"])
    pack_visual_cells(prompt,columns=len(prompt),gutter=0).save(out/"prompt.png")
    base=[sys.executable,"scripts/infer_native_pixel_language.py"]
    common=["--prompt-image",str(out/"prompt.png")]
    bundle=out/"model_bundle"
    with (out/"original_inference.log").open("w") as log:
        subprocess.run(base+["--checkpoint",str(checkpoint),*common,"--out",str(out/"answer.png"),
                             "--export",str(bundle)],check=True,stdout=log,stderr=subprocess.STDOUT)
    # A new process gets empty source/model caches and no network. Only the
    # self-contained learned bundle may supply weights or author model code.
    env=dict(os.environ,HF_HUB_OFFLINE="1",TRANSFORMERS_OFFLINE="1",
             HF_HUB_CACHE=str(out/"empty_hub"),HF_MODULES_CACHE=str(out/"isolated_modules"))
    with (out/"bundle_replay.log").open("w") as log:
        subprocess.run(base+["--checkpoint",str(bundle),"--snapshot","/source-snapshot-not-required",
                             *common,"--out",str(out/"replay.png")],env=env,check=True,
                       stdout=log,stderr=subprocess.STDOUT)
    original=torch.load(out/"answer.pt",map_location="cpu",weights_only=False)["images"]
    replay=torch.load(out/"replay.pt",map_location="cpu",weights_only=False)["images"]
    first=json.loads((out/"answer.json").read_text());second=json.loads((out/"replay.json").read_text())
    manifest=json.loads((bundle/"deployment_manifest.json").read_text())
    checks=dict(tensor=torch.equal(original,replay),png=(out/"answer.png").read_bytes()==(out/"replay.png").read_bytes(),
        events=first["events"]==second["events"],empty_cache_offline_replay=True,
        bundled_model_code=(bundle/"modeling_llama_pixar.py").exists(),
        bundle_hashes=all(sha256(bundle/name)==digest for name,digest in manifest["files_sha256"].items()))
    if not all(checks.values()):
        raise AssertionError(checks)
    receipt=dict(checkpoint_sha256=sha256(checkpoint),checks=checks,prompt_id=fixed,
        generated_cells=len(original),qualitative_accuracy_not_scored=True,
        source_sha256={name:sha256(name) for name in [__file__,"scripts/infer_native_pixel_language.py",
                      "ilm/visual_lm/native_pixel_runtime.py","ilm/visual_lm/native_pixel_instruction.py"]})
    _atomic_json(receipt,out/"verification.json")
    print(json.dumps(receipt,indent=2),flush=True)


if __name__=="__main__":
    main()

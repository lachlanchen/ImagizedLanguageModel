#!/usr/bin/env python3
"""Replay both completed learned actors from self-contained offline bundles."""

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
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run",default="artifacts/correlated_native_density_v62_20261105")
    args=parser.parse_args();run=Path(args.run).resolve()
    # Complete the matched experiment before generating development answers.
    for k in (1,8):
        report=json.loads((run/f"k{k}/training_report.json").read_text())
        if report["smoke_only"] or report["update"]!=1000 or report["components"]!=k:
            raise ValueError("both production arms must finish first")
    fixed=json.loads(Path("artifacts/question_first_reading_v60_20261105/development/report.json").read_text())["primary_ids"][0]
    record=next(r for r in read_questions("data/visual_qa/cmrc2018_short/development.jsonl") if r["id"]==fixed)
    for k in (1,8):
        out=run/f"standalone/k{k}";out.mkdir(parents=True,exist_ok=True)
        if (out/"verification.json").exists():raise FileExistsError("completed replay is immutable")
        checkpoint=run/f"k{k}/model_final.pt";bundle=out/"model_bundle"
        pack_visual_cells(VisualQARenderer()(record["prompt"]),columns=len(record["prompt"]),gutter=0).save(out/"prompt.png")
        with (out/"original.log").open("w") as log:
            subprocess.run([sys.executable,"scripts/infer_correlated_native_language.py",
                "--checkpoint",str(checkpoint),"--prompt-image",str(out/"prompt.png"),
                "--out",str(out/"answer.png"),"--export",str(bundle)],check=True,
                stdout=log,stderr=subprocess.STDOUT)
        empty=out/"empty_working_directory";empty.mkdir(exist_ok=True)
        env=dict(os.environ,PYTHONPATH="",HF_HUB_OFFLINE="1",TRANSFORMERS_OFFLINE="1",
                 HF_HUB_CACHE=str(out/"empty_hub"),HF_MODULES_CACHE=str(out/"isolated_modules"))
        with (out/"offline.log").open("w") as log:
            subprocess.run([sys.executable,str(bundle/"run.py"),"--prompt-image",str(out/"prompt.png"),
                "--out",str(out/"replay.png")],env=env,cwd=empty,check=True,stdout=log,stderr=subprocess.STDOUT)
        original=torch.load(out/"answer.pt",map_location="cpu",weights_only=False)["images"]
        replay=torch.load(out/"replay.pt",map_location="cpu",weights_only=False)["images"]
        first=json.loads((out/"answer.json").read_text());second=json.loads((out/"replay.json").read_text())
        manifest=json.loads((bundle/"deployment_manifest.json").read_text())
        checks=dict(tensor=torch.equal(original,replay),png=(out/"answer.png").read_bytes()==(out/"replay.png").read_bytes(),
            events=first["events"]==second["events"],empty_cache_offline=True,empty_repository_path=True,
            bundled_code=all((bundle/path).exists() for path in
                ("actor/correlated_native_runtime.py","actor/correlated_native_density.py","architecture/modeling_llama_pixar.py")),
            bundle_hashes=all(sha256(bundle/name)==digest for name,digest in manifest["files_sha256"].items()))
        if not all(checks.values()):raise AssertionError(checks)
        receipt=dict(checkpoint_sha256=sha256(checkpoint),checks=checks,components=k,prompt_id=fixed,
            generated_cells=len(original),qualitative_accuracy_not_scored=True,
            source_sha256={name:sha256(name) for name in [__file__,"scripts/infer_correlated_native_language.py",
                "scripts/run_correlated_native_bundle.py","ilm/visual_lm/correlated_native_runtime.py"]})
        _atomic_json(receipt,out/"verification.json");print(json.dumps(receipt),flush=True)


if __name__=="__main__":main()

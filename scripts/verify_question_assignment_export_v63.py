#!/usr/bin/env python3
"""Replay both completed V63 image actors in fresh standalone offline processes."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import torch

from ilm.visual_lm.visual_cell_data import pack_visual_cells
from ilm.visual_lm.visual_qa_data import VisualQARenderer,read_questions
from scripts.prepare_native_pixel_qa_v61 import sha256
from scripts.train_canonical_glyph_binding_v43 import _atomic_json


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run",default="artifacts/question_assignment_v63_20261105")
    run=Path(parser.parse_args().run).resolve()
    for arm in ("control","aligned"):
        report=json.loads((run/arm/"training_report.json").read_text())
        if report["update"]!=2000 or report["smoke_only"] or report["arm"]!=arm:raise ValueError("both arms must finish")
    fixed=json.loads(Path("artifacts/correlated_native_density_v62_20261105/development/report.json").read_text())["primary_ids"][0]
    record=next(r for r in read_questions("data/visual_qa/cmrc2018_short/development.jsonl") if r["id"]==fixed)
    for arm in ("control","aligned"):
        out=run/"standalone"/arm;out.mkdir(parents=True,exist_ok=True)
        checkpoint=run/arm/"model_final.pt";receipt=out/"verification.json"
        if receipt.exists():
            prior=json.loads(receipt.read_text())
            if prior["checkpoint_sha256"]!=sha256(checkpoint) or not all(prior["checks"].values()):raise ValueError("completed replay differs")
            continue
        bundle=out/"model_bundle"
        pack_visual_cells(VisualQARenderer()(record["prompt"]),columns=len(record["prompt"]),gutter=0).save(out/"prompt.png")
        command=[sys.executable,"scripts/infer_correlated_native_language.py","--checkpoint",str(checkpoint),
            "--prompt-image",str(out/"prompt.png"),"--out",str(out/"answer.png")]
        if not bundle.exists():command.extend(["--export",str(bundle)])
        with (out/"original.log").open("w") as log:subprocess.run(command,check=True,stdout=log,stderr=subprocess.STDOUT)
        manifest=json.loads((bundle/"deployment_manifest.json").read_text())
        if manifest["source_checkpoint_sha256"]!=sha256(checkpoint) or not all(sha256(bundle/p)==h for p,h in manifest["files_sha256"].items()):
            raise ValueError("bundle identity differs; preserve partial evidence for inspection")
        empty=out/"empty_working_directory";empty.mkdir(exist_ok=True)
        env=dict(os.environ,PYTHONPATH="",HF_HUB_OFFLINE="1",TRANSFORMERS_OFFLINE="1",
            HF_HUB_CACHE=str(out/"empty_hub"),HF_MODULES_CACHE=str(out/"isolated_modules"))
        with (out/"offline.log").open("w") as log:
            subprocess.run([sys.executable,str(bundle/"run.py"),"--prompt-image",str(out/"prompt.png"),
                "--out",str(out/"replay.png")],env=env,cwd=empty,check=True,stdout=log,stderr=subprocess.STDOUT)
        a=torch.load(out/"answer.pt",map_location="cpu",weights_only=False)["images"]
        b=torch.load(out/"replay.pt",map_location="cpu",weights_only=False)["images"]
        checks=dict(tensor=torch.equal(a,b),png=(out/"answer.png").read_bytes()==(out/"replay.png").read_bytes(),
            events=json.loads((out/"answer.json").read_text())["events"]==json.loads((out/"replay.json").read_text())["events"],
            empty_cache_offline=True,empty_repository_path=True,bundle_hashes=True)
        if not all(checks.values()):raise AssertionError(checks)
        _atomic_json(dict(arm=arm,checkpoint_sha256=sha256(checkpoint),checks=checks,prompt_id=fixed,
            generated_cells=len(a),qualitative_accuracy_not_scored=True,verifier_sha256=sha256(__file__)),receipt)
        print(json.dumps(dict(arm=arm,checks=checks)),flush=True)


if __name__=="__main__":main()

#!/usr/bin/env python3
"""Exercise real completed-model CLI calls and typed/image input equivalence."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch

from ilm.visual_lm.native_prompt_interface import validate_native_answer
from scripts.audit_native_chinese_v80 import training_state
from scripts.cache_semantic_images_v72 import ROOT, atomic_json, committed_identity, sha
from scripts.prepare_copy_transfer_v75 import CASES
from scripts.verify_codec_language_v68 import physical_scores

COPY = ROOT/"data/general_instructions/copy_transfer_v75_20261106"
COPY_SHA = "71fdda2aa89cd176965a8bdb5900facce8ead2ad49bbc5bc22f544e75eebda7d"
SOURCES = ("scripts/check_completed_native_prompt_routes_v80.py", "scripts/infer_native_chinese_v80.py",
    "ilm/visual_lm/native_prompt_interface.py", "ilm/visual_lm/codec_image_interface.py",
    "ilm/visual_lm/full_text_renderer.py", "ilm/visual_lm/native_image_actor.py",
    "ilm/visual_lm/native_image_continuation.py", "scripts/prepare_copy_transfer_v75.py")


def run(run_dir):
    # Final benchmark and independent pixel scoring must finish first. No model
    # is trained or selected by these additional interface calls.
    if not (run_dir/"verification.json").is_file():
        raise ValueError("finish the original full output audit before standalone interface checks")
    verified = json.loads((run_dir/"verification.json").read_text())
    if not verified["offline_checks_pass"]:
        raise ValueError("original raw-output verification is incomplete")
    training, checkpoint = training_state(run_dir)
    if sha(COPY/"report.json") != COPY_SHA:
        raise ValueError("the existing copy-prompt fixture changed")
    fixture = json.loads((COPY/"report.json").read_text())
    strip = COPY/"strips/00.png"
    if sha(strip) != fixture["files_sha256"]["strips/00.png"]:
        raise ValueError("actual copy input changed")
    out = run_dir/"standalone_inference"; out.mkdir(exist_ok=True)
    identity = dict(**committed_identity(SOURCES), checkpoint_sha256=sha(checkpoint),
        original_verification_sha256=sha(run_dir/"verification.json"), copy_fixture_sha256=COPY_SHA,
        seed=20268005, maximum_blocks=128, same_image_and_text_prompt=True,
        mixed_instruction="请执行下方图片中的要求。", optimizer_updates=0, quality_selection=False)
    if (out/"identity.json").exists():
        previous = json.loads((out/"identity.json").read_text())
        if {k:v for k,v in previous.items() if k!="source_commit"} != {k:v for k,v in identity.items() if k!="source_commit"}:
            raise ValueError("standalone checkpoint or interface changed")
        identity = previous
    else:
        atomic_json(out/"identity.json", identity)
    routes = dict(image=["--prompt",str(strip),"--layout","strip"], text=["--text",CASES[0][1]],
        mixed=["--prompt",str(strip),"--layout","strip","--instruction",identity["mixed_instruction"]])
    results, reports = {}, {}
    for route, arguments in routes.items():
        destination = out/route
        if not (destination/"report.json").exists():
            if destination.exists():
                raise ValueError("partial standalone output exists; preserve it and investigate before retrying")
            command = [sys.executable,"scripts/infer_native_chinese_v80.py",*arguments,
                "--checkpoint",str(checkpoint),"--output",str(destination),
                "--seed",str(identity["seed"]),"--maximum-blocks","128"]
            with (out/(route+".log")).open("w") as log:
                subprocess.run(command,cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES="0"),
                    stdout=log,stderr=subprocess.STDOUT,check=True)
        report = json.loads((destination/"report.json").read_text())
        if (not report["generation_performed"] or report["checkpoint_sha256"]!=identity["checkpoint_sha256"]
                or report["output"]["raw_sha256"]!=sha(destination/"output/answer.pt")
                or report["seed"]!=identity["seed"] or report["maximum_blocks"]!=128):
            raise ValueError("actual standalone model output differs from its receipt")
        result = torch.load(destination/"output/answer.pt",map_location="cpu",weights_only=True)
        validate_native_answer(result,128)
        results[route],reports[route] = result,report
    a,b = results["image"],results["text"]
    exact = (reports["image"]["input"]["actor_pixels_sha256"]==reports["text"]["input"]["actor_pixels_sha256"]
        and torch.equal(a["images"],b["images"]) and a["stop_probabilities"]==b["stop_probabilities"]
        and a["feedback_sha256"]==b["feedback_sha256"] and a["reader_memory_sha256"]==b["reader_memory_sha256"])
    # References are consulted only after all three autonomous CLI processes
    # have exited. They never enter the actor's input or output path.
    reference = COPY/"references/00.png"
    if sha(reference)!=fixture["files_sha256"]["references/00.png"]:
        raise ValueError("owning copy reference changed")
    from PIL import Image
    target = torch.from_numpy(np.asarray(Image.open(reference)).copy())
    scores = {}
    for route,result in results.items():
        images = result["images"]
        raster = np.concatenate([v[0].numpy() for v in images],axis=1) if len(images) else np.empty((32,0))
        scores[route] = physical_scores(raster<0,[target],result["stopped"])
    atomic_json(out/"verification.json", dict(identity=identity, actual_completed_checkpoint_calls=3,
        typed_image_input_and_generation_bit_exact=exact, copy_reference_scores=scores,
        whole_forward_independently_implemented=False, generalization_claim=False, language_quality_claim=False,
        full_benchmark_quality_gates=verified["gates"], general_ilm_goal_complete=False))
    if not exact:
        raise AssertionError("typed and image representations of identical pixels changed actual model generation")
    print(json.dumps(dict(actual_cli_calls=3,typed_image_equivalence=exact,copy_scores=scores),indent=2),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run",type=Path,required=True)
    run(parser.parse_args().run.resolve())

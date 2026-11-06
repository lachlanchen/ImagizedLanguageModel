#!/usr/bin/env python3
"""Prepare one fixed owning prompt and verify a later standalone actor replay."""
import argparse
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from ilm.visual_lm.codec_image_interface import prepare_image_prompt
from scripts.audit_native_neural_transcripts_v80 import write
from scripts.cache_semantic_images_v72 import ROOT, sha, pixels_sha, committed_identity
from scripts.probe_native_corpus_fit_v80 import cases_and_data, prompt
from scripts.score_native_training_samples_v80 import check_answer

PLAN = "references/decoder_metric_v82_runtime_replay_protocol.md"
SOURCES = ("scripts/check_metric_runtime_replay_v82.py",PLAN,
    "scripts/infer_decoder_metric_v82.py","ilm/visual_lm/metric_image_actor.py",
    "ilm/visual_lm/codec_image_interface.py","ilm/visual_lm/native_prompt_interface.py")


def cpu_only():
    if os.environ.get("CUDA_VISIBLE_DEVICES")!="" or torch.cuda.is_initialized():raise ValueError("explicit CPU-only fixture/verification required")
    torch.set_num_threads(2)


def prepare(folder):
    cpu_only()
    if folder.exists():raise FileExistsError("preserve the prepared fixture")
    cases,data,indices=cases_and_data();case=cases[0]
    image=prompt(data,indices[case["source"]],case)
    if not torch.equal(image,prompt(data,indices[case["source"]],case,independent=True)):
        raise ValueError("independent source prompt differs")
    folder.mkdir(parents=True)
    pixels=((image[0,0]+1)*127.5).round().to(torch.uint8).numpy()
    Image.fromarray(pixels).save(folder/"prompt.png")
    prepared=prepare_image_prompt(folder/"prompt.png",layout="page")
    if not torch.equal(prepared.pixels,image):raise ValueError("PNG input adapter changed owning source pixels")
    write(folder/"fixture.json",dict(**committed_identity(SOURCES),position=0,source=case["source"],
        seed=20267420,prompt_pixels_sha256=pixels_sha(image),prompt_png_sha256=sha(folder/"prompt.png"),
        prompt_shape=list(image.shape),png_preparation_bit_exact=True,checkpoint_loaded=False,
        cuda_initialized=False,target_supplied_to_actor=False,model_generation_performed=False,
        language_quality_claim=False))
    print("Prepared exact original case00 prompt pixels; no model loaded.")


def inspect_access(path,run):
    if path is None:return dict(observed=False)
    content=path.read_text(errors="replace")
    if "openat(" not in content and "open(" not in content:raise ValueError("access log lacks file-open coverage")
    forbidden=["data/","artifacts/native_chinese_v80_20261106/","artifacts/adaptive_readout_v81r2_20261106/",
        "artifacts/adaptive_readout_v81eval_20261107/"]
    violations=[];internet=[]
    for line in content.splitlines():
        if "openat(" in line or "open(" in line:
            if any(('"'+name) in line or str(ROOT/name) in line for name in forbidden):violations.append(line)
        if "connect(" in line and ("AF_INET" in line or "AF_INET6" in line):internet.append(line)
    if violations or internet:raise ValueError("actor accessed forbidden training/ancestor paths or an internet socket; retain raw log")
    expected=str(run/"metric/checkpoint_latest.pt")
    if expected not in content and '"'+str((run/"metric/checkpoint_latest.pt").relative_to(ROOT))+'"' not in content:
        raise ValueError("access log does not identify the invoked final checkpoint")
    return dict(observed=True,access_log_sha256=sha(path),forbidden_path_opens=0,
        internet_family_connect_calls=0,raw_trace_retained_privately=True,
        coverage="standalone actor subprocess file opens and connect calls; ordinary libraries/caches/driver sockets allowed")


def verify(folder,output,run,access_log):
    cpu_only()
    fixture=json.loads((folder/"fixture.json").read_text())
    for name,digest in fixture["sources"].items():
        if sha(ROOT/name)!=digest:raise ValueError("fixture source changed")
    if sha(folder/"prompt.png")!=fixture["prompt_png_sha256"]:raise ValueError("fixture PNG changed")
    verified=json.loads((run/"verification.json").read_text())
    if not verified["all_64_final_outputs_independently_scored"]:raise ValueError("finish original matched CPU verification first")
    before=run/"metric/after/predictions/00"
    header=json.loads((before/"header.json").read_text())
    if header["source"]!=fixture["source"]:raise ValueError("original cohort case00 differs")
    for name,digest in header["files_sha256"].items():
        if sha(before/name)!=digest:raise ValueError("completed source answer changed")
    reference=torch.load(before/"answer.pt",map_location="cpu",weights_only=True)
    report=json.loads((output/"report.json").read_text())
    checkpoint=run/"metric/checkpoint_latest.pt"
    if (report["checkpoint_sha256"]!=sha(checkpoint) or report["checkpoint_sha256"]!=header["checkpoint_binding"]
            or report["seed"]!=fixture["seed"] or report["maximum_blocks"]!=128 or report["flow_steps"]!=32
            or report["reference_images_used"] or report["output_repair"]
            or not report["generation_performed"] or report["language_quality_claim"]):
        raise ValueError("standalone invocation differs from fixed replay")
    input_pixels=torch.load(output/"input/prompt.pt",map_location="cpu",weights_only=True)
    if pixels_sha(input_pixels)!=fixture["prompt_pixels_sha256"]:raise ValueError("actor input pixels differ")
    actual=torch.load(output/"output/answer.pt",map_location="cpu",weights_only=True)
    pixels=check_answer(actual);check_answer(reference)
    if (not torch.equal(actual["images"],reference["images"])
            or any(actual[k]!=reference[k] for k in ("stopped","capped","stop_probabilities","feedback_sha256"))
            or actual["reader_memory_sha256"]!=header["reader_memory_sha256"]):
        raise ValueError("restored actor images, events, feedback or reader memory differ")
    if sha(output/"output/answer.pt")!=report["output"]["raw_sha256"]:raise ValueError("standalone raw output hash differs")
    if len(actual["images"]):
        if not np.array_equal(pixels,np.asarray(Image.open(output/"output/answer.png"))):raise ValueError("standalone PNG differs from raw bits")
    elif (output/"output/answer.png").exists():raise ValueError("empty output has a stale PNG")
    access=inspect_access(access_log,run)
    write(folder/"verification.json",dict(**committed_identity(SOURCES),fixture_sha256=sha(folder/"fixture.json"),
        standalone_report_sha256=sha(output/"report.json"),original_answer_sha256=sha(before/"answer.pt"),
        matched_verification_sha256=sha(run/"verification.json"),checkpoint_sha256=sha(checkpoint),
        cases=1,original_training_case=True,images_stops_feedback_and_reader_memory_bit_exact=True,
        standalone_complete_actor_forward_executed=True,independent_neural_implementation=False,
        asset_access=access,language_quality_claim=False,general_ilm_goal_complete=False))
    print("Verified one exact standalone checkpoint replay; language quality is a separate result.")


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage",choices=("prepare","verify"),required=True)
    parser.add_argument("--fixture",type=Path,default=ROOT/"artifacts/decoder_metric_v82r2_20261107/runtime_replay_fixture")
    parser.add_argument("--output",type=Path,default=ROOT/"artifacts/decoder_metric_v82r2_20261107/runtime_replay_output")
    parser.add_argument("--run",type=Path,default=ROOT/"artifacts/decoder_metric_v82r2_20261107")
    parser.add_argument("--access-log",type=Path)
    args=parser.parse_args()
    if args.stage=="prepare":prepare(args.fixture.resolve())
    else:verify(args.fixture.resolve(),args.output.resolve(),args.run.resolve(),args.access_log.resolve() if args.access_log else None)

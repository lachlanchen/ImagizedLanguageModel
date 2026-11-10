#!/usr/bin/env python3
"""Replay one fixed evaluated image in a traced fresh inference process."""

import argparse
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys

import numpy as np

from ilm.image_edit.baseline import ROOT, save_json, sha
from scripts.review_compact_visual_span import collect_run, registered_inputs, verified_answer
from scripts.verify_glyph_tail_cold_inference import trace_audit


FIXED_CASE = "book_01_normal"
CODEC = ROOT / "artifacts/continuous_glyph_codec_v34_20260814/checkpoint_latest.pt"


def audit_access(output, working, prompt, checkpoint):
    receipt = trace_audit(output, working, prompt)
    artifacts = ROOT / "artifacts"
    allowed = {prompt, checkpoint.resolve(), CODEC.resolve(), (artifacts / "ilm_gpu0.lock").resolve()}
    opened = set()
    for trace in output.glob("access.*"):
        for line in trace.read_text(errors="replace").splitlines():
            match = re.search(r"^openat2?\(.*\)\s+=\s+\d+<(/[^>]+)>$", line)
            if match:
                opened.add(Path(match[1]).resolve())
    if not {prompt, checkpoint.resolve(), CODEC.resolve()} <= opened:
        raise ValueError("Trace lacks a successful prompt, image-head or codec open")
    if any(path.is_relative_to(artifacts) and not path.is_relative_to(output) and path not in allowed
           for path in opened):
        raise ValueError("Cold inference read an undeclared project artifact")
    return {**receipt, "required_model_and_prompt_opens_verified": True,
            "other_project_artifacts_read": False}


def compare_outputs(expected_path, expected, actual_path, actual):
    verified_answer(expected_path, expected)
    verified_answer(actual_path, actual)
    fields = ("input_sha256", "input_glyphs", "emitted_glyphs", "stopped", "capped",
              "prompt_span_lengths", "emitted_pixel_feedback", "image_sha256")
    if any(expected[name] != actual[name] for name in fields):
        raise ValueError("Cold generation changed the display, stop decision or image feedback trace")
    with np.load(expected_path / "ink.npz", allow_pickle=False) as left, np.load(
        actual_path / "ink.npz", allow_pickle=False
    ) as right:
        if not np.array_equal(left["ink"], right["ink"]):
            raise ValueError("Cold generation changed the raw glyph stream")


def verify(args):
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("Run the verifier with CUDA hidden; its child uses the project GPU")
    _, inputs = registered_inputs(args.evaluation)
    request, report, rows = collect_run(args.run, inputs)
    if report["optimizer_updates"] <= 0 or request["checkpoint_sha256"] != sha(args.checkpoint):
        raise ValueError("Replay requires the evaluated trained image-head checkpoint")
    if request["device"] == "cpu":
        raise ValueError("Replay requires the single-GPU evaluation run")
    expected, = [actual for _, actual, _, _ in rows if actual["id"] == FIXED_CASE]
    prompt = (args.evaluation / "inputs" / inputs[FIXED_CASE]["image"]).resolve()
    if sha(prompt) != expected["input_sha256"]:
        raise ValueError("Fixed evaluation input changed")
    output = args.output.resolve()
    working = output / "empty"
    working.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ, CUDA_DEVICE_ORDER="PCI_BUS_ID", CUDA_VISIBLE_DEVICES="0",
               CUBLAS_WORKSPACE_CONFIG=":4096:8", PYTHONPATH=str(ROOT),
               HF_HOME=str(output / "empty_hf_cache"), HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    command = [
        "strace", "-ff", "-yy", "-s", "4096", "-e", "trace=openat,openat2,connect", "-o", str(output / "access"),
        sys.executable, "-u", str(ROOT / "scripts/infer_compact_visual_span.py"), "--image", str(prompt),
        "--glyph-count", str(expected["input_glyphs"]), "--checkpoint", str(args.checkpoint.resolve()),
        "--output", str(output / "inference"), "--maximum-glyphs", str(request["maximum_glyphs"]),
        "--flow-steps", str(request["flow_steps"]), "--seed", str(request["seed"]),
    ]
    # Launch in a separate process group so a timeout can stop the traced child
    # along with strace. The child acquires the normal project GPU lock.
    with (output / "inference.log").open("x") as log:
        child = subprocess.Popen(command, cwd=working, env=env, stdout=log, stderr=subprocess.STDOUT,
                                 start_new_session=True)
        try:
            if child.wait(timeout=args.seconds) != 0:
                raise RuntimeError("Cold inference child failed; inspect inference.log")
        except BaseException:
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait()
            raise
    access = audit_access(output, working, prompt, args.checkpoint)
    actual_report = json.loads((output / "inference/report.json").read_text())
    actual_request = json.loads((output / "inference/request.json").read_text())
    if actual_request["device"] != request["device"] or actual_request["checkpoint_sha256"] != request["checkpoint_sha256"]:
        raise ValueError("Cold inference device or checkpoint differs from the evaluation")
    actual, = actual_report["records"]
    compare_outputs(args.run / FIXED_CASE, expected, output / "inference/answer", actual)
    save_json(output / "verification.json", {
        "fixed_case": FIXED_CASE, "optimizer_updates": report["optimizer_updates"],
        "checkpoint_sha256": request["checkpoint_sha256"], "suite_report_sha256": sha(args.run / "report.json"),
        "cold_report_sha256": sha(output / "inference/report.json"), "emitted_glyphs": actual["emitted_glyphs"],
        "raw_images_stopping_and_feedback_hashes_identical": True,
        "feedback_events": len(actual["emitted_pixel_feedback"]), **access,
    })
    print((output / "verification.json").read_text())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", type=Path, default=ROOT / "artifacts/compact_visual_span_evaluation_v1")
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seconds", type=float, default=1800)
    args = parser.parse_args()
    if args.seconds <= 0:
        parser.error("Set a positive replay deadline")
    verify(args)

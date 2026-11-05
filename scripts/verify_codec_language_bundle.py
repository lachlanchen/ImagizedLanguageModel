#!/usr/bin/env python3
"""Replay the first fixed audited answer in an isolated bundle-local process."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from PIL import Image
import torch

from ilm.visual_lm.codec_image_interface import image_bytes, json_file, sha
from ilm.visual_lm.codec_language_audit import CodecLanguageAuditData
from scripts.infer_codec_language_v68 import audited_identity

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--run", type=Path, default=ROOT / "artifacts/codec_language_v68_20261105")
    args = parser.parse_args()
    identity = audited_identity(args.run)
    bundle = args.bundle.resolve()
    manifest = json.loads((bundle / "manifest.json").read_text())
    if manifest["identity"] != identity or not identity["bounded_chinese_development_pass"]:
        raise ValueError("bundle must match the actual independently qualified final adaptation")
    args.out.mkdir(parents=True, exist_ok=False)
    out = args.out.resolve()
    data = CodecLanguageAuditData(ROOT / "data/general_instructions/codec_language_audit_v68_20261105")
    row = data.cases[0]
    prompt = data.prompt(0)
    Image.fromarray(image_bytes(prompt)).save(out / "prompt.png")
    relative = Path("predictions/normal") / (row["source"] + "_" + row["id"] + ".pt")
    audit_report = json.loads((args.run / "development/report.json").read_text())
    original = args.run / "development" / relative
    if sha(original) != audit_report["raw_files_sha256"][str(relative)]:
        raise ValueError("original autonomous answer changed")
    expected = torch.load(original, map_location="cpu", weights_only=True)["result"]
    empty = out / "empty_working_directory"
    empty.mkdir()
    environment = dict(os.environ, PYTHONPATH="", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
        HF_HUB_DISABLE_TELEMETRY="1", HF_HOME=str(out / "empty_hub"),
        HF_HUB_CACHE=str(out / "empty_hub"), TRANSFORMERS_CACHE=str(out / "empty_transformers"),
        HF_MODULES_CACHE=str(out / "empty_modules"))
    command = [sys.executable, "-I", "-B", str(bundle / "run.py"), "--image", str(out / "prompt.png"),
        "--out", str(out / "inference"), "--maximum-blocks", "128", "--deny-root", str(ROOT),
        "--deny-root", str(ROOT.parent / "LocalLLM"), "--deny-root", str(Path.home() / ".cache/huggingface")]
    with (out / "isolated_console.log").open("w") as log:
        subprocess.run(command, cwd=empty, env=environment, stdout=log, stderr=subprocess.STDOUT, check=True)
    result = torch.load(out / "inference/generation/raw_generation.pt", map_location="cpu", weights_only=True)
    checks = dict(block_count=len(result["images"]) == len(expected["images"]),
        actual_events=result["events"] == expected["events"],
        actual_stopping=result["stopped"] == expected["stopped"] and result["capped"] == expected["capped"],
        actual_binary_blocks=len(result["images"]) == len(expected["images"]) and all(
            torch.equal(a, b) for a,b in zip(result["images"], expected["images"])))
    child = json.loads((out / "inference/report.json").read_text())
    checks.update(bundle_local_sources=child["bundle_local_runtime_modules"],
        python_source_network_subprocess_guards=child["python_isolation_guard_active"],
        exact_prompt_pixels=child["input"]["actor_pixels_sha256"] ==
            hashlib.sha256(prompt.numpy().tobytes()).hexdigest(),
        final_checkpoint=child["final_checkpoint_sha256"] == identity["checkpoint_sha256"],
        manifest=child["bundle_manifest_sha256"] == sha(bundle / "manifest.json"))
    if expected["images"]:
        checks["native_png_bytes"] = original.with_suffix(".png").read_bytes() == (out / "inference/generation/answer.png").read_bytes()
    else:
        checks["empty_has_no_placeholder_png"] = not (out / "inference/generation/answer.png").exists()
    receipt = dict(checks=checks, all_checks_pass=all(checks.values()),
        bundle_manifest_sha256=sha(bundle / "manifest.json"), identity=identity,
        original_generation_sha256=sha(original), original_prompt_pixels_sha256=child["input"]["actor_pixels_sha256"],
        actual_answer_blocks=len(result["images"]), verifier_sha256=sha(Path(__file__)),
        operating_system_sandbox=False, general_language_qualified=False,
        language_quality_scored_here=False, general_ilm_goal_complete=False)
    json_file(out / "verification.json", receipt)
    if not all(checks.values()):
        raise AssertionError(checks)
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""CPU-check real prompt-file round trips and native output serialization."""

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

import numpy as np
from PIL import Image
import torch

from ilm.visual_lm.codec_image_interface import (
    image_bytes, json_file, prepare_image_prompt, save_generation, save_prepared_prompt, sha,
)
from ilm.visual_lm.codec_language_audit import AUDIT_DATA_SHA, CodecLanguageAuditData
from scripts.infer_codec_language_v68 import audited_identity

ROOT = Path(__file__).resolve().parents[1]
SOURCES = ("ilm/visual_lm/codec_image_interface.py", "scripts/infer_codec_language_v68.py",
    "scripts/check_codec_image_interface.py", "references/codec_image_interface_plan.md")
PREFLIGHT_SHA = "960a79cd02eab2b9f084a926dc137bef275356cec3ae9df26a567acc33d7f0ba"


def rejected(call, expected):
    try:
        call()
    except expected:
        return True
    raise AssertionError("invalid interface input was accepted")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--run", type=Path, default=ROOT / "artifacts/codec_language_v68_20261105")
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "" or torch.cuda.is_initialized():
        raise ValueError("CPU-only interface check required")
    args.out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    sources = {name: sha(ROOT / name) for name in SOURCES}
    for name, digest in sources.items():
        if hashlib.sha256(subprocess.check_output(["git", "show", revision + ":" + name], cwd=ROOT)).hexdigest() != digest:
            raise ValueError("commit the interface implementation before CPU validation")
    begun = time.perf_counter()
    data = CodecLanguageAuditData(ROOT / "data/general_instructions/codec_language_audit_v68_20261105")
    checks = {}
    with tempfile.TemporaryDirectory(prefix="image-input-", dir=args.out) as temporary:
        temporary = Path(temporary)
        for index in range(192):
            payload = data.payload(index)
            for blank in (False, True):
                expected = data.prompt(index, blank=blank)
                strip = payload["blank_prompt" if blank else "prompt"]
                Image.fromarray(image_bytes(strip)).save(temporary / "strip.png")
                folded = prepare_image_prompt(temporary / "strip.png", layout="strip")
                if not torch.equal(folded.pixels, expected):
                    raise ValueError("file strip differs from actual training-layout prompt")
                Image.fromarray(image_bytes(expected)).save(temporary / "page.png")
                page = prepare_image_prompt(temporary / "page.png", layout="page")
                if not torch.equal(page.pixels, expected):
                    raise ValueError("file page differs from actual owning prompt tensor")
        checks["all_384_real_prompts_exact_in_both_explicit_layouts"] = True

        # File-format witnesses only: alpha on white and all grayscale levels.
        # They are not training examples, model output or semantic evidence.
        rgba = np.zeros((33, 257, 4), dtype=np.uint8)
        rgba[:, :256, :3] = np.arange(256, dtype=np.uint8)[None, :, None]
        rgba[:, :256, 3] = 255
        Image.fromarray(rgba).save(temporary / "rgba.png")
        prepared = prepare_image_prompt(temporary / "rgba.png")
        expected = np.full((64, 288), 255, dtype=np.uint8)
        expected[:33, :256] = np.arange(256, dtype=np.uint8)[None]
        if not np.array_equal(image_bytes(prepared.pixels), expected):
            raise ValueError("grayscale/alpha/padding conversion changed physical intensities")
        saved = save_prepared_prompt(prepared, temporary / "prepared")
        recovered = torch.load(temporary / "prepared/prompt.pt", map_location="cpu", weights_only=True)
        if not torch.equal(recovered, prepared.pixels) or saved["actor_pixels_sha256"] != prepared.receipt["actor_pixels_sha256"]:
            raise ValueError("input record does not preserve the actor tensor")
        checks["grayscale_alpha_padding_and_owning_tensor_round_trip"] = True
        checks["completed_input_record_immutable"] = rejected(
            lambda: save_prepared_prompt(prepared, temporary / "prepared"), FileExistsError)

        Image.new("L", (1024, 544), 255).save(temporary / "large.png")
        rejected(lambda: prepare_image_prompt(temporary / "large.png"), ValueError)
        rejected(lambda: prepare_image_prompt(temporary / "rgba.png", layout="strip"), ValueError)
        Image.new("L", (32, 32), 0).save(temporary / "frames.gif", save_all=True,
            append_images=[Image.new("L", (32, 32), 255)], duration=100)
        rejected(lambda: prepare_image_prompt(temporary / "frames.gif"), ValueError)
        checks["oversized_ambiguous_and_nonbinary_strips_rejected"] = True

        native = ROOT / "artifacts/codec_visual_actor_preflight_20261105/attempt01"
        if sha(native / "report.json") != PREFLIGHT_SHA:
            raise ValueError("original real-model execution receipt changed")
        witness = json.loads((native / "report.json").read_text())["evidence"]["generation"]
        with Image.open(native / "untrained_execution.png") as image:
            original = np.array(image.convert("L"), copy=True)
        if original.shape != (32, 128) or witness["blocks"] != 2:
            raise ValueError("two actual native output blocks required")
        pixels = torch.from_numpy(original).float().div(127.5).sub(1)[None, None]
        result = dict(images=tuple(x.clone() for x in pixels.split(64, dim=-1)),
            events=witness["events"], stopped=False, capped=True)
        output = save_generation(result, temporary / "output", maximum_blocks=2)
        with Image.open(temporary / "output/answer.png") as image:
            if not np.array_equal(np.asarray(image), original):
                raise ValueError("native generated stream changed during PNG serialization")
        for index in range(2):
            with Image.open(temporary / f"output/blocks/{index:03d}.png") as image:
                if not np.array_equal(np.asarray(image), original[:, index*64:(index+1)*64]):
                    raise ValueError("individual generated block differs from source")
        raw = torch.load(temporary / "output/raw_generation.pt", map_location="cpu", weights_only=True)
        if raw["events"] != witness["events"] or not torch.equal(torch.cat(raw["images"], -1), pixels):
            raise ValueError("native events or binary blocks changed")
        checks["two_actual_execution_blocks_and_events_serialized_exactly"] = True
        rejected(lambda: save_generation(result, temporary / "output", maximum_blocks=2), FileExistsError)
        invalid = copy.deepcopy(result)
        invalid["events"][0]["stop_probability"] = .9
        rejected(lambda: save_generation(invalid, temporary / "invalid", maximum_blocks=2), ValueError)
        if (temporary / "invalid").exists():
            raise ValueError("invalid output created a misleading artifact directory")
        # A format boundary fixture, not a claimed model answer.
        empty = dict(images=(), events=[dict(block=0, stop_probability=.5, stopped=True)],
            stopped=True, capped=False)
        save_generation(empty, temporary / "empty", maximum_blocks=128)
        if list((temporary / "empty").rglob("*.png")):
            raise ValueError("empty output received a synthetic PNG")
        checks["stopping_empty_output_and_immutability_distinctions"] = True

        subprocess.run([sys.executable, str(ROOT / "scripts/infer_codec_language_v68.py"),
            "--image", str(temporary / "rgba.png"), "--out", str(temporary / "cli"),
            "--prepare-only"], cwd=ROOT, check=True, capture_output=True, text=True)
        cli = json.loads((temporary / "cli/report.json").read_text())
        if cli["model_calls"] or cli["gpu_allocations"] or cli["language_quality_claim"]:
            raise ValueError("preparation command made a model claim")
        checks["actual_cpu_preparation_command"] = True
        if (args.run / "training_report.json").exists():
            raise RuntimeError("this admission check requires the actual still-running instruction stage")
        checks["actual_unfinished_training_rejected_before_cuda"] = rejected(
            lambda: audited_identity(args.run), RuntimeError) and not torch.cuda.is_initialized()
    if not all(checks.values()):
        raise AssertionError(checks)
    report = dict(checks=checks, source_commit=revision, source_sha256=sources,
        prepared_data_sha256=AUDIT_DATA_SHA, actual_prompt_paths=384,
        explicit_input_layouts=2, actual_prior_execution_blocks_serialized=2,
        original_execution_report_sha256=PREFLIGHT_SHA,
        original_execution_png_sha256=sha(native / "untrained_execution.png"),
        output_serialization_does_not_reexecute_actor=True,
        elapsed_seconds=time.perf_counter() - begun, model_calls=0, gpu_allocations=0,
        final_actor_inference_tested=False, standalone_export_verified=False,
        language_quality_claim=False, general_ilm_goal_complete=False)
    json_file(args.out / "report.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

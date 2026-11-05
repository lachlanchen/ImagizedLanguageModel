#!/usr/bin/env python3
"""Measure frozen writing-code sensitivity on the fixed V65 training blocks."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from ilm.visual_lm.codec_block_writer import (
    CODEC_SHA, CodecBlockWriter, load_qualified_writing_codec,
)
from ilm.visual_lm.complete_writing_data import CompleteWritingImageData

ROOT = Path(__file__).resolve().parents[1]
TRAIN_SHA = "0ffcfd101ea26d79c4084bfd166bd146ff5560f5fa83b7aa57c6b48558377d7e"
MANIFEST_SHA = "2fc36c1da4344351dd0f97fa4f928fd99f9b5ca7e16b01c50ad439ecbbdbc50c"
LEVELS = (0., .01, .03, .1, .3, 1.)
SOURCES = (
    "scripts/audit_continuous_writing_geometry.py",
    "references/continuous_writing_geometry_protocol.md",
    "ilm/visual_lm/codec_block_writer.py",
    "ilm/visual_lm/continuous_glyph_codec.py",
    "ilm/visual_lm/complete_writing_data.py",
)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def scores(prediction, target, source_pixels, code, source_code):
    if not bool(torch.isfinite(prediction).all()):
        raise ValueError("non-finite native decoder output")
    ink, actual = prediction < 0, target < 0
    axes = (1, 2, 3)
    numerator = 2 * (ink & actual).sum(axes)
    denominator = ink.sum(axes) + actual.sum(axes)
    f1 = torch.where(denominator > 0, numerator / denominator, 1.)
    return dict(
        foreground_f1=f1.tolist(),
        exact=(ink == actual).flatten(1).all(1).tolist(),
        changed_pixel_fraction=((prediction < 0) != (source_pixels < 0)).float().mean(axes).tolist(),
        pixel_mse=(prediction - target).square().mean(axes).tolist(),
        pixel_mse_from_source=(prediction - source_pixels).square().mean(axes).tolist(),
        latent_mse=(code - source_code).square().mean((1, 2)).tolist(),
    )


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path,
        default=ROOT / "artifacts/continuous_writing_geometry_20261105/attempt01")
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "" or torch.cuda.is_initialized():
        raise ValueError("CPU-only diagnostic required")
    # Each attempt owns a new directory; never overwrite an interrupted run.
    args.out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    begun = time.perf_counter()
    source_hashes = {name: sha(ROOT / name) for name in SOURCES}
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    for name, digest in source_hashes.items():
        committed = subprocess.check_output(["git", "show", revision + ":" + name], cwd=ROOT)
        if hashlib.sha256(committed).hexdigest() != digest:
            raise ValueError("commit diagnostic sources before execution")
    run = ROOT / "artifacts/writer_factorial_v65_20261105"
    if sha(run / "manifest.json") != MANIFEST_SHA:
        raise ValueError("original training manifest changed")
    manifest = json.loads((run / "manifest.json").read_text())
    if len(manifest) != 64:
        raise ValueError("all 64 original training blocks required")
    data = CompleteWritingImageData(
        ROOT / "data/general_instructions/complete_lines_train_20261105",
        purpose="train", expected_report_sha256=TRAIN_SHA,
    )
    images, bindings = [], {}
    for i, row in enumerate(manifest):
        name = data.names[row["index"]]
        if name != row["file"] or data.report["files_sha256"][name] != row["source_sha256"]:
            raise ValueError("training source identity changed")
        strip = data.target(row["index"])[..., :-128]
        block = strip[..., row["block"] * 64:(row["block"] + 1) * 64].clone()
        reference_path = run / f"reference/{i:03}.png"
        with Image.open(reference_path) as opened:
            reference = np.array(opened.convert("L"))
        if (reference.shape != (32, 64) or not set(np.unique(reference)) <= {0, 255}
                or not np.array_equal(reference < 128, block[0, 0].numpy() < 0)):
            raise ValueError("saved training reference differs from source pixels")
        images.append(block)
        bindings[str(i)] = dict(source=row, reference_sha256=sha(reference_path))
    images = torch.cat(images)
    writer = CodecBlockWriter(load_qualified_writing_codec(
        ROOT / "artifacts/continuous_glyph_codec_v34_20260814/checkpoint_latest.pt",
        torch.device("cpu"),
    ))
    code = writer.encode_targets(images)
    source_pixels = writer.predict(code)
    generator = torch.Generator(device="cpu").manual_seed(2026110507)
    directions = torch.randn((4, *code.shape), generator=generator)
    directions -= directions.mean(-1, keepdim=True)
    centered = code - code.mean(-1, keepdim=True)
    directions -= (directions * centered).sum(-1, keepdim=True) / centered.square().sum(-1, keepdim=True) * centered
    directions /= directions.square().mean(-1, keepdim=True).sqrt()
    levels, retained = [], {}
    for sigma in LEVELS:
        cases, predictions, perturbed = [], [], []
        for direction in directions:
            changed = F.layer_norm(code + sigma * direction, (768,), eps=1e-5)
            prediction = writer.predict(changed)
            cases.append(scores(prediction, images, source_pixels, changed, code))
            predictions.append(prediction)
            perturbed.append(changed)
        keys = cases[0].keys()
        levels.append(dict(sigma=sigma, means={
            key: float(np.mean([case[key] for case in cases])) for key in keys
        }, cases=cases))
        retained[str(sigma)] = dict(codes=torch.stack(perturbed), scores=torch.stack(predictions))
    small = levels[1]["cases"]
    sensitivity = np.asarray([x["pixel_mse_from_source"] for x in small]) / np.asarray([x["latent_mse"] for x in small])
    tensors = args.out / "measurements.pt"
    torch.save(dict(images=images, codes=code, source_scores=source_pixels,
        directions=directions, perturbed=retained), tensors)
    report = dict(source_commit=revision, sources=source_hashes, manifest_sha256=MANIFEST_SHA,
        training_cache_sha256=TRAIN_SHA, codec_sha256=CODEC_SHA, reference_bindings=bindings,
        examples=64, training_only=True, development_examples=0, sealed_examples=0,
        optimization_updates=0, directions_per_example=4, seed=2026110507,
        baseline=scores(source_pixels, images, source_pixels, code, code),
        levels=levels, finite_difference_directional_sensitivity=dict(
            sigma=.01, per_block_min=sensitivity.min(0).tolist(),
            per_block_max=sensitivity.max(0).tolist()),
        measurements_sha256=sha(tensors), elapsed_seconds=time.perf_counter() - begun,
        language_quality_claim=False, general_ilm_goal_complete=False)
    if source_hashes != {name: sha(ROOT / name) for name in SOURCES}:
        raise ValueError("diagnostic sources changed during execution")
    (args.out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"baseline_f1":float(np.mean(report["baseline"]["foreground_f1"])),
        "levels":[dict(sigma=x["sigma"], **x["means"]) for x in levels],
        "report_sha256":sha(args.out / "report.json"),
        "elapsed_seconds":report["elapsed_seconds"]}, indent=2))


if __name__ == "__main__":
    main()

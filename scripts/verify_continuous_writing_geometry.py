#!/usr/bin/env python3
"""Independently replay every saved continuous-writing perturbation and score."""

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from ilm.visual_lm.codec_block_writer import load_qualified_writing_codec
from ilm.visual_lm.complete_writing_data import CompleteWritingImageData

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path,
        default=ROOT / "artifacts/continuous_writing_geometry_20261105/attempt01")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "" or torch.cuda.is_initialized():
        raise ValueError("CPU-only verification required")
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    report = json.loads((args.run / "report.json").read_text())
    if (not report["training_only"] or report["development_examples"] != 0
            or report["sealed_examples"] != 0 or report["optimization_updates"] != 0
            or report["language_quality_claim"] or report["general_ilm_goal_complete"]):
        raise ValueError("fixed training-only diagnostic scope changed")
    if sha(args.run / "measurements.pt") != report["measurements_sha256"]:
        raise ValueError("raw measurements changed")
    for name, digest in report["sources"].items():
        if sha(ROOT / name) != digest:
            raise ValueError("diagnostic source changed")
    saved = torch.load(args.run / "measurements.pt", map_location="cpu", weights_only=True)
    data = CompleteWritingImageData(
        ROOT / "data/general_instructions/complete_lines_train_20261105",
        purpose="train", expected_report_sha256=report["training_cache_sha256"],
    )
    for i in range(64):
        row = report["reference_bindings"][str(i)]["source"]
        if data.names[row["index"]] != row["file"]:
            raise ValueError("training identity changed")
        strip = data.target(row["index"])[..., :-128]
        expected = strip[..., row["block"] * 64:(row["block"] + 1) * 64]
        if not torch.equal(expected, saved["images"][i:i+1]):
            raise ValueError("original training pixels differ")
    codec = load_qualified_writing_codec(
        ROOT / "artifacts/continuous_glyph_codec_v34_20260814/checkpoint_latest.pt",
        torch.device("cpu"),
    )
    white = (saved["images"] + 1) / 2
    # Encode all left tiles before all right tiles, independently of the wrapper.
    native = codec.encode(torch.cat((white[..., :32], white[..., 32:])))
    expected = torch.stack((native[:64], native[64:]), 1)
    if not torch.equal(expected, saved["codes"]):
        raise ValueError("native source codes differ")
    centered = expected - expected.mean(-1, keepdim=True)
    generator = torch.Generator().manual_seed(2026110507)
    direction = torch.randn((4, 64, 2, 768), generator=generator)
    direction -= direction.mean(-1, keepdim=True)
    direction -= (direction * centered).sum(-1, keepdim=True) / centered.square().sum(-1, keepdim=True) * centered
    direction /= direction.square().mean(-1, keepdim=True).sqrt()
    if not torch.equal(direction, saved["directions"]):
        raise ValueError("seeded directions differ")
    reference = saved["images"].numpy() < 0
    source = saved["source_scores"].numpy()
    checks = 0
    for level in report["levels"]:
        values = saved["perturbed"][str(level["sigma"])]
        for k in range(4):
            code = F.layer_norm(expected + level["sigma"] * direction[k], (768,), eps=1e-5)
            if not torch.equal(code, values["codes"][k]):
                raise ValueError("actual perturbed code differs")
            decoded = codec.decode(code.reshape(-1, 768)).sigmoid() * 2 - 1
            tiles = decoded.reshape(64, 2, 1, 32, 32)
            pixels = torch.cat((tiles[:, 0], tiles[:, 1]), -1)
            if not torch.equal(pixels, values["scores"][k]):
                raise ValueError("raw native decoder replay differs")
            array = pixels.numpy()
            ink = array < 0
            denominator = (ink + reference.astype(np.int64)).sum((1, 2, 3))
            numerator = 2 * (ink & reference).sum((1, 2, 3))
            f1 = np.divide(numerator, denominator, out=np.ones(64), where=denominator != 0)
            actual = dict(foreground_f1=f1,
                exact=(ink == reference).reshape(64, -1).all(1),
                changed_pixel_fraction=(ink != (source < 0)).mean((1, 2, 3)),
                pixel_mse=((array - saved["images"].numpy()) ** 2).mean((1, 2, 3)),
                pixel_mse_from_source=((array - source) ** 2).mean((1, 2, 3)),
                latent_mse=((code.numpy() - expected.numpy()) ** 2).mean((1, 2)))
            for key, value in actual.items():
                if not np.allclose(value, level["cases"][k][key], atol=6e-8, rtol=2e-6):
                    raise ValueError("independent raw score differs: " + key)
            checks += 64
        for key, mean in level["means"].items():
            if not np.isclose(mean, np.mean([x[key] for x in level["cases"]]), atol=1e-12):
                raise ValueError("summary aggregation differs")
    result = dict(report_sha256=sha(args.run / "report.json"),
        measurements_sha256=sha(args.run / "measurements.pt"),
        verifier_sha256=sha(Path(__file__)), original_training_blocks_verified=64,
        native_perturbed_outputs_replayed=checks, raw_scores_independently_verified=checks,
        all_checks_pass=True, language_quality_claim=False, general_ilm_goal_complete=False)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.out.exists():
        raise FileExistsError("verification receipt already exists")
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

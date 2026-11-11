#!/usr/bin/env python3
"""Measure the flow head's invariant noise subspace using actual saved weights."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import time

import torch

from ilm.image_edit.baseline import ROOT, save_json, sha
from ilm.visual_lm.causal_glyph_flow import V35_CODEC_CHECKPOINT_SHA256


def stable_checkpoint(path):
    # Training replaces latest atomically. Keep one inode open for both hash
    # and deserialization so a concurrent checkpoint cannot mix identities.
    with path.open("rb") as stream:
        digest = hashlib.sha256()
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(block)
        stream.seek(0)
        state = torch.load(stream, map_location="cpu", weights_only=True)
    return state, digest.hexdigest()


def audit(checkpoint, codec, output):
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("Hide CUDA for this CPU matrix audit")
    if output.exists():
        raise FileExistsError(output)
    torch.set_num_threads(2)
    began = time.perf_counter()
    state, checkpoint_sha = stable_checkpoint(checkpoint)
    writer = state["image_heads"]["writer"]
    weight = writer["output.weight"].double()
    bias = writer["output.bias"].double()
    dimensions, channels = weight.shape
    if (bias.shape != (dimensions,) or writer["input.weight"].shape != (channels, dimensions)
            or not bool(torch.isfinite(weight).all()) or not bool(torch.isfinite(bias).all())):
        raise ValueError("The saved flow matrices are not the expected finite linear maps")
    left, singular, _ = torch.linalg.svd(weight, full_matrices=True)
    tolerance = max(weight.shape) * torch.finfo(torch.float64).eps * float(singular[0])
    rank = int((singular > tolerance).sum())
    projection = torch.eye(dimensions, dtype=torch.float64) - left[:, :rank] @ left[:, :rank].T
    annihilation = float((projection @ weight).abs().max())
    idempotence = float((projection @ projection - projection).abs().max())
    if max(annihilation, idempotence) > 1e-10:
        raise ValueError("Computed complement does not annihilate the trained output map")

    # This identity holds for every hidden trajectory, including all contexts.
    # It exercises only small matrix algebra, not neural/image generation.
    generator = torch.Generator().manual_seed(202611155)
    noise = torch.randn(dimensions, generator=generator, dtype=torch.float64)
    endpoints = []
    for _ in range(2):
        hidden = torch.randn((8, channels), generator=generator, dtype=torch.float64)
        endpoint = noise.clone()
        for value in hidden:
            endpoint += (weight @ value + bias) / 8
        endpoints.append(endpoint)
    expected = projection @ (noise + bias)
    invariant_error = max(float((projection @ point - expected).abs().max()) for point in endpoints)
    if invariant_error > 1e-10:
        raise ValueError("Eight-step Euler integration changed the invariant projection")

    if sha(codec) != V35_CODEC_CHECKPOINT_SHA256:
        raise ValueError("Use the authenticated project-trained V34 codec")
    # Match the existing codec loader: this hash-verified local checkpoint
    # includes TorchVersion metadata as well as the EMA tensors.
    codec_state = torch.load(codec, map_location="cpu", weights_only=False)
    decoder = codec_state["ema"]["shadow"]["decoder.projection.weight"].double()
    if decoder.shape[1] != dimensions:
        raise ValueError("Writer and actual decoder latent dimensions differ")
    total_energy = float(decoder.square().sum())
    projected_energy = float((decoder @ projection).square().sum())
    if total_energy <= 0:
        raise ValueError("The decoder projection is empty")
    feedback = state["image_heads"]["encoder"]["feedback.1.weight"]
    receipt = {
        "checkpoint_sha256": checkpoint_sha, "optimizer_updates": state["updates"],
        "checkpoint_was_live_latest": checkpoint.name == "checkpoint_latest.pt",
        "codec_sha256": V35_CODEC_CHECKPOINT_SHA256,
        "flow_latent_dimensions": dimensions, "flow_hidden_channels": channels,
        "actual_output_rank": rank, "rank_tolerance_float64": tolerance,
        "smallest_output_singular_value": float(singular[-1]),
        "invariant_noise_dimensions": dimensions - rank,
        "fresh_gaussian_expected_velocity_mse_floor": (dimensions - rank) / dimensions,
        "bound_scope": "Expected fresh-noise loss of the fixed linear-output model in exact arithmetic; finite-batch loss need not exceed this number",
        "matrix_calculation_dtype": "float64",
        "hardware_sampler_identity_measured": False,
        "projection_annihilation_max_error": annihilation,
        "projection_idempotence_max_error": idempotence,
        "eight_step_endpoint_invariant_max_error": invariant_error,
        "decoder_first_projection_shape": list(decoder.shape),
        "decoder_first_projection_noise_energy_fraction": projected_energy / total_energy,
        "decoder_measure_is_pixel_quality": False,
        "feedback_linear_output_shape": list(feedback.shape),
        "feedback_affine_dimension_upper_bound": min(feedback.shape),
        "feedback_target_approximation_error_measured": False,
        "neural_generation_performed": False, "language_quality_measured": False,
        "training_or_bound_pipeline_source_changed": False,
        "cuda_initialized": torch.cuda.is_initialized(),
        "elapsed_seconds": time.perf_counter() - began,
        "maximum_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2,
        "source_sha256": {"scripts/audit_compact_visual_flow_rank.py": sha(Path(__file__)),
                          "ilm/visual_lm/compact_visual_span_model.py": sha(ROOT / "ilm/visual_lm/compact_visual_span_model.py")},
    }
    if receipt["cuda_initialized"]:
        raise ValueError("Matrix audit initialized CUDA")
    output.mkdir(parents=True)
    save_json(output / "report.json", receipt)
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "artifacts/compact_visual_span_fit_v1/checkpoint_latest.pt")
    parser.add_argument("--codec", type=Path, default=ROOT / "artifacts/continuous_glyph_codec_v34_20260814/checkpoint_latest.pt")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    audit(args.checkpoint, args.codec, args.output)

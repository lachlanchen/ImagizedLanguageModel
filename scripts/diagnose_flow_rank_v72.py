#!/usr/bin/env python3
"""Authenticate and reproduce the V72 output-span obstruction on CPU."""

import argparse
import contextlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from ilm.visual_lm.semantic_writing_model import SemanticWritingConfig, SemanticWritingModel
from scripts.cache_semantic_images_v72 import ROOT, atomic_json, sha, committed_identity


SOURCES = ("scripts/diagnose_flow_rank_v72.py", "ilm/visual_lm/semantic_writing_model.py")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=ROOT/"artifacts/semantic_ilm_v72_20261105")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError("retain existing diagnoses; supply a fresh output directory")
    torch.set_num_threads(4)
    begun = time.perf_counter()
    run = args.run
    training = json.loads((run/"training/training_report.json").read_text())
    verification = json.loads((run/"verification.json").read_text())
    if not verification["all_384_outputs_independently_scored"]:
        raise ValueError("completed source audit required")
    checkpoint = run/"training/checkpoint_latest.pt"
    if sha(checkpoint) != training["checkpoint_sha256"]:
        raise ValueError("source checkpoint changed")
    for name, digest in training["identity"]["sources"].items():
        if sha(ROOT/name) != digest:
            raise ValueError("source implementation changed: "+name)
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model = SemanticWritingModel(SemanticWritingConfig(**saved["model_config"])).eval()
    model.load_state_dict(saved["model"], strict=True)
    del saved
    weight = model.flow.output.weight.detach().to(torch.bfloat16).float().double().numpy()
    bias = model.flow.output.bias.detach().to(torch.bfloat16).float().double().numpy()
    matrix = np.column_stack((weight, bias))
    u, singular, _ = np.linalg.svd(matrix, full_matrices=False)
    tolerance = max(matrix.shape)*np.finfo(np.float64).eps*singular[0]
    rank = int((singular > tolerance).sum())
    basis = u[:, :rank]
    dimension = matrix.shape[0]
    cache = run/"cache"
    inventory = json.loads((cache/"report.json").read_text())
    if sha(cache/"report.json") != training["identity"]["cache_report_sha256"]:
        raise ValueError("source cache report changed")
    names = sorted(inventory["files_sha256"])[:128]
    energies, count = [], 0
    first = None
    for name in names:
        path = cache/name
        if sha(path) != inventory["files_sha256"][name]:
            raise ValueError("source payload changed")
        row = torch.load(path, map_location="cpu", weights_only=True)
        if first is None:
            first = row
        z = row["codes"].double().flatten(1).numpy()
        residual = np.maximum(0, np.square(z).sum(1)-np.square(z@basis).sum(1))
        energies.append(float(residual.mean()/dimension))
        count += len(z)
    noise_floor = (dimension-rank)/dimension
    traces = []
    initial = torch.randn((1, dimension), generator=torch.Generator().manual_seed(20267300))
    with torch.inference_mode():
        for bf16 in (False, True):
            def convert(value):
                value = value.detach()
                return (value.to(torch.bfloat16).float() if bf16 else value).double().numpy()
            # QR is independent of the SVD basis used for the loss floor.
            q, _ = np.linalg.qr(np.column_stack((
                convert(model.flow.output.weight), convert(model.flow.output.bias))), mode="reduced")

            def perpendicular(value):
                value = value.detach().double().numpy()
                return value-(value@q)@q.T
            value = initial.clone()
            before = perpendicular(value)
            precision = torch.autocast("cpu", dtype=torch.bfloat16) if bf16 else contextlib.nullcontext()
            with precision:
                condition = model.conditions(first["memory"][None], torch.empty(1, 0, 2, 768))[:, -1]
                for i in range(32):
                    velocity = model.flow(value, torch.full((1,), i/32), condition)
                    middle = value+velocity/64
                    value = value+model.flow(middle, torch.full((1,), (i+.5)/32), condition)/32
            after = perpendicular(value)
            traces.append(dict(precision="CPU BF16" if bf16 else "CPU FP32",
                initial_perpendicular_rms=float(np.sqrt(np.mean(before**2))),
                final_perpendicular_rms=float(np.sqrt(np.mean(after**2))),
                perpendicular_drift_rms=float(np.sqrt(np.mean((after-before)**2)))))
    report = dict(**committed_identity(SOURCES), checkpoint_sha256=sha(checkpoint),
        source_verification_sha256=sha(run/"verification.json"), output_weight_shape=list(weight.shape),
        affine_output_span_rank=rank, invariant_noise_dimensions=dimension-rank,
        noise_only_expected_velocity_mse_floor=noise_floor,
        example_balanced_target_orthogonal_energy=float(np.mean(energies)),
        real_arithmetic_expected_velocity_mse_floor=noise_floor+float(np.mean(energies)),
        examples=len(names), blocks=count, sample_names=names, trajectories=traces,
        scope="post-hoc architecture diagnosis before final LayerNorm; CPU trace, not GPU audit replay",
        arithmetic_caveat="BF16 elementwise rounding gives small departures from the real-arithmetic invariant",
        language_quality_claim=False, elapsed_seconds=time.perf_counter()-begun)
    args.out.mkdir(parents=True)
    atomic_json(args.out/"report.json", report)
    print(json.dumps({k:v for k,v in report.items() if k not in ("sample_names", "sources")}, indent=2))


if __name__ == "__main__":
    main()

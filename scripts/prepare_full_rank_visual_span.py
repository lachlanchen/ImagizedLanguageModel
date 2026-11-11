#!/usr/bin/env python3
"""Register the controlled writer comparison and audit its initial capacity."""

import argparse
import copy
import json
import os
from pathlib import Path
import resource
import time

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from ilm.image_edit.baseline import ROOT, save_json, sha
from ilm.visual_lm.compact_visual_span_model import VisualSpanWriter
from ilm.visual_lm.full_rank_visual_span import (
    ARCHITECTURE, LATENT_WIDTH, MODEL_SOURCES, TRAINING_SEED, WRITER_WIDTH,
    tensor_state_sha, widen_writer,
)
from scripts.train_compact_visual_span import SOURCES as BASELINE_SOURCES, training_source


SOURCES = tuple(dict.fromkeys((*BASELINE_SOURCES, *MODEL_SOURCES,
    "scripts/prepare_full_rank_visual_span.py", "scripts/train_full_rank_visual_span.py",
    "scripts/infer_compact_visual_span.py", "scripts/infer_full_rank_visual_span.py",
    "scripts/run_compact_visual_span_pipeline.py", "scripts/run_full_rank_visual_span_pipeline.py",
)))


def velocity_math(writer, noisy, clock, context, valid):
    # Algebraically identical to the deployed FP32 velocity. Omitting the
    # context.float() cast allows a float64 mathematical Jacobian audit.
    phase = torch.stack((clock, clock.square()), dim=-1)
    condition = writer.condition(context) + writer.time(phase)
    hidden = writer.input(noisy) + writer.positions[:noisy.shape[1]] + condition[:, None]
    return writer.output(writer.flow(hidden, src_key_padding_mask=~valid))


def matrix_rank(matrix):
    values = torch.linalg.svdvals(matrix.double())
    tolerance = max(matrix.shape) * torch.finfo(torch.float64).eps * float(values[0])
    return {"shape": list(matrix.shape), "rank": int((values > tolerance).sum()),
            "smallest_singular_value": float(values[-1]),
            "largest_singular_value": float(values[0]), "float64_rank_tolerance": tolerance}


def capacity_audit(writer):
    generator = torch.Generator().manual_seed(202611156)
    noisy = torch.randn((1, 1, LATENT_WIDTH), generator=generator)
    context = torch.randn((1, 2048), generator=generator)
    valid = torch.ones((1, 1), dtype=torch.bool)
    clock = torch.tensor([.5])
    writer.eval()
    # Check the audited expression against the actual implementation before
    # promoting its tensors. The float64 result is not a BF16 sampler identity.
    with sdpa_kernel(SDPBackend.MATH), torch.no_grad():
        exact = torch.equal(writer.velocity(noisy, clock, context, valid),
                            velocity_math(writer, noisy, clock, context, valid))
    if not exact:
        raise ValueError("Mathematical audit expression differs from the actual FP32 velocity")
    wide = copy.deepcopy(writer).double()
    weight = wide.output.weight.detach()
    gamma = wide.flow.norm.weight.detach()
    # LN(x) = gamma * centered(x)/rms + beta. Centering removes one
    # direction; output rank alone is therefore an insufficient audit.
    centered = weight * gamma
    centered = centered - centered.mean(dim=1, keepdim=True)
    output_rank, centered_rank = matrix_rank(weight), matrix_rank(centered)
    if output_rank["rank"] != LATENT_WIDTH or centered_rank["rank"] != LATENT_WIDTH:
        raise ValueError("The writer still has a fixed output or LayerNorm affine restriction")
    points = []
    for time_value in (0., .5, .875):
        t = torch.tensor([time_value], dtype=torch.float64)
        def function(value):
            return velocity_math(wide, value.reshape(1, 1, -1), t, context.double(), valid).flatten()
        with sdpa_kernel(SDPBackend.MATH):
            jacobian = torch.autograd.functional.jacobian(function, noisy.double().flatten(), vectorize=True)
        measured = matrix_rank(jacobian)
        if measured["rank"] != LATENT_WIDTH or not torch.isfinite(jacobian).all():
            raise ValueError("The initial local velocity Jacobian is not full-dimensional")
        points.append({"time": time_value, **measured})
    return {"actual_fp32_velocity_matches_audited_expression": exact,
            "output_projection": output_rank, "layernorm_centered_output": centered_rank,
            "single_glyph_velocity_jacobians": points,
            "jacobian_dtype": "float64 mathematical expression with actual initial weights",
            "global_invertibility_or_hardware_sampler_proved": False,
            "training_or_language_quality_measured": False}


def validate_registration(path):
    registered = json.loads(Path(path).read_text())
    if registered["architecture"] != ARCHITECTURE or registered["writer_width"] != WRITER_WIDTH:
        raise ValueError("Registration identifies a different writer")
    for name, digest in registered["sources_sha256"].items():
        if sha(ROOT / name) != digest:
            raise ValueError(f"Registered source changed: {name}")
    for name, digest in registered["files_sha256"].items():
        if sha(ROOT / name) != digest:
            raise ValueError(f"Registered data or control changed: {name}")
    return registered


def prepare(args):
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("Hide CUDA during CPU registration and mathematical audit")
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(2)
    torch.manual_seed(TRAINING_SEED)
    began = time.perf_counter()
    corpus, plan, cache, _, _ = training_source(args.teacher)
    control = args.baseline / "image_heads_step_0000000.pt"
    initial = torch.load(control, weights_only=True, map_location="cpu")
    if initial["updates"] != 0:
        raise ValueError("Require the actual unchanged V1 initial snapshot")
    for name, digest in initial["identity"]["code_sha256"].items():
        if sha(ROOT / name) != digest:
            raise ValueError(f"The preserved V1 training implementation changed: {name}")
    original = VisualSpanWriter(LATENT_WIDTH, 2048, 16)
    original.load_state_dict(initial["image_heads"]["writer"], strict=True)
    before = torch.get_rng_state()
    writer = widen_writer(original)
    assert torch.equal(before, torch.get_rng_state())
    capacity = capacity_audit(writer)
    evaluation = json.loads((args.evaluation / "preparation.json").read_text())
    if (evaluation["visual_corpus_report_sha256"] != corpus.report_sha256
            or sha(args.evaluation / "inputs/manifest.json") != evaluation["input_manifest_sha256"]
            or sha(args.evaluation / "references.json") != evaluation["references_sha256"]):
        raise ValueError("The unchanged held-out evaluation differs")
    bindings = [args.teacher / "preparation.json", args.teacher / "cache/report.json",
                args.baseline / "report.json", control,
                args.evaluation / "preparation.json", args.evaluation / "inputs/manifest.json",
                args.evaluation / "references.json",
                ROOT / "references/evidence/compact_visual_span_learning_20261111.json"]
    registered = {
        "architecture": ARCHITECTURE, "writer_width": WRITER_WIDTH,
        "baseline_writer_width": 384, "latent_dimensions": LATENT_WIDTH,
        "teacher_run": str(args.teacher.resolve()), "baseline_fit": str(args.baseline.resolve()),
        "evaluation": str(args.evaluation.resolve()), "corpus_report_sha256": corpus.report_sha256,
        "training_sequences": len(corpus), "teacher_shards": len(cache), "causal_state_targets": plan["state_rows"],
        "training_configuration": {"epochs": 1, "padded_unit_budget": 1024, "learning_rate": .0002,
                                   "seed": TRAINING_SEED, "seconds": 10800.},
        "evaluation_configuration": {"seed": 202611153, "flow_steps": 8, "maximum_glyphs": 128,
                                     "all_endpoint_cases": 120, "intermediate_normal_cases": 44},
        "initial_encoder_sha256": tensor_state_sha(initial["image_heads"]["encoder"]),
        "initial_length_head_sha256": tensor_state_sha(original.length.state_dict()),
        "initial_wide_writer_sha256": tensor_state_sha(writer.state_dict()),
        "writer_parameters": sum(p.numel() for p in writer.parameters()),
        "baseline_writer_parameters": sum(p.numel() for p in original.parameters()),
        "initial_capacity_audit": capacity,
        "sources_sha256": {name: sha(ROOT / name) for name in SOURCES},
        "files_sha256": {str(path.relative_to(ROOT)): sha(path) for path in bindings},
        "gpu_resource_limits": {"device": 0, "maximum_allocated_gib": 22.,
                                "minimum_entry_available_ram_gib": 48., "minimum_runtime_available_ram_gib": 24.,
                                "fit_stage_seconds": 10800., "pressure_wait_seconds": 1800.},
        "primary_measurement": "Final-step complete exact QA/32 and both-correct question pairs/16, with raw-image readability and normal-minus-erased controls; same 12 book continuations and five checkpoints.",
        "decision_rule": "No credited readable conditional complete answers means the repair has not solved writing/language. Preserve failure and diagnose visual access, conditional writing and generated-history drift before another change. A development improvement requires new untouched confirmation groups before a generalization claim.",
        "remaining_recipe_unchanged": True, "length_and_encoder_initialization_bound_to_actual_control": True,
        "new_writer_construction_restores_cpu_rng": True, "offline_teacher_reextracted": False,
        "language_success_claimed": False, "registered_unix": time.time(),
        "cpu_preparation_seconds": time.perf_counter() - began,
        "maximum_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2,
        "cuda_initialized": torch.cuda.is_initialized(),
    }
    if registered["cuda_initialized"]:
        raise ValueError("Registration initialized CUDA")
    args.output.mkdir(parents=True)
    save_json(args.output / "registration.json", registered)
    print(json.dumps({key: registered[key] for key in ("architecture", "writer_parameters", "baseline_writer_parameters",
                      "initial_capacity_audit", "cpu_preparation_seconds", "maximum_rss_gib", "cuda_initialized")}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher", type=Path, default=ROOT / "artifacts/visual_sequence_teacher_v1")
    parser.add_argument("--baseline", type=Path, default=ROOT / "artifacts/compact_visual_span_fit_v1")
    parser.add_argument("--evaluation", type=Path, default=ROOT / "artifacts/compact_visual_span_evaluation_v1")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/compact_visual_span_full_rank_v1")
    prepare(parser.parse_args())

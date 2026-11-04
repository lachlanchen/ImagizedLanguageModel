#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import label
from torch.utils.data import DataLoader

from ilm.visual_lm.canonical_glyph_language_data import (
    CanonicalGlyphAuditDataset,
    CanonicalGlyphRenderConfig,
    canonical_glyph_audit_collate,
    render_canonical_character_bank,
)
from ilm.visual_lm.canonical_glyph_language_evaluation import _shuffle_prefix
from ilm.visual_lm.causal_visual_block_diffusion_v53 import (
    V53_ARCHITECTURE,
    CausalVisualBlockDiffusion,
    ConditionalStripDenoiser,
    causal_visual_block_boundary_receipt,
    causal_visual_block_config_from_payload,
)
from ilm.visual_lm.causal_visual_block_diffusion_v53_training import _visual_fields
from ilm.visual_lm.visual_cell_data import (
    load_v25_records,
    pack_visual_cells,
    verify_v25_manifest,
)
from ilm.visual_lm.visual_cell_eval_data import (
    build_visual_cell_audit_windows,
    build_visual_character_statistics,
)
from ilm.visual_lm.visual_posterior_language_v51_evaluation import (
    isolated_ink_fraction,
    pixel_f1,
)
from scripts.eval_noisy_visual_energy_guidance_v52 import _numeric_tree_is_finite
from scripts.train_canonical_glyph_binding_v43 import _atomic_json, _resolve_device
from scripts.train_causal_visual_block_diffusion_v53 import (
    DEFAULT_MANIFEST,
    DEFAULT_OUTPUT,
    DEFAULT_READER,
    load_frozen_reader,
)
from scripts.train_visual_state_actuator import file_sha256, seed_everything

AUDIT_SEED = 20265320


def parse_args():
    parser = argparse.ArgumentParser(description="Locked paired-image audit of V53.")
    parser.add_argument("--checkpoint", default=DEFAULT_OUTPUT + "/checkpoint_final.pt")
    parser.add_argument("--reader-checkpoint", default=DEFAULT_READER)
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--out", default=DEFAULT_OUTPUT + "/development")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--windows", type=int, default=512)
    parser.add_argument("--generated-examples", type=int, default=256)
    parser.add_argument("--feedback-examples", type=int, default=64)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def grouped_bootstrap(values, groups, *, replicates=2000):
    """Resample source records, retaining all overlapping windows together."""
    values = np.asarray(values, dtype=float)
    unique = sorted(set(groups))
    rows = [np.flatnonzero(np.asarray(groups) == group) for group in unique]
    sums = np.asarray([values[row].sum() for row in rows])
    counts = np.asarray([len(row) for row in rows])
    rng = np.random.default_rng(AUDIT_SEED + 99)
    draws = rng.integers(0, len(unique), (replicates, len(unique)))
    estimates = sums[draws].sum(1) / counts[draws].sum(1)
    return {
        "mean": float(values.mean()),
        "ci95": np.quantile(estimates, [0.025, 0.975]).tolist(),
        "source_records": len(unique),
        "windows": len(values),
        "replicates": replicates,
    }


@torch.no_grad()
def denoising_audit(model, loader, device):
    totals = {route: [] for route in ("ordered", "shuffled", "zero_condition")}
    offset = 0
    for raw in loader:
        context = raw["context"][:, :60].to(device)
        targets = raw["context"][:, 60:64].to(device)
        target_fields = _visual_fields(model, targets * 2 - 1)
        shuffled = _shuffle_prefix(context, first_index=offset)
        conditions = {
            "ordered": model.read_condition(context),
            "shuffled": model.read_condition(shuffled),
        }
        conditions["zero_condition"] = tuple(
            torch.zeros_like(x) for x in conditions["ordered"]
        )
        generator = torch.Generator(device=device).manual_seed(AUDIT_SEED + offset)
        noise = torch.randn(targets.shape, device=device, generator=generator)
        for step in (75, 95):
            times = torch.full((len(context), 4), step, device=device, dtype=torch.long)
            noisy = model.add_noise(targets, times, noise)
            for route, state in conditions.items():
                velocity, _ = model.denoiser(noisy, *state, times)
                clean, _ = model.velocity_to_clean_noise(noisy, velocity, times)
                totals[route].append(
                    (_visual_fields(model, clean) * target_fields).sum(-1).mean(1).cpu()
                )
        offset += len(context)
    return {
        route: {
            "field_cosine": float(torch.cat(values).mean()),
            "probe_times": [75, 95],
        }
        for route, values in totals.items()
    }


@torch.no_grad()
def generate_audit(model, loader, device, maximum, *, feedback=False):
    routes = (
        ("ordered", "shuffled")
        if feedback
        else ("ordered", "shuffled", "zero_condition")
    )
    outputs = {route: [] for route in routes}
    targets, identifiers, plans = [], [], []
    offset = 0
    for raw in loader:
        count = min(maximum - offset, len(raw["context"]))
        if count <= 0:
            break
        context = raw["context"][:count, :60].to(device)
        target = raw["context"][:count, 60:64]
        if feedback:
            target = torch.cat((target, raw["continuation"][:count, :4]), dim=1)
        targets.append(target)
        identifiers.extend(raw["identifier"][:count])
        plans.append(model.read_condition(context)[-1].cpu())
        for route in routes:
            visible = (
                _shuffle_prefix(context, first_index=offset)
                if route == "shuffled"
                else context
            )
            generated_blocks = []
            for block in range(2 if feedback else 1):
                generator = torch.Generator(device=device).manual_seed(
                    AUDIT_SEED + offset * 104729 + block * 10007
                )
                generated, trace = model.sample_next_block(
                    visible,
                    generator=generator,
                    conditioned=route != "zero_condition",
                )
                if not bool(torch.isfinite(trace["continuous"]).all()):
                    raise FloatingPointError("nonfinite generated image")
                generated_blocks.append(generated.cpu())
                visible = torch.cat((visible, generated), dim=1)[:, -60:]
            outputs[route].append(torch.cat(generated_blocks, dim=1))
        offset += count
    return {
        "outputs": {key: torch.cat(value) for key, value in outputs.items()},
        "targets": torch.cat(targets),
        "identifiers": identifiers,
        "plans": torch.cat(plans),
    }


@torch.no_grad()
def score_outputs(model, generated, bank, device):
    target = generated["targets"]
    bank_fields = model.reader.field.encode_unit(bank.to(device)).cpu()
    target_fields = model.reader.field.encode_unit(
        target.flatten(0, 1).to(device)
    ).cpu()
    similarities = target_fields @ bank_fields.T
    target_id = similarities.argmax(-1)
    represented = similarities.max(-1).values >= 1 - 1e-6
    b, t = target.shape[:2]
    reports, rows = {}, {}
    for route, images in generated["outputs"].items():
        flat = images.flatten(0, 1)
        fields = model.reader.field.encode_unit(flat.to(device)).cpu()
        identity = (fields @ bank_fields.T).argmax(-1)
        correct = ((identity == target_id) & represented).reshape(b, t).float()
        f1 = pixel_f1(flat, target.flatten(0, 1)).reshape(b, t)
        components = [
            label(image[0].numpy() >= 0.5, structure=np.ones((3, 3)))[1]
            for image in flat
        ]
        reports[route] = {
            "examples": b,
            "cells": t,
            "identity_top1_by_position": correct.mean(0).tolist(),
            "mean_identity_top1": float(correct.mean()),
            "pixel_f1_by_position": f1.mean(0).tolist(),
            "mean_pixel_f1": float(f1.mean()),
            "blank_rate": float((flat.flatten(1).sum(-1) < 2).float().mean()),
            "ink_ratio": float(images.mean() / target.mean()),
            "isolated_ink_fraction": float(isolated_ink_fraction(flat).mean()),
            "mean_connected_components": float(np.mean(components)),
            "distinct_evaluator_identities": int(len(identity.unique())),
            "evaluator_bank_coverage_by_position": represented.reshape(b, t)
            .float()
            .mean(0)
            .tolist(),
            "identity_ci95_grouped": grouped_bootstrap(
                correct.mean(1), generated["identifiers"]
            ),
        }
        rows[route] = correct
    comparisons = {}
    for route in reports:
        if route == "ordered":
            continue
        comparisons[f"ordered_minus_{route}"] = grouped_bootstrap(
            (rows["ordered"] - rows[route]).mean(1), generated["identifiers"]
        )
        comparisons[f"first_position_ordered_minus_{route}"] = grouped_bootstrap(
            rows["ordered"][:, 0] - rows[route][:, 0], generated["identifiers"]
        )
        reports[route]["pixel_disagreement_from_ordered"] = float(
            (generated["outputs"][route] != generated["outputs"]["ordered"])
            .float()
            .mean()
        )
    return reports, comparisons, rows


def main():
    args = parse_args()
    if not args.smoke and (
        args.windows,
        args.generated_examples,
        args.feedback_examples,
    ) != (512, 256, 64):
        raise ValueError("locked V53 audit requires 512/256/64 examples")
    if (
        min(
            args.windows,
            args.generated_examples,
            args.feedback_examples,
            args.batch_size,
        )
        < 1
    ):
        raise ValueError("audit sizes must be positive")
    output = Path(args.out)
    if (output / "development_report.json").exists():
        raise FileExistsError("audit report already exists")
    output.mkdir(parents=True, exist_ok=True)
    device = _resolve_device(args.device)
    seed_everything(AUDIT_SEED)
    torch.set_float32_matmul_precision("high")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if checkpoint["architecture"] != V53_ARCHITECTURE or (
        checkpoint["smoke_only"] and not args.smoke
    ):
        raise ValueError("ineligible V53 checkpoint")
    for category in checkpoint["provenance"].values():
        for path, digest in category.items():
            if file_sha256(path) != digest:
                raise ValueError(f"source/protocol changed: {path}")
    if file_sha256(args.reader_checkpoint) != checkpoint["reader_checkpoint_sha256"]:
        raise ValueError("reader hash differs from training")
    manifest = verify_v25_manifest(args.manifest, strict=True)
    if manifest["sha256"] != checkpoint["manifest"]["sha256"]:
        raise ValueError("corpus differs from training")
    reader = load_frozen_reader(args.reader_checkpoint, device)
    denoiser = ConditionalStripDenoiser(
        causal_visual_block_config_from_payload(checkpoint["model_config"])
    )
    denoiser.load_state_dict(checkpoint["ema_denoiser"], strict=True)
    model = (
        CausalVisualBlockDiffusion(reader, denoiser)
        .to(device)
        .requires_grad_(False)
        .eval()
    )
    records = load_v25_records(args.manifest, strict_manifest=True)
    render = CanonicalGlyphRenderConfig(**checkpoint["render_config"])
    statistics = build_visual_character_statistics(
        records, bank_size=1024, script_views_mode=render.script_views
    )
    windows = build_visual_cell_audit_windows(
        records,
        statistics,
        count=args.windows,
        continuation_cells=4,
        seed=AUDIT_SEED,
        script_views_mode=render.script_views,
    )
    dataset = CanonicalGlyphAuditDataset(windows, statistics, render_config=render)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=canonical_glyph_audit_collate,
    )
    started = time.perf_counter()
    denoising = denoising_audit(model, loader, device)
    generated = generate_audit(model, loader, device, args.generated_examples)
    feedback_images = generate_audit(
        model, loader, device, args.feedback_examples, feedback=True
    )
    # No candidate bank has existed during any denoising or generation above.
    bank = render_canonical_character_bank(statistics, render_config=render)
    generation, comparisons, per_example = score_outputs(model, generated, bank, device)
    feedback, feedback_comparisons, _ = score_outputs(
        model, feedback_images, bank, device
    )
    count = min(16, len(generated["targets"]))
    # Each row: 4 target, 4 ordered, 4 shuffled, 4 zero-condition, first plan.
    gallery = torch.cat(
        [
            generated["targets"][:count],
            *[
                generated["outputs"][r][:count]
                for r in ("ordered", "shuffled", "zero_condition")
            ],
            generated["plans"][:count, None],
        ],
        dim=1,
    )
    pack_visual_cells(gallery.flatten(0, 1), columns=17, gutter=2).save(
        output / "target_ordered_shuffled_zero_plan.png"
    )
    torch.save(generated, output / "generated_image_evidence.pt")
    torch.save(feedback_images, output / "feedback_image_evidence.pt")
    ordered = generation["ordered"]
    boundary = causal_visual_block_boundary_receipt(model)
    peak = (
        torch.cuda.max_memory_allocated(device) / 1024**3
        if device.type == "cuda"
        else 0.0
    )
    metrics = {"denoising": denoising, "generation": generation, "feedback": feedback}
    gates = {
        "image_only_boundary_and_hashes": not boundary[
            "parameter_names_with_forbidden_fragments"
        ],
        "parameter_budget": boundary["denoiser_parameters"] <= 8_000_000
        and boundary["combined_parameters"] <= 26_000_000,
        "finite_and_memory": _numeric_tree_is_finite(metrics) and peak < 18,
        "high_noise_order_advantage": denoising["ordered"]["field_cosine"]
        - denoising["shuffled"]["field_cosine"]
        > 0.01,
        "first_position_identity": ordered["identity_top1_by_position"][0] >= 0.05,
        "first_position_condition_advantage": all(
            comparisons[f"first_position_ordered_minus_{route}"]["mean"] > 0.02
            for route in ("shuffled", "zero_condition")
        ),
        "block_identity": ordered["mean_identity_top1"] >= 0.03
        and comparisons["ordered_minus_shuffled"]["mean"] > 0.01,
        "pixel_f1": ordered["mean_pixel_f1"] >= 0.48,
        "surface_quality": ordered["blank_rate"] < 0.02
        and ordered["isolated_ink_fraction"] <= 0.01
        and 0.7 <= ordered["ink_ratio"] <= 1.5,
        "identity_diversity": ordered["distinct_evaluator_identities"] >= 64,
        "visible_feedback_advantage": feedback_comparisons["ordered_minus_shuffled"][
            "mean"
        ]
        > 0,
        "bank_excluded_from_generation": True,
    }
    report = {
        "architecture": V53_ARCHITECTURE,
        "decision": ("retain" if all(gates.values()) else "reject")
        if not args.smoke
        else "smoke-only",
        "checkpoint_sha256": file_sha256(args.checkpoint),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "evaluator_sha256": file_sha256(__file__),
        "audit_seed": AUDIT_SEED,
        "training_provenance": checkpoint["provenance"],
        "manifest": manifest,
        **metrics,
        "comparisons": comparisons,
        "feedback_comparisons": feedback_comparisons,
        "boundary": boundary,
        "gates": gates,
        "elapsed_seconds": time.perf_counter() - started,
        "peak_allocated_vram_gib": peak,
        "frozen_partition_opened": False,
        "surface_identity_is_posthoc_nearest_raster_not_ocr": True,
        "gallery_layout": "4 target | 4 ordered | 4 shuffled | 4 zero condition | first V42 plan",
    }
    _atomic_json(report, output / "development_report.json")
    _atomic_json(
        {
            k: v
            for k, v in report.items()
            if k not in ("manifest", "training_provenance")
        },
        output / "development_summary.json",
    )
    _atomic_json(
        {
            "identifiers": generated["identifiers"],
            "correct_by_route": {k: v.tolist() for k, v in per_example.items()},
        },
        output / "per_example_identity.json",
    )
    print(
        json.dumps(
            {
                "decision": report["decision"],
                "gates": gates,
                "generation": generation,
                "comparisons": comparisons,
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()

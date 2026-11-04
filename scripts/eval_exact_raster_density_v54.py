#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from ilm.visual_lm.canonical_glyph_language import (
    canonical_glyph_language_config_from_payload,
)
from ilm.visual_lm.canonical_glyph_language_data import (
    CanonicalGlyphAuditDataset,
    CanonicalGlyphRenderConfig,
    canonical_glyph_audit_collate,
)
from ilm.visual_lm.canonical_glyph_language_evaluation import _shuffle_prefix
from ilm.visual_lm.exact_raster_density_v54 import (
    V54_ARCHITECTURE,
    ExactRasterDensity,
    ExactRasterDensityConfig,
)
from ilm.visual_lm.hierarchical_raster_language import CausalRasterReader
from ilm.visual_lm.visual_cell_data import (
    iter_split_writing,
    load_v25_records,
    pack_visual_cells,
    render_visual_cell_stream,
)
from ilm.visual_lm.visual_cell_eval_data import (
    build_visual_cell_audit_windows,
    build_visual_character_statistics,
)
from ilm.visual_lm.visual_posterior_language_v51_evaluation import (
    isolated_ink_fraction,
    pixel_f1,
)
from scripts.eval_causal_visual_block_diffusion_v53 import grouped_bootstrap
from scripts.eval_noisy_visual_energy_guidance_v52 import _numeric_tree_is_finite
from scripts.train_canonical_glyph_binding_v43 import _atomic_json, _resolve_device
from scripts.train_exact_raster_density_v54 import DEFAULT_OUT
from scripts.train_visual_state_actuator import file_sha256, seed_everything

AUDIT_SEED = 20265420


def arguments():
    parser = argparse.ArgumentParser(
        description="Audit the exact normalized raster language distribution."
    )
    parser.add_argument("--checkpoint", default=DEFAULT_OUT + "/checkpoint_final.pt")
    parser.add_argument("--out", default=DEFAULT_OUT + "/development")
    parser.add_argument(
        "--manifest",
        default="data/visual_grammar/chinese_wikisource_public_domain.jsonl",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--windows", type=int, default=512)
    parser.add_argument("--generation-examples", type=int, default=256)
    parser.add_argument("--feedback-examples", type=int, default=64)
    parser.add_argument("--motor-examples", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def load_student(checkpoint, device):
    reader = CausalRasterReader(
        canonical_glyph_language_config_from_payload(checkpoint["reader_config"])
    )
    model = ExactRasterDensity(
        reader, ExactRasterDensityConfig(**checkpoint["model_config"])
    )
    model.load_state_dict(checkpoint["ema_model"], strict=True)
    return model.to(device).requires_grad_(False).eval()


@torch.no_grad()
def evaluate_likelihood(model, loader, device):
    nll = {route: [] for route in ("ordered", "shuffled", "zero", "independent_plan")}
    ids = []
    offset = 0
    for raw in loader:
        context = raw["context"].to(device)
        target = (raw["continuation"][:, 0].to(device) >= 0.5).float()
        hidden, fields = model.read(context)
        sh, sa = model.read(_shuffle_prefix(context, first_index=offset))
        conditions = {
            "ordered": (hidden, fields),
            "shuffled": (sh, sa),
            "zero": (torch.zeros_like(hidden), torch.zeros_like(fields)),
        }
        for route, state in conditions.items():
            nll[route].append(model.image_nll(target, *state).cpu())
        bias = model.config.plan_logit_scale * model.plan(fields)[:, None]
        nll["independent_plan"].append(
            torch.nn.functional.binary_cross_entropy_with_logits(
                bias, target, reduction="none"
            )
            .flatten(1)
            .sum(1)
            .cpu()
        )
        ids.extend(raw["identifier"])
        offset += len(context)
    nll = {route: torch.cat(rows) for route, rows in nll.items()}
    result = {
        route: {
            "nats_per_image": float(values.mean()),
            "bits_per_pixel": float(values.mean()) / (1024 * math.log(2)),
            "examples": len(values),
        }
        for route, values in nll.items()
    }
    advantage = grouped_bootstrap(nll["shuffled"] - nll["ordered"], ids)
    result["ordered_nll_advantage_over_shuffled"] = advantage
    return result, {"nll": nll, "identifiers": ids}


@torch.no_grad()
def generate(model, loader, device, maximum, *, feedback=False, motor=False):
    names = (
        ("ordered", "shuffled")
        if feedback
        else ("copy_sample", "copy_greedy", "deranged_greedy")
        if motor
        else ("ordered", "shuffled", "zero", "greedy")
    )
    output = {name: [] for name in names}
    targets = []
    ids = []
    offset = 0
    for raw in loader:
        count = min(maximum - offset, len(raw["context"]))
        if count <= 0:
            break
        context = raw["context"][:count].to(device)
        target = (raw["continuation"][:count, : 4 if feedback else 1] >= 0.5).float()
        targets.append(target)
        ids.extend(raw["identifier"][:count])
        for route in names:
            visible = (
                _shuffle_prefix(context, first_index=offset)
                if route == "shuffled"
                else context
            )
            rng = torch.Generator(device=device).manual_seed(
                AUDIT_SEED + offset * 104729
            )
            cells = []
            for step in range(4 if feedback else 1):
                if motor:
                    fields = model.reader.field.encode_unit(target[:, 0].to(device))
                    if route == "deranged_greedy":
                        fields = fields.roll(1, 0)
                    image = model.sample_state(
                        torch.zeros(count, 384, device=device),
                        fields,
                        generator=rng,
                        greedy=route != "copy_sample",
                    )
                else:
                    image = model.sample_next(
                        visible,
                        generator=rng,
                        greedy=route == "greedy",
                        conditioned=route != "zero",
                    )
                if not bool(torch.isfinite(image).all()):
                    raise FloatingPointError("nonfinite generated raster")
                cells.append(image.cpu())
                visible = torch.cat((visible, image[:, None]), dim=1)[:, -64:]
            output[route].append(torch.stack(cells, dim=1))
        offset += count
        print(
            json.dumps(
                {
                    "phase": "motor"
                    if motor
                    else "feedback"
                    if feedback
                    else "generation",
                    "examples": offset,
                }
            ),
            flush=True,
        )
    return {
        "outputs": {route: torch.cat(rows) for route, rows in output.items()},
        "targets": torch.cat(targets),
        "identifiers": ids,
    }


def evaluator_bank(records, render, evidence):
    forms = set()
    for _, _, writing in iter_split_writing(
        records, split="train", script_views_mode=render.script_views
    ):
        forms.update(writing)
    raster = (
        render_visual_cell_stream(
            "".join(sorted(forms)),
            config=render.visual_cell_config(),
            font_path=render.font_path,
            variant=0,
        )
        >= 0.5
    ).float()
    targets = [data["targets"].flatten(0, 1) for data in evidence]
    # Pixel deduplication admits punctuation and any target form absent from
    # the training bank. This post-generation bank is evaluator-only.
    raster = torch.cat((raster, *targets))
    unique = torch.from_numpy(
        __import__("numpy").unique(raster.numpy().reshape(len(raster), -1), axis=0)
    )
    return unique.reshape(-1, 1, 32, 32)


@torch.no_grad()
def measure(model, data, bank_fields, device):
    target = data["targets"]
    b, t = target.shape[:2]
    target_fields = model.reader.field.encode_unit(
        target.flatten(0, 1).to(device)
    ).cpu()
    target_cos = target_fields @ bank_fields.T
    if not bool((target_cos.max(-1).values > 1 - 1e-5).all()):
        raise ValueError("post-generation bank does not cover every target raster")
    target_id = target_cos.argmax(-1)
    result = {}
    correct_rows = {}
    for route, images in data["outputs"].items():
        flat = images.flatten(0, 1)
        fields = model.reader.field.encode_unit(flat.to(device)).cpu()
        identity = (fields @ bank_fields.T).argmax(-1)
        correct = (identity == target_id).reshape(b, t).float()
        f1 = pixel_f1(flat, target.flatten(0, 1)).reshape(b, t)
        result[route] = {
            "examples": b,
            "cells": t,
            "identity_top1": float(correct.mean()),
            "identity_by_step": correct.mean(0).tolist(),
            "pixel_f1": float(f1.mean()),
            "pixel_f1_by_step": f1.mean(0).tolist(),
            "blank_rate": float((flat.flatten(1).sum(-1) < 2).float().mean()),
            "isolated_ink_fraction": float(isolated_ink_fraction(flat).mean()),
            "ink_ratio": float(images.mean() / target.mean()),
            "exact_image_match": float(
                (flat == target.flatten(0, 1)).flatten(1).all(-1).float().mean()
            ),
            "distinct_nearest_identities": len(identity.unique()),
            "target_bank_coverage": 1.0,
            "identity_ci95": grouped_bootstrap(correct.mean(1), data["identifiers"]),
        }
        correct_rows[route] = correct
    comparisons = {}
    if "ordered" in correct_rows:
        for other in ("shuffled", "zero"):
            if other in correct_rows:
                comparisons["ordered_minus_" + other] = grouped_bootstrap(
                    (correct_rows["ordered"] - correct_rows[other]).mean(1),
                    data["identifiers"],
                )
    return (
        result,
        comparisons,
        {key: value.tolist() for key, value in correct_rows.items()},
    )


def main():
    args = arguments()
    if not args.smoke and (
        args.windows,
        args.generation_examples,
        args.feedback_examples,
        args.motor_examples,
    ) != (512, 256, 64, 256):
        raise ValueError("production V54 requires the registered audit sizes")
    out = Path(args.out)
    if (out / "development_report.json").exists():
        raise FileExistsError("audit report already exists")
    out.mkdir(parents=True, exist_ok=True)
    device = _resolve_device(args.device)
    seed_everything(AUDIT_SEED)
    torch.set_float32_matmul_precision("high")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if checkpoint["architecture"] != V54_ARCHITECTURE or (
        checkpoint["smoke_only"] and not args.smoke
    ):
        raise ValueError("ineligible checkpoint")
    for path, sha in checkpoint["provenance"].items():
        if file_sha256(path) != sha:
            raise ValueError(f"training source/protocol changed: {path}")
    manifest_path = args.manifest
    if file_sha256(manifest_path) != checkpoint["manifest"]["sha256"]:
        raise ValueError("corpus changed")
    model = load_student(checkpoint, device)
    records = load_v25_records(manifest_path, strict_manifest=True)
    render = CanonicalGlyphRenderConfig(**checkpoint["render_config"])
    stats = build_visual_character_statistics(
        records, bank_size=1024, script_views_mode=render.script_views
    )
    windows = build_visual_cell_audit_windows(
        records,
        stats,
        count=args.windows,
        continuation_cells=4,
        seed=AUDIT_SEED,
        script_views_mode=render.script_views,
    )
    dataset = CanonicalGlyphAuditDataset(windows, stats, render_config=render)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=canonical_glyph_audit_collate,
    )
    start = time.perf_counter()
    likelihood, likelihood_rows = evaluate_likelihood(model, loader, device)
    generated = generate(model, loader, device, args.generation_examples)
    feedback = generate(model, loader, device, args.feedback_examples, feedback=True)
    motor = generate(model, loader, device, args.motor_examples, motor=True)
    # Every primary output exists before any evaluator image bank is rendered.
    bank = evaluator_bank(records, render, (generated, feedback, motor))
    bank_fields = model.reader.field.encode_unit(bank.to(device)).cpu()
    generation, comparisons, per_example = measure(
        model, generated, bank_fields, device
    )
    feedback_report, feedback_comparisons, _ = measure(
        model, feedback, bank_fields, device
    )
    motor_report, _, _ = measure(model, motor, bank_fields, device)
    primary = generation["ordered"]
    difference = comparisons["ordered_minus_shuffled"]
    peak = (
        torch.cuda.max_memory_allocated(device) / 1024**3
        if device.type == "cuda"
        else 0.0
    )
    boundary = model.boundary_receipt()
    metrics = {
        "likelihood": likelihood,
        "generation": generation,
        "motor": motor_report,
        "feedback": feedback_report,
    }
    gates = {
        "runtime_boundary": all(
            boundary[key] is False
            for key in (
                "text_tokenizer",
                "symbol_ids",
                "glyph_inventory",
                "codebook",
                "ocr",
                "external_runtime_model",
            )
        ),
        "motor": motor_report["copy_greedy"]["pixel_f1"] >= 0.995
        and motor_report["copy_sample"]["pixel_f1"] >= 0.98
        and motor_report["copy_greedy"]["pixel_f1"]
        - motor_report["deranged_greedy"]["pixel_f1"]
        >= 0.30,
        "ordered_likelihood": likelihood["ordered_nll_advantage_over_shuffled"]["ci95"][
            0
        ]
        > 0,
        "sample_quality": primary["pixel_f1"] >= 0.48
        and primary["identity_top1"] >= 0.03,
        "ordered_generation": difference["mean"] > 0.01 and difference["ci95"][0] > 0,
        "surface": primary["isolated_ink_fraction"] <= 0.01
        and primary["blank_rate"] < 0.02
        and 0.7 <= primary["ink_ratio"] <= 1.5,
        "feedback": feedback_comparisons["ordered_minus_shuffled"]["mean"] > 0,
        "finite_and_resource": _numeric_tree_is_finite(metrics)
        and peak < 18
        and boundary["density_parameters"] <= 5_000_000
        and boundary["runtime_parameters"] <= 23_000_000,
    }
    report = {
        "architecture": V54_ARCHITECTURE,
        "decision": "smoke-only"
        if args.smoke
        else "retain"
        if all(gates.values())
        else "reject",
        "checkpoint_sha256": file_sha256(args.checkpoint),
        "audit_seed": AUDIT_SEED,
        "evaluator_sha256": file_sha256(__file__),
        "training_provenance": checkpoint["provenance"],
        "boundary": boundary,
        "manifest_sha256": checkpoint["manifest"]["sha256"],
        **metrics,
        "comparisons": comparisons,
        "feedback_comparisons": feedback_comparisons,
        "gates": gates,
        "elapsed_seconds": time.perf_counter() - start,
        "peak_allocated_vram_gib": peak,
        "evaluator_bank_size": len(bank),
        "evaluator_bank_created_after_generation": True,
        "frozen_partition_opened": False,
        "supplied_image_motor_is_not_language_evidence": True,
    }
    _atomic_json(report, out / "development_report.json")
    _atomic_json(report, out / "development_summary.json")
    _atomic_json(
        {"identifiers": generated["identifiers"], "correct_by_route": per_example},
        out / "per_example_identity.json",
    )
    torch.save(likelihood_rows, out / "per_example_likelihood.pt")
    for name, data in (
        ("generation", generated),
        ("feedback", feedback),
        ("motor", motor),
    ):
        torch.save(data, out / f"{name}_image_evidence.pt")
        routes = list(data["outputs"])
        rows = torch.cat(
            (data["targets"][:16], *[data["outputs"][route][:16] for route in routes]),
            dim=1,
        )
        pack_visual_cells(rows.flatten(0, 1), columns=rows.shape[1], gutter=2).save(
            out / f"{name}_gallery.png"
        )
    print(
        json.dumps(
            {
                "decision": report["decision"],
                "gates": gates,
                **metrics,
                "comparisons": comparisons,
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()

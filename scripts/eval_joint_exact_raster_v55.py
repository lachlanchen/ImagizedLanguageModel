#!/usr/bin/env python3
"""Locked matched audit of joint versus frozen exact-raster language learning."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from ilm.visual_lm.canonical_glyph_language_data import (
    CanonicalGlyphAuditDataset,
    CanonicalGlyphRenderConfig,
    canonical_glyph_audit_collate,
)
from ilm.visual_lm.joint_exact_raster_v55 import ARCHITECTURE, model_from_checkpoint
from ilm.visual_lm.visual_cell_data import load_v25_records, pack_visual_cells
from ilm.visual_lm.visual_cell_eval_data import (
    build_visual_cell_audit_windows,
    build_visual_character_statistics,
)
from scripts.eval_exact_raster_density_v54 import (
    AUDIT_SEED,
    _numeric_tree_is_finite,
    evaluate_likelihood,
    evaluator_bank,
    generate,
    grouped_bootstrap,
    measure,
)
from scripts.train_canonical_glyph_binding_v43 import _atomic_json, _resolve_device
from scripts.train_joint_exact_raster_v55 import DEFAULT_OUT, SOURCE_SHA
from scripts.train_visual_state_actuator import file_sha256, seed_everything


def qualification_gates(metrics, boundary, peak):
    primary = metrics["generation"]["ordered"]
    motor = metrics["motor"]
    return {
        "runtime_boundary": all(
            boundary[k] is False
            for k in [
                "text_tokenizer",
                "symbol_ids",
                "glyph_inventory",
                "codebook",
                "ocr",
                "external_runtime_model",
            ]
        ),
        "motor": motor["copy_greedy"]["pixel_f1"] >= 0.995
        and motor["copy_sample"]["pixel_f1"] >= 0.98
        and motor["copy_greedy"]["pixel_f1"] - motor["deranged_greedy"]["pixel_f1"]
        >= 0.30,
        "ordered_likelihood": metrics["likelihood"][
            "ordered_nll_advantage_over_shuffled"
        ]["ci95"][0]
        > 0,
        "sample_quality": primary["pixel_f1"] >= 0.48
        and primary["identity_top1"] >= 0.03,
        "ordered_generation": metrics["comparisons"]["ordered_minus_shuffled"]["mean"]
        > 0.01
        and metrics["comparisons"]["ordered_minus_shuffled"]["ci95"][0] > 0,
        "surface": primary["isolated_ink_fraction"] <= 0.01
        and primary["blank_rate"] < 0.02
        and 0.7 <= primary["ink_ratio"] <= 1.5,
        "feedback": metrics["feedback_comparisons"]["ordered_minus_shuffled"]["mean"]
        > 0,
        "finite_and_resource": _numeric_tree_is_finite(metrics)
        and peak < 18
        and boundary["density_parameters"] <= 5_000_000
        and boundary["runtime_parameters"] <= 23_000_000,
    }


@torch.no_grad()
def audit_arm(root, arm, checkpoint, device):
    out = root / arm / "development"
    if (out / "development_report.json").exists():
        raise FileExistsError("completed arm audit is immutable")
    out.mkdir(parents=True, exist_ok=True)
    for path, digest in checkpoint["provenance"].items():
        if file_sha256(path) != digest:
            raise ValueError(f"trained source changed: {path}")
    manifest = "data/visual_grammar/chinese_wikisource_public_domain.jsonl"
    if file_sha256(manifest) != checkpoint["manifest"]["sha256"]:
        raise ValueError("corpus changed")
    seed_everything(AUDIT_SEED)
    torch.set_float32_matmul_precision("high")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    model = model_from_checkpoint(checkpoint, device)
    records = load_v25_records(manifest, strict_manifest=True)
    render = CanonicalGlyphRenderConfig(**checkpoint["render_config"])
    stats = build_visual_character_statistics(
        records, bank_size=1024, script_views_mode=render.script_views
    )
    windows = build_visual_cell_audit_windows(
        records,
        stats,
        count=512,
        continuation_cells=4,
        seed=AUDIT_SEED,
        script_views_mode=render.script_views,
    )
    loader = DataLoader(
        CanonicalGlyphAuditDataset(windows, stats, render_config=render),
        batch_size=32,
        num_workers=4,
        collate_fn=canonical_glyph_audit_collate,
    )
    started = time.perf_counter()
    likelihood, likelihood_rows = evaluate_likelihood(model, loader, device)
    generated = generate(model, loader, device, 256)
    feedback = generate(model, loader, device, 64, feedback=True)
    motor = generate(model, loader, device, 256, motor=True)
    bank = evaluator_bank(records, render, (generated, feedback, motor))
    bank_fields = model.reader.field.encode_unit(bank.to(device)).cpu()
    generation, comparisons, correct = measure(model, generated, bank_fields, device)
    feedback_report, feedback_comparisons, _ = measure(
        model, feedback, bank_fields, device
    )
    motor_report, _, _ = measure(model, motor, bank_fields, device)
    metrics = {
        "likelihood": likelihood,
        "generation": generation,
        "comparisons": comparisons,
        "feedback": feedback_report,
        "feedback_comparisons": feedback_comparisons,
        "motor": motor_report,
    }
    peak = (
        torch.cuda.max_memory_allocated(device) / 1024**3
        if device.type == "cuda"
        else 0
    )
    boundary = model.boundary_receipt()
    gates = qualification_gates(metrics, boundary, peak)
    report = {
        "architecture": ARCHITECTURE,
        "arm": arm,
        "decision": "retain" if all(gates.values()) else "reject",
        "checkpoint_sha256": file_sha256(root / arm / "checkpoint_final.pt"),
        "audit_seed": AUDIT_SEED,
        "evaluator_sha256": file_sha256(__file__),
        "shared_v54_evaluator_sha256": file_sha256(
            "scripts/eval_exact_raster_density_v54.py"
        ),
        "training_provenance": checkpoint["provenance"],
        "boundary": boundary,
        "training_boundary": checkpoint["boundary"],
        "manifest_sha256": checkpoint["manifest"]["sha256"],
        **metrics,
        "gates": gates,
        "elapsed_seconds": time.perf_counter() - started,
        "peak_allocated_vram_gib": peak,
        "evaluator_bank_size": len(bank),
        "evaluator_bank_created_after_generation": True,
        "frozen_partition_opened": False,
        "supplied_image_motor_is_not_language_evidence": True,
    }
    torch.save(likelihood_rows, out / "per_example_likelihood.pt")
    _atomic_json(
        {"identifiers": generated["identifiers"], "correct_by_route": correct},
        out / "per_example_identity.json",
    )
    for name, data in [
        ("generation", generated),
        ("feedback", feedback),
        ("motor", motor),
    ]:
        torch.save(data, out / f"{name}_image_evidence.pt")
        cells = torch.cat(
            (data["targets"][:16], *[v[:16] for v in data["outputs"].values()]), dim=1
        )
        pack_visual_cells(cells.flatten(0, 1), columns=cells.shape[1], gutter=2).save(
            out / f"{name}_gallery.png"
        )
    _atomic_json(report, out / "development_report.json")
    print(
        json.dumps(
            {
                "arm": arm,
                "decision": report["decision"],
                "gates": gates,
                "ordered_nll": likelihood["ordered"],
                "sample": generation["ordered"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return report, likelihood_rows, generated["identifiers"], correct


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=DEFAULT_OUT)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    root = Path(args.root)
    if (root / "matched_development_report.json").exists():
        raise FileExistsError("matched audit already exists")
    # Verify both arms completed before opening either development result.
    checkpoints = {
        arm: torch.load(
            root / arm / "checkpoint_final.pt", map_location="cpu", weights_only=False
        )
        for arm in ["frozen", "joint"]
    }
    for arm, checkpoint in checkpoints.items():
        if (
            checkpoint["architecture"] != ARCHITECTURE
            or checkpoint["arm"] != arm
            or checkpoint["smoke_only"]
            or checkpoint["update"] != 8000
            or checkpoint["initialization_sha256"] != SOURCE_SHA
        ):
            raise ValueError(
                "both registered production arms must finish before evaluation"
            )
    device = _resolve_device(args.device)
    outcomes = {
        arm: audit_arm(root, arm, checkpoint, device)
        for arm, checkpoint in checkpoints.items()
    }
    control, candidate = outcomes["frozen"], outcomes["joint"]
    if (
        control[1]["identifiers"] != candidate[1]["identifiers"]
        or control[2] != candidate[2]
    ):
        raise ValueError("matched audit windows differ")
    nll_gain = grouped_bootstrap(
        control[1]["nll"]["ordered"] - candidate[1]["nll"]["ordered"],
        candidate[1]["identifiers"],
    )
    identity_gain = grouped_bootstrap(
        np.array(candidate[3]["ordered"]).mean(1)
        - np.array(control[3]["ordered"]).mean(1),
        candidate[2],
    )
    qualifies = all(candidate[0]["gates"].values()) and nll_gain["ci95"][0] > 0
    report = {
        "architecture": ARCHITECTURE,
        "decision": "retain" if qualifies else "reject",
        "joint_nll_advantage_over_matched_frozen": nll_gain,
        "joint_identity_advantage_over_matched_frozen": identity_gain,
        "arm_reports": {
            arm: str(Path(arm) / "development/development_report.json")
            for arm in outcomes
        },
        "checkpoint_sha256": {
            arm: result[0]["checkpoint_sha256"] for arm, result in outcomes.items()
        },
        "frozen_partition_opened": False,
        "audit_seed": AUDIT_SEED,
    }
    _atomic_json(report, root / "matched_development_report.json")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()

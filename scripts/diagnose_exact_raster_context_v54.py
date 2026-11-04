#!/usr/bin/env python3
"""Read-only, post-result attribution of exact raster likelihood to context."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from ilm.visual_lm.canonical_glyph_language_data import (
    CanonicalGlyphAuditDataset,
    CanonicalGlyphRenderConfig,
    canonical_glyph_audit_collate,
)
from ilm.visual_lm.canonical_glyph_language_evaluation import _shuffle_prefix
from ilm.visual_lm.visual_cell_data import load_v25_records
from ilm.visual_lm.visual_cell_eval_data import (
    build_visual_cell_audit_windows,
    build_visual_character_statistics,
)
from scripts.eval_exact_raster_density_v54 import (
    AUDIT_SEED,
    grouped_bootstrap,
    load_student,
)
from scripts.train_canonical_glyph_binding_v43 import _atomic_json, _resolve_device
from scripts.train_exact_raster_density_v54 import DEFAULT_OUT
from scripts.train_visual_state_actuator import file_sha256, seed_everything

CHECKPOINT = "f2c3fab74981541243a963c8cffab6bbbd8c7f9d051512b228347db04a57a9ee"


def cross_record_donors(identifiers):
    count = len(identifiers)
    indices = torch.tensor(sorted(range(count), key=lambda i: identifiers[i]))
    donors = torch.empty_like(indices)
    donors[indices] = indices.roll(count // 2)
    if any(identifiers[i] == identifiers[j] for i, j in enumerate(donors.tolist())):
        raise ValueError("complete-condition donor is from the same source record")
    return donors


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=DEFAULT_OUT + "/checkpoint_final.pt")
    parser.add_argument("--out", default=DEFAULT_OUT + "/context_attribution")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()
    out = Path(args.out)
    if (out / "report.json").exists():
        raise FileExistsError("completed diagnostic is immutable")
    if file_sha256(args.checkpoint) != CHECKPOINT:
        raise ValueError("diagnostic requires the unchanged production checkpoint")
    out.mkdir(parents=True, exist_ok=True)
    device = _resolve_device(args.device)
    seed_everything(AUDIT_SEED)
    torch.set_float32_matmul_precision("high")
    start = time.perf_counter()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    for path, digest in checkpoint["provenance"].items():
        if file_sha256(path) != digest:
            raise ValueError(f"trained source changed: {path}")
    manifest = "data/visual_grammar/chinese_wikisource_public_domain.jsonl"
    if file_sha256(manifest) != checkpoint["manifest"]["sha256"]:
        raise ValueError("corpus changed")
    model = load_student(checkpoint, device)
    render = CanonicalGlyphRenderConfig(**checkpoint["render_config"])
    records = load_v25_records(manifest, strict_manifest=True)
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
        num_workers=args.num_workers,
        collate_fn=canonical_glyph_audit_collate,
    )
    chunks = {
        key: []
        for key in ("hidden", "fields", "shuffled_hidden", "shuffled_fields", "targets")
    }
    identifiers = []
    for raw in loader:
        context = raw["context"].to(device)
        hidden, fields = model.read(context)
        sh, sf = model.read(_shuffle_prefix(context, first_index=len(identifiers)))
        target = (raw["continuation"][:, 0] >= 0.5).float()
        for key, value in zip(chunks, (hidden, fields, sh, sf, target)):
            chunks[key].append(value.cpu())
        identifiers.extend(raw["identifier"])
    arrays = {key: torch.cat(value).to(device) for key, value in chunks.items()}
    donors = cross_record_donors(identifiers).to(device)
    hidden, fields = arrays["hidden"], arrays["fields"]
    targets = arrays["targets"]
    supplied = model.reader.field.encode_unit(targets)
    conditions = {
        "ordered": (hidden, fields),
        "shuffled_earlier": (arrays["shuffled_hidden"], arrays["shuffled_fields"]),
        "deranged_complete": (hidden[donors], fields[donors]),
        "zero": (torch.zeros_like(hidden), torch.zeros_like(fields)),
        "supplied_true": (torch.zeros_like(hidden), supplied),
        "supplied_deranged": (torch.zeros_like(hidden), supplied[donors]),
    }
    rows = {}
    for name, (h, f) in conditions.items():
        losses = []
        for offset in range(0, len(targets), 32):
            x = targets[offset : offset + 32]
            logits = model.logits(x, h[offset : offset + 32], f[offset : offset + 32])
            losses.append(
                F.binary_cross_entropy_with_logits(logits, x, reduction="none")[:, 0]
                .sum(-1)
                .cpu()
            )
        rows[name] = torch.cat(losses)
        if not torch.isfinite(rows[name]).all():
            raise FloatingPointError("nonfinite row NLL")
    original_path = (
        Path(args.checkpoint).parent / "development/per_example_likelihood.pt"
    )
    original = torch.load(original_path, map_location="cpu", weights_only=False)
    if original["identifiers"] != identifiers:
        raise ValueError("diagnostic source windows differ from the locked audit")
    reconstruction_errors = {}
    for name, primary in (
        ("ordered", "ordered"),
        ("shuffled_earlier", "shuffled"),
        ("zero", "zero"),
    ):
        error = (rows[name].sum(-1) - original["nll"][primary]).abs().max().item()
        reconstruction_errors[name] = error
        if error > 1e-4:
            raise ValueError("row decomposition does not reproduce locked NLL")
    summary = {
        name: {
            "nats_per_image": float(value.sum(-1).mean()),
            "mean_nats_by_row": value.mean(0).tolist(),
            "nats_by_eight_row_band": value.reshape(512, 4, 8).sum(-1).mean(0).tolist(),
        }
        for name, value in rows.items()
    }
    advantages = {}
    for name in ("shuffled_earlier", "deranged_complete"):
        delta = rows[name] - rows["ordered"]
        advantages[name] = {
            "total": grouped_bootstrap(delta.sum(-1), identifiers),
            "bands": [
                grouped_bootstrap(
                    delta[:, band * 8 : (band + 1) * 8].sum(-1), identifiers
                )
                for band in range(4)
            ],
        }
    report = {
        "post_result_exploratory": True,
        "original_decision_unchanged": "reject",
        "checkpoint_sha256": CHECKPOINT,
        "diagnostic_source_sha256": file_sha256(__file__),
        "diagnostic_plan_sha256": file_sha256(
            "references/v54_context_attribution_plan.md"
        ),
        "primary_likelihood_sha256": file_sha256(original_path),
        "examples": 512,
        "source_records": len(set(identifiers)),
        "same_source_donors": 0,
        "summary": summary,
        "ordered_advantages": advantages,
        "locked_nll_reconstruction_max_abs": reconstruction_errors,
        "elapsed_seconds": time.perf_counter() - start,
        "not_mutual_information_estimate": True,
        "frozen_partition_opened": False,
    }
    torch.save(
        {"rows": rows, "identifiers": identifiers, "donor_indices": donors.cpu()},
        out / "per_example_rows.pt",
    )
    _atomic_json(report, out / "report.json")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(7.25, 2.7), layout="constrained")
    for name, label, color in (
        ("ordered", "Ordered", "#28638b"),
        ("shuffled_earlier", "Earlier shuffled", "#939b9f"),
        ("deranged_complete", "Whole context swapped", "#ae593f"),
    ):
        axes[0].plot(
            np.arange(1, 33),
            summary[name]["mean_nats_by_row"],
            label=label,
            color=color,
        )
    axes[0].set(xlabel="Target image row", ylabel="Teacher-forced NLL (nats)")
    axes[0].legend(frameon=False, fontsize=7)
    for name, color in (
        ("shuffled_earlier", "#939b9f"),
        ("deranged_complete", "#ae593f"),
    ):
        axes[1].plot(
            np.arange(1, 33),
            np.array(summary[name]["mean_nats_by_row"])
            - np.array(summary["ordered"]["mean_nats_by_row"]),
            color=color,
        )
    axes[1].axhline(0, color="black", lw=0.5)
    axes[1].set(xlabel="Target image row", ylabel="Ordered NLL advantage (nats)")
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    for suffix in ("pdf", "png"):
        fig.savefig(out / f"context_attribution.{suffix}", dpi=200)
    plt.close(fig)
    print(
        json.dumps(
            {
                "summary": {k: v["nats_per_image"] for k, v in summary.items()},
                "advantages": {k: v["total"] for k, v in advantages.items()},
                "reconstruction": reconstruction_errors,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

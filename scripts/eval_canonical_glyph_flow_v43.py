#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from ilm.visual_lm.canonical_glyph_flow_v43 import (
    CanonicalGlyphFlowV43,
    canonical_glyph_flow_v43_boundary_receipt,
    canonical_glyph_flow_v43_config_from_payload,
)
from ilm.visual_lm.canonical_glyph_flow_v43_data import V43_ARCHITECTURE
from ilm.visual_lm.canonical_glyph_flow_v43_evaluation import (
    V43_AUDIT_SEED,
    canonical_glyph_flow_v43_boundary_is_clean,
    evaluate_v43_generated_fields,
)
from ilm.visual_lm.canonical_glyph_language import (
    canonical_glyph_language_config_from_payload,
)
from ilm.visual_lm.canonical_glyph_language_data import (
    CanonicalGlyphAuditDataset,
    CanonicalGlyphPairAuditDataset,
    CanonicalGlyphRenderConfig,
    canonical_glyph_audit_collate,
    canonical_glyph_pair_audit_collate,
    render_canonical_character_bank,
)
from ilm.visual_lm.canonical_glyph_language_evaluation import (
    canonical_language_gate_report,
    evaluate_canonical_language,
    evaluate_counterfactual_pairs,
)
from ilm.visual_lm.factorized_visual_context_data import (
    build_factorized_suffix_pairs,
)
from ilm.visual_lm.visual_cell_data import (
    load_v25_records,
    pack_visual_cells,
    verify_v25_manifest,
    visual_cell_partition_receipt,
)
from ilm.visual_lm.visual_cell_eval_data import (
    build_visual_cell_audit_windows,
    build_visual_character_statistics,
    visual_character_statistics_receipt,
)
from scripts.train_canonical_glyph_binding_v43 import (
    PROTOCOL_DOCUMENT,
    _atomic_json,
    _resolve_device,
)
from scripts.train_visual_state_actuator import (
    autocast_context,
    file_sha256,
    seed_everything,
)


DEFAULT_MANIFEST = "data/visual_grammar/chinese_wikisource_public_domain.jsonl"
DEFAULT_CHECKPOINT = (
    "artifacts/canonical_glyph_flow_v43_20260814/writer/checkpoint_final.pt"
)
DEFAULT_OUTPUT = "artifacts/canonical_glyph_flow_v43_20260814/development"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the V43 image-only reader and spatial flow writer."
    )
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--out", default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--precision", choices=("fp32", "fp16", "bf16"), default="bf16")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--windows", type=int, default=2_048)
    parser.add_argument("--bank-size", type=int, default=1_024)
    parser.add_argument("--pairs", type=int, default=512)
    parser.add_argument("--generated-examples", type=int, default=256)
    parser.add_argument("--allow-smoke", action="store_true")
    return parser.parse_args()


@torch.no_grad()
def _save_sample_sheet(
    model: CanonicalGlyphFlowV43,
    dataset: CanonicalGlyphAuditDataset,
    path: Path,
    *,
    device: torch.device,
    precision: str,
) -> None:
    count = min(16, len(dataset))
    context = torch.stack([dataset[index]["context"] for index in range(count)]).to(
        device
    )
    target = torch.stack(
        [dataset[index]["continuation"][0] for index in range(count)]
    ).to(device)
    generator = torch.Generator(device=device).manual_seed(V43_AUDIT_SEED + 77)
    with autocast_context(device, precision):
        generated, _ = model.sample_next(context, generator=generator)
    interleaved = torch.stack((target, generated), dim=1).flatten(0, 1)
    path.parent.mkdir(parents=True, exist_ok=True)
    pack_visual_cells(interleaved.cpu(), columns=8, gutter=2).save(path)


def main() -> None:
    args = parse_args()
    if (
        min(
            args.batch_size,
            args.windows,
            args.bank_size,
            args.pairs,
            args.generated_examples,
        )
        < 1
    ):
        raise ValueError("V43 audit sizes must be positive")
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    device = _resolve_device(args.device)
    seed_everything(V43_AUDIT_SEED)
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if (
        checkpoint.get("architecture") != V43_ARCHITECTURE
        or checkpoint.get("stage") != "writer"
    ):
        raise ValueError("audit checkpoint is not a V43 writer")
    if checkpoint.get("smoke_only") and not args.allow_smoke:
        raise PermissionError("smoke checkpoints require --allow-smoke")
    model = CanonicalGlyphFlowV43(
        canonical_glyph_language_config_from_payload(checkpoint["language_config"]),
        canonical_glyph_flow_v43_config_from_payload(checkpoint["v43_config"]),
    )
    model.load_state_dict(checkpoint["model"], strict=True)
    model.to(device).eval()
    render_config = CanonicalGlyphRenderConfig(**checkpoint["render_config"])

    strict_manifest = not checkpoint.get("smoke_only")
    manifest_receipt = verify_v25_manifest(args.manifest, strict=strict_manifest)
    if manifest_receipt["sha256"] != checkpoint["manifest"]["sha256"]:
        raise ValueError("V43 audit corpus differs from its training corpus")
    records = load_v25_records(args.manifest, strict_manifest=strict_manifest)
    statistics = build_visual_character_statistics(
        records,
        bank_size=args.bank_size,
        script_views_mode=render_config.script_views,
    )
    windows = build_visual_cell_audit_windows(
        records,
        statistics,
        count=args.windows,
        continuation_cells=16,
        seed=V43_AUDIT_SEED,
        script_views_mode=render_config.script_views,
    )
    audit_dataset = CanonicalGlyphAuditDataset(
        windows,
        statistics,
        render_config=render_config,
    )
    audit_loader = DataLoader(
        audit_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
        collate_fn=canonical_glyph_audit_collate,
    )
    pair_records = build_factorized_suffix_pairs(
        records,
        split="development",
        suffix_cells=4,
        count=args.pairs,
        seed=V43_AUDIT_SEED + 1,
        require_different_identifiers=True,
        allowed_targets=set(statistics.characters),
        script_views_mode=render_config.script_views,
    )
    pair_dataset = CanonicalGlyphPairAuditDataset(
        pair_records,
        render_config=render_config,
    )
    pair_loader = DataLoader(
        pair_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
        collate_fn=canonical_glyph_pair_audit_collate,
    )
    bank_images = render_canonical_character_bank(
        statistics,
        render_config=render_config,
    )

    started = time.perf_counter()
    training_peak = float(checkpoint.get("peak_allocated_vram_gib", 0.0))
    language = evaluate_canonical_language(
        model.language_model,
        audit_loader,
        statistics,
        bank_images,
        device=device,
        precision=args.precision,
        checkpoint_peak_vram_gib=training_peak,
    )
    pairs = evaluate_counterfactual_pairs(
        model.language_model,
        pair_loader,
        device=device,
        precision=args.precision,
    )
    generated = evaluate_v43_generated_fields(
        model,
        audit_loader,
        bank_images,
        device=device,
        precision=args.precision,
        maximum_examples=args.generated_examples,
    )
    boundary_clean = canonical_glyph_flow_v43_boundary_is_clean(model)
    gates = canonical_language_gate_report(
        language,
        pairs,
        generated,
        boundary_clean=boundary_clean,
    )
    _save_sample_sheet(
        model,
        audit_dataset,
        output / "target_generated_pairs.png",
        device=device,
        precision=args.precision,
    )
    elapsed = time.perf_counter() - started
    report: dict[str, Any] = {
        "experiment": "canonical-glyph-flow-v43-development",
        "claim_status": "passed" if all(gates.values()) else "rejected-or-partial",
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_sha256": file_sha256(args.checkpoint),
        "binding_checkpoint_sha256": checkpoint["binding_checkpoint_sha256"],
        "protocol": {
            "document": PROTOCOL_DOCUMENT,
            "sha256": file_sha256(PROTOCOL_DOCUMENT),
        },
        "manifest": manifest_receipt,
        "partition": visual_cell_partition_receipt(records),
        "statistics": visual_character_statistics_receipt(statistics),
        "language": language,
        "counterfactual_pairs": pairs,
        "generated": generated,
        "gates": gates,
        "boundary_clean": boundary_clean,
        "model_boundary": canonical_glyph_flow_v43_boundary_receipt(model),
        "elapsed_seconds": elapsed,
        "evaluator_peak_allocated_vram_gib": (
            torch.cuda.max_memory_allocated(device) / 1024**3
            if device.type == "cuda"
            else 0.0
        ),
        "frozen_partition_opened": False,
    }
    _atomic_json(report, output / "development_report.json")
    _atomic_json(
        {
            "claim_status": report["claim_status"],
            "gates": gates,
            "language": language,
            "counterfactual_pairs": pairs,
            "generated": generated,
            "checkpoint_sha256": report["checkpoint_sha256"],
        },
        output / "development_summary.json",
    )
    print(json.dumps(gates, sort_keys=True), flush=True)
    print(json.dumps(language, sort_keys=True), flush=True)
    print(json.dumps(pairs, sort_keys=True), flush=True)
    print(json.dumps(generated, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()

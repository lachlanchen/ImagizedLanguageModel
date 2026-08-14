#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Mapping

import torch
from torch.utils.data import DataLoader

from ilm.visual_lm.canonical_glyph_language import (
    CanonicalGlyphLanguageModel,
    canonical_glyph_language_boundary_receipt,
    canonical_glyph_language_config_from_payload,
)
from ilm.visual_lm.canonical_glyph_language_data import (
    V42_ARCHITECTURE,
    CanonicalGlyphAuditDataset,
    CanonicalGlyphPairAuditDataset,
    CanonicalGlyphRenderConfig,
    canonical_glyph_audit_collate,
    canonical_glyph_data_boundary_receipt,
    canonical_glyph_pair_audit_collate,
    render_canonical_character_bank,
)
from ilm.visual_lm.canonical_glyph_language_evaluation import (
    canonical_language_boundary_is_clean,
    canonical_language_gate_report,
    evaluate_canonical_language,
    evaluate_counterfactual_pairs,
    evaluate_generated_fields,
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
from scripts.train_visual_state_actuator import (
    autocast_context,
    choose_device,
    file_sha256,
    seed_everything,
)


DEFAULT_MANIFEST = "data/visual_grammar/chinese_wikisource_public_domain.jsonl"
DEFAULT_CHECKPOINT = (
    "artifacts/canonical_glyph_language_v42_20260814/checkpoint_final.pt"
)
DEFAULT_OUTPUT = "artifacts/canonical_glyph_language_v42_20260814/development"
AUDIT_SEED = 20264220


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the V42 canonical image-stream language core."
    )
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--out", default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--precision", choices=("fp32", "fp16", "bf16"), default="bf16"
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--windows", type=int, default=2_048)
    parser.add_argument("--bank-size", type=int, default=1_024)
    parser.add_argument("--pairs", type=int, default=512)
    parser.add_argument("--generated-examples", type=int, default=256)
    parser.add_argument("--samples", type=int, default=4)
    parser.add_argument("--noise-scale", type=float, default=1.0)
    parser.add_argument("--allow-smoke", action="store_true")
    return parser.parse_args()


def _atomic_json(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(path)


@torch.no_grad()
def _save_sample_sheet(
    model: CanonicalGlyphLanguageModel,
    dataset: CanonicalGlyphAuditDataset,
    path: Path,
    *,
    device: torch.device,
    precision: str,
    samples: int,
    noise_scale: float,
) -> None:
    count = min(16, len(dataset))
    context = torch.stack([dataset[index]["context"] for index in range(count)]).to(
        device
    )
    target = torch.stack(
        [dataset[index]["continuation"][0] for index in range(count)]
    ).to(device)
    generator = torch.Generator(device=device).manual_seed(AUDIT_SEED + 77)
    with autocast_context(device, precision):
        generated, _ = model.sample_next(
            context,
            samples=samples,
            generator=generator,
            noise_scale=noise_scale,
        )
    interleaved = torch.stack((target, generated), dim=1).flatten(0, 1)
    path.parent.mkdir(parents=True, exist_ok=True)
    pack_visual_cells(interleaved.cpu(), columns=8, gutter=2).save(path)


def main() -> None:
    args = parse_args()
    if min(
        args.batch_size,
        args.windows,
        args.bank_size,
        args.pairs,
        args.generated_examples,
        args.samples,
    ) < 1:
        raise ValueError("V42 audit sizes must be positive")
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    device = choose_device(args.device)
    if device.type == "cuda":
        cuda_index = device.index
        if cuda_index is None:
            cuda_index = torch.cuda.current_device()
        torch.cuda.set_device(cuda_index)
        device = torch.device("cuda", cuda_index)
    seed_everything(AUDIT_SEED)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if checkpoint.get("architecture") != V42_ARCHITECTURE:
        raise ValueError("audit checkpoint is not V42")
    if checkpoint.get("smoke_only") and not args.allow_smoke:
        raise PermissionError("smoke checkpoints require --allow-smoke")
    model = CanonicalGlyphLanguageModel(
        canonical_glyph_language_config_from_payload(checkpoint["model_config"])
    )
    model.load_state_dict(checkpoint["model"], strict=True)
    model.to(device).eval()
    render_config = CanonicalGlyphRenderConfig(**checkpoint["render_config"])

    strict_manifest = not checkpoint.get("exploratory") and not checkpoint.get(
        "smoke_only"
    )
    manifest_receipt = verify_v25_manifest(args.manifest, strict=strict_manifest)
    if manifest_receipt["sha256"] != checkpoint["manifest"]["sha256"]:
        raise ValueError("V42 audit corpus differs from its training corpus")
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
        seed=AUDIT_SEED,
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
        seed=AUDIT_SEED + 1,
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
    language = evaluate_canonical_language(
        model,
        audit_loader,
        statistics,
        bank_images,
        device=device,
        precision=args.precision,
        checkpoint_peak_vram_gib=float(
            checkpoint.get("peak_allocated_vram_gib", 0.0)
        ),
    )
    pairs = evaluate_counterfactual_pairs(
        model,
        pair_loader,
        device=device,
        precision=args.precision,
    )
    generated = evaluate_generated_fields(
        model,
        audit_loader,
        bank_images,
        device=device,
        precision=args.precision,
        maximum_examples=args.generated_examples,
        samples=args.samples,
        noise_scale=args.noise_scale,
    )
    boundary_clean = canonical_language_boundary_is_clean(model)
    gates = canonical_language_gate_report(
        language,
        pairs,
        generated,
        boundary_clean=boundary_clean,
    )
    elapsed = time.perf_counter() - started
    report = {
        "experiment": "canonical-glyph-language-v42-development-audit",
        "architecture": V42_ARCHITECTURE,
        "checkpoint": args.checkpoint,
        "checkpoint_sha256": file_sha256(args.checkpoint),
        "checkpoint_update": checkpoint["update"],
        "checkpoint_smoke_only": bool(checkpoint.get("smoke_only")),
        "checkpoint_exploratory": bool(checkpoint.get("exploratory")),
        "manifest": manifest_receipt,
        "partition": visual_cell_partition_receipt(records),
        "statistics": visual_character_statistics_receipt(statistics),
        "data_boundary": canonical_glyph_data_boundary_receipt(),
        "model_boundary": canonical_glyph_language_boundary_receipt(model),
        "language": language,
        "counterfactual_pairs": pairs,
        "generated": generated,
        "gates": gates,
        "all_gates_pass": all(gates.values()),
        "frozen_partition_opened": False,
        "audit_elapsed_seconds": elapsed,
        "effective_arguments": vars(args),
    }
    _atomic_json(report, output / "development_report.json")
    _atomic_json(
        {
            "all_gates_pass": report["all_gates_pass"],
            "checkpoint_update": checkpoint["update"],
            "full_top1": language["full_top1"],
            "unigram_top1": language["unigram_top1"],
            "bigram_top1": language["bigram_top1"],
            "full_minus_shuffled_top1": (
                language["full_top1"] - language["shuffled_top1"]
            ),
            "counterfactual_arm_accuracy": pairs["full_arm_accuracy"],
            "generated_identity_top1": generated["generated_identity_top1"],
            "generated_pixel_f1": generated["generated_pixel_f1"],
            "peak_allocated_vram_gib": language["peak_allocated_vram_gib"],
        },
        output / "development_summary.json",
    )
    _save_sample_sheet(
        model,
        audit_dataset,
        output / "target_generated_pairs.png",
        device=device,
        precision=args.precision,
        samples=args.samples,
        noise_scale=args.noise_scale,
    )
    print(json.dumps(report["gates"], sort_keys=True), flush=True)
    print(json.dumps(report["language"], sort_keys=True), flush=True)
    print(json.dumps(report["generated"], sort_keys=True), flush=True)


if __name__ == "__main__":
    main()

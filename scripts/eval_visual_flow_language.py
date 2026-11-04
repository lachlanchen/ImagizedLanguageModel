#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from ilm.visual_lm.canonical_glyph_flow_v43 import (
    canonical_glyph_flow_v43_config_from_payload,
)
from ilm.visual_lm.canonical_glyph_language import (
    canonical_glyph_language_config_from_payload,
)
from ilm.visual_lm.canonical_glyph_language_data import (
    CanonicalGlyphAuditDataset,
    CanonicalGlyphRenderConfig,
    canonical_glyph_audit_collate,
    render_canonical_character_bank,
)
from ilm.visual_lm.canonical_glyph_language_evaluation import (
    evaluate_canonical_language,
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
from ilm.visual_lm.visual_flow_language import (
    VISUAL_FLOW_LANGUAGE_ARCHITECTURE,
    VisualFlowLanguageModel,
    visual_flow_language_boundary_is_clean,
    visual_flow_language_boundary_receipt,
)
from ilm.visual_lm.visual_flow_language_evaluation import (
    VISUAL_FLOW_LANGUAGE_AUDIT_SEED,
    evaluate_visual_flow_generation,
)
from scripts.train_canonical_glyph_binding_v43 import (
    _atomic_json,
    _resolve_device,
)
from scripts.train_visual_state_actuator import (
    autocast_context,
    file_sha256,
    seed_everything,
)


DEFAULT_CHECKPOINT = "artifacts/visual_flow_language_20261104/checkpoint_final.pt"
DEFAULT_MANIFEST = "data/visual_grammar/chinese_wikisource_public_domain.jsonl"
DEFAULT_OUTPUT = "artifacts/visual_flow_language_20261104/development"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the image-native Visual Flow Language model."
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
    parser.add_argument("--windows", type=int, default=512)
    parser.add_argument("--bank-size", type=int, default=1_024)
    parser.add_argument("--generated-examples", type=int, default=256)
    parser.add_argument("--samples", type=int, default=4)
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--guidance-scale", type=float, default=1.25)
    parser.add_argument("--allow-smoke", action="store_true")
    return parser.parse_args()


@torch.no_grad()
def _save_sample_sheet(
    model: VisualFlowLanguageModel,
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
    generator = torch.Generator(device=device).manual_seed(
        VISUAL_FLOW_LANGUAGE_AUDIT_SEED + 91
    )
    with autocast_context(device, precision):
        generated, _ = model.sample_next(context, generator=generator)
    interleaved = torch.stack((target, generated), dim=1).flatten(0, 1)
    path.parent.mkdir(parents=True, exist_ok=True)
    pack_visual_cells(interleaved.cpu(), columns=8, gutter=2).save(path)


def main() -> None:
    args = parse_args()
    if min(
        args.batch_size,
        args.windows,
        args.bank_size,
        args.generated_examples,
        args.samples,
        args.steps,
    ) < 1:
        raise ValueError("visual-flow evaluation sizes must be positive")
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    device = _resolve_device(args.device)
    seed_everything(VISUAL_FLOW_LANGUAGE_AUDIT_SEED)
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if checkpoint.get("architecture") != VISUAL_FLOW_LANGUAGE_ARCHITECTURE:
        raise ValueError("evaluation checkpoint is not Visual Flow Language")
    if checkpoint.get("smoke_only") and not args.allow_smoke:
        raise PermissionError("smoke checkpoints require --allow-smoke")
    model = VisualFlowLanguageModel(
        canonical_glyph_language_config_from_payload(checkpoint["language_config"]),
        canonical_glyph_flow_v43_config_from_payload(checkpoint["flow_config"]),
    )
    model.load_state_dict(checkpoint["model"], strict=True)
    model.to(device).eval()
    render_config = CanonicalGlyphRenderConfig(**checkpoint["render_config"])

    strict_manifest = not checkpoint.get("smoke_only") and not checkpoint.get(
        "exploratory"
    )
    manifest = verify_v25_manifest(args.manifest, strict=strict_manifest)
    if manifest["sha256"] != checkpoint["manifest"]["sha256"]:
        raise ValueError("evaluation corpus differs from the training corpus")
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
        continuation_cells=4,
        seed=VISUAL_FLOW_LANGUAGE_AUDIT_SEED,
        script_views_mode=render_config.script_views,
    )
    dataset = CanonicalGlyphAuditDataset(
        windows,
        statistics,
        render_config=render_config,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
        collate_fn=canonical_glyph_audit_collate,
    )
    bank_images = render_canonical_character_bank(
        statistics,
        render_config=render_config,
    )

    started = time.perf_counter()
    language = evaluate_canonical_language(
        model.language_model,
        loader,
        statistics,
        bank_images,
        device=device,
        precision=args.precision,
        checkpoint_peak_vram_gib=float(
            checkpoint.get("peak_allocated_vram_gib", 0.0)
        ),
    )
    generated = evaluate_visual_flow_generation(
        model,
        loader,
        bank_images,
        device=device,
        precision=args.precision,
        maximum_examples=args.generated_examples,
        samples=args.samples,
        steps=args.steps,
        guidance_scale=args.guidance_scale,
    )
    boundary_clean = visual_flow_language_boundary_is_clean(model)
    _save_sample_sheet(
        model,
        dataset,
        output / "target_generated_pairs.png",
        device=device,
        precision=args.precision,
    )
    comparisons = {
        "selected_identity_gain_over_v43": (
            generated["ordered"]["selected_identity_top1"] - 0.04297
        ),
        "selected_pixel_f1_gain_over_v43": (
            generated["ordered"]["selected_pixel_f1"] - 0.44507
        ),
        "ordered_first_identity_gain_over_shuffled": (
            generated["ordered"]["first_identity_top1"]
            - generated["shuffled"]["first_identity_top1"]
        ),
        "ordered_selected_identity_gain_over_shuffled": (
            generated["ordered"]["selected_identity_top1"]
            - generated["shuffled"]["selected_identity_top1"]
        ),
    }
    report = {
        "experiment": "visual-flow-language-development",
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_sha256": file_sha256(args.checkpoint),
        "manifest": manifest,
        "partition": visual_cell_partition_receipt(records),
        "statistics": visual_character_statistics_receipt(statistics),
        "language": language,
        "generated": generated,
        "comparisons": comparisons,
        "boundary_clean": boundary_clean,
        "model_boundary": visual_flow_language_boundary_receipt(model),
        "elapsed_seconds": time.perf_counter() - started,
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
            "checkpoint_sha256": report["checkpoint_sha256"],
            "language": language,
            "generated": generated,
            "comparisons": comparisons,
            "boundary_clean": boundary_clean,
        },
        output / "development_summary.json",
    )
    print(json.dumps(language, sort_keys=True), flush=True)
    print(json.dumps(generated, sort_keys=True), flush=True)
    print(json.dumps(comparisons, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()

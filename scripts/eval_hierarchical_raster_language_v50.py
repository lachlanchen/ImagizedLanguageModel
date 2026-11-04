#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
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
    render_canonical_character_bank,
)
from ilm.visual_lm.canonical_glyph_language_evaluation import (
    _shuffle_prefix,
    evaluate_canonical_language,
)
from ilm.visual_lm.hierarchical_raster_language import (
    HIERARCHICAL_RASTER_LANGUAGE_ARCHITECTURE,
    HierarchicalRasterLanguageModel,
    hierarchical_raster_language_boundary_receipt,
    hierarchical_raster_language_config_from_payload,
)
from ilm.visual_lm.hierarchical_raster_language_evaluation import (
    V50_AUDIT_SEED,
    evaluate_raster_generation,
    evaluate_raster_likelihood,
    evaluate_visible_feedback,
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
    _atomic_json,
    _resolve_device,
)
from scripts.train_visual_state_actuator import (
    autocast_context,
    file_sha256,
    seed_everything,
)


DEFAULT_CHECKPOINT = (
    "artifacts/hierarchical_raster_language_v50_20261104/checkpoint_final.pt"
)
DEFAULT_MANIFEST = "data/visual_grammar/chinese_wikisource_public_domain.jsonl"
DEFAULT_OUTPUT = (
    "artifacts/hierarchical_raster_language_v50_20261104/development"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit V50 likelihood, visible generation, and feedback."
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
    parser.add_argument("--likelihood-examples", type=int, default=512)
    parser.add_argument("--feedback-examples", type=int, default=128)
    parser.add_argument("--samples", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--allow-smoke", action="store_true")
    return parser.parse_args()


def _boundary_clean(receipt: dict[str, object]) -> bool:
    prohibited = (
        "text_tokenizer",
        "unicode_or_character_ids_in_model",
        "ocr_in_model",
        "glyph_vocabulary_or_candidate_bank_in_model",
        "vector_quantizer_or_codebook",
        "external_model_at_runtime",
    )
    return all(receipt[key] is False for key in prohibited)


@torch.no_grad()
def _save_sample_sheet(
    model: HierarchicalRasterLanguageModel,
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
    shuffled = _shuffle_prefix(context, first_index=0)
    generator = torch.Generator(device=device).manual_seed(V50_AUDIT_SEED + 91)
    with autocast_context(device, precision):
        ordered, _ = model.sample_next(context, stochastic=False)
        shuffled_output, _ = model.sample_next(shuffled, stochastic=False)
        sampled, _ = model.sample_next(
            context,
            stochastic=True,
            generator=generator,
        )
    rows = torch.stack((target, ordered, shuffled_output, sampled), dim=1)
    path.parent.mkdir(parents=True, exist_ok=True)
    pack_visual_cells(rows.flatten(0, 1).cpu(), columns=8, gutter=2).save(path)


def main() -> None:
    args = parse_args()
    if min(
        args.batch_size,
        args.windows,
        args.bank_size,
        args.generated_examples,
        args.likelihood_examples,
        args.feedback_examples,
        args.samples,
    ) < 1:
        raise ValueError("V50 evaluation sizes must be positive")
    if args.temperature <= 0.0:
        raise ValueError("V50 temperature must be positive")
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    device = _resolve_device(args.device)
    seed_everything(V50_AUDIT_SEED)
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if checkpoint.get("architecture") != HIERARCHICAL_RASTER_LANGUAGE_ARCHITECTURE:
        raise ValueError("evaluation checkpoint is not V50")
    if checkpoint.get("smoke_only") and not args.allow_smoke:
        raise PermissionError("smoke checkpoints require --allow-smoke")
    model = HierarchicalRasterLanguageModel(
        canonical_glyph_language_config_from_payload(checkpoint["reader_config"]),
        hierarchical_raster_language_config_from_payload(
            checkpoint["writer_config"]
        ),
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
        seed=V50_AUDIT_SEED,
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
        model.reader,
        loader,
        statistics,
        bank_images,
        device=device,
        precision=args.precision,
        checkpoint_peak_vram_gib=float(
            checkpoint.get("peak_allocated_vram_gib", 0.0)
        ),
    )
    likelihood = evaluate_raster_likelihood(
        model,
        loader,
        device=device,
        precision=args.precision,
        maximum_examples=args.likelihood_examples,
    )
    generation = evaluate_raster_generation(
        model,
        loader,
        bank_images,
        device=device,
        precision=args.precision,
        maximum_examples=args.generated_examples,
        samples=args.samples,
        temperature=args.temperature,
    )
    feedback = evaluate_visible_feedback(
        model,
        loader,
        bank_images,
        device=device,
        precision=args.precision,
        maximum_examples=args.feedback_examples,
        steps=4,
    )
    _save_sample_sheet(
        model,
        dataset,
        output / "target_ordered_shuffled_sampled.png",
        device=device,
        precision=args.precision,
    )
    boundary = hierarchical_raster_language_boundary_receipt(model)
    comparisons = {
        "first_sample_identity_gain_over_v43": (
            generation["ordered"]["first_sample_identity_top1"] - 0.04297
        ),
        "first_sample_pixel_f1_gain_over_v43": (
            generation["ordered"]["first_sample_pixel_f1"] - 0.44507
        ),
        "ordered_identity_gain_over_shuffled": (
            generation["ordered"]["first_sample_identity_top1"]
            - generation["shuffled"]["first_sample_identity_top1"]
        ),
        "ordered_feedback_gain_over_shuffled": (
            feedback["ordered"]["mean_identity_top1"]
            - feedback["shuffled"]["mean_identity_top1"]
        ),
    }
    gates = {
        "coarse_counterfactual_accuracy": likelihood["ordered_better_coarse"]
        > 0.60,
        "coarse_nll_gain_positive": likelihood[
            "ordered_coarse_nll_gain_over_shuffled"
        ]
        > 0.0,
        "identity_beats_v43": comparisons["first_sample_identity_gain_over_v43"]
        > 0.0,
        "ordered_identity_margin": comparisons[
            "ordered_identity_gain_over_shuffled"
        ]
        >= 0.01,
        "pixel_f1_beats_v43": comparisons["first_sample_pixel_f1_gain_over_v43"]
        > 0.0,
        "nonblank": generation["ordered"]["self_selected_blank_rate"] < 0.02,
        "feedback_order_sensitive": comparisons[
            "ordered_feedback_gain_over_shuffled"
        ]
        > 0.0,
        "image_only_boundary": _boundary_clean(boundary),
    }
    decision = "retain" if all(gates.values()) else "reject-or-revise"
    report = {
        "experiment": "hierarchical-raster-language-v50-development",
        "decision": decision,
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_sha256": file_sha256(args.checkpoint),
        "manifest": manifest,
        "partition": visual_cell_partition_receipt(records),
        "statistics": visual_character_statistics_receipt(statistics),
        "language": language,
        "likelihood": likelihood,
        "generation": generation,
        "feedback": feedback,
        "comparisons": comparisons,
        "gates": gates,
        "boundary": boundary,
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
            "decision": decision,
            "checkpoint_sha256": report["checkpoint_sha256"],
            "language": language,
            "likelihood": likelihood,
            "generation": generation,
            "feedback": feedback,
            "comparisons": comparisons,
            "gates": gates,
        },
        output / "development_summary.json",
    )
    print(json.dumps(likelihood, sort_keys=True), flush=True)
    print(json.dumps(generation, sort_keys=True), flush=True)
    print(json.dumps(feedback, sort_keys=True), flush=True)
    print(json.dumps({"decision": decision, "gates": gates}, sort_keys=True))


if __name__ == "__main__":
    main()

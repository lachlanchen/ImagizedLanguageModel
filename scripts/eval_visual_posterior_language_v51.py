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
from ilm.visual_lm.canonical_glyph_language_evaluation import _shuffle_prefix
from ilm.visual_lm.hierarchical_raster_language import CausalRasterReader
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
from ilm.visual_lm.visual_posterior_language_v51 import (
    V51_ARCHITECTURE,
    VisualGlyphScorePrior,
    visual_posterior_language_boundary_receipt,
    visual_posterior_language_config_from_payload,
)
from ilm.visual_lm.visual_posterior_language_v51_evaluation import (
    V51_AUDIT_SEED,
    evaluate_posterior_feedback,
    evaluate_posterior_generation,
)
from scripts.train_canonical_glyph_binding_v43 import _atomic_json, _resolve_device
from scripts.train_visual_state_actuator import file_sha256, seed_everything


DEFAULT_CHECKPOINT = (
    "artifacts/visual_posterior_language_v51_20261104/checkpoint_final.pt"
)
DEFAULT_READER = "artifacts/canonical_glyph_language_v42_20260814/checkpoint_final.pt"
DEFAULT_MANIFEST = "data/visual_grammar/chinese_wikisource_public_domain.jsonl"
DEFAULT_OUTPUT = "artifacts/visual_posterior_language_v51_20261104/development"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit V51 posterior generation and visible feedback."
    )
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--reader-checkpoint", default=DEFAULT_READER)
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--out", default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--windows", type=int, default=512)
    parser.add_argument("--bank-size", type=int, default=1_024)
    parser.add_argument("--generated-examples", type=int, default=256)
    parser.add_argument("--feedback-examples", type=int, default=64)
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
        "prior_receives_sequence_or_character_labels",
    )
    return all(receipt[key] is False for key in prohibited)


@torch.no_grad()
def _save_sample_sheet(
    reader: CausalRasterReader,
    prior: VisualGlyphScorePrior,
    dataset: CanonicalGlyphAuditDataset,
    path: Path,
    *,
    device: torch.device,
) -> None:
    count = min(16, len(dataset))
    context = torch.stack([dataset[index]["context"] for index in range(count)]).to(
        device
    )
    target = torch.stack(
        [dataset[index]["continuation"][0] for index in range(count)]
    ).to(device)
    shuffled = _shuffle_prefix(context, first_index=0)
    ordered_language = reader.language(context)
    shuffled_language = reader.language(shuffled)
    ordered_anchor = ordered_language["anchor_fields"][:, -1].float()
    shuffled_anchor = shuffled_language["anchor_fields"][:, -1].float()
    ordered_plan = reader.field.binary(ordered_anchor)
    shuffled_plan = reader.field.binary(shuffled_anchor)
    ordered_generator = torch.Generator(device=device).manual_seed(V51_AUDIT_SEED)
    shuffled_generator = torch.Generator(device=device).manual_seed(V51_AUDIT_SEED)
    with torch.enable_grad():
        ordered, _ = prior.posterior_sample(
            ordered_plan,
            ordered_anchor,
            generator=ordered_generator,
        )
        shuffled_output, _ = prior.posterior_sample(
            shuffled_plan,
            shuffled_anchor,
            generator=shuffled_generator,
        )
    rows = torch.stack(
        (target, ordered_plan, ordered, shuffled_plan, shuffled_output), dim=1
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    # One audit example per row: target, ordered plan, ordered posterior,
    # shuffled plan, shuffled posterior.  Keeping the five semantic columns
    # intact makes identity loss visible without relying on evaluator scores.
    pack_visual_cells(rows.flatten(0, 1).cpu(), columns=5, gutter=2).save(path)


def main() -> None:
    args = parse_args()
    if min(
        args.batch_size,
        args.windows,
        args.bank_size,
        args.generated_examples,
        args.feedback_examples,
    ) < 1:
        raise ValueError("V51 evaluation sizes must be positive")
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    device = _resolve_device(args.device)
    seed_everything(V51_AUDIT_SEED)
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if checkpoint.get("architecture") != V51_ARCHITECTURE:
        raise ValueError("evaluation checkpoint is not V51")
    if checkpoint.get("smoke_only") and not args.allow_smoke:
        raise PermissionError("smoke checkpoints require --allow-smoke")
    reader_checkpoint = torch.load(
        args.reader_checkpoint, map_location="cpu", weights_only=False
    )
    if reader_checkpoint.get("architecture") != "canonical-glyph-language-v42":
        raise ValueError("V51 evaluation requires the V42 reader")
    reader_sha256 = file_sha256(args.reader_checkpoint)
    if reader_sha256 != checkpoint["reader_checkpoint_sha256"]:
        raise ValueError("V51 evaluation reader differs from the training receipt")

    reader_config = canonical_glyph_language_config_from_payload(
        reader_checkpoint["model_config"]
    )
    reader = CausalRasterReader(reader_config)
    reader.load_v42_state(reader_checkpoint["model"])
    reader.to(device).requires_grad_(False).eval()
    prior = VisualGlyphScorePrior(
        visual_posterior_language_config_from_payload(checkpoint["model_config"])
    )
    prior.load_state_dict(checkpoint["ema_model"], strict=True)
    prior.to(device).requires_grad_(False).eval()
    render_config = CanonicalGlyphRenderConfig(**checkpoint["render_config"])

    strict_manifest = not checkpoint.get("smoke_only") and not checkpoint.get(
        "exploratory"
    )
    manifest = verify_v25_manifest(args.manifest, strict=strict_manifest)
    if manifest["sha256"] != checkpoint["manifest"]["sha256"]:
        raise ValueError("evaluation corpus differs from prior training")
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
        seed=V51_AUDIT_SEED,
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
    generation = evaluate_posterior_generation(
        reader,
        prior,
        loader,
        bank_images,
        device=device,
        maximum_examples=args.generated_examples,
    )
    feedback = evaluate_posterior_feedback(
        reader,
        prior,
        loader,
        bank_images,
        device=device,
        maximum_examples=args.feedback_examples,
        steps=4,
    )
    _save_sample_sheet(
        reader,
        prior,
        dataset,
        output / "target_plan_posterior_shuffled.png",
        device=device,
    )
    boundary = visual_posterior_language_boundary_receipt(prior)
    boundary.update(
        {
            "causal_reader_parameters": sum(
                parameter.numel() for parameter in reader.parameters()
            ),
            "combined_parameters": sum(
                parameter.numel() for parameter in reader.parameters()
            )
            + sum(parameter.numel() for parameter in prior.parameters()),
            "reader_is_frozen": not any(
                parameter.requires_grad for parameter in reader.parameters()
            ),
        }
    )
    comparisons = {
        "ordered_posterior_identity_gain_over_shuffled": (
            generation["ordered"]["posterior_identity_top1"]
            - generation["shuffled"]["posterior_identity_top1"]
        ),
        "ordered_feedback_gain_over_shuffled": (
            feedback["ordered"]["mean_identity_top1"]
            - feedback["shuffled"]["mean_identity_top1"]
        ),
    }
    gates = {
        "image_only_boundary": _boundary_clean(boundary),
        "finite_nonblank_output": generation["ordered"]["posterior_blank_rate"]
        < 0.02,
        "identity_retention": generation["ordered"][
            "posterior_identity_retention_of_plan"
        ]
        >= 0.90,
        "f1_retention": generation["ordered"][
            "posterior_pixel_f1_delta_from_plan"
        ]
        >= -0.01,
        "ordered_identity_advantage": comparisons[
            "ordered_posterior_identity_gain_over_shuffled"
        ]
        > 0.0,
        "ordered_feedback_advantage": comparisons[
            "ordered_feedback_gain_over_shuffled"
        ]
        > 0.0,
        "isolated_ink_control": generation["ordered"][
            "posterior_isolated_ratio_to_target"
        ]
        <= 2.0,
        "bank_excluded_from_student": True,
    }
    decision = "retain" if all(gates.values()) else "reject"
    report = {
        "experiment": "visual-posterior-language-v51-development",
        "decision": decision,
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_sha256": file_sha256(args.checkpoint),
        "reader_checkpoint": str(Path(args.reader_checkpoint).resolve()),
        "reader_checkpoint_sha256": reader_sha256,
        "manifest": manifest,
        "partition": visual_cell_partition_receipt(records),
        "statistics": visual_character_statistics_receipt(statistics),
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
            "generation": generation,
            "feedback": feedback,
            "comparisons": comparisons,
            "gates": gates,
        },
        output / "development_summary.json",
    )
    print(json.dumps(generation, sort_keys=True), flush=True)
    print(json.dumps(feedback, sort_keys=True), flush=True)
    print(json.dumps({"decision": decision, "gates": gates}, sort_keys=True))


if __name__ == "__main__":
    main()

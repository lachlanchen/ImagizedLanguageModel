#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

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
from ilm.visual_lm.noisy_visual_energy_guidance_v52 import (
    V52_ARCHITECTURE,
    GuidedVisualLanguageSampler,
    NoisyVisualEnergyCritic,
    noisy_visual_energy_boundary_receipt,
    noisy_visual_energy_config_from_payload,
)
from ilm.visual_lm.noisy_visual_energy_guidance_v52_evaluation import (
    V52_AUDIT_SEED,
    evaluate_clean_visual_energy,
    evaluate_guided_feedback,
    evaluate_guided_generation,
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
from ilm.visual_lm.visual_posterior_language_v51 import (
    V51_ARCHITECTURE,
    VisualGlyphScorePrior,
    visual_posterior_language_config_from_payload,
)
from scripts.train_canonical_glyph_binding_v43 import _atomic_json, _resolve_device
from scripts.train_visual_state_actuator import file_sha256, seed_everything


DEFAULT_CHECKPOINT = (
    "artifacts/noisy_visual_energy_guidance_v52_20261104/checkpoint_final.pt"
)
DEFAULT_READER = "artifacts/canonical_glyph_language_v42_20260814/checkpoint_final.pt"
DEFAULT_PRIOR = (
    "artifacts/visual_posterior_language_v51_calibration_20261104/"
    "checkpoint_final.pt"
)
DEFAULT_MANIFEST = "data/visual_grammar/chinese_wikisource_public_domain.jsonl"
DEFAULT_OUTPUT = (
    "artifacts/noisy_visual_energy_guidance_v52_20261104/development"
)
FIXED_AUDIT = {
    "windows": 512,
    "bank_size": 1_024,
    "generated_examples": 256,
    "feedback_examples": 64,
}
EVALUATOR_SOURCE_FILES = (
    "ilm/visual_lm/noisy_visual_energy_guidance_v52_evaluation.py",
    "scripts/eval_noisy_visual_energy_guidance_v52.py",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the frozen V52 visual-energy development audit."
    )
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--reader-checkpoint", default=DEFAULT_READER)
    parser.add_argument("--prior-checkpoint", default=DEFAULT_PRIOR)
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--out", default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--windows", type=int, default=FIXED_AUDIT["windows"])
    parser.add_argument("--bank-size", type=int, default=FIXED_AUDIT["bank_size"])
    parser.add_argument(
        "--generated-examples",
        type=int,
        default=FIXED_AUDIT["generated_examples"],
    )
    parser.add_argument(
        "--feedback-examples",
        type=int,
        default=FIXED_AUDIT["feedback_examples"],
    )
    parser.add_argument(
        "--allow-smoke",
        action="store_true",
        help="Permit reduced audit sizes; the resulting decision is smoke-only.",
    )
    return parser.parse_args()


def _boundary_clean(receipt: dict[str, Any]) -> bool:
    prohibited = (
        "text_tokenizer",
        "unicode_or_character_ids_in_model",
        "ocr_in_model",
        "glyph_vocabulary_or_candidate_bank_in_model",
        "vector_quantizer_or_codebook",
        "external_model_at_runtime",
        "critic_receives_strings_or_labels",
    )
    return (
        all(receipt[key] is False for key in prohibited)
        and receipt["parameter_names_with_forbidden_fragments"] == []
    )


def _numeric_tree_is_finite(value: Any) -> bool:
    if isinstance(value, dict):
        return all(_numeric_tree_is_finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_numeric_tree_is_finite(item) for item in value)
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return True
    if isinstance(value, (int, float)):
        return math.isfinite(float(value))
    return True


def _source_hashes_unchanged(checkpoint: dict[str, Any]) -> bool:
    expected = checkpoint["protocol"]["source_files_sha256"]
    return all(
        Path(path).is_file() and file_sha256(path) == digest
        for path, digest in expected.items()
    ) and file_sha256(checkpoint["protocol"]["document"]) == checkpoint[
        "protocol"
    ]["sha256"]


@torch.no_grad()
def _save_sample_sheet(
    sampler: GuidedVisualLanguageSampler,
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
    routes = {
        "ordered": (context, None),
        "shuffled": (shuffled, None),
        "zero_guidance": (context, 0.0),
    }
    outputs: dict[str, torch.Tensor] = {}
    first_ordered: torch.Tensor | None = None
    for name, (visible, strength) in routes.items():
        generator = torch.Generator(device=device).manual_seed(V52_AUDIT_SEED)
        with torch.enable_grad():
            selected, trace = sampler.sample_next(
                visible,
                generator=generator,
                guidance_strength=strength,
            )
        outputs[name] = selected
        if name == "ordered":
            first_ordered = trace["first_samples"]
    if first_ordered is None:
        raise RuntimeError("V52 ordered gallery output was not produced")
    rows = torch.stack(
        (
            target,
            first_ordered,
            outputs["ordered"],
            outputs["shuffled"],
            outputs["zero_guidance"],
        ),
        dim=1,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    pack_visual_cells(rows.flatten(0, 1).cpu(), columns=5, gutter=2).save(path)


def main() -> None:
    args = parse_args()
    sizes = {
        "windows": args.windows,
        "bank_size": args.bank_size,
        "generated_examples": args.generated_examples,
        "feedback_examples": args.feedback_examples,
    }
    if min(args.batch_size, *sizes.values()) < 1:
        raise ValueError("V52 evaluation sizes must be positive")
    if not args.allow_smoke and sizes != FIXED_AUDIT:
        raise ValueError(f"production V52 audit requires {FIXED_AUDIT!r}")
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    device = _resolve_device(args.device)
    seed_everything(V52_AUDIT_SEED)
    torch.set_float32_matmul_precision("high")
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if checkpoint.get("architecture") != V52_ARCHITECTURE:
        raise ValueError("evaluation checkpoint is not V52")
    if (checkpoint.get("smoke_only") or checkpoint.get("exploratory")) and not (
        args.allow_smoke
    ):
        raise PermissionError("non-production checkpoints require --allow-smoke")
    reader_sha256 = file_sha256(args.reader_checkpoint)
    prior_sha256 = file_sha256(args.prior_checkpoint)
    if reader_sha256 != checkpoint["reader_checkpoint_sha256"]:
        raise ValueError("V52 evaluation reader differs from critic training")
    if prior_sha256 != checkpoint["prior_checkpoint_sha256"]:
        raise ValueError("V52 evaluation prior differs from critic training")
    reader_payload = torch.load(
        args.reader_checkpoint, map_location="cpu", weights_only=False
    )
    prior_payload = torch.load(
        args.prior_checkpoint, map_location="cpu", weights_only=False
    )
    if reader_payload.get("architecture") != "canonical-glyph-language-v42":
        raise ValueError("V52 evaluation requires the production V42 reader")
    if prior_payload.get("architecture") != V51_ARCHITECTURE:
        raise ValueError("V52 evaluation requires the measured V51 prior")

    reader = CausalRasterReader(
        canonical_glyph_language_config_from_payload(reader_payload["model_config"])
    )
    reader.load_v42_state(reader_payload["model"])
    reader.to(device).requires_grad_(False).eval()
    prior = VisualGlyphScorePrior(
        visual_posterior_language_config_from_payload(prior_payload["model_config"])
    )
    prior.load_state_dict(prior_payload["ema_model"], strict=True)
    prior.to(device).requires_grad_(False).eval()
    critic = NoisyVisualEnergyCritic(
        noisy_visual_energy_config_from_payload(checkpoint["model_config"])
    )
    critic.load_state_dict(checkpoint["ema_critic"], strict=True)
    critic.to(device).requires_grad_(False).eval()
    sampler = GuidedVisualLanguageSampler(reader, prior, critic).to(device).eval()
    render_config = CanonicalGlyphRenderConfig(**checkpoint["render_config"])

    strict_manifest = not args.allow_smoke
    manifest = verify_v25_manifest(args.manifest, strict=strict_manifest)
    if manifest["sha256"] != checkpoint["manifest"]["sha256"]:
        raise ValueError("V52 evaluation corpus differs from critic training")
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
        seed=V52_AUDIT_SEED,
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
    clean_energy = evaluate_clean_visual_energy(
        reader,
        critic,
        loader,
        bank_images,
        device=device,
        maximum_examples=args.windows,
    )
    generation = evaluate_guided_generation(
        sampler,
        loader,
        bank_images,
        device=device,
        maximum_examples=args.generated_examples,
    )
    feedback = evaluate_guided_feedback(
        sampler,
        loader,
        bank_images,
        device=device,
        maximum_examples=args.feedback_examples,
        steps=4,
    )
    _save_sample_sheet(
        sampler,
        dataset,
        output / "target_ordered_first_selected_shuffled_zero.png",
        device=device,
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    boundary = noisy_visual_energy_boundary_receipt(critic)
    reader_parameters = sum(parameter.numel() for parameter in reader.parameters())
    prior_parameters = sum(parameter.numel() for parameter in prior.parameters())
    boundary.update(
        {
            "causal_reader_parameters": reader_parameters,
            "glyph_prior_parameters": prior_parameters,
            "combined_parameters": (
                boundary["critic_parameters"]
                + reader_parameters
                + prior_parameters
            ),
            "reader_is_frozen": not any(
                parameter.requires_grad for parameter in reader.parameters()
            ),
            "prior_is_frozen": not any(
                parameter.requires_grad for parameter in prior.parameters()
            ),
            "critic_is_frozen_during_audit": not any(
                parameter.requires_grad for parameter in critic.parameters()
            ),
            "evaluator_bank_used_after_generation_only": True,
        }
    )
    comparisons = {
        "clean_ordered_gain_over_shuffled": (
            clean_energy["ordered"]["identity_top1"]
            - clean_energy["shuffled"]["identity_top1"]
        ),
        "ordered_first_gain_over_zero_guidance": (
            generation["ordered"]["first_identity_top1"]
            - generation["zero_guidance"]["first_identity_top1"]
        ),
        "ordered_selected_gain_over_shuffled": (
            generation["ordered"]["selected_identity_top1"]
            - generation["shuffled"]["selected_identity_top1"]
        ),
        "ordered_feedback_gain_over_shuffled": (
            feedback["ordered"]["mean_identity_top1"]
            - feedback["shuffled"]["mean_identity_top1"]
        ),
    }
    peak_vram_gib = (
        torch.cuda.max_memory_allocated(device) / 1024**3
        if device.type == "cuda"
        else 0.0
    )
    metric_tree = {
        "clean_energy": clean_energy,
        "generation": generation,
        "feedback": feedback,
        "comparisons": comparisons,
        "training_metrics": checkpoint["training_metrics"],
    }
    hash_integrity = (
        reader_sha256 == checkpoint["reader_checkpoint_sha256"]
        and prior_sha256 == checkpoint["prior_checkpoint_sha256"]
        and _source_hashes_unchanged(checkpoint)
    )
    ordered = generation["ordered"]
    gates = {
        "image_only_boundary": _boundary_clean(boundary),
        "parameter_budget": (
            boundary["critic_parameters"] <= 6_000_000
            and boundary["combined_parameters"] <= 27_000_000
        ),
        "clean_rank": clean_energy["ordered"]["identity_top1"] >= 0.15,
        "clean_order_advantage": comparisons[
            "clean_ordered_gain_over_shuffled"
        ]
        > 0.01,
        "first_generated_identity": ordered["first_identity_top1"] >= 0.05,
        "first_guidance_gain": comparisons[
            "ordered_first_gain_over_zero_guidance"
        ]
        > 0.02,
        "selected_generated_identity": (
            ordered["selected_identity_top1"] >= 0.08
            and comparisons["ordered_selected_gain_over_shuffled"] > 0.015
        ),
        "ordered_feedback_advantage": comparisons[
            "ordered_feedback_gain_over_shuffled"
        ]
        > 0.0,
        "surface_quality": (
            ordered["selected_blank_rate"] < 0.02
            and ordered["selected_isolated_ink_fraction"] <= 0.01
            and 0.70 <= ordered["selected_to_target_ink_ratio"] <= 1.50
        ),
        "identity_diversity": ordered["distinct_selected_identities"] >= 32,
        "finite_resource_and_hash_integrity": (
            _numeric_tree_is_finite(metric_tree)
            and peak_vram_gib < 18.0
            and hash_integrity
        ),
        "bank_excluded_from_model": boundary[
            "evaluator_bank_used_after_generation_only"
        ],
    }
    decision = "retain" if all(gates.values()) else "reject"
    if args.allow_smoke:
        decision = f"smoke-{decision}"
    report = {
        "experiment": "noisy-visual-energy-guidance-v52-development",
        "decision": decision,
        "audit_seed": V52_AUDIT_SEED,
        "audit_sizes": sizes,
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_sha256": file_sha256(args.checkpoint),
        "reader_checkpoint": str(Path(args.reader_checkpoint).resolve()),
        "reader_checkpoint_sha256": reader_sha256,
        "prior_checkpoint": str(Path(args.prior_checkpoint).resolve()),
        "prior_checkpoint_sha256": prior_sha256,
        "source_and_protocol_hashes_unchanged": _source_hashes_unchanged(
            checkpoint
        ),
        "evaluator_source_sha256": {
            path: file_sha256(path) for path in EVALUATOR_SOURCE_FILES
        },
        "manifest": manifest,
        "partition": visual_cell_partition_receipt(records),
        "statistics": visual_character_statistics_receipt(statistics),
        "clean_energy": clean_energy,
        "generation": generation,
        "feedback": feedback,
        "comparisons": comparisons,
        "gates": gates,
        "boundary": boundary,
        "elapsed_seconds": time.perf_counter() - started,
        "evaluator_peak_allocated_vram_gib": peak_vram_gib,
        "gallery_columns": [
            "target",
            "ordered first sample",
            "ordered critic-selected",
            "shuffled critic-selected",
            "zero-guidance critic-selected",
        ],
        "frozen_partition_opened": False,
    }
    _atomic_json(report, output / "development_report.json")
    _atomic_json(
        {
            "decision": decision,
            "checkpoint_sha256": report["checkpoint_sha256"],
            "clean_energy": clean_energy,
            "generation": generation,
            "feedback": feedback,
            "comparisons": comparisons,
            "gates": gates,
        },
        output / "development_summary.json",
    )
    print(json.dumps(clean_energy, sort_keys=True), flush=True)
    print(json.dumps(generation, sort_keys=True), flush=True)
    print(json.dumps(feedback, sort_keys=True), flush=True)
    print(json.dumps({"decision": decision, "gates": gates}, sort_keys=True))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import json
import time
from pathlib import Path
from typing import Any, Iterable

import torch
from torch.utils.data import DataLoader

from ilm.visual_lm.canonical_glyph_flow_v43 import (
    CanonicalGlyphFlowV43,
    canonical_glyph_flow_v43_config_from_payload,
)
from ilm.visual_lm.canonical_glyph_flow_v43_evaluation import V43_AUDIT_SEED
from ilm.visual_lm.canonical_glyph_language import (
    CanonicalGlyphLanguageModel,
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
    evaluate_counterfactual_pairs,
)
from ilm.visual_lm.factorized_visual_context_data import (
    FactorizedVisualSuffixPair,
    build_factorized_suffix_pairs,
)
from ilm.visual_lm.ink_writer import sample_foveal_ink
from ilm.visual_lm.visual_cell_data import (
    load_v25_records,
    verify_v25_manifest,
)
from ilm.visual_lm.visual_cell_eval_data import (
    build_visual_cell_audit_windows,
    build_visual_character_statistics,
)
from scripts.train_canonical_glyph_binding_v43 import (
    PINNED_V42_SHA256,
    _atomic_json,
    _resolve_device,
)
from scripts.train_visual_state_actuator import file_sha256, seed_everything


DEFAULT_MANIFEST = "data/visual_grammar/chinese_wikisource_public_domain.jsonl"
DEFAULT_V42 = "artifacts/canonical_glyph_language_v42_20260814/checkpoint_final.pt"
DEFAULT_BINDING = (
    "artifacts/canonical_glyph_flow_v43_20260814/binding/checkpoint_final.pt"
)
DEFAULT_V43 = "artifacts/canonical_glyph_flow_v43_20260814/writer/checkpoint_final.pt"
DEFAULT_OUTPUT = (
    "artifacts/canonical_glyph_flow_v43_20260814/diagnostic/diagnostic_report.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run post-result V43 binding and motor diagnostics."
    )
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--v42-checkpoint", default=DEFAULT_V42)
    parser.add_argument("--binding-checkpoint", default=DEFAULT_BINDING)
    parser.add_argument("--v43-checkpoint", default=DEFAULT_V43)
    parser.add_argument("--out", default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--precision", choices=("fp32", "fp16", "bf16"), default="bf16")
    parser.add_argument("--pair-examples", type=int, default=512)
    parser.add_argument("--writer-examples", type=int, default=256)
    parser.add_argument("--bank-size", type=int, default=1_024)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    return parser.parse_args()


def _autocast(
    device: torch.device,
    precision: str,
) -> contextlib.AbstractContextManager[Any]:
    if device.type != "cuda" or precision == "fp32":
        return contextlib.nullcontext()
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


def _pair_loader(
    pairs: tuple[FactorizedVisualSuffixPair, ...],
    *,
    render_config: CanonicalGlyphRenderConfig,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
) -> DataLoader[dict[str, Any]]:
    return DataLoader(
        CanonicalGlyphPairAuditDataset(pairs, render_config=render_config),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
        collate_fn=canonical_glyph_pair_audit_collate,
    )


def _pixel_f1(
    predicted: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    predicted_ink = predicted >= 0.5
    target_ink = target >= 0.5
    true_positive = (predicted_ink & target_ink).flatten(-3).sum(dim=-1).float()
    predicted_count = predicted_ink.flatten(-3).sum(dim=-1).clamp_min(1)
    target_count = target_ink.flatten(-3).sum(dim=-1).clamp_min(1)
    precision = true_positive / predicted_count
    recall = true_positive / target_count
    return 2.0 * precision * recall / (precision + recall).clamp_min(1e-8)


@torch.no_grad()
def _evaluate_direct_anchor(
    model: CanonicalGlyphLanguageModel,
    loader: Iterable[dict[str, Any]],
    bank_images: torch.Tensor,
    *,
    device: torch.device,
    precision: str,
    maximum_examples: int,
) -> dict[str, float]:
    model.eval()
    bank_fields = model.field.encode_unit(bank_images.to(device)).float()
    totals = {"f1": 0.0, "identity": 0.0, "density_ratio": 0.0}
    examples = 0
    for raw in loader:
        remaining = maximum_examples - examples
        if remaining <= 0:
            break
        context = raw["context"][:remaining].to(device, non_blocking=True)
        target = raw["continuation"][:remaining, 0].to(device, non_blocking=True)
        target_index = raw["target_index"][:remaining].to(device, non_blocking=True)
        with _autocast(device, precision):
            anchor = model(context)["anchor_fields"][:, -1]
        pixels = model.field.binary(anchor)
        predicted_fields = model.field.encode_unit(pixels).float()
        predicted_index = (predicted_fields @ bank_fields.transpose(0, 1)).argmax(1)
        predicted_density = (pixels >= 0.5).flatten(1).sum(1).float()
        target_density = (target >= 0.5).flatten(1).sum(1).float().clamp_min(1)
        totals["f1"] += float(_pixel_f1(pixels, target).sum())
        totals["identity"] += float((predicted_index == target_index).sum())
        totals["density_ratio"] += float((predicted_density / target_density).sum())
        examples += len(context)
    if examples < 1:
        raise ValueError("V43 anchor diagnostic received no examples")
    return {
        "examples": float(examples),
        "pixel_f1": totals["f1"] / examples,
        "identity_top1": totals["identity"] / examples,
        "ink_density_ratio": totals["density_ratio"] / examples,
    }


def _selection_metrics(
    candidates: torch.Tensor,
    candidate_fields: torch.Tensor,
    target: torch.Tensor,
    target_fields: torch.Tensor,
    target_index: torch.Tensor,
    bank_fields: torch.Tensor,
    selection: torch.Tensor,
) -> dict[str, float]:
    rows = torch.arange(len(candidates), device=candidates.device)
    pixels = candidates[rows, selection]
    fields = candidate_fields[rows, selection]
    predicted_index = (fields.float() @ bank_fields.transpose(0, 1)).argmax(1)
    predicted_density = (pixels >= 0.5).flatten(1).sum(1).float()
    target_density = (target >= 0.5).flatten(1).sum(1).float().clamp_min(1)
    return {
        "examples": float(len(candidates)),
        "pixel_f1_sum": float(_pixel_f1(pixels, target).sum()),
        "identity_correct": float((predicted_index == target_index).sum()),
        "target_cosine_sum": float((fields.float() * target_fields.float()).sum()),
        "ink_density_ratio_sum": float((predicted_density / target_density).sum()),
    }


def _accumulate(
    destination: dict[str, float],
    source: dict[str, float],
) -> None:
    for key, value in source.items():
        destination[key] = destination.get(key, 0.0) + value


@torch.no_grad()
def _evaluate_writer_plan(
    model: CanonicalGlyphFlowV43,
    loader: Iterable[dict[str, Any]],
    bank_images: torch.Tensor,
    *,
    device: torch.device,
    precision: str,
    maximum_examples: int,
    exact_target_plan: bool,
) -> dict[str, dict[str, float]]:
    model.eval()
    bank_fields = model.language_model.field.encode_unit(bank_images.to(device)).float()
    generator = torch.Generator(device=device).manual_seed(V43_AUDIT_SEED + 77)
    totals: dict[str, dict[str, float]] = {
        name: {} for name in ("runtime_anchor", "oracle_field", "oracle_pixel")
    }
    examples = 0
    sample_count = model.config.generated_candidates
    for raw in loader:
        remaining = maximum_examples - examples
        if remaining <= 0:
            break
        context = raw["context"][:remaining].to(device, non_blocking=True)
        target = raw["continuation"][:remaining, 0].to(device, non_blocking=True)
        target_index = raw["target_index"][:remaining].to(device, non_blocking=True)
        with _autocast(device, precision):
            hidden, anchor, predicted_plan = model.flow_inputs(context)
        target_fields = model.language_model.field.encode_unit(target).float()
        plan = target.float().mul(2.0).sub(1.0) if exact_target_plan else predicted_plan
        repeated_hidden = hidden[:, None].expand(-1, sample_count, -1).flatten(0, 1)
        repeated_plan = plan[:, None].expand(-1, sample_count, -1, -1, -1).flatten(0, 1)
        with _autocast(device, precision):
            signed = sample_foveal_ink(
                model.writer,
                repeated_hidden,
                repeated_plan,
                steps=model.config.flow_steps,
                guidance_scale=model.config.guidance_scale,
                generator=generator,
            )
        candidates = (
            (signed.add(1.0).div(2.0) >= 0.5)
            .to(target.dtype)
            .reshape(len(context), sample_count, 1, 32, 32)
        )
        fields = model.language_model.field.encode_unit(
            candidates.flatten(0, 1)
        ).reshape(len(context), sample_count, -1)
        runtime_selection = torch.einsum("bsd,bd->bs", fields, anchor.float()).argmax(1)
        oracle_field_selection = torch.einsum(
            "bsd,bd->bs", fields, target_fields
        ).argmax(1)
        candidate_f1 = _pixel_f1(candidates, target[:, None])
        oracle_pixel_selection = candidate_f1.argmax(1)
        for name, selection in (
            ("runtime_anchor", runtime_selection),
            ("oracle_field", oracle_field_selection),
            ("oracle_pixel", oracle_pixel_selection),
        ):
            _accumulate(
                totals[name],
                _selection_metrics(
                    candidates,
                    fields,
                    target,
                    target_fields,
                    target_index,
                    bank_fields,
                    selection,
                ),
            )
        examples += len(context)
    if examples < 1:
        raise ValueError("V43 writer diagnostic received no examples")
    result: dict[str, dict[str, float]] = {}
    for name, values in totals.items():
        count = values["examples"]
        result[name] = {
            "examples": count,
            "pixel_f1": values["pixel_f1_sum"] / count,
            "identity_top1": values["identity_correct"] / count,
            "target_cosine": values["target_cosine_sum"] / count,
            "ink_density_ratio": values["ink_density_ratio_sum"] / count,
        }
    return result


def main() -> None:
    args = parse_args()
    if (
        min(
            args.pair_examples,
            args.writer_examples,
            args.bank_size,
            args.batch_size,
        )
        < 1
    ):
        raise ValueError("V43 diagnostic sizes must be positive")
    device = _resolve_device(args.device)
    seed_everything(V43_AUDIT_SEED + 900)
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()

    v42_sha256 = file_sha256(args.v42_checkpoint)
    if v42_sha256 != PINNED_V42_SHA256:
        raise ValueError("V43 diagnostic requires the pinned V42 checkpoint")
    v42_payload = torch.load(
        args.v42_checkpoint, map_location="cpu", weights_only=False
    )
    v42 = CanonicalGlyphLanguageModel(
        canonical_glyph_language_config_from_payload(v42_payload["model_config"])
    )
    v42.load_state_dict(v42_payload["model"], strict=True)

    binding_sha256 = file_sha256(args.binding_checkpoint)
    binding = torch.load(
        args.binding_checkpoint, map_location="cpu", weights_only=False
    )
    v43_sha256 = file_sha256(args.v43_checkpoint)
    v43_payload = torch.load(
        args.v43_checkpoint, map_location="cpu", weights_only=False
    )
    if v43_payload.get("binding_checkpoint_sha256") != binding_sha256:
        raise ValueError("V43 diagnostic binding and writer checkpoints do not match")
    v43 = CanonicalGlyphFlowV43(
        canonical_glyph_language_config_from_payload(v43_payload["language_config"]),
        canonical_glyph_flow_v43_config_from_payload(v43_payload["v43_config"]),
    )
    v43.load_state_dict(v43_payload["model"], strict=True)
    v42.to(device).eval()
    v43.to(device).eval()

    manifest = verify_v25_manifest(args.manifest, strict=True)
    if manifest["sha256"] != v43_payload["manifest"]["sha256"]:
        raise ValueError("V43 diagnostic corpus differs from the writer corpus")
    records = load_v25_records(args.manifest, strict_manifest=True)
    render_config = CanonicalGlyphRenderConfig(**v43_payload["render_config"])
    pair_pool = int(binding["pair_pool"]["count"])
    pair_seed = int(binding["pair_pool"]["seed"])
    train_pairs = build_factorized_suffix_pairs(
        records,
        split="train",
        suffix_cells=4,
        count=pair_pool + args.pair_examples,
        seed=pair_seed,
        require_different_identifiers=True,
        script_views_mode=render_config.script_views,
    )
    seen_pairs = train_pairs[: args.pair_examples]
    unseen_train_pairs = train_pairs[pair_pool : pair_pool + args.pair_examples]
    development_pairs = build_factorized_suffix_pairs(
        records,
        split="development",
        suffix_cells=4,
        count=args.pair_examples,
        seed=V43_AUDIT_SEED + 1,
        require_different_identifiers=True,
        script_views_mode=render_config.script_views,
    )
    pair_sets = {
        "seen_train": seen_pairs,
        "unseen_train": unseen_train_pairs,
        "development": development_pairs,
    }
    pair_results: dict[str, dict[str, dict[str, float]]] = {}
    for name, pairs in pair_sets.items():
        pair_results[name] = {}
        for model_name, model in (
            ("v42", v42),
            ("v43", v43.language_model),
        ):
            loader = _pair_loader(
                pairs,
                render_config=render_config,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                pin_memory=device.type == "cuda",
            )
            pair_results[name][model_name] = evaluate_counterfactual_pairs(
                model,
                loader,
                device=device,
                precision=args.precision,
            )

    statistics = build_visual_character_statistics(
        records,
        bank_size=args.bank_size,
        script_views_mode=render_config.script_views,
    )
    windows = build_visual_cell_audit_windows(
        records,
        statistics,
        count=args.writer_examples,
        continuation_cells=16,
        seed=V43_AUDIT_SEED,
        script_views_mode=render_config.script_views,
    )
    audit_dataset = CanonicalGlyphAuditDataset(
        windows,
        statistics,
        render_config=render_config,
    )

    def audit_loader() -> DataLoader[dict[str, Any]]:
        return DataLoader(
            audit_dataset,
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
    direct_anchor = {
        "v42": _evaluate_direct_anchor(
            v42,
            audit_loader(),
            bank_images,
            device=device,
            precision=args.precision,
            maximum_examples=args.writer_examples,
        ),
        "v43": _evaluate_direct_anchor(
            v43.language_model,
            audit_loader(),
            bank_images,
            device=device,
            precision=args.precision,
            maximum_examples=args.writer_examples,
        ),
    }
    writer = {
        "predicted_plan": _evaluate_writer_plan(
            v43,
            audit_loader(),
            bank_images,
            device=device,
            precision=args.precision,
            maximum_examples=args.writer_examples,
            exact_target_plan=False,
        ),
        "evaluator_exact_target_plan": _evaluate_writer_plan(
            v43,
            audit_loader(),
            bank_images,
            device=device,
            precision=args.precision,
            maximum_examples=args.writer_examples,
            exact_target_plan=True,
        ),
    }
    report = {
        "experiment": "canonical-glyph-flow-v43-post-result-diagnostic",
        "claim_effect": "none; post-result diagnostics do not change V43 gates",
        "v42_checkpoint_sha256": v42_sha256,
        "binding_checkpoint_sha256": binding_sha256,
        "v43_checkpoint_sha256": v43_sha256,
        "manifest": manifest,
        "pair_pool": {
            "trained_pairs": pair_pool,
            "available_train_pairs": 215_340,
            "seen_sample": args.pair_examples,
            "unseen_train_sample": args.pair_examples,
            "development_sample": args.pair_examples,
        },
        "counterfactual_pairs": pair_results,
        "direct_anchor": direct_anchor,
        "writer": writer,
        "elapsed_seconds": time.perf_counter() - started,
        "peak_allocated_vram_gib": (
            torch.cuda.max_memory_allocated(device) / 1024**3
            if device.type == "cuda"
            else 0.0
        ),
        "evaluator_target_used_only_in_named_oracle_routes": True,
        "frozen_partition_opened": False,
    }
    output = Path(args.out)
    _atomic_json(report, output)
    print(
        json.dumps(
            {
                "counterfactual_full_arm_accuracy": {
                    name: {
                        model_name: metrics["full_arm_accuracy"]
                        for model_name, metrics in models.items()
                    }
                    for name, models in pair_results.items()
                },
                "direct_anchor": direct_anchor,
                "writer": writer,
                "report": str(output),
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()

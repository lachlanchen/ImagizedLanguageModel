#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import tempfile
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ilm.visual_lm.canonical_glyph_binding_v44_evaluation import V44_AUDIT_SEED
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
from ilm.visual_lm.factorized_visual_context_data import (
    build_factorized_suffix_pairs,
)
from ilm.visual_lm.noise_limited_retinal_field_v45 import (
    V45_ARCHITECTURE,
    V45_PROTOCOL,
    NoiseLimitedRetinalFieldV45,
    NoiseLimitedRetinalFieldV45Config,
    fit_noise_limited_retinal_field_v45,
    noise_limited_retinal_field_v45_boundary_is_clean,
    noise_limited_retinal_field_v45_boundary_receipt,
    noise_limited_retinal_field_v45_from_checkpoint_payload,
    noise_limited_retinal_field_v45_state_sha256,
)
from ilm.visual_lm.noise_limited_retinal_field_v45_evaluation import (
    V45_AUDIT_SEED,
    combine_roundtrip_metrics,
    field_geometry_metrics,
    finite_report,
    matrix_power_control_directions,
    noise_limited_retinal_field_v45_gate_report,
    pair_displacement_metrics,
    retrieval_metrics,
    roundtrip_metrics,
)
from ilm.visual_lm.visual_cell_data import (
    V25_DEVELOPMENT_FONTS,
    VisualCellRenderConfig,
    load_v25_records,
    render_visual_cell_stream,
    verify_v25_manifest,
    visual_cell_partition_receipt,
)
from ilm.visual_lm.visual_cell_eval_data import (
    build_visual_cell_audit_windows,
    build_visual_character_statistics,
)
from scripts.train_canonical_glyph_binding_v44 import (
    DEFAULT_BASE,
    DEFAULT_MANIFEST,
    PINNED_V42_SHA256,
    pair_sequence_receipt,
)
from scripts.train_visual_state_actuator import (
    autocast_context,
    file_sha256,
    seed_everything,
)


DEFAULT_V44 = "artifacts/canonical_glyph_binding_v44_20260814/checkpoint_final.pt"
DEFAULT_OUTPUT = "artifacts/noise_limited_retinal_field_v45_20260814"
PINNED_HOLDOUT_SHA256 = (
    "e41637c5e3846e47d19ade6312205cd30c2c96c65730bce75ee8ab4a0745154c"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit and run the preregistered V45 raster-field audit."
    )
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--v42-checkpoint", default=DEFAULT_BASE)
    parser.add_argument("--v44-checkpoint", default=DEFAULT_V44)
    parser.add_argument("--out", default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--precision", choices=("fp32", "fp16", "bf16"), default="bf16")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def _resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def _atomic_json(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        json.dump(payload, handle, ensure_ascii=True, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def _atomic_torch_save(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
    try:
        torch.save(dict(payload), temporary)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _sha256_json(payload: object) -> str:
    serialized = json.dumps(
        payload,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def _fit_sequence_receipt(
    characters: Sequence[str],
    counts: Sequence[int],
) -> dict[str, Any]:
    if len(characters) != len(counts):
        raise ValueError("V45 fit characters and counts must align")
    return {
        "count": len(characters),
        "characters_sha256": _sha256_json(list(characters)),
        "counts_sha256": _sha256_json(list(counts)),
        "joint_sha256": _sha256_json(
            [[character, count] for character, count in zip(characters, counts)]
        ),
        "retained_instances": int(sum(counts)),
    }


def _font_receipt(path: str) -> dict[str, Any]:
    value = Path(path)
    return {
        "path": str(value),
        "bytes": value.stat().st_size,
        "sha256": file_sha256(value),
    }


def _fixed_view_config() -> VisualCellRenderConfig:
    return VisualCellRenderConfig(
        cell_size=32,
        sequence_cells=65,
        minimum_font_size=26,
        maximum_font_size=26,
        augment=False,
        script_views="original+simplified",
    )


def _translate_one_pixel(
    cells: torch.Tensor,
    *,
    y: int,
    x: int,
) -> torch.Tensor:
    if (y, x) not in ((0, 1), (0, -1), (1, 0), (-1, 0)):
        raise ValueError("V45 audit permits exactly one cardinal-pixel shift")
    output = torch.zeros_like(cells)
    source_y0 = max(0, -y)
    source_y1 = cells.shape[-2] - max(0, y)
    source_x0 = max(0, -x)
    source_x1 = cells.shape[-1] - max(0, x)
    target_y0 = max(0, y)
    target_y1 = cells.shape[-2] - max(0, -y)
    target_x0 = max(0, x)
    target_x1 = cells.shape[-1] - max(0, -x)
    output[..., target_y0:target_y1, target_x0:target_x1] = cells[
        ..., source_y0:source_y1, source_x0:source_x1
    ]
    return output


def _field_directions(
    field: NoiseLimitedRetinalFieldV45,
    pixels: torch.Tensor,
    *,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    dct = field.dct.encode(pixels.to(device, non_blocking=True)).float()
    raw = F.normalize(dct, dim=-1)
    transformed = field.encode_dct(dct)
    return raw, transformed.direction, transformed.radius


def _held_pair_fields(
    field: NoiseLimitedRetinalFieldV45,
    pairs,
    *,
    render_config: CanonicalGlyphRenderConfig,
    device: torch.device,
    batch_size: int,
    num_workers: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    loader = DataLoader(
        CanonicalGlyphPairAuditDataset(pairs, render_config=render_config),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
        collate_fn=canonical_glyph_pair_audit_collate,
    )
    raw_fields = []
    transformed_fields = []
    for batch in loader:
        candidates = batch["candidates"].to(device, non_blocking=True)
        dct = field.dct.encode(candidates).float()
        raw_fields.append(F.normalize(dct, dim=-1))
        transformed_fields.append(field.encode_dct(dct).direction)
    return torch.cat(raw_fields), torch.cat(transformed_fields)


@torch.no_grad()
def _retrofit_v42_diagnostic(
    field: NoiseLimitedRetinalFieldV45,
    *,
    checkpoint_path: str,
    records,
    render_config: CanonicalGlyphRenderConfig,
    device: torch.device,
    precision: str,
    batch_size: int,
    num_workers: int,
    windows_count: int,
    bank_size: int,
) -> dict[str, Any]:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = CanonicalGlyphLanguageModel(
        canonical_glyph_language_config_from_payload(payload["model_config"])
    )
    model.load_state_dict(payload["model"], strict=True)
    model.requires_grad_(False).to(device).eval()
    statistics = build_visual_character_statistics(
        records,
        bank_size=bank_size,
        script_views_mode=render_config.script_views,
    )
    windows = build_visual_cell_audit_windows(
        records,
        statistics,
        count=windows_count,
        continuation_cells=16,
        seed=V44_AUDIT_SEED,
        script_views_mode=render_config.script_views,
    )
    loader = DataLoader(
        CanonicalGlyphAuditDataset(
            windows,
            statistics,
            render_config=render_config,
        ),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
        collate_fn=canonical_glyph_audit_collate,
    )
    bank_pixels = render_canonical_character_bank(
        statistics,
        render_config=render_config,
    ).to(device)
    bank_dct = field.dct.encode(bank_pixels).float()
    raw_bank = F.normalize(bank_dct, dim=-1)
    retinal_bank = field.encode_dct(bank_dct).direction
    totals = {
        "raw": {"examples": 0.0, "correct": 0.0, "log_probability": 0.0},
        "v45": {"examples": 0.0, "correct": 0.0, "log_probability": 0.0},
    }
    for batch in loader:
        context = batch["context"].to(device, non_blocking=True)
        targets = batch["target_index"].to(device, non_blocking=True)
        with autocast_context(device, precision):
            anchors = model.language(context)["anchor_fields"][:, -1]
        rows = torch.arange(len(context), device=device)
        variants = {
            "raw": (anchors.float(), raw_bank),
            "v45": (
                field.encode_dct(anchors.float() * math.sqrt(1024.0)).direction,
                retinal_bank,
            ),
        }
        for name, (query, bank) in variants.items():
            logits = model.contrastive_scale.float() * (
                query @ bank.transpose(0, 1)
            )
            target_log_probability = logits.log_softmax(dim=-1)[rows, targets]
            total = totals[name]
            total["examples"] += len(context)
            total["correct"] += float((logits.argmax(dim=-1) == targets).sum())
            total["log_probability"] += float(target_log_probability.sum())
    report = {}
    for name, total in totals.items():
        count = total["examples"]
        report[name] = {
            "examples": count,
            "top1": total["correct"] / count,
            "target_log_probability": total["log_probability"] / count,
        }
    del model
    return report


def main() -> None:
    args = parse_args()
    if args.batch_size < 1 or args.num_workers < 0:
        raise ValueError("V45 loader settings are invalid")
    seed_everything(V45_AUDIT_SEED)
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    device = _resolve_device(args.device)
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()

    config = (
        NoiseLimitedRetinalFieldV45Config(
            fit_bank_size=64,
            identity_bank_size=32,
        )
        if args.smoke
        else NoiseLimitedRetinalFieldV45Config()
    )
    windows_count = 32 if args.smoke else 2_048
    pair_count = 16 if args.smoke else 1_024
    manifest_receipt = verify_v25_manifest(args.manifest, strict=not args.smoke)
    if not args.smoke and manifest_receipt["sha256"] != (
        "76048753b52735d14c98f0d9e4eb8d751401fe8810e82eaaaebff517f6866c03"
    ):
        raise ValueError("V45 production manifest is not the preregistered corpus")
    records = load_v25_records(args.manifest, strict_manifest=not args.smoke)
    render_config = CanonicalGlyphRenderConfig()
    statistics = build_visual_character_statistics(
        records,
        bank_size=config.fit_bank_size,
        script_views_mode=render_config.script_views,
    )
    fit_receipt = _fit_sequence_receipt(statistics.characters, statistics.counts)
    fit_receipt["retained_fraction_of_han_instances"] = (
        fit_receipt["retained_instances"] / statistics.han_character_count
    )
    fit_pixels = render_canonical_character_bank(
        statistics,
        render_config=render_config,
    )
    field = fit_noise_limited_retinal_field_v45(
        fit_pixels,
        statistics.counts,
        config=config,
    )
    boundary = noise_limited_retinal_field_v45_boundary_receipt(field)
    boundary_clean = noise_limited_retinal_field_v45_boundary_is_clean(field)
    fit_font = _font_receipt(render_config.font_path)

    identity_characters = "".join(statistics.characters[: config.identity_bank_size])
    identity_pixels = fit_pixels[: config.identity_bank_size]
    view_config = _fixed_view_config()
    held_font_pixels = {
        Path(font).name: render_visual_cell_stream(
            identity_characters,
            config=view_config,
            font_path=font,
            variant=0,
        )
        for font in V25_DEVELOPMENT_FONTS
    }
    shifted_pixels = {
        "right": _translate_one_pixel(identity_pixels, y=0, x=1),
        "left": _translate_one_pixel(identity_pixels, y=0, x=-1),
        "down": _translate_one_pixel(identity_pixels, y=1, x=0),
        "up": _translate_one_pixel(identity_pixels, y=-1, x=0),
    }

    roundtrip_banks = {"fit": fit_pixels, **held_font_pixels, **shifted_pixels}
    roundtrips = {
        name: roundtrip_metrics(field, pixels)
        for name, pixels in roundtrip_banks.items()
    }
    roundtrip = combine_roundtrip_metrics(roundtrips)

    field.to(device).eval()
    fit_dct = field.dct.encode(fit_pixels.to(device, non_blocking=True)).float()
    raw_fit = F.normalize(fit_dct, dim=-1)
    retinal_fit = field.encode_dct(fit_dct)
    weights = torch.tensor(statistics.counts, device=device, dtype=torch.float32)
    raw_geometry = field_geometry_metrics(
        raw_fit,
        fit_dct.norm(dim=-1),
        weights=weights,
    )
    field_geometry = field_geometry_metrics(
        retinal_fit.direction,
        retinal_fit.radius,
        weights=weights,
    )
    center_direction, center_radius = matrix_power_control_directions(
        field,
        fit_dct,
        power=0.0,
    )
    full_direction, full_radius = matrix_power_control_directions(
        field,
        fit_dct,
        power=0.5,
    )
    controls = {
        "mean_center_only": field_geometry_metrics(
            center_direction,
            center_radius,
            weights=weights,
        ),
        "full_zca": field_geometry_metrics(
            full_direction,
            full_radius,
            weights=weights,
        ),
    }

    raw_identity = raw_fit[: config.identity_bank_size]
    retinal_identity = retinal_fit.direction[: config.identity_bank_size]
    held_fonts = {}
    for name, pixels in held_font_pixels.items():
        raw_query, retinal_query, _ = _field_directions(
            field,
            pixels,
            device=device,
        )
        held_fonts[name] = {
            "raw": retrieval_metrics(raw_query, raw_identity),
            "v45": retrieval_metrics(retinal_query, retinal_identity),
        }
    shifts = {}
    for name, pixels in shifted_pixels.items():
        raw_query, retinal_query, _ = _field_directions(
            field,
            pixels,
            device=device,
        )
        shifts[name] = {
            "raw": retrieval_metrics(raw_query, raw_identity),
            "v45": retrieval_metrics(retinal_query, retinal_identity),
        }

    v44_payload = torch.load(args.v44_checkpoint, map_location="cpu", weights_only=False)
    pool = v44_payload["pair_pool"]
    all_pairs = build_factorized_suffix_pairs(
        records,
        split="train",
        suffix_cells=int(pool["suffix_cells"]),
        count=int(pool["training"]["count"]) + int(pool["holdout"]["count"]),
        seed=int(pool["seed"]),
        require_different_identifiers=True,
        script_views_mode=render_config.script_views,
    )
    holdout_pairs = all_pairs[int(pool["training"]["count"]) :]
    holdout_receipt = pair_sequence_receipt(holdout_pairs)
    if holdout_receipt != pool["holdout"]:
        raise ValueError("V45 reconstructed V44 holdout receipt differs")
    if not args.smoke and holdout_receipt["sha256"] != PINNED_HOLDOUT_SHA256:
        raise ValueError("V45 pair holdout is not the preregistered receipt")
    pair_subset = holdout_pairs[:pair_count]
    raw_pair_fields, retinal_pair_fields = _held_pair_fields(
        field,
        pair_subset,
        render_config=render_config,
        device=device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    raw_pairs = pair_displacement_metrics(raw_pair_fields)
    field_pairs = pair_displacement_metrics(retinal_pair_fields)

    del fit_dct, raw_fit, retinal_fit, center_direction, full_direction
    del raw_pair_fields, retinal_pair_fields, v44_payload
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    v42_sha256 = file_sha256(args.v42_checkpoint)
    if not args.smoke and v42_sha256 != PINNED_V42_SHA256:
        raise ValueError("V45 retrofit diagnostic requires the pinned V42 checkpoint")
    retrofit = _retrofit_v42_diagnostic(
        field,
        checkpoint_path=args.v42_checkpoint,
        records=records,
        render_config=render_config,
        device=device,
        precision=args.precision,
        batch_size=min(args.batch_size, 32),
        num_workers=args.num_workers,
        windows_count=windows_count,
        bank_size=config.identity_bank_size,
    )
    elapsed = time.perf_counter() - started
    peak_vram = (
        torch.cuda.max_memory_allocated(device) / 1024**3
        if device.type == "cuda"
        else 0.0
    )
    gates = noise_limited_retinal_field_v45_gate_report(
        roundtrip=roundtrip,
        raw_geometry=raw_geometry,
        field_geometry=field_geometry,
        held_fonts=held_fonts,
        shifts=shifts,
        raw_pairs=raw_pairs,
        field_pairs=field_pairs,
        fit_boundary_clean=boundary_clean,
        frozen_partition_opened=False,
        peak_allocated_vram_gib=peak_vram,
        elapsed_seconds=elapsed,
    )
    if args.smoke:
        claim_status = "smoke-only"
    else:
        claim_status = (
            "qualified-retinal-field" if all(gates.values()) else "rejected-or-partial"
        )

    protocol_path = Path(V45_PROTOCOL)
    checkpoint_path = output / "field.pt"
    checkpoint = {
        "experiment": V45_ARCHITECTURE,
        "architecture": V45_ARCHITECTURE,
        "config": asdict(config),
        "field": field.cpu().state_dict(),
        "field_state_sha256": noise_limited_retinal_field_v45_state_sha256(
            field.cpu()
        ),
        "manifest": manifest_receipt,
        "partition": visual_cell_partition_receipt(records),
        "fit_sequence": fit_receipt,
        "fit_font": fit_font,
        "boundary": boundary,
        "protocol": {
            "document": V45_PROTOCOL,
            "sha256": file_sha256(protocol_path),
        },
        "v42_checkpoint_sha256": v42_sha256,
        "v44_checkpoint_sha256": file_sha256(args.v44_checkpoint),
        "holdout_pair_receipt": holdout_receipt,
        "smoke_only": args.smoke,
    }
    _atomic_torch_save(checkpoint, checkpoint_path)
    restored_payload = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    restored_field = noise_limited_retinal_field_v45_from_checkpoint_payload(
        restored_payload
    )
    checkpoint_reload_verified = (
        noise_limited_retinal_field_v45_state_sha256(restored_field)
        == checkpoint["field_state_sha256"]
    )
    if not checkpoint_reload_verified:
        raise RuntimeError("V45 checkpoint reload verification failed")
    report = {
        "experiment": V45_ARCHITECTURE,
        "claim_status": claim_status,
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "checkpoint_reload_verified": checkpoint_reload_verified,
        "field_state_sha256": checkpoint["field_state_sha256"],
        "config": asdict(config),
        "manifest": manifest_receipt,
        "fit_sequence": fit_receipt,
        "fit_font": fit_font,
        "development_fonts": [
            _font_receipt(font) for font in V25_DEVELOPMENT_FONTS
        ],
        "holdout_pair_receipt": holdout_receipt,
        "roundtrip_banks": roundtrips,
        "roundtrip": roundtrip,
        "raw_geometry": raw_geometry,
        "field_geometry": field_geometry,
        "descriptive_controls": controls,
        "held_fonts": held_fonts,
        "one_pixel_shifts": shifts,
        "raw_pair_geometry": raw_pairs,
        "field_pair_geometry": field_pairs,
        "frozen_v42_retrofit_report_only": retrofit,
        "gates": gates,
        "gates_passed": sum(gates.values()),
        "gates_total": len(gates),
        "boundary": boundary,
        "boundary_clean": boundary_clean,
        "frozen_partition_opened": False,
        "writer_opened": False,
        "peak_allocated_vram_gib": peak_vram,
        "elapsed_seconds": elapsed,
        "smoke_only": args.smoke,
    }
    if not finite_report(report):
        raise RuntimeError("V45 official report contains a non-finite value")
    _atomic_json(report, output / "report.json")
    print(json.dumps(report, ensure_ascii=True, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()

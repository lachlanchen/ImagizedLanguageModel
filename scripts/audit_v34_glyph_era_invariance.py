#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn.functional as F

from ilm.visual_lm.continuous_glyph_codec import (
    V34_ARCHITECTURE,
    ContinuousGlyphCodec,
    ContinuousGlyphCodecConfig,
)
from ilm.visual_lm.continuous_glyph_codec_data import (
    file_sha256,
    load_historic_glyph_records,
)
from ilm.visual_lm.glyph_era_invariance import (
    cross_era_pair_sha256,
    cross_era_retrieval_metrics,
    select_cross_era_glyph_pairs,
)


EXPERIMENT = "v34-cross-era-latent-audit"
DEFAULT_CHECKPOINT = (
    "artifacts/continuous_glyph_codec_v34_20260814/checkpoint_latest.pt"
)
EXPECTED_CHECKPOINT_SHA256 = (
    "a138c9cb3b0502e43d1227f689c020893d56b468742c32e1840e44d299662f33"
)
DEFAULT_DATABASE = (
    "/home/lachlan/ProjectsLFS/incoder/data/historic/etymology.sqlite3"
)
EXPECTED_DATABASE_SHA256 = (
    "c563e8587d7dcacf73704c0fb7816f6d830db11122e0a3da62678b3a7119f738"
)
DEFAULT_CACHE = "artifacts/cache/v34_historic_glyph_rasters_32.pt"
EXPECTED_MANIFEST_SHA256 = (
    "3c4064441563c88dffe0c36d42cce0c381bf8b401b764b87484edfb4aa7db99c"
)
DEFAULT_OUTPUT = (
    "artifacts/v34_cross_era_invariance_20260814/development_report.json"
)
SEED = 20_264_000
SOURCE_FILES = (
    "ilm/visual_lm/glyph_era_invariance.py",
    "scripts/audit_v34_glyph_era_invariance.py",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit frozen V34 latent invariance across historical stages."
    )
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--database", default=DEFAULT_DATABASE)
    parser.add_argument("--cache", default=DEFAULT_CACHE)
    parser.add_argument(
        "--split",
        choices=("train", "development", "sealed"),
        default="development",
    )
    parser.add_argument("--maximum-families", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", default=DEFAULT_OUTPUT)
    parser.add_argument("--allow-unverified", action="store_true")
    return parser.parse_args()


def atomic_write_json(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=path.name,
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        json.dump(dict(payload), handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def load_v34_model(path: Path, *, verified: bool) -> ContinuousGlyphCodec:
    digest = file_sha256(path)
    if verified and digest != EXPECTED_CHECKPOINT_SHA256:
        raise RuntimeError("V34 checkpoint hash changed")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, Mapping):
        raise TypeError("V34 checkpoint must contain a mapping")
    if checkpoint.get("architecture") != V34_ARCHITECTURE:
        raise ValueError("cross-era audit received another architecture")
    ema = checkpoint.get("ema")
    if not isinstance(ema, Mapping) or not isinstance(ema.get("shadow"), Mapping):
        raise ValueError("V34 checkpoint lacks its selected EMA state")
    model = ContinuousGlyphCodec(ContinuousGlyphCodecConfig())
    model.load_state_dict(ema["shadow"], strict=True)
    return model.requires_grad_(False).eval()


def encode_pixels(
    model: ContinuousGlyphCodec,
    pixels: torch.Tensor,
    *,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    if pixels.ndim != 4 or tuple(pixels.shape[1:]) != (1, 32, 32):
        raise ValueError("historic cache pixels have another geometry")
    values: list[torch.Tensor] = []
    model.to(device)
    with torch.inference_mode():
        for batch in pixels.split(batch_size):
            states = model.encode(batch.to(device=device, dtype=torch.float32))
            values.append(F.normalize(states.float(), dim=-1).cpu())
    return torch.cat(values)


def main() -> None:
    args = parse_args()
    if args.maximum_families < 0 or args.batch_size < 1:
        raise ValueError("cross-era audit arguments are invalid")
    started = time.perf_counter()
    verified = not args.allow_unverified
    checkpoint_path = Path(args.checkpoint)
    database_path = Path(args.database)
    cache_path = Path(args.cache)
    if verified and file_sha256(database_path) != EXPECTED_DATABASE_SHA256:
        raise RuntimeError("historical database hash changed")
    records = load_historic_glyph_records(database_path)
    cache = torch.load(cache_path, map_location="cpu", weights_only=False)
    if not isinstance(cache, Mapping) or not isinstance(cache.get("pixels"), torch.Tensor):
        raise TypeError("V34 historical cache is incomplete")
    expected_cache = {
        "database_sha256": file_sha256(database_path),
        "manifest_sha256": EXPECTED_MANIFEST_SHA256,
    }
    changed = [key for key, value in expected_cache.items() if cache.get(key) != value]
    if verified and changed:
        raise RuntimeError(f"V34 historical cache provenance changed: {changed}")
    pixels = cache["pixels"]
    if len(pixels) != len(records):
        raise ValueError("V34 historical cache and database do not align")
    pairs = select_cross_era_glyph_pairs(
        records,
        split=args.split,
        seed=args.seed,
        maximum_families=args.maximum_families,
    )
    anchor_indices = torch.tensor([pair.anchor_index for pair in pairs])
    positive_indices = torch.tensor([pair.positive_index for pair in pairs])
    model = load_v34_model(checkpoint_path, verified=verified)
    device = torch.device(args.device)
    anchor = encode_pixels(
        model,
        pixels[anchor_indices].float(),
        batch_size=args.batch_size,
        device=device,
    )
    positive = encode_pixels(
        model,
        pixels[positive_indices].float(),
        batch_size=args.batch_size,
        device=device,
    )
    report = {
        "experiment": EXPERIMENT,
        "status": "diagnostic-not-language-evidence",
        "split": args.split,
        "seed": args.seed,
        "maximum_families": args.maximum_families,
        "pair_sha256": cross_era_pair_sha256(pairs),
        "metrics": cross_era_retrieval_metrics(anchor, positive),
        "stage_pairs": {
            f"{left}->{right}": sum(
                pair.anchor_stage == left and pair.positive_stage == right
                for pair in pairs
            )
            for left, right in sorted(
                {(pair.anchor_stage, pair.positive_stage) for pair in pairs}
            )
        },
        "provenance": {
            "checkpoint": str(checkpoint_path.resolve()),
            "checkpoint_sha256": file_sha256(checkpoint_path),
            "checkpoint_route": "ema-shadow",
            "database": str(database_path.resolve()),
            "database_sha256": file_sha256(database_path),
            "cache": str(cache_path.resolve()),
            "cache_database_sha256": cache.get("database_sha256"),
            "cache_manifest_sha256": cache.get("manifest_sha256"),
            "source_sha256": {path: file_sha256(path) for path in SOURCE_FILES},
            "verified": verified,
        },
        "boundary": {
            "model_inputs": ["glyph_pixels"],
            "model_received_strings": False,
            "model_received_character_ids": False,
            "family_labels_used_by_host_evaluator": True,
            "runtime_claim": False,
        },
        "interpretation": (
            "Measures whether a frozen reconstruction codec preserves matching "
            "geometry across historical stages. It does not measure recognition, "
            "semantics, language generation, or zero-shot decipherment."
        ),
        "device": str(device),
        "elapsed_seconds": time.perf_counter() - started,
    }
    output = Path(args.out)
    atomic_write_json(report, output)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

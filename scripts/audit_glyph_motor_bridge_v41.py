#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from fontTools.ttLib import TTFont
from PIL import Image

from ilm.visual_lm.continuous_glyph_codec_data import file_sha256
from ilm.visual_lm.glyph_motor_bridge import (
    binary_ink_f1,
    load_mxfont_generator,
    load_qualified_v34_codec,
    load_unit_grayscale,
    mxfont_render,
    noise_condition_name,
    render_centered_glyph,
    unit_grayscale_to_pil,
    v34_project_source,
)


DEFAULT_MXFONT_ROOT = "artifacts/external/mxfont"
DEFAULT_V34_CHECKPOINT = "artifacts/continuous_glyph_codec_v34_20260814/checkpoint_latest.pt"
DEFAULT_OUTPUT = "artifacts/glyph_motor_bridge_v41_audit"
DEFAULT_CHARACTERS = "语言学中天地人心书法"
PINNED_SOURCE_FONT_SHA256 = (
    "995cd290eb86e8ae2a7fd2631824047444d2434b9de6f65727423d010ff5db1d"
)
PINNED_TARGET_FONT_SHA256 = (
    "2770d29ce713e03ea310c2874957da0172f21d53afec30fc398c533483eaa995"
)
PINNED_STYLE_REFERENCE_SHA256 = {
    "圄.png": "ff16087cf3ab596437893f0e2b4c7aec26b41e18dabff4cf6febd7d6109b9aa9",
    "檎.png": "044c22c2d8a5fc4ebd65e7d47307d18a97de4de03f005a13ffa92b3115312fdc",
    "泷.png": "25bc879da49ce8e6c2b626581097a65b959e1286010b732731dea364b32e0b29",
    "涠.png": "1a5dc8d086a35e0bc770b6f5fa6609fa9499f8e8ebcf3f0622c3b812cecb1b80",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit a real image-conditioned glyph motor behind V34 projections."
    )
    parser.add_argument("--mxfont-root", default=DEFAULT_MXFONT_ROOT)
    parser.add_argument("--v34-checkpoint", default=DEFAULT_V34_CHECKPOINT)
    parser.add_argument("--out", default=DEFAULT_OUTPUT)
    parser.add_argument("--characters", default=DEFAULT_CHARACTERS)
    parser.add_argument("--source-font")
    parser.add_argument("--target-font")
    parser.add_argument("--style-reference-dir")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--noise-sigmas", type=float, nargs="+", default=(0.0, 0.03, 0.05, 0.10))
    parser.add_argument("--seed", type=int, default=20_264_100)
    parser.add_argument("--skip-hash-verification", action="store_true")
    return parser.parse_args()


def validate_characters(value: str) -> tuple[str, ...]:
    glyphs = tuple(dict.fromkeys(value))
    if len(glyphs) < 2 or any(character.isspace() for character in glyphs):
        raise ValueError("V41 audit requires at least two unique non-space glyphs")
    return glyphs


def validate_font_support(path: Path, glyphs: tuple[str, ...]) -> None:
    font = TTFont(path)
    codepoints = {
        codepoint
        for table in font["cmap"].tables
        for codepoint in table.cmap
    }
    missing = [glyph for glyph in glyphs if ord(glyph) not in codepoints]
    if missing:
        raise ValueError(f"font {path} lacks requested glyphs: {''.join(missing)}")


def condition_metrics(
    source: torch.Tensor,
    output: torch.Tensor,
    target: torch.Tensor,
) -> dict[str, Any]:
    target_f1 = binary_ink_f1(output, target)
    source_f1 = binary_ink_f1(source, target)
    return {
        "target_ink_f1_mean": float(target_f1.mean()),
        "target_ink_f1_min": float(target_f1.min()),
        "source_before_motor_ink_f1_mean": float(source_f1.mean()),
        "motor_delta_ink_f1": float(target_f1.mean() - source_f1.mean()),
        "pixel_mae": float((output - target).abs().mean()),
        "nonblank_fraction": float((output < 0.5).flatten(1).any(dim=1).float().mean()),
    }


def evaluate_motor_gate(metrics: dict[str, dict[str, Any]]) -> dict[str, Any]:
    required = (
        "exact_source",
        "coarse_32",
        "v34_sigma_0p000",
        "v34_sigma_0p030",
        "v34_sigma_0p050",
        "v34_sigma_0p100",
    )
    missing = [name for name in required if name not in metrics]
    if missing:
        raise ValueError(f"V41 motor gate lacks required conditions: {missing}")
    clean_f1 = float(metrics["v34_sigma_0p000"]["target_ink_f1_mean"])
    if clean_f1 <= 0.0 or not math.isfinite(clean_f1):
        raise ValueError("V41 clean projected-motor F1 must be finite and positive")
    retention = {
        name: float(metrics[name]["target_ink_f1_mean"]) / clean_f1
        for name in ("v34_sigma_0p030", "v34_sigma_0p050")
    }
    checks = {
        "all_metrics_finite": all(
            math.isfinite(float(value))
            for name in required
            for value in metrics[name].values()
        ),
        "all_outputs_nonblank": all(
            float(metrics[name]["nonblank_fraction"]) >= 1.0
            for name in required
        ),
        "motor_improves_every_condition": all(
            float(metrics[name]["motor_delta_ink_f1"]) > 0.0
            for name in required
        ),
        "sigma_0p03_retention_at_least_0p90": retention["v34_sigma_0p030"]
        >= 0.90,
        "sigma_0p05_retention_at_least_0p90": retention["v34_sigma_0p050"]
        >= 0.90,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "noise_retention": retention,
        "minimum_noise_retention": 0.90,
    }


def verify_pinned_visual_assets(
    source_font: Path,
    target_font: Path,
    reference_paths: list[Path],
) -> None:
    source_hash = file_sha256(source_font)
    target_hash = file_sha256(target_font)
    reference_hashes = {
        path.name: file_sha256(path)
        for path in reference_paths
    }
    if source_hash != PINNED_SOURCE_FONT_SHA256:
        raise ValueError(f"unexpected V41 source-font SHA-256: {source_hash}")
    if target_hash != PINNED_TARGET_FONT_SHA256:
        raise ValueError(f"unexpected V41 target-font SHA-256: {target_hash}")
    if reference_hashes != PINNED_STYLE_REFERENCE_SHA256:
        raise ValueError(
            f"unexpected V41 style-reference SHA-256 values: {reference_hashes}"
        )


def save_contact_sheet(
    output_path: Path,
    rows: list[tuple[str, torch.Tensor]],
) -> dict[str, int]:
    if not rows:
        raise ValueError("V41 contact sheet requires at least one row")
    batch = len(rows[0][1])
    if any(len(values) != batch for _name, values in rows):
        raise ValueError("V41 contact-sheet rows do not align")
    cell = 128
    canvas = Image.new("L", (cell * batch, cell * len(rows)), 255)
    for row, (_name, values) in enumerate(rows):
        for column, value in enumerate(values):
            canvas.paste(unit_grayscale_to_pil(value), (cell * column, cell * row))
    canvas.save(output_path)
    return {"rows": len(rows), "columns": batch, "cell_pixels": cell}


def main() -> None:
    args = parse_args()
    glyphs = validate_characters(args.characters)
    root = Path(args.mxfont_root).expanduser().resolve()
    source_font = Path(args.source_font or root / "data/ttfs/val/MaShanZheng-Regular.ttf").resolve()
    target_font = Path(args.target_font or root / "data/ttfs/val/ZCOOLKuaiLe-Regular.ttf").resolve()
    reference_dir = Path(
        args.style_reference_dir or root / "data/images/test/ZCOOLKuaiLe-Regular"
    ).resolve()
    output = Path(args.out).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    validate_font_support(source_font, glyphs)
    validate_font_support(target_font, glyphs)
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision("highest")
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("V41 requested CUDA but CUDA is unavailable")
        torch.cuda.set_device(device)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.cuda.reset_peak_memory_stats()
    if any(sigma < 0.0 for sigma in args.noise_sigmas) or args.seed < 0:
        raise ValueError("V41 noise settings are invalid")

    started = time.monotonic()
    mxfont, mxfont_receipt = load_mxfont_generator(
        root,
        device=device,
        verify_hash=not args.skip_hash_verification,
    )
    codec, codec_receipt = load_qualified_v34_codec(
        args.v34_checkpoint,
        device=device,
        verify_hash=not args.skip_hash_verification,
    )
    sources = torch.stack(
        [render_centered_glyph(source_font, glyph) for glyph in glyphs]
    ).to(device)
    targets = torch.stack(
        [render_centered_glyph(target_font, glyph) for glyph in glyphs]
    ).to(device)
    reference_paths = sorted(reference_dir.glob("*.png"))
    if len(reference_paths) < 1:
        raise FileNotFoundError("V41 found no style reference PNGs")
    if not args.skip_hash_verification:
        verify_pinned_visual_assets(source_font, target_font, reference_paths)
    references = torch.stack(
        [load_unit_grayscale(path) for path in reference_paths]
    ).to(device)

    coarse = F.interpolate(sources, size=(32, 32), mode="area")
    coarse = F.interpolate(
        coarse,
        size=(128, 128),
        mode="bilinear",
        align_corners=False,
    )
    conditions: dict[str, torch.Tensor] = {
        "exact_source": sources,
        "coarse_32": coarse,
    }
    latent_norms: dict[str, float] = {}
    for sigma in dict.fromkeys(float(value) for value in args.noise_sigmas):
        name = noise_condition_name(sigma)
        projected, latent = v34_project_source(
            codec,
            sources,
            latent_noise_sigma=sigma,
            seed=args.seed + round(sigma * 1_000_000),
        )
        conditions[name] = projected
        latent_norms[name] = float(latent.float().norm(dim=1).mean())

    rows: list[tuple[str, torch.Tensor]] = [
        ("source", sources.cpu()),
        ("target", targets.cpu()),
    ]
    for name, values in (("source", sources), ("target", targets)):
        directory = output / name
        directory.mkdir(parents=True, exist_ok=True)
        for glyph, pixels in zip(glyphs, values):
            unit_grayscale_to_pil(pixels).save(directory / f"u{ord(glyph):04x}.png")
    metrics: dict[str, Any] = {}
    with torch.inference_mode():
        for name, source in conditions.items():
            generated = mxfont_render(mxfont, source, references)
            metrics[name] = condition_metrics(source, generated, targets)
            rows.append((name, generated.cpu()))
            condition_dir = output / name
            condition_dir.mkdir(parents=True, exist_ok=True)
            for glyph, pixels in zip(glyphs, generated):
                unit_grayscale_to_pil(pixels).save(
                    condition_dir / f"u{ord(glyph):04x}.png"
                )

    contact_sheet = save_contact_sheet(output / "contact_sheet.png", rows)
    finite = all(
        all(
            isinstance(value, (int, float)) and math.isfinite(float(value))
            for value in values.values()
        )
        for values in metrics.values()
    ) and all(
        torch.isfinite(value).all()
        for value in (
            *conditions.values(),
            *(values for _name, values in rows),
            sources,
            targets,
            references,
        )
    )
    gate = evaluate_motor_gate(metrics)
    receipt = {
        "experiment": "image-conditioned-glyph-motor-bridge-v41-audit",
        "claim_status": "mechanism-audit-only",
        "glyphs": list(glyphs),
        "model_input": "glyph raster plus style-reference rasters",
        "model_output": "glyph raster",
        "host_only_labels": True,
        "uses_token_ids": False,
        "uses_unicode_ids_in_model": False,
        "uses_ocr": False,
        "uses_retrieval": False,
        "mxfont": mxfont_receipt,
        "v34": codec_receipt,
        "fonts": {
            "source": str(source_font),
            "source_sha256": file_sha256(source_font),
            "target": str(target_font),
            "target_sha256": file_sha256(target_font),
            "style_reference_dir": str(reference_dir),
            "style_reference_sha256": {
                path.name: file_sha256(path) for path in reference_paths
            },
        },
        "noise_sigmas": list(dict.fromkeys(float(value) for value in args.noise_sigmas)),
        "latent_norms": latent_norms,
        "metrics": metrics,
        "motor_gate": gate,
        "contact_sheet": contact_sheet,
        "resources": {
            "device": str(device),
            "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
            "peak_vram_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0,
            "elapsed_seconds": time.monotonic() - started,
        },
        "determinism": {
            "seed": args.seed,
            "deterministic_algorithms": True,
            "cudnn_benchmark": False,
            "allow_tf32": False,
        },
        "finite": bool(finite),
    }
    receipt_path = output / "receipt.json"
    receipt_path.write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(receipt, ensure_ascii=False, indent=2))
    if not gate["passed"]:
        raise RuntimeError("V41 motor gate failed; inspect the written receipt")


if __name__ == "__main__":
    main()

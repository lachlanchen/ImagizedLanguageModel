#!/usr/bin/env python3
"""Independent image-prompt continuation using raw-pixel beam decoding."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import torch

from ilm.visual_lm.canonical_glyph_language_data import CanonicalGlyphRenderConfig
from ilm.visual_lm.joint_exact_raster_v55 import ARCHITECTURE, model_from_checkpoint
from ilm.visual_lm.raster_beam import raster_beam_search
from ilm.visual_lm.visual_cell_data import pack_visual_cells, render_visual_cell_stream
from scripts.infer_exact_raster_density_v54 import load_aligned_image
from scripts.train_canonical_glyph_binding_v43 import (
    _atomic_json,
    _atomic_save,
    _resolve_device,
)
from scripts.train_visual_state_actuator import file_sha256


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        default="artifacts/joint_exact_raster_v55_20261105/joint/checkpoint_final.pt",
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--prompt-image")
    source.add_argument(
        "--prompt-text", help="Optional host rasterization; no strings enter the model"
    )
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--cells", type=int, default=8)
    parser.add_argument("--beam-width", type=int, default=8)
    parser.add_argument(
        "--copy",
        action="store_true",
        help="Supplied-image motor control, not autonomous language",
    )
    parser.add_argument("--export")
    args = parser.parse_args()
    if args.cells < 1 or args.beam_width < 1:
        raise ValueError("output length and beam width must be positive")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if checkpoint["architecture"] != ARCHITECTURE or checkpoint.get(
        "smoke_only", False
    ):
        raise ValueError("production V55-compatible checkpoint required")
    device = _resolve_device(args.device)
    model = model_from_checkpoint(checkpoint, device)
    torch.set_float32_matmul_precision("highest")
    if args.prompt_image:
        images = load_aligned_image(args.prompt_image)
    else:
        render = CanonicalGlyphRenderConfig(**checkpoint.get("render_config", {}))
        images = render_visual_cell_stream(
            args.prompt_text,
            config=render.visual_cell_config(),
            font_path=render.font_path,
            variant=0,
        )
    if not len(images):
        raise ValueError("empty image prompt")
    probabilities = []
    if args.copy:
        observed = images.to(device)
        fields = model.reader.field.encode_unit(observed)
        result = raster_beam_search(
            model,
            torch.zeros(len(observed), 384, device=device),
            fields,
            beam_width=args.beam_width,
        )
        generated = result.images.cpu()
        probabilities = result.log_probability.cpu().tolist()
    else:
        context = images[-64:][None].to(device)
        generated = []
        for _ in range(args.cells):
            hidden, fields = model.read(context)
            result = raster_beam_search(
                model, hidden, fields, beam_width=args.beam_width
            )
            generated.append(result.images[0].cpu())
            probabilities.append(float(result.log_probability[0]))
            context = torch.cat((context, result.images[:, None]), dim=1)[:, -64:]
        generated = torch.stack(generated)
    destination = Path(args.out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    pack_visual_cells(generated, columns=min(32, len(generated)), gutter=0).save(
        destination
    )
    pack_visual_cells(images, columns=min(32, len(images)), gutter=0).save(
        destination.with_name(destination.stem + "_prompt.png")
    )
    receipt = {
        "architecture": ARCHITECTURE,
        "decoder": "raw_pixel_beam",
        "beam_width": args.beam_width,
        "float32_matmul_precision": "highest",
        "checkpoint_sha256": file_sha256(args.checkpoint),
        "decoder_sha256": file_sha256("ilm/visual_lm/raster_beam.py"),
        "mode": "supplied_image_copy" if args.copy else "autonomous_image_continuation",
        "generated_cells": len(generated),
        "image_log_probabilities": probabilities,
        "tokenizer": False,
        "ocr": False,
        "glyph_inventory": False,
        "external_model_calls": False,
        "qualification": checkpoint.get(
            "qualification", "unqualified research checkpoint"
        ),
    }
    _atomic_json(receipt, destination.with_suffix(".json"))
    if args.export:
        _atomic_save(
            {
                "architecture": ARCHITECTURE,
                "arm": checkpoint["arm"],
                "model_config": asdict(model.config),
                "reader_config": asdict(model.reader.config),
                "model": model.state_dict(),
                "render_config": checkpoint.get("render_config", {}),
                "source_checkpoint_sha256": receipt["checkpoint_sha256"],
                "decoder": {
                    "type": "raw_pixel_beam",
                    "width": args.beam_width,
                    "source_sha256": receipt["decoder_sha256"],
                },
                "qualification": receipt["qualification"],
            },
            Path(args.export),
        )
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()

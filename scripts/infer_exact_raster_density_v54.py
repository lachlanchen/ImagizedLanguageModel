#!/usr/bin/env python3
"""Independent raster inference; optional text rendering remains host-side."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from ilm.visual_lm.canonical_glyph_language_data import CanonicalGlyphRenderConfig
from ilm.visual_lm.exact_raster_density_v54 import V54_ARCHITECTURE
from ilm.visual_lm.visual_cell_data import pack_visual_cells, render_visual_cell_stream
from scripts.eval_exact_raster_density_v54 import load_student
from scripts.train_canonical_glyph_binding_v43 import (
    _atomic_json,
    _atomic_save,
    _resolve_device,
)
from scripts.train_exact_raster_density_v54 import DEFAULT_OUT
from scripts.train_visual_state_actuator import file_sha256


def load_aligned_image(path):
    array = np.asarray(Image.open(path).convert("L"), dtype=np.float32)
    height, width = array.shape
    if height % 32 or width % 32:
        raise ValueError("native input must be an aligned grid of 32px cells")
    pixels = 1 - array / 255
    return torch.from_numpy(
        pixels.reshape(height // 32, 32, width // 32, 32).transpose(0, 2, 1, 3).copy()
    ).reshape(-1, 1, 32, 32)


def main():
    parser = argparse.ArgumentParser(
        description="Read aligned glyph images and generate visible writing independently."
    )
    parser.add_argument("--checkpoint", default=DEFAULT_OUT + "/checkpoint_final.pt")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--prompt-image")
    source.add_argument(
        "--prompt-text", help="Optional host renderer; strings never enter the model"
    )
    parser.add_argument("--out", required=True)
    parser.add_argument("--cells", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20265430)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--greedy", action="store_true")
    parser.add_argument(
        "--copy",
        action="store_true",
        help="Supplied-image motor mode; not autonomous language",
    )
    parser.add_argument(
        "--export", help="Write an independent inference-only checkpoint"
    )
    args = parser.parse_args()
    if args.cells < 1:
        raise ValueError("output length must be positive")
    device = _resolve_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if checkpoint["architecture"] != V54_ARCHITECTURE:
        raise ValueError("checkpoint is not V54")
    if "ema_model" not in checkpoint:
        checkpoint["ema_model"] = checkpoint["model"]
    model = load_student(checkpoint, device)
    torch.set_float32_matmul_precision("high")
    if args.prompt_image:
        images = load_aligned_image(args.prompt_image)
    else:
        render = CanonicalGlyphRenderConfig()
        images = render_visual_cell_stream(
            args.prompt_text,
            config=render.visual_cell_config(),
            font_path=render.font_path,
            variant=0,
        )
    if not len(images):
        raise ValueError("prompt contains no image cells")
    generator = torch.Generator(device=device).manual_seed(args.seed)
    generated = []
    with torch.no_grad():
        if args.copy:
            source = images.to(device)
            fields = model.reader.field.encode_unit(source)
            result = model.sample_state(
                torch.zeros(len(source), 384, device=device),
                fields,
                generator=generator,
                greedy=args.greedy,
            )
            generated = result.cpu()
        else:
            context = images[-64:][None].to(device)
            for _ in range(args.cells):
                result = model.sample_next(
                    context, generator=generator, greedy=args.greedy
                )
                generated.append(result[0].cpu())
                context = torch.cat((context, result[:, None]), dim=1)[:, -64:]
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
        "architecture": V54_ARCHITECTURE,
        "checkpoint_sha256": file_sha256(args.checkpoint),
        "seed": args.seed,
        "mode": "supplied_image_copy" if args.copy else "autonomous_image_continuation",
        "greedy": args.greedy,
        "generated_cells": len(generated),
        "input_cells": len(images),
        "tokenizer_in_model": False,
        "ocr_in_model": False,
        "external_model_calls": False,
        "qualification": checkpoint.get(
            "qualification", "unqualified research checkpoint"
        ),
    }
    _atomic_json(receipt, destination.with_suffix(".json"))
    if args.export:
        exported = {
            "architecture": V54_ARCHITECTURE,
            "model_config": asdict(model.config),
            "reader_config": asdict(model.reader.config),
            "model": model.state_dict(),
            "source_checkpoint_sha256": receipt["checkpoint_sha256"],
            "qualification": receipt["qualification"],
            "runtime_boundary": model.boundary_receipt(),
        }
        _atomic_save(exported, Path(args.export))
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()

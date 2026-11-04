#!/usr/bin/env python3
"""Independent visual continuation and export for the jointly trained raster law."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import torch

from ilm.visual_lm.canonical_glyph_language_data import CanonicalGlyphRenderConfig
from ilm.visual_lm.joint_exact_raster_v55 import ARCHITECTURE, model_from_checkpoint
from ilm.visual_lm.visual_cell_data import pack_visual_cells, render_visual_cell_stream
from scripts.infer_exact_raster_density_v54 import load_aligned_image
from scripts.train_canonical_glyph_binding_v43 import (
    _atomic_json,
    _atomic_save,
    _resolve_device,
)
from scripts.train_joint_exact_raster_v55 import DEFAULT_OUT
from scripts.train_visual_state_actuator import file_sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint", default=DEFAULT_OUT + "/joint/checkpoint_final.pt"
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--prompt-image")
    source.add_argument(
        "--prompt-text",
        help="Optional host-side rasterization; strings never enter the model",
    )
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--cells", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20265530)
    parser.add_argument("--greedy", action="store_true")
    parser.add_argument(
        "--copy",
        action="store_true",
        help="Supplied-image motor exercise; not autonomous language",
    )
    parser.add_argument("--export")
    args = parser.parse_args()
    if args.cells < 1:
        raise ValueError("output length must be positive")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if checkpoint["architecture"] != ARCHITECTURE:
        raise ValueError("checkpoint is not V55")
    if checkpoint.get("smoke_only"):
        raise ValueError("smoke checkpoint is not a production model")
    device = _resolve_device(args.device)
    model = model_from_checkpoint(checkpoint, device)
    torch.set_float32_matmul_precision("high")
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
        raise ValueError("prompt contains no image cells")
    rng = torch.Generator(device=device).manual_seed(args.seed)
    with torch.no_grad():
        if args.copy:
            observed = images.to(device)
            fields = model.reader.field.encode_unit(observed)
            output = model.sample_state(
                torch.zeros(len(observed), 384, device=device),
                fields,
                generator=rng,
                greedy=args.greedy,
            ).cpu()
        else:
            context = images[-64:][None].to(device)
            generated = []
            for _ in range(args.cells):
                cell = model.sample_next(context, generator=rng, greedy=args.greedy)
                generated.append(cell[0].cpu())
                context = torch.cat((context, cell[:, None]), dim=1)[:, -64:]
            output = torch.stack(generated)
    destination = Path(args.out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    pack_visual_cells(output, columns=min(32, len(output)), gutter=0).save(destination)
    pack_visual_cells(images, columns=min(32, len(images)), gutter=0).save(
        destination.with_name(destination.stem + "_prompt.png")
    )
    receipt = {
        "architecture": ARCHITECTURE,
        "arm": checkpoint["arm"],
        "checkpoint_sha256": file_sha256(args.checkpoint),
        "seed": args.seed,
        "mode": "supplied_image_copy" if args.copy else "autonomous_image_continuation",
        "greedy": args.greedy,
        "generated_cells": len(output),
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
        _atomic_save(
            {
                "architecture": ARCHITECTURE,
                "arm": checkpoint["arm"],
                "model_config": asdict(model.config),
                "reader_config": asdict(model.reader.config),
                "model": model.state_dict(),
                "render_config": checkpoint.get("render_config", {}),
                "source_checkpoint_sha256": receipt["checkpoint_sha256"],
                "qualification": receipt["qualification"],
                "runtime_boundary": model.boundary_receipt(),
            },
            Path(args.export),
        )
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()

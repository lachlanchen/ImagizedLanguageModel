#!/usr/bin/env python3
"""Independent image-block reading with a retained free raster continuation path."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import torch

from ilm.visual_lm.canonical_glyph_language_data import CanonicalGlyphRenderConfig
from ilm.visual_lm.visual_cell_data import pack_visual_cells
from ilm.visual_lm.visual_answer_blocks import ARCHITECTURE, model_from_checkpoint
from ilm.visual_lm.visual_qa_data import VisualQARenderer
from ilm.visual_lm.visual_qa_evaluation import save_answer_image
from scripts.infer_exact_raster_density_v54 import load_aligned_image
from scripts.train_canonical_glyph_binding_v43 import (
    _atomic_json,
    _atomic_save,
    _resolve_device,
)
from scripts.train_visual_state_actuator import file_sha256


@torch.no_grad()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--prompt-image")
    source.add_argument(
        "--prompt-text", help="Host rasterizer only; no string enters the network"
    )
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="auto")
    p.add_argument("--maximum-output-cells", type=int, default=17)
    p.add_argument("--export")
    p.add_argument(
        "--continuation",
        action="store_true",
        help="Use the retained V57 glyph continuation path",
    )
    p.add_argument("--allow-smoke", action="store_true")
    a = p.parse_args()
    if not 1 <= a.maximum_output_cells <= 31:
        raise ValueError("output safety cap must be 1..31")
    checkpoint = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
    if checkpoint.get("smoke_only") and not a.allow_smoke:
        raise ValueError("smoke checkpoint excluded")
    device = _resolve_device(a.device)
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("highest")
    model = model_from_checkpoint(checkpoint, device)
    render = CanonicalGlyphRenderConfig(**checkpoint["render_config"])
    prompt = (
        load_aligned_image(a.prompt_image)
        if a.prompt_image
        else VisualQARenderer(render)(a.prompt_text)
    )
    prompt = (prompt >= 0.5).float().to(device)
    if model.block_enabled and not a.continuation:
        result = model.generate_block(prompt)
    else:
        result = model.generate(prompt, maximum_output_cells=a.maximum_output_cells)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    save_answer_image(result["images"], out)
    # A one-row strip contains exactly the prompt's cells, with no padding ambiguity.
    pack_visual_cells(prompt.cpu(), columns=len(prompt), gutter=0).save(
        out.with_name(out.stem + "_prompt.png")
    )
    _atomic_save({"images": result["images"].cpu()}, out.with_suffix(".pt"))
    receipt = {k: v for k, v in result.items() if k != "images"}
    receipt.update(
        architecture=ARCHITECTURE,
        checkpoint_sha256=file_sha256(a.checkpoint),
        prompt_cells=len(prompt),
        generated_cells=len(result["images"]),
        output_placeholder_only=len(result["images"]) == 0,
        boundary=model.boundary_receipt(),
        qualification=checkpoint.get(
            "qualification", "unqualified research checkpoint"
        ),
        smoke_only=checkpoint.get("smoke_only", False),
    )
    _atomic_json(receipt, out.with_suffix(".json"))
    if a.export:
        _atomic_save(
            {
                "architecture": ARCHITECTURE,
                "arm": checkpoint["arm"],
                "model_config": asdict(model.config),
                "reader_config": asdict(model.reader.config),
                "render_config": asdict(render),
                "model": model.state_dict(),
                "qualification": receipt["qualification"],
                "smoke_only": receipt["smoke_only"],
                "source_checkpoint_sha256": receipt["checkpoint_sha256"],
            },
            Path(a.export),
        )
    print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    main()

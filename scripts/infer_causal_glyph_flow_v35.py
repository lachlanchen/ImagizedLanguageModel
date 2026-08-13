#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from ilm.visual_lm.causal_glyph_flow import (
    V35_ARCHITECTURE,
    CausalGlyphFlowConfig,
    CausalGlyphFlowLM,
    causal_glyph_flow_boundary_receipt,
    file_sha256,
)
from ilm.visual_lm.causal_glyph_flow_development import (
    TesseractStripOCR,
    patches_to_image,
)
from ilm.visual_lm.direct_visual_patch_training import module_state_sha256
from ilm.visual_lm.visual_semantic_raster_data import normalize_visible_text
from scripts.export_causal_glyph_flow_v35 import (
    STANDALONE_ARTIFACT,
    standalone_checkpoint_is_clean,
)


DEFAULT_CHECKPOINT = "artifacts/causal_glyph_flow_v35_20260814/ilm_v35_ema_standalone.pt"
DEFAULT_FONT = "/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run independent raster-in/raster-out V35 inference."
    )
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--text")
    source.add_argument("--image")
    parser.add_argument("--out", required=True)
    parser.add_argument("--font", default=DEFAULT_FONT)
    parser.add_argument("--font-size", type=int, default=26)
    parser.add_argument("--writer", choices=("anchor", "flow"), default=None)
    parser.add_argument("--maximum-new-patches", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--precision", choices=("fp32", "fp16", "bf16"), default="bf16")
    parser.add_argument("--ocr-sidecar", action="store_true")
    return parser.parse_args()


def choose_device(value: str) -> torch.device:
    if value == "auto":
        value = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = torch.device(value)
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda:0")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("V35 inference requested CUDA but CUDA is unavailable")
    return device


def _autocast(device: torch.device, precision: str):
    if device.type != "cuda" or precision == "fp32":
        return nullcontext()
    dtype = torch.float16 if precision == "fp16" else torch.bfloat16
    return torch.autocast(device_type="cuda", dtype=dtype)


def _image_to_prompt(image: Image.Image, *, maximum_patches: int) -> torch.Tensor:
    if maximum_patches < 1:
        raise ValueError("V35 prompt has no available visual context")
    image = image.convert("L")
    if image.height != 32:
        width = max(1, round(image.width * 32 / image.height))
        image = image.resize((width, 32), Image.Resampling.LANCZOS)
    patch_count = max(1, math.ceil(image.width / 32))
    if patch_count > maximum_patches:
        raise ValueError(
            f"V35 prompt requires {patch_count} patches; maximum is {maximum_patches}"
        )
    canvas = Image.new("L", (patch_count * 32, 32), 255)
    canvas.paste(image, (0, 0))
    pixels = (np.asarray(canvas, dtype=np.float32) >= 127.5).astype(np.float32)
    return torch.from_numpy(pixels.copy()).unsqueeze(0)


def render_text_prompt(
    text: str,
    *,
    font_path: str | Path,
    font_size: int,
    maximum_patches: int,
) -> torch.Tensor:
    normalized = normalize_visible_text(text)
    if not normalized:
        raise ValueError("V35 cannot render an empty text prompt")
    if not 16 <= font_size <= 31:
        raise ValueError("V35 prompt font size must be in [16,31]")
    font_path = Path(font_path)
    if not font_path.is_file():
        raise FileNotFoundError(font_path)
    font = ImageFont.truetype(str(font_path), size=font_size)
    probe = Image.new("L", (1, 32), 255)
    draw = ImageDraw.Draw(probe)
    left, top, right, bottom = draw.textbbox((0, 0), normalized, font=font)
    origin = 4
    width = origin + right - left
    patch_count = max(1, math.ceil(width / 32))
    if patch_count > maximum_patches:
        raise ValueError(
            f"V35 text prompt requires {patch_count} patches; maximum is {maximum_patches}"
        )
    image = Image.new("L", (patch_count * 32, 32), 255)
    draw = ImageDraw.Draw(image)
    y = (32 - (bottom - top)) // 2 - top
    draw.text((origin - left, y), normalized, font=font, fill=0)
    return _image_to_prompt(image, maximum_patches=maximum_patches)


def load_image_prompt(
    path: str | Path,
    *,
    maximum_patches: int,
) -> torch.Tensor:
    with Image.open(path) as image:
        return _image_to_prompt(image, maximum_patches=maximum_patches)


def load_standalone_model(
    path: str | Path,
    *,
    device: torch.device,
) -> tuple[CausalGlyphFlowLM, dict[str, Any]]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("artifact") != STANDALONE_ARTIFACT:
        raise ValueError("V35 inference checkpoint is not standalone")
    if checkpoint.get("architecture") != V35_ARCHITECTURE:
        raise ValueError("V35 standalone checkpoint has the wrong architecture")
    if not standalone_checkpoint_is_clean(checkpoint):
        raise ValueError("V35 standalone checkpoint failed its clean-state audit")
    model = CausalGlyphFlowLM(CausalGlyphFlowConfig(**checkpoint["model_config"]))
    model.load_state_dict(checkpoint["model"], strict=True)
    if module_state_sha256(model) != checkpoint.get("model_state_sha256"):
        raise ValueError("V35 standalone model state hash does not match")
    model.requires_grad_(False).eval().to(device)
    boundary = causal_glyph_flow_boundary_receipt(model)
    if boundary != checkpoint.get("boundary"):
        raise ValueError("V35 standalone runtime boundary does not match")
    return model, checkpoint


@torch.no_grad()
def main() -> None:
    args = parse_args()
    device = choose_device(args.device)
    model, checkpoint = load_standalone_model(args.checkpoint, device=device)
    generation = dict(checkpoint["generation"])
    maximum_new = int(
        args.maximum_new_patches
        if args.maximum_new_patches is not None
        else generation["maximum_new_patches"]
    )
    if not 1 <= maximum_new < model.config.maximum_patches:
        raise ValueError("V35 maximum new patch count is outside the model context")
    maximum_prompt = model.config.maximum_patches - maximum_new
    if args.text is not None:
        prompt = render_text_prompt(
            args.text,
            font_path=args.font,
            font_size=args.font_size,
            maximum_patches=maximum_prompt,
        )
        source = {"type": "text-wrapper", "text_length": len(args.text)}
    else:
        prompt = load_image_prompt(args.image, maximum_patches=maximum_prompt)
        source = {"type": "image", "path": str(Path(args.image).resolve())}
    prompt_mask = torch.ones(prompt.shape[-1] // 32)
    writer = args.writer or str(checkpoint["writer"])
    seed = int(args.seed if args.seed is not None else generation["seed"])
    with _autocast(device, args.precision):
        result = model.generate(
            prompt.unsqueeze(0).to(device),
            prompt_mask.unsqueeze(0).to(device),
            maximum_new_patches=maximum_new,
            minimum_new_patches=int(generation["minimum_new_patches"]),
            stop_threshold=float(generation["stop_threshold"]),
            raster_threshold=float(generation["raster_threshold"]),
            flow_steps=int(generation["flow_steps"]),
            seed=seed,
            use_flow=writer == "flow",
        )
    length = int(result.lengths[0])
    image = patches_to_image(result.patches[0].float().cpu(), length)
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output)
    receipt = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_sha256": file_sha256(args.checkpoint),
        "status": checkpoint["decision"]["status"],
        "source": source,
        "model_inputs": {
            "pixels": list(prompt.unsqueeze(0).shape),
            "patch_mask": list(prompt_mask.unsqueeze(0).shape),
        },
        "writer": writer,
        "seed": seed,
        "generated_patches": length,
        "output": str(output.resolve()),
        "finite": bool(
            torch.isfinite(result.patches).all()
            and torch.isfinite(result.feedback_latents).all()
            and torch.isfinite(result.stop_probabilities).all()
        ),
        "ocr_used_by_model": False,
    }
    if args.ocr_sidecar:
        observed = TesseractStripOCR()(image)
        sidecar = output.with_suffix(output.suffix + ".txt")
        sidecar.write_text(observed + "\n", encoding="utf-8")
        receipt["ocr_sidecar"] = str(sidecar.resolve())
        receipt["ocr_text"] = observed
        receipt["ocr_role"] = "post-hoc evaluator only"
    receipt_path = output.with_suffix(output.suffix + ".json")
    receipt_path.write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(receipt, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

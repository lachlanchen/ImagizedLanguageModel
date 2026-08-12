#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from ilm.visual_lm.ink_stream import InkStreamLM, ink_stream_config_from_payload
from ilm.visual_lm.ink_stream_data import (
    InkRibbonConfig,
    render_prompt_stream,
    split_ink_strips,
    strips_to_image,
    visual_separator,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a writing image from an image-native InkStream model.")
    parser.add_argument("--checkpoint", required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--prompt", help="UI text rasterized before the student model")
    source.add_argument("--input-image", help="single-line writing image passed directly")
    parser.add_argument("--out", default="artifacts/ink_stream_inference")
    parser.add_argument("--maximum-new-strips", type=int, default=96)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument("--feedback-mode", choices=("soft", "hard"), default="soft")
    parser.add_argument("--variant", type=int, default=2_000_003)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def choose_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def image_prefix(path: str, config: InkRibbonConfig) -> torch.Tensor:
    image = Image.open(path).convert("L")
    scale = config.height / image.height
    image = image.resize((max(1, round(image.width * scale)), config.height), Image.Resampling.LANCZOS)
    ink = torch.from_numpy(1.0 - np.asarray(image, dtype=np.float32) / 255.0)
    blank = torch.zeros((config.height, config.strip_width), dtype=torch.float32)
    field = torch.cat((blank, ink, blank, visual_separator(config), blank), dim=1)
    return split_ink_strips(field, config.strip_width)[: config.maximum_strips]


def main() -> None:
    args = parse_args()
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    device = choose_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    if checkpoint.get("architecture") != "ink-stream-v1":
        raise ValueError("checkpoint is not an ink-stream-v1 model")
    model_config = ink_stream_config_from_payload(checkpoint["model_config"])
    ribbon_config = InkRibbonConfig(**checkpoint["ribbon_config"])
    ribbon_config = InkRibbonConfig(**{**ribbon_config.__dict__, "augment": False})
    model = InkStreamLM(model_config).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    if args.input_image:
        prefix = image_prefix(args.input_image, ribbon_config)
        adapter = "direct_writing_image"
    else:
        prefix = render_prompt_stream(
            args.prompt or "",
            config=ribbon_config,
            variant=args.variant,
        )
        adapter = "deterministic_ui_rasterizer"
    prefix = prefix.unsqueeze(0).to(device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    with torch.inference_mode():
        generated = model.generate(
            prefix,
            maximum_new_strips=args.maximum_new_strips,
            threshold=args.threshold,
            temperature=args.temperature,
            stochastic=args.stochastic,
            feedback_mode=args.feedback_mode,
        )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    answer = generated[:, prefix.shape[1] :]
    strips_to_image(prefix).save(output / "prompt_field.png", optimize=True)
    strips_to_image(answer).save(output / "answer_field.png", optimize=True)
    strips_to_image(generated).save(output / "complete_field.png", optimize=True)
    receipt = {
        "architecture": "ink-stream-v1",
        "adapter": adapter,
        "student_boundary_input": "continuous grayscale image strips",
        "student_used_text_tokens": False,
        "student_used_ocr": False,
        "student_used_external_llm": False,
        "generated_strips": args.maximum_new_strips,
        "elapsed_seconds": elapsed,
        "strips_per_second": args.maximum_new_strips / max(elapsed, 1e-9),
        "peak_vram_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0,
        "answer_image": str(output / "answer_field.png"),
    }
    (output / "receipt.json").write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(receipt, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

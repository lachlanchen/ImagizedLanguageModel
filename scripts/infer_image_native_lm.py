#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from contextlib import nullcontext
from pathlib import Path

import torch
from PIL import Image, ImageOps

from ilm.visual_lm import (
    ConditionalVisualFlow,
    InstructionRenderConfig,
    VisualFlowConfig,
    VisualPageVAE,
    VisualVAEConfig,
    render_instruction_page,
)
from ilm.visual_lm.dataset import pil_to_tensor, tensor_to_pil
from ilm.visual_lm.flow import sample_heun


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run independent pixel-to-pixel inference with a trained ILM checkpoint."
    )
    parser.add_argument("--checkpoint", required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--prompt", help="Typed question; rasterized before the model boundary.")
    source.add_argument("--input-image", help="Question/page/glyph image used directly.")
    parser.add_argument("--language", default="auto")
    parser.add_argument("--out", default="artifacts/ilm_inference")
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--guidance-scale", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--precision", choices=("fp32", "fp16", "bf16"), default="bf16")
    return parser.parse_args()


def choose_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def autocast_context(device: torch.device, precision: str):
    if device.type != "cuda" or precision == "fp32":
        return nullcontext()
    dtype = torch.float16 if precision == "fp16" else torch.bfloat16
    return torch.amp.autocast("cuda", dtype=dtype)


def detect_language(text: str) -> str:
    cjk = sum(1 for character in text if ord(character) >= 0x3400)
    return "zh" if cjk >= max(1, len(text) // 8) else "en"


def normalize_input_image(path: Path, size: int) -> Image.Image:
    source = Image.open(path).convert("RGB")
    contained = ImageOps.contain(source, (size, size), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (size, size), "white")
    canvas.paste(contained, ((size - contained.width) // 2, (size - contained.height) // 2))
    return canvas


def main() -> None:
    args = parse_args()
    device = choose_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if checkpoint.get("format") != "ilm-visual-flow-v1":
        raise ValueError("Checkpoint is not an ilm-visual-flow-v1 model")
    vae = VisualPageVAE(VisualVAEConfig.from_dict(checkpoint["vae_config"]))
    flow = ConditionalVisualFlow(VisualFlowConfig.from_dict(checkpoint["flow_config"]))
    vae.load_state_dict(checkpoint["vae"])
    flow.load_state_dict(checkpoint.get("flow_ema", checkpoint["flow"]))
    vae.to(device).eval().requires_grad_(False)
    flow.to(device).eval().requires_grad_(False)

    image_size = int(checkpoint["args"]["image_size"])
    if args.prompt is not None:
        language = detect_language(args.prompt) if args.language == "auto" else args.language
        prompt_image, render_metadata = render_instruction_page(
            args.prompt,
            role="prompt",
            language=language,
            config=InstructionRenderConfig(image_size=image_size, augment=False),
            variant=args.seed,
        )
        input_source = "typed-raster"
    else:
        prompt_image = normalize_input_image(Path(args.input_image), image_size)
        render_metadata = {"language": args.language, "truncated": False}
        input_source = "image"

    prompt = pil_to_tensor(prompt_image).unsqueeze(0).to(device)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    with torch.inference_mode(), autocast_context(device, args.precision):
        condition = vae.encode(prompt, sample=False, normalize=True)
        generated_latent = sample_heun(
            flow,
            condition,
            steps=args.steps,
            guidance_scale=args.guidance_scale,
            generator=generator,
        )
        answer = vae.decode(generated_latent, normalized=True)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started

    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    prompt_image.save(output / "prompt.png")
    tensor_to_pil(answer[0]).save(output / "answer.png")
    metadata = {
        "format": "ilm-image-answer-v1",
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "input_source": input_source,
        "render": render_metadata,
        "steps": args.steps,
        "guidance_scale": args.guidance_scale,
        "seed": args.seed,
        "device": str(device),
        "precision": args.precision,
        "latency_seconds": elapsed,
        "peak_vram_bytes": (
            torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
        ),
        "primary_output": str((output / "answer.png").resolve()),
    }
    (output / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, ensure_ascii=False))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import torch
from PIL import Image, ImageDraw

from ilm.visual_lm.dataset import pil_to_tensor
from ilm.visual_lm.instruction_data import InstructionRenderConfig, render_instruction_page
from ilm.visual_lm.retinal_memory import (
    VisualAssociativeReader,
    VisualEpisodeMemory,
    retinal_config_from_payload,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Address image-valued memory using only a writing image at the student boundary."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--memory", required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--prompt", help="UI text; rasterized before the model call")
    source.add_argument("--input-image", help="writing image passed directly to the model")
    parser.add_argument("--out", default="artifacts/visual_field_inference")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--minimum-score", type=float, default=-1.0)
    parser.add_argument("--minimum-margin", type=float, default=-1.0)
    parser.add_argument("--render-variant", type=int, default=2_147_483_001)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def choose_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def fit_canvas(image: Image.Image, size: int) -> Image.Image:
    image = image.convert("RGB")
    image.thumbnail((size, size), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (size, size), "white")
    canvas.paste(image, ((size - image.width) // 2, (size - image.height) // 2))
    return canvas


def diagnostic_pair(prompt: Image.Image, answer: Image.Image, score: float) -> Image.Image:
    size = max(prompt.height, answer.height)
    canvas = Image.new("RGB", (prompt.width + answer.width, size + 34), "#f8fafc")
    canvas.paste(prompt, (0, 34))
    canvas.paste(answer, (prompt.width, 34))
    draw = ImageDraw.Draw(canvas)
    draw.text((12, 9), "input image", fill="#111827")
    draw.text((prompt.width + 12, 9), f"retrieved answer image · cosine {score:.4f}", fill="#111827")
    return canvas


def main() -> None:
    args = parse_args()
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    device = choose_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    if checkpoint.get("architecture") != "visual-field-memory-v1":
        raise ValueError("checkpoint is not a visual-field-memory-v1 model")
    config = retinal_config_from_payload(checkpoint["retinal_config"])
    model = VisualAssociativeReader(config).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    memory = VisualEpisodeMemory.load(args.memory)

    if args.input_image:
        prompt_image = fit_canvas(Image.open(args.input_image), config.image_size)
        adapter = "direct_image"
    else:
        prompt_image, _ = render_instruction_page(
            args.prompt or "",
            role="prompt",
            language="visual",
            config=InstructionRenderConfig(image_size=config.image_size, augment=False),
            variant=args.render_variant,
        )
        adapter = "deterministic_ui_rasterizer"
    prompt_path = output / "prompt.png"
    prompt_image.save(prompt_path, optimize=True)

    image_tensor = pil_to_tensor(prompt_image).unsqueeze(0).to(device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    with torch.inference_mode():
        query = model.encode_query(image_tensor)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    encode_seconds = time.perf_counter() - started
    hits = memory.search(query, top_k=max(2, args.top_k))[0]
    if not hits:
        raise RuntimeError("visual memory is empty")
    best = hits[0]
    margin = best.score - hits[1].score if len(hits) > 1 else float("inf")
    accepted = best.score >= args.minimum_score and margin >= args.minimum_margin
    answer_source = memory.answer_path(best)
    answer_path = output / "answer.png"
    if accepted:
        shutil.copyfile(answer_source, answer_path)
        answer_image = Image.open(answer_path).convert("RGB")
        status = "retrieved"
    else:
        answer_image = Image.new("RGB", (config.image_size, config.image_size), "white")
        draw = ImageDraw.Draw(answer_image)
        draw.text((24, 24), "VISUAL MEMORY: LOW CONFIDENCE", fill="#7f1d1d")
        answer_image.save(answer_path)
        status = "abstained"
    diagnostic_pair(prompt_image, answer_image, best.score).save(output / "retrieval_pair.png", optimize=True)

    receipt = {
        "status": status,
        "model_boundary_input": "RGB image tensor",
        "ui_adapter": adapter,
        "student_used_text_tokens": False,
        "student_used_ocr": False,
        "student_used_external_llm": False,
        "encode_seconds": encode_seconds,
        "best_score": best.score,
        "score_margin": margin,
        "answer_image": str(answer_path),
        "retrieved": [
            {
                "score": hit.score,
                "entry_index": hit.entry_index,
                "metadata": hit.metadata,
            }
            for hit in hits[: args.top_k]
        ],
    }
    (output / "receipt.json").write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(receipt, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

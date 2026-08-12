#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from ilm.visual_lm.folio import FolioRetina, folio_config_from_payload
from ilm.visual_lm.folio_data import FolioRenderConfig, folio_tensor_to_image, render_folio
from ilm.visual_lm.folio_memory import FolioMemory


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Image-first inference with the independent visual folio.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--memory", required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--text", help="Outer UI text; rasterized before the student boundary.")
    source.add_argument("--image", help="Writing image supplied directly to the student.")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--variant", type=int, default=104729)
    parser.add_argument("--out", default="artifacts/visual_folio_inference")
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def choose_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def load_model(path: str | Path, device: torch.device) -> tuple[FolioRetina, dict[str, Any]]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if checkpoint.get("architecture") != "visual-folio-retina-v1":
        raise ValueError("checkpoint is not a visual folio retina")
    model = FolioRetina(folio_config_from_payload(checkpoint["model_config"])).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, checkpoint


def render_config(checkpoint: dict[str, Any]) -> FolioRenderConfig:
    payload = dict(checkpoint["render_config"])
    payload["augment"] = False
    return FolioRenderConfig(**payload)


def image_to_ink(path: str | Path, config: FolioRenderConfig) -> torch.Tensor:
    source = Image.open(path).convert("L")
    source.thumbnail((config.width, config.height), Image.Resampling.LANCZOS)
    canvas = Image.new("L", (config.width, config.height), 255)
    left = (config.width - source.width) // 2
    top = (config.height - source.height) // 2
    canvas.paste(source, (left, top))
    array = np.asarray(canvas, dtype=np.float32) / 255.0
    return torch.from_numpy(1.0 - array)[None]


def combine_pages(paths: list[Path], output: Path) -> None:
    images = [Image.open(path).convert("RGB") for path in paths]
    width = max(image.width for image in images)
    gap = 16
    height = sum(image.height for image in images) + gap * max(0, len(images) - 1)
    canvas = Image.new("RGB", (width, height), "white")
    y = 0
    for image in images:
        canvas.paste(image, ((width - image.width) // 2, y))
        y += image.height + gap
    canvas.save(output, optimize=True)


def main() -> None:
    args = parse_args()
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    device = choose_device(args.device)
    model, checkpoint = load_model(args.checkpoint, device)
    config = render_config(checkpoint)
    if args.text is not None:
        query = render_folio(args.text, config=config, variant=args.variant, augment=False)
        adapter = "deterministic_text_rasterizer"
        input_digest = hashlib.sha256(args.text.encode("utf-8")).hexdigest()
    else:
        query = image_to_ink(args.image, config)
        adapter = "direct_writing_image"
        input_digest = hashlib.sha256(Path(args.image).read_bytes()).hexdigest()
    folio_tensor_to_image(query).save(output / "query.png", optimize=True)
    memory = FolioMemory.load(args.memory)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    with torch.inference_mode():
        field = model(query[None].to(device))
    hits = memory.search(field, top_k=args.top_k)[0]
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    if not hits:
        raise RuntimeError("folio memory returned no answer")

    best_paths = memory.image_paths(hits[0])
    copied_paths = []
    for index, source in enumerate(best_paths, 1):
        suffix = source.suffix.lower() or ".png"
        destination = output / f"answer_{index:02d}{suffix}"
        shutil.copy2(source, destination)
        copied_paths.append(destination)
    combine_pages(copied_paths, output / "answer.png")
    result = {
        "architecture": "visual-folio-memory-v1",
        "input_adapter": adapter,
        "input_sha256": input_digest,
        "query_image": "query.png",
        "primary_output": "answer.png",
        "answer_pages": [path.name for path in copied_paths],
        "retrieved": [
            {
                "rank": rank,
                "score": hit.score,
                "entry_index": hit.entry_index,
                "metadata": hit.metadata,
            }
            for rank, hit in enumerate(hits, 1)
        ],
        "elapsed_seconds": elapsed,
        "peak_vram_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0,
        "student_received_text": False,
        "student_used_tokens": False,
        "student_used_ocr": False,
        "student_called_external_model": False,
    }
    (output / "result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

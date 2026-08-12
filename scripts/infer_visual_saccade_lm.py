#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageOps

from ilm.visual_lm.ink_jepa_data import (
    RetinalRenderConfig,
    advance_retinal_cursor,
    extract_retinal_fovea,
    infer_retinal_cursor,
    place_retinal_fovea,
    render_retinal_page,
    retinal_cell_bounds,
    retinal_cursor_after_text,
    retinal_layout,
)
from ilm.visual_lm.saccade_lm import VisualSaccadeLM, visual_saccade_config_from_payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate writing as a recurrent sequence of continuous visual fixations."
    )
    parser.add_argument("--checkpoint", required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--text", help="UI text rasterized before the student model boundary")
    source.add_argument("--input-image", help="fixed-layout writing page passed directly as pixels")
    parser.add_argument("--out", default="artifacts/visual_saccade_inference")
    parser.add_argument("--new-cells", type=int, default=24)
    parser.add_argument("--row", type=int)
    parser.add_argument("--column", type=int)
    parser.add_argument("--minimum-cell-ink", type=float, default=0.002)
    parser.add_argument("--feedback", choices=("soft", "hard"), default="soft")
    parser.add_argument("--ink-threshold", type=float, default=0.5)
    parser.add_argument("--render-variant", type=int, default=2_000_003)
    parser.add_argument("--precision", choices=("fp32", "fp16", "bf16"), default="bf16")
    parser.add_argument("--device", default="auto")
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


def render_config_from_checkpoint(checkpoint: dict[str, Any]) -> RetinalRenderConfig:
    payload = dict(checkpoint["render_config"])
    payload["augment"] = False
    return RetinalRenderConfig(**payload)


def image_to_retinal_ink(path: str, config: RetinalRenderConfig) -> torch.Tensor:
    source = ImageOps.exif_transpose(Image.open(path)).convert("L")
    if source.size != (config.width, config.height):
        source.thumbnail(
            (config.width - 2 * config.margin, config.height - 2 * config.margin),
            Image.Resampling.LANCZOS,
        )
        canvas = Image.new("L", (config.width, config.height), 255)
        canvas.paste(source, (config.margin, config.margin))
        source = canvas
    array = np.asarray(source, dtype=np.float32) / 255.0
    return torch.from_numpy(1.0 - array)[None]


def save_ink(page: torch.Tensor, path: Path) -> None:
    array = (255.0 * (1.0 - page.detach().float().cpu()[0].clamp(0, 1))).round().byte().numpy()
    Image.fromarray(array, mode="L").save(path, optimize=True)


def occupied_foveas(
    page: torch.Tensor,
    config: RetinalRenderConfig,
    *,
    fovea_size: int,
    minimum_mean_ink: float,
) -> list[torch.Tensor]:
    output: list[torch.Tensor] = []
    for row in range(config.rows):
        for column in range(config.columns):
            left, top, right, bottom = retinal_cell_bounds(row=row, column=column, config=config)
            if float(page[:, top:bottom, left:right].float().mean()) < minimum_mean_ink:
                continue
            output.append(
                extract_retinal_fovea(
                    page,
                    row=row,
                    column=column,
                    config=config,
                    fovea_size=fovea_size,
                )
            )
    return output


def prompt_page_foveas_cursor(
    args: argparse.Namespace,
    config: RetinalRenderConfig,
    *,
    fovea_size: int,
) -> tuple[torch.Tensor, list[torch.Tensor], tuple[int, int], str]:
    if (args.row is None) != (args.column is None):
        raise ValueError("--row and --column must be supplied together")
    if args.input_image:
        page = image_to_retinal_ink(args.input_image, config)
        foveas = occupied_foveas(
            page,
            config,
            fovea_size=fovea_size,
            minimum_mean_ink=args.minimum_cell_ink,
        )
        inferred = infer_retinal_cursor(page, config, minimum_mean_ink=args.minimum_cell_ink)
        adapter = "direct_fixed-layout_writing_image"
    else:
        text = args.text or ""
        page = render_retinal_page(text, config=config, variant=args.render_variant)
        foveas = [
            extract_retinal_fovea(
                page,
                row=row,
                column=column,
                config=config,
                fovea_size=fovea_size,
            )
            for _, row, column in retinal_layout(text, config)
        ]
        inferred = retinal_cursor_after_text(text, config)
        adapter = "deterministic_boundary_rasterizer"
    if not foveas:
        raise ValueError("the prompt contains no visible writing fixations")
    cursor = (args.row, args.column) if args.row is not None else inferred
    if cursor is None:
        raise ValueError("the prompt page has no remaining visual cells")
    return page, foveas, (int(cursor[0]), int(cursor[1])), adapter


def main() -> None:
    args = parse_args()
    if args.new_cells < 1:
        raise ValueError("new-cells must be positive")
    output = Path(args.out)
    step_root = output / "steps"
    output.mkdir(parents=True, exist_ok=True)
    step_root.mkdir(parents=True, exist_ok=True)
    device = choose_device(args.device)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if checkpoint.get("architecture") != "visual-saccade-language-model-v1":
        raise ValueError("checkpoint is not a visual saccade language model")
    model = VisualSaccadeLM(visual_saccade_config_from_payload(checkpoint["model_config"]))
    model.load_state_dict(checkpoint["model"])
    model.to(device).eval().requires_grad_(False)
    render_config = render_config_from_checkpoint(checkpoint)
    page, foveas, cursor, adapter = prompt_page_foveas_cursor(
        args,
        render_config,
        fovea_size=model.config.fovea_size,
    )
    page = page.to(device)
    answer = torch.zeros_like(page)
    context = torch.stack(foveas)[None].to(device)
    save_ink(page, output / "prompt_page.png")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    traces: list[dict[str, Any]] = []
    stop_reason = "requested_cells_generated"

    with torch.inference_mode(), autocast_context(device, args.precision):
        current_visual = model.encode_sequence(context)
        state, recurrent = model.dynamics(current_visual)
        current_visual = current_visual[:, -1:]
        state = state[:, -1:]
        for step in range(args.new_cells):
            row, column = cursor
            logits = model.ink(state, current_visual)[:, 0]
            soft_fovea = logits.float().sigmoid()[0]
            feedback_fovea = (
                (soft_fovea >= args.ink_threshold).to(soft_fovea.dtype)
                if args.feedback == "hard"
                else soft_fovea
            )
            page = place_retinal_fovea(
                page,
                soft_fovea,
                row=row,
                column=column,
                config=render_config,
            )
            answer = place_retinal_fovea(
                answer,
                soft_fovea,
                row=row,
                column=column,
                config=render_config,
            )
            save_ink(soft_fovea, step_root / f"step_{step:04d}.png")
            traces.append(
                {
                    "step": step,
                    "row": row,
                    "column": column,
                    "mean_ink": float(soft_fovea.mean()),
                    "binary_ink_fraction": float((soft_fovea >= args.ink_threshold).float().mean()),
                }
            )
            next_cursor = advance_retinal_cursor(row, column, render_config)
            if next_cursor is None:
                stop_reason = "page_full"
                break
            cursor = next_cursor
            current_visual = model.online_retina(feedback_fovea[None])[:, None]
            state, recurrent = model.dynamics(current_visual, recurrent)

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    save_ink(answer, output / "answer_page.png")
    save_ink(page, output / "complete_page.png")
    receipt = {
        "architecture": "recurrent-visual-saccade-generation-v1",
        "checkpoint": args.checkpoint,
        "boundary_adapter": adapter,
        "student_boundary_input": "ordered continuous ink foveas",
        "student_used_text_tokens": False,
        "student_used_unicode_ids": False,
        "student_used_ocr": False,
        "student_used_codebook": False,
        "student_used_external_language_model": False,
        "primary_output": "continuous ink image",
        "initial_visual_fixations": len(foveas),
        "generated_cells": len(traces),
        "requested_cells": args.new_cells,
        "feedback": args.feedback,
        "stop_reason": stop_reason,
        "elapsed_seconds": elapsed,
        "cells_per_second": len(traces) / max(elapsed, 1e-9),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "peak_vram_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0,
        "prompt_image": str(output / "prompt_page.png"),
        "answer_image": str(output / "answer_page.png"),
        "complete_image": str(output / "complete_page.png"),
        "trace": traces,
    }
    (output / "receipt.json").write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(receipt, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

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
import torch.nn.functional as F
from PIL import Image, ImageOps

from ilm.visual_lm.ink_jepa import InkJEPA, hide_retinal_regions, ink_jepa_config_from_payload
from ilm.visual_lm.ink_jepa_data import (
    RetinalRenderConfig,
    advance_retinal_cursor,
    extract_retinal_fovea,
    future_retinal_masks,
    infer_retinal_cursor,
    place_retinal_fovea,
    render_retinal_page,
    retinal_cursor_after_text,
)
from ilm.visual_lm.ink_writer import (
    FovealInkFlow,
    foveal_writer_config_from_payload,
    retinal_foveal_prediction,
    sample_foveal_ink,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Autonomously continue a writing image through retinal prediction, "
            "continuous ink flow, and visual re-reading."
        )
    )
    parser.add_argument("--writer-checkpoint", required=True)
    parser.add_argument(
        "--foundation-checkpoint",
        help="override the frozen foundation recorded by the writer checkpoint",
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--text", help="UI text rasterized before the student model boundary")
    source.add_argument("--input-image", help="writing page passed directly as pixels")
    parser.add_argument("--out", default="artifacts/ink_language_inference")
    parser.add_argument("--new-cells", type=int, default=16)
    parser.add_argument("--candidates", type=int, default=4)
    parser.add_argument("--flow-steps", type=int, default=8)
    parser.add_argument("--guidance-scale", type=float, default=1.0)
    parser.add_argument("--plan-score-weight", type=float, default=0.15)
    parser.add_argument("--row", type=int)
    parser.add_argument("--column", type=int)
    parser.add_argument("--minimum-cell-ink", type=float, default=0.002)
    parser.add_argument("--render-variant", type=int, default=2_000_003)
    parser.add_argument("--seed", type=int, default=20260812)
    parser.add_argument("--precision", choices=("fp32", "fp16", "bf16"), default="bf16")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--save-candidates", action="store_true")
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


def load_foundation_checkpoint(
    writer_checkpoint: dict[str, Any],
    override: str | None,
) -> tuple[dict[str, Any], Path]:
    recorded = Path(override or str(writer_checkpoint["foundation_checkpoint"]))
    if not recorded.exists():
        raise FileNotFoundError(f"foundation checkpoint does not exist: {recorded}")
    checkpoint = torch.load(recorded, map_location="cpu", weights_only=False)
    expected_step = int(writer_checkpoint.get("foundation_step", -1))
    actual_step = int(checkpoint.get("global_step", -2))
    if actual_step != expected_step and override is None:
        immutable = recorded.parent / f"checkpoint_step_{expected_step:07d}.pt"
        if immutable.exists():
            recorded = immutable
            checkpoint = torch.load(recorded, map_location="cpu", weights_only=False)
            actual_step = int(checkpoint.get("global_step", -2))
    if actual_step != expected_step:
        raise ValueError(
            f"writer expects foundation step {expected_step}, but {recorded} contains step {actual_step}"
        )
    if checkpoint.get("architecture") != "ink-jepa-retinal-predictive-field-v1":
        raise ValueError("foundation checkpoint is not an InkJEPA retinal field")
    if checkpoint.get("model_config") != writer_checkpoint.get("foundation_model_config"):
        raise ValueError("writer and foundation model configurations do not match")
    return checkpoint, recorded


def load_models(
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[FovealInkFlow, InkJEPA, dict[str, Any], Path]:
    writer_checkpoint = torch.load(args.writer_checkpoint, map_location="cpu", weights_only=False)
    if writer_checkpoint.get("architecture") != "foveal-continuous-ink-flow-v2":
        raise ValueError("writer checkpoint is not a coarse-plan foveal ink flow")
    foundation_checkpoint, foundation_path = load_foundation_checkpoint(
        writer_checkpoint,
        args.foundation_checkpoint,
    )
    foundation = InkJEPA(ink_jepa_config_from_payload(foundation_checkpoint["model_config"]))
    foundation.load_state_dict(foundation_checkpoint["model"])
    foundation.to(device).eval().requires_grad_(False)
    writer = FovealInkFlow(
        foveal_writer_config_from_payload(writer_checkpoint["writer_config"])
    )
    writer.load_state_dict(writer_checkpoint["writer"])
    writer.to(device).eval().requires_grad_(False)
    return writer, foundation, writer_checkpoint, foundation_path


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


def masked_local_state(
    local_field: torch.Tensor,
    target_mask: torch.Tensor,
) -> torch.Tensor:
    weights = target_mask.to(local_field.dtype)
    state = (local_field * weights[None, ..., None]).sum(dim=(1, 2))
    return state / weights.sum().clamp_min(1.0)


def select_candidate(
    foundation: InkJEPA,
    candidate_pages: torch.Tensor,
    candidate_foveas: torch.Tensor,
    predicted_local: torch.Tensor,
    plan_fovea: torch.Tensor,
    target_mask: torch.Tensor,
    *,
    plan_score_weight: float,
) -> tuple[int, torch.Tensor, torch.Tensor, torch.Tensor]:
    reread = foundation.target_encoder(candidate_pages)["local"]
    actual_local = masked_local_state(reread, target_mask)
    expected = predicted_local.float().expand(actual_local.shape[0], -1)
    cycle_score = F.cosine_similarity(actual_local.float(), expected, dim=-1)
    plan_error = (candidate_foveas.float() - plan_fovea.float()).abs().flatten(1).mean(dim=1)
    scores = cycle_score - plan_score_weight * plan_error
    selected = int(scores.argmax())
    return selected, scores, cycle_score, plan_error


def initial_page_and_cursor(
    args: argparse.Namespace,
    config: RetinalRenderConfig,
) -> tuple[torch.Tensor, tuple[int, int], str]:
    if (args.row is None) != (args.column is None):
        raise ValueError("--row and --column must be supplied together")
    if args.input_image:
        page = image_to_retinal_ink(args.input_image, config)
        inferred = infer_retinal_cursor(page, config, minimum_mean_ink=args.minimum_cell_ink)
        adapter = "direct_continuous_writing_image"
    else:
        page = render_retinal_page(args.text or "", config=config, variant=args.render_variant)
        inferred = retinal_cursor_after_text(args.text or "", config)
        adapter = "deterministic_boundary_rasterizer"
    cursor = (args.row, args.column) if args.row is not None else inferred
    if cursor is None:
        raise ValueError("the prompt page has no remaining visual cells")
    return page, (int(cursor[0]), int(cursor[1])), adapter


def main() -> None:
    args = parse_args()
    if args.new_cells < 1 or args.candidates < 1 or args.flow_steps < 1:
        raise ValueError("new-cells, candidates, and flow-steps must all be positive")
    output = Path(args.out)
    step_root = output / "steps"
    output.mkdir(parents=True, exist_ok=True)
    step_root.mkdir(parents=True, exist_ok=True)
    device = choose_device(args.device)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    writer, foundation, writer_checkpoint, foundation_path = load_models(args, device)
    render_config = render_config_from_checkpoint(writer_checkpoint)
    if (render_config.height, render_config.width) != (
        foundation.config.image_height,
        foundation.config.image_width,
    ):
        raise ValueError("render and retinal model dimensions do not match")
    page, cursor, adapter = initial_page_and_cursor(args, render_config)
    page = page.to(device)
    answer = torch.zeros_like(page)
    save_ink(page, output / "prompt_page.png")
    generator = torch.Generator(device=device).manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    traces: list[dict[str, Any]] = []
    stop_reason = "requested_cells_generated"

    with torch.inference_mode():
        for generation_step in range(args.new_cells):
            row, column = cursor
            hidden_mask, target_mask = future_retinal_masks(
                row=row,
                column=column,
                config=render_config,
                patch_size=foundation.config.patch_size,
            )
            hidden_batch = hidden_mask[None].to(device)
            target_mask = target_mask.to(device)
            context = hide_retinal_regions(page[None], hidden_batch, foundation.config.patch_size)
            with autocast_context(device, args.precision):
                condition, page_plan_logits = retinal_foveal_prediction(
                    foundation,
                    context,
                    hidden_batch,
                    target_mask[None],
                )
                plan_fovea = extract_retinal_fovea(
                    page_plan_logits[0].sigmoid(),
                    row=row,
                    column=column,
                    config=render_config,
                    fovea_size=writer.config.fovea_size,
                )
                plan_for_writer = plan_fovea.mul(2.0).sub(1.0)
                candidate_condition = condition.expand(args.candidates, -1)
                candidate_plan = plan_for_writer.expand(args.candidates, -1, -1, -1)
                sampled = sample_foveal_ink(
                    writer,
                    candidate_condition,
                    candidate_plan,
                    steps=args.flow_steps,
                    guidance_scale=args.guidance_scale,
                    generator=generator,
                )
                candidate_foveas = sampled.float().add(1.0).mul(0.5).clamp(0, 1)
                candidate_pages = torch.stack(
                    [
                        place_retinal_fovea(
                            page,
                            candidate_foveas[index],
                            row=row,
                            column=column,
                            config=render_config,
                        )
                        for index in range(args.candidates)
                    ]
                )
                selected, scores, cycle_scores, plan_errors = select_candidate(
                    foundation,
                    candidate_pages,
                    candidate_foveas,
                    condition[:, : foundation.config.representation_dim],
                    plan_fovea,
                    target_mask,
                    plan_score_weight=args.plan_score_weight,
                )
            selected_fovea = candidate_foveas[selected]
            page = candidate_pages[selected]
            answer = place_retinal_fovea(
                answer,
                selected_fovea,
                row=row,
                column=column,
                config=render_config,
            )
            save_ink(plan_fovea, step_root / f"step_{generation_step:04d}_plan.png")
            save_ink(selected_fovea, step_root / f"step_{generation_step:04d}_selected.png")
            if args.save_candidates:
                for candidate_index, candidate in enumerate(candidate_foveas):
                    save_ink(
                        candidate,
                        step_root / f"step_{generation_step:04d}_candidate_{candidate_index:02d}.png",
                    )
            traces.append(
                {
                    "step": generation_step,
                    "row": row,
                    "column": column,
                    "selected_candidate": selected,
                    "selection_score": float(scores[selected]),
                    "cycle_cosine": float(cycle_scores[selected]),
                    "plan_l1": float(plan_errors[selected]),
                    "all_scores": [float(value) for value in scores],
                }
            )
            next_cursor = advance_retinal_cursor(row, column, render_config)
            if next_cursor is None:
                stop_reason = "page_full"
                break
            cursor = next_cursor

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    save_ink(answer, output / "answer_page.png")
    save_ink(page, output / "complete_page.png")
    receipt = {
        "architecture": "image-native-retinal-read-write-loop-v1",
        "writer_checkpoint": args.writer_checkpoint,
        "foundation_checkpoint": str(foundation_path),
        "boundary_adapter": adapter,
        "student_boundary_input": "continuous single-channel ink image",
        "student_used_text_tokens": False,
        "student_used_unicode_ids": False,
        "student_used_ocr": False,
        "student_used_external_language_model": False,
        "primary_output": "continuous ink image",
        "generated_cells": len(traces),
        "requested_cells": args.new_cells,
        "candidates_per_cell": args.candidates,
        "flow_steps": args.flow_steps,
        "stop_reason": stop_reason,
        "elapsed_seconds": elapsed,
        "cells_per_second": len(traces) / max(elapsed, 1e-9),
        "writer_parameters": sum(parameter.numel() for parameter in writer.parameters()),
        "foundation_parameters": sum(parameter.numel() for parameter in foundation.parameters()),
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

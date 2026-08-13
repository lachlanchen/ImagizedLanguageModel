from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import DataLoader, Dataset

from .causal_glyph_flow import CausalGlyphFlowLM
from .causal_glyph_flow_data import causal_glyph_flow_collate
from .continuous_glyph_codec_training import glyph_sobel_edges


def _autocast(device: torch.device, precision: str):
    if device.type != "cuda" or precision == "fp32":
        return nullcontext()
    dtype = torch.float16 if precision == "fp16" else torch.bfloat16
    return torch.autocast(device_type="cuda", dtype=dtype)


def _binary_counts(
    predicted: torch.Tensor,
    target: torch.Tensor,
) -> tuple[int, int, int]:
    predicted = predicted.bool()
    target = target.bool()
    return (
        int((predicted & target).sum()),
        int(predicted.sum()),
        int(target.sum()),
    )


def _f1(true_positive: int, predicted: int, target: int) -> float:
    denominator = predicted + target
    return 1.0 if denominator == 0 else 2.0 * true_positive / denominator


def _patch_image(patch: torch.Tensor, *, scale: int = 2) -> Image.Image:
    if patch.shape != (1, 32, 32):
        raise ValueError("V35 alignment gallery requires a 32-pixel patch")
    array = patch[0].detach().float().cpu().clamp(0, 1).numpy()
    image = Image.fromarray((array * 255).astype(np.uint8), mode="L")
    return image.resize((32 * scale, 32 * scale), Image.Resampling.NEAREST)


def save_v35_alignment_gallery(
    targets: torch.Tensor,
    reconstructions: torch.Tensor,
    path: str | Path,
    *,
    title: str = "V35 held-out visual interface alignment",
) -> None:
    count = min(32, len(targets), len(reconstructions))
    if count < 1:
        return
    columns = 8
    rows = (count + columns - 1) // columns
    cell_width = 150
    cell_height = 102
    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    font = ImageFont.truetype(font_path, 12) if Path(font_path).is_file() else None
    canvas = Image.new(
        "RGB",
        (columns * cell_width, 34 + rows * cell_height),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 8), title, fill="black", font=font)
    for index in range(count):
        left = index % columns * cell_width
        top = 34 + index // columns * cell_height
        canvas.paste(_patch_image(targets[index]).convert("RGB"), (left + 4, top))
        canvas.paste(
            _patch_image(reconstructions[index]).convert("RGB"),
            (left + 78, top),
        )
        draw.text((left + 4, top + 68), "input", fill="black", font=font)
        draw.text((left + 78, top + 68), "V34", fill="black", font=font)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)


@torch.no_grad()
def evaluate_visual_interface_alignment(
    model: CausalGlyphFlowLM,
    teacher_projection: nn.Conv2d,
    dataset: Dataset[dict[str, Any]],
    *,
    device: torch.device,
    precision: str,
    minimum_patches: int = 2_048,
    batch_size: int = 8,
    num_workers: int = 0,
    gallery_path: str | Path | None = None,
) -> dict[str, Any]:
    if minimum_patches < 1 or batch_size < 1 or num_workers < 0:
        raise ValueError("V35 alignment evaluation settings are invalid")
    if teacher_projection.kernel_size != (
        model.config.patch_size,
        model.config.patch_size,
    ):
        raise ValueError("V35 alignment teacher has the wrong patch shape")
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=causal_glyph_flow_collate,
    )
    teacher_projection = teacher_projection.to(device).requires_grad_(False).eval()
    was_training = model.training
    model.eval()
    patches_seen = 0
    squared_error = 0.0
    cosine_total = 0.0
    finite = True
    ink_true_positive = 0
    predicted_ink = 0
    target_ink = 0
    edge_true_positive = 0
    predicted_edge = 0
    target_edge = 0
    gallery_targets: list[torch.Tensor] = []
    gallery_reconstructions: list[torch.Tensor] = []
    try:
        for batch in loader:
            mask = batch["patch_mask"].to(device, non_blocking=True)
            visible = int(mask.sum(dim=1).max().item())
            mask = mask[:, :visible]
            pixels = batch["pixels"][..., : visible * model.config.patch_size].to(
                device,
                non_blocking=True,
            )
            with _autocast(device, precision):
                latents = model.encode_patches(pixels)
                predicted = model.input_adapter(latents)
                normalized = pixels.clamp(0, 1).mul(2.0).sub(1.0)
                target = teacher_projection(normalized).squeeze(2).transpose(1, 2)
                reconstruction_logits = model.decode_latents(latents)
            active = mask.bool()
            active_count = int(active.sum())
            if active_count < 1:
                continue
            values = (predicted.float() - target.float()).square().mean(dim=-1)
            cosine = F.cosine_similarity(predicted.float(), target.float(), dim=-1)
            squared_error += float(values[active].sum().cpu())
            cosine_total += float(cosine[active].sum().cpu())
            target_patches = model.patchify(pixels)[active].float().cpu()
            reconstructed = (
                reconstruction_logits[active].float().sigmoid().ge(0.5).float().cpu()
            )
            predicted_ink_tensor = reconstructed < 0.5
            target_ink_tensor = target_patches < 0.5
            counts = _binary_counts(predicted_ink_tensor, target_ink_tensor)
            ink_true_positive += counts[0]
            predicted_ink += counts[1]
            target_ink += counts[2]
            predicted_edges_tensor = glyph_sobel_edges(reconstructed).abs() > 0.05
            target_edges_tensor = glyph_sobel_edges(target_patches).abs() > 0.05
            counts = _binary_counts(predicted_edges_tensor, target_edges_tensor)
            edge_true_positive += counts[0]
            predicted_edge += counts[1]
            target_edge += counts[2]
            finite = finite and all(
                bool(torch.isfinite(value).all())
                for value in (latents, predicted, target, reconstruction_logits)
            )
            needed = max(0, 32 - len(gallery_targets))
            if needed:
                gallery_targets.extend(target_patches[:needed])
                gallery_reconstructions.extend(reconstructed[:needed])
            patches_seen += active_count
            if patches_seen >= minimum_patches:
                break
    finally:
        model.train(was_training)
    if patches_seen < minimum_patches:
        raise ValueError(
            f"V35 alignment dataset supplied {patches_seen} patches, "
            f"need {minimum_patches}"
        )
    if gallery_path is not None:
        save_v35_alignment_gallery(
            torch.stack(gallery_targets),
            torch.stack(gallery_reconstructions),
            gallery_path,
        )
    return {
        "finite": finite,
        "patches": patches_seen,
        "mean_squared_error": squared_error / patches_seen,
        "mean_cosine_similarity": cosine_total / patches_seen,
        "codec_ink_f1": _f1(ink_true_positive, predicted_ink, target_ink),
        "codec_edge_f1": _f1(edge_true_positive, predicted_edge, target_edge),
        "gallery": str(gallery_path) if gallery_path is not None else None,
    }


def v35_stage_a_gate(
    alignment: Mapping[str, Any],
    boundary: Mapping[str, Any],
    *,
    initial_core_sha256: str,
    observed_core_sha256: str,
    initial_codec_sha256: str,
    observed_codec_sha256: str,
) -> dict[str, Any]:
    forbidden_flags = (
        "uses_strings",
        "uses_token_ids",
        "uses_unicode_ids",
        "uses_character_ids",
        "uses_embedding_table",
        "uses_vocabulary_logits",
        "uses_ocr",
        "uses_visual_codebook",
        "uses_quantization",
        "uses_retrieval",
        "uses_runtime_teacher",
    )
    checks = {
        "minimum_patches": int(alignment.get("patches", 0)) >= 2_048,
        "finite": bool(alignment.get("finite", False)),
        "core_unchanged": initial_core_sha256 == observed_core_sha256,
        "codec_unchanged": initial_codec_sha256 == observed_codec_sha256,
        "cosine_at_least_0_95": float(
            alignment.get("mean_cosine_similarity", float("-inf"))
        )
        >= 0.95,
        "mse_at_most_0_035": float(
            alignment.get("mean_squared_error", float("inf"))
        )
        <= 0.035,
        "runtime_boundary_clean": not boundary.get(
            "parameter_names_with_forbidden_fragments",
            ["missing"],
        )
        and not any(bool(boundary.get(name, True)) for name in forbidden_flags),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "alignment": dict(alignment),
        "initial_core_sha256": initial_core_sha256,
        "observed_core_sha256": observed_core_sha256,
        "initial_codec_sha256": initial_codec_sha256,
        "observed_codec_sha256": observed_codec_sha256,
    }

from __future__ import annotations

import json
import math
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import DataLoader, Dataset

from .continuous_glyph_codec import ContinuousGlyphCodec
from .continuous_glyph_codec_data import historic_glyph_collate
from .direct_visual_patch_data import direct_patch_collate
from .direct_visual_patch_evaluation import (
    _ocr_character_accuracy,
    _tesseract_text,
    tesseract_identity,
)
from .direct_visual_patch_training import sobel_edges, strip_to_patches


def _autocast(device: torch.device, precision: str):
    if device.type != "cuda" or precision == "fp32":
        return nullcontext()
    dtype = torch.float16 if precision == "fp16" else torch.bfloat16
    return torch.autocast(device_type="cuda", dtype=dtype)


@dataclass
class BinaryPatchMetrics:
    patches: int = 0
    ink_true_positive: int = 0
    predicted_ink: int = 0
    target_ink: int = 0
    edge_true_positive: int = 0
    predicted_edge: int = 0
    target_edge: int = 0
    exact_patches: int = 0

    def update(self, predicted_white: torch.Tensor, target_white: torch.Tensor) -> None:
        if predicted_white.shape != target_white.shape:
            raise ValueError("V34 metric prediction and target do not align")
        if predicted_white.ndim != 4 or predicted_white.shape[1:] != (1, 32, 32):
            raise ValueError("V34 patch metrics require [B,1,32,32]")
        predicted_white = predicted_white.bool()
        target_white = target_white.bool()
        predicted_ink = ~predicted_white
        target_ink = ~target_white
        self.ink_true_positive += int((predicted_ink & target_ink).sum())
        self.predicted_ink += int(predicted_ink.sum())
        self.target_ink += int(target_ink.sum())

        predicted_edge = (
            sobel_edges(predicted_white.float().unsqueeze(1)).abs() > 0.05
        ).squeeze(1)
        target_edge = (
            sobel_edges(target_white.float().unsqueeze(1)).abs() > 0.05
        ).squeeze(1)
        self.edge_true_positive += int((predicted_edge & target_edge).sum())
        self.predicted_edge += int(predicted_edge.sum())
        self.target_edge += int(target_edge.sum())
        self.exact_patches += int(
            predicted_white.eq(target_white).flatten(1).all(dim=1).sum()
        )
        self.patches += predicted_white.shape[0]

    @staticmethod
    def _f1(true_positive: int, predicted: int, target: int) -> float:
        denominator = predicted + target
        return 1.0 if denominator == 0 else 2.0 * true_positive / denominator

    def report(self) -> dict[str, Any]:
        if self.patches < 1:
            raise ValueError("V34 cannot report empty patch metrics")
        return {
            "patches": self.patches,
            "ink_pixel_f1": self._f1(
                self.ink_true_positive,
                self.predicted_ink,
                self.target_ink,
            ),
            "edge_f1": self._f1(
                self.edge_true_positive,
                self.predicted_edge,
                self.target_edge,
            ),
            "exact_patch_rate": self.exact_patches / self.patches,
            "predicted_ink_pixels": self.predicted_ink,
            "target_ink_pixels": self.target_ink,
        }


class LatentMoments:
    def __init__(self) -> None:
        self.count = 0
        self.total: torch.Tensor | None = None
        self.square_total: torch.Tensor | None = None
        self.finite = True

    def update(self, latents: torch.Tensor) -> None:
        if latents.ndim != 2:
            raise ValueError("V34 latent moments require [B,D]")
        values = latents.detach().float().cpu().double()
        self.finite = self.finite and bool(torch.isfinite(values).all())
        batch_total = values.sum(dim=0)
        batch_square_total = values.square().sum(dim=0)
        if self.total is None:
            self.total = batch_total
            self.square_total = batch_square_total
        else:
            if values.shape[1] != self.total.shape[0]:
                raise ValueError("V34 latent dimensions changed during evaluation")
            self.total += batch_total
            assert self.square_total is not None
            self.square_total += batch_square_total
        self.count += values.shape[0]

    def report(self) -> dict[str, Any]:
        if self.count < 2 or self.total is None or self.square_total is None:
            raise ValueError("V34 latent audit requires at least two patches")
        mean = self.total / self.count
        variance = (self.square_total / self.count - mean.square()).clamp_min(0.0)
        standard_deviation = variance.sqrt()
        return {
            "samples": self.count,
            "dimensions": standard_deviation.shape[0],
            "finite": self.finite,
            "mean_per_dimension_std": float(standard_deviation.mean()),
            "minimum_per_dimension_std": float(standard_deviation.min()),
            "maximum_per_dimension_std": float(standard_deviation.max()),
        }


def _patches_to_image(patches: torch.Tensor) -> Image.Image:
    if patches.ndim != 4 or patches.shape[1:] != (1, 32, 32):
        raise ValueError("V34 strip image requires [L,1,32,32]")
    strip = patches.permute(1, 2, 0, 3).reshape(1, 32, -1)
    array = strip[0].detach().float().cpu().clamp(0, 1).numpy()
    return Image.fromarray((array * 255).astype(np.uint8), mode="L")


def _model_patch_batch(
    model: ContinuousGlyphCodec,
    patches: torch.Tensor,
    *,
    device: torch.device,
    precision: str,
    noise_sigma: float | None,
    noise_generator: torch.Generator | None,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor, bool]:
    patches = patches.to(device, non_blocking=True)
    with _autocast(device, precision):
        latents = model.encode(patches)
        clean_logits = model.decode(latents)
        noisy_logits = None
        if noise_sigma is not None:
            if noise_generator is None:
                raise ValueError("V34 noisy evaluation requires a generator")
            noise = torch.randn(
                latents.shape,
                generator=noise_generator,
                device=device,
                dtype=latents.dtype,
            ).mul(noise_sigma)
            noisy_logits = model.decode(latents + noise)
    finite = bool(torch.isfinite(latents).all()) and bool(
        torch.isfinite(clean_logits).all()
    )
    if noisy_logits is not None:
        finite = finite and bool(torch.isfinite(noisy_logits).all())
    clean = clean_logits.float().sigmoid().ge(0.5).cpu()
    noisy = (
        noisy_logits.float().sigmoid().ge(0.5).cpu()
        if noisy_logits is not None
        else None
    )
    return clean, noisy, latents, finite


def save_rendered_codec_gallery(
    rows: Sequence[Mapping[str, Any]],
    path: str | Path,
    *,
    title: str,
) -> None:
    if not rows:
        return
    selected = list(rows[:8])
    width = 1400
    label_width = 110
    row_height = 122
    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    font = ImageFont.truetype(font_path, 13) if Path(font_path).is_file() else None
    canvas = Image.new("RGB", (width, 34 + row_height * len(selected)), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 8), title, fill="black", font=font)
    for index, row in enumerate(selected):
        top = 34 + index * row_height
        for line, (label, key) in enumerate(
            (("target", "target"), ("clean", "clean"), ("noise .03", "noisy"))
        ):
            draw.text((8, top + line * 38 + 9), label, fill="black", font=font)
            image = row[key]
            maximum_width = width - label_width - 8
            if image.width > maximum_width:
                image = image.crop((0, 0, maximum_width, image.height))
            canvas.paste(image.convert("RGB"), (label_width, top + line * 38))
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)


def save_historic_codec_gallery(
    targets: torch.Tensor,
    predictions: torch.Tensor,
    metadata: Sequence[Mapping[str, Any]],
    path: str | Path,
    *,
    title: str,
) -> None:
    count = min(32, targets.shape[0], predictions.shape[0], len(metadata))
    if count < 1:
        return
    columns = 8
    rows = math.ceil(count / columns)
    cell_width = 150
    cell_height = 104
    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    font = ImageFont.truetype(font_path, 11) if Path(font_path).is_file() else None
    canvas = Image.new(
        "RGB",
        (columns * cell_width, 34 + rows * cell_height),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 8), title, fill="black", font=font)
    for index in range(count):
        left = (index % columns) * cell_width
        top = 34 + (index // columns) * cell_height
        target = Image.fromarray(
            (targets[index, 0].float().cpu().numpy() * 255).astype(np.uint8),
            mode="L",
        ).resize((64, 64), Image.Resampling.NEAREST)
        prediction = Image.fromarray(
            (predictions[index, 0].float().cpu().numpy() * 255).astype(np.uint8),
            mode="L",
        ).resize((64, 64), Image.Resampling.NEAREST)
        canvas.paste(target.convert("RGB"), (left + 4, top))
        canvas.paste(prediction.convert("RGB"), (left + 76, top))
        label = f"{metadata[index].get('stage', '')}:{metadata[index].get('label', '')}"
        draw.text((left + 4, top + 70), label[:22], fill="black", font=font)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)


@torch.no_grad()
def evaluate_rendered_codec(
    model: ContinuousGlyphCodec,
    dataset: Dataset[dict[str, Any]],
    *,
    device: torch.device,
    precision: str,
    minimum_patches: int = 4_096,
    strip_batch_size: int = 8,
    num_workers: int = 0,
    noise_sigma: float = 0.03,
    noise_seed: int = 20_263_434,
    gallery_path: str | Path | None = None,
    gallery_title: str = "V34 rendered glyph codec",
    ocr_language: str = "chi_tra",
) -> tuple[dict[str, Any], LatentMoments]:
    if minimum_patches < 1 or strip_batch_size < 1 or noise_sigma < 0.0:
        raise ValueError("V34 rendered evaluation settings are invalid")
    loader = DataLoader(
        dataset,
        batch_size=strip_batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        collate_fn=direct_patch_collate,
    )
    clean_metrics = BinaryPatchMetrics()
    noisy_metrics = BinaryPatchMetrics()
    moments = LatentMoments()
    target_ocr: list[float] = []
    reconstruction_ocr: list[float] = []
    paired_ocr: list[float] = []
    examples: list[dict[str, Any]] = []
    gallery_rows: list[dict[str, Any]] = []
    noise_generator = torch.Generator(device=device)
    noise_generator.manual_seed(noise_seed)
    finite = True
    samples = 0
    started = time.monotonic()
    was_training = model.training
    model.eval()
    try:
        for batch in loader:
            lengths = batch["patch_mask"].sum(dim=1).long()
            source = strip_to_patches(batch["pixels"], 32)
            active = torch.cat(
                [source[index, : int(length)] for index, length in enumerate(lengths)]
            )
            clean, noisy, latents, batch_finite = _model_patch_batch(
                model,
                active,
                device=device,
                precision=precision,
                noise_sigma=noise_sigma,
                noise_generator=noise_generator,
            )
            assert noisy is not None
            finite = finite and batch_finite
            moments.update(latents)
            target_white = active >= 0.5
            clean_metrics.update(clean, target_white)
            noisy_metrics.update(noisy, target_white)

            offset = 0
            for index, length_tensor in enumerate(lengths):
                length = int(length_tensor)
                clean_strip = clean[offset : offset + length].float()
                noisy_strip = noisy[offset : offset + length].float()
                target_strip = source[index, :length]
                offset += length
                target_image = _patches_to_image(target_strip)
                clean_image = _patches_to_image(clean_strip)
                noisy_image = _patches_to_image(noisy_strip)
                expected = str(batch["metadata"][index].get("text", ""))
                target_observed = _tesseract_text(target_image, language=ocr_language)
                reconstruction_observed = _tesseract_text(
                    clean_image,
                    language=ocr_language,
                )
                target_accuracy = _ocr_character_accuracy(expected, target_observed)
                reconstruction_accuracy = _ocr_character_accuracy(
                    expected,
                    reconstruction_observed,
                )
                agreement = _ocr_character_accuracy(
                    target_observed,
                    reconstruction_observed,
                )
                target_ocr.append(target_accuracy)
                reconstruction_ocr.append(reconstruction_accuracy)
                paired_ocr.append(agreement)
                if len(examples) < 8:
                    examples.append(
                        {
                            "identifier": batch["metadata"][index]["identifier"],
                            "expected": expected,
                            "target_observed": target_observed,
                            "reconstruction_observed": reconstruction_observed,
                            "target_character_accuracy": target_accuracy,
                            "reconstruction_character_accuracy": reconstruction_accuracy,
                            "paired_ocr_agreement": agreement,
                        }
                    )
                    gallery_rows.append(
                        {
                            "target": target_image,
                            "clean": clean_image,
                            "noisy": noisy_image,
                        }
                    )
                samples += 1
            if clean_metrics.patches >= minimum_patches:
                break
    finally:
        model.train(was_training)
    if clean_metrics.patches < minimum_patches:
        raise ValueError(
            f"V34 rendered split supplied {clean_metrics.patches} patches; "
            f"need {minimum_patches}"
        )
    if gallery_path is not None:
        save_rendered_codec_gallery(
            gallery_rows,
            gallery_path,
            title=gallery_title,
        )
    mean_target_ocr = sum(target_ocr) / len(target_ocr)
    mean_reconstruction_ocr = sum(reconstruction_ocr) / len(reconstruction_ocr)
    elapsed = time.monotonic() - started
    return {
        "samples": samples,
        "finite": finite,
        "clean": clean_metrics.report(),
        "noisy": {"sigma": noise_sigma, **noisy_metrics.report()},
        "ocr": {
            "evaluator": tesseract_identity(ocr_language),
            "target_character_accuracy": mean_target_ocr,
            "reconstruction_character_accuracy": mean_reconstruction_ocr,
            "retention": (
                mean_reconstruction_ocr / mean_target_ocr
                if mean_target_ocr > 0.0
                else 0.0
            ),
            "paired_agreement": sum(paired_ocr) / len(paired_ocr),
        },
        "examples": examples,
        "elapsed_seconds": elapsed,
        "patches_per_second": clean_metrics.patches / elapsed,
    }, moments


@torch.no_grad()
def evaluate_historic_codec(
    model: ContinuousGlyphCodec,
    dataset: Dataset[dict[str, Any]],
    *,
    device: torch.device,
    precision: str,
    minimum_patches: int = 4_096,
    batch_size: int = 512,
    num_workers: int = 0,
    gallery_path: str | Path | None = None,
    gallery_title: str = "V34 historical glyph codec",
) -> tuple[dict[str, Any], LatentMoments]:
    if minimum_patches < 1 or batch_size < 1:
        raise ValueError("V34 historical evaluation settings are invalid")
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        collate_fn=historic_glyph_collate,
    )
    metrics = BinaryPatchMetrics()
    moments = LatentMoments()
    finite = True
    gallery_targets: list[torch.Tensor] = []
    gallery_predictions: list[torch.Tensor] = []
    gallery_metadata: list[Mapping[str, Any]] = []
    started = time.monotonic()
    was_training = model.training
    model.eval()
    try:
        for batch in loader:
            targets = batch["pixels"]
            predictions, _, latents, batch_finite = _model_patch_batch(
                model,
                targets,
                device=device,
                precision=precision,
                noise_sigma=None,
                noise_generator=None,
            )
            finite = finite and batch_finite
            moments.update(latents)
            metrics.update(predictions, targets >= 0.5)
            if len(gallery_metadata) < 32:
                take = min(32 - len(gallery_metadata), targets.shape[0])
                gallery_targets.append(targets[:take])
                gallery_predictions.append(predictions[:take].float())
                gallery_metadata.extend(batch["metadata"][:take])
            if metrics.patches >= minimum_patches:
                break
    finally:
        model.train(was_training)
    required = min(minimum_patches, len(dataset))
    if metrics.patches < required:
        raise ValueError(
            f"V34 historical split supplied {metrics.patches} patches; need {required}"
        )
    if gallery_path is not None:
        save_historic_codec_gallery(
            torch.cat(gallery_targets),
            torch.cat(gallery_predictions),
            gallery_metadata,
            gallery_path,
            title=gallery_title,
        )
    elapsed = time.monotonic() - started
    return {
        "finite": finite,
        **metrics.report(),
        "elapsed_seconds": elapsed,
        "patches_per_second": metrics.patches / elapsed,
    }, moments


@torch.no_grad()
def evaluate_blank_codec(
    model: ContinuousGlyphCodec,
    *,
    device: torch.device,
    precision: str,
    patches: int = 512,
    batch_size: int = 512,
) -> dict[str, Any]:
    if patches < 1 or batch_size < 1:
        raise ValueError("V34 blank evaluation settings are invalid")
    false_ink = 0
    finite = True
    completed = 0
    was_training = model.training
    model.eval()
    try:
        while completed < patches:
            count = min(batch_size, patches - completed)
            targets = torch.ones(count, 1, 32, 32)
            predictions, _, _, batch_finite = _model_patch_batch(
                model,
                targets,
                device=device,
                precision=precision,
                noise_sigma=None,
                noise_generator=None,
            )
            finite = finite and batch_finite
            false_ink += int((~predictions).sum())
            completed += count
    finally:
        model.train(was_training)
    return {
        "patches": patches,
        "finite": finite,
        "false_ink_rate": false_ink / (patches * 32 * 32),
    }


def merge_latent_moments(*moments: LatentMoments) -> LatentMoments:
    merged = LatentMoments()
    for source in moments:
        if source.count < 1 or source.total is None or source.square_total is None:
            continue
        if merged.total is None:
            merged.total = source.total.clone()
            merged.square_total = source.square_total.clone()
        else:
            if source.total.shape != merged.total.shape:
                raise ValueError("V34 latent audits use different dimensions")
            merged.total += source.total
            assert merged.square_total is not None
            merged.square_total += source.square_total
        merged.count += source.count
        merged.finite = merged.finite and source.finite
    return merged


def v34_development_gate(
    report: Mapping[str, Any],
    *,
    updates_complete: bool,
    checkpoint_finite: bool,
    peak_vram_bytes: int,
) -> dict[str, Any]:
    rendered = report["rendered"]
    historical = report["historical"]
    blank = report["blank"]
    latent = report["latent"]
    gates = {
        "finite_model_latent_and_checkpoint": bool(report["finite"])
        and bool(latent["finite"])
        and checkpoint_finite,
        "clean_rendered_ink_f1_at_least_0_985": (
            rendered["clean"]["ink_pixel_f1"] >= 0.985
        ),
        "clean_rendered_edge_f1_at_least_0_980": (
            rendered["clean"]["edge_f1"] >= 0.980
        ),
        "clean_rendered_ocr_retention_at_least_0_95": (
            rendered["ocr"]["retention"] >= 0.95
        ),
        "noisy_rendered_ink_f1_at_least_0_970": (
            rendered["noisy"]["ink_pixel_f1"] >= 0.970
        ),
        "noisy_rendered_edge_f1_at_least_0_950": (
            rendered["noisy"]["edge_f1"] >= 0.950
        ),
        "historical_ink_f1_at_least_0_960": historical["ink_pixel_f1"] >= 0.960,
        "historical_edge_f1_at_least_0_940": historical["edge_f1"] >= 0.940,
        "blank_false_ink_below_0_005": blank["false_ink_rate"] < 0.005,
        "mean_latent_dimension_std_at_least_0_10": (
            latent["mean_per_dimension_std"] >= 0.10
        ),
        "updates_complete": updates_complete,
        "peak_vram_below_12_gib": peak_vram_bytes < 12 * 1024**3,
    }
    return {
        "gates": gates,
        "pass": all(gates.values()),
        "updates_complete": updates_complete,
        "checkpoint_finite": checkpoint_finite,
        "peak_vram_bytes": peak_vram_bytes,
    }


def v34_sealed_transfer_gate(
    development: Mapping[str, Any],
    sealed: Mapping[str, Any],
) -> dict[str, Any]:
    paths = {
        "clean_rendered_ink_f1": ("rendered", "clean", "ink_pixel_f1"),
        "clean_rendered_edge_f1": ("rendered", "clean", "edge_f1"),
        "clean_rendered_ocr_retention": ("rendered", "ocr", "retention"),
        "noisy_rendered_ink_f1": ("rendered", "noisy", "ink_pixel_f1"),
        "noisy_rendered_edge_f1": ("rendered", "noisy", "edge_f1"),
        "historical_ink_f1": ("historical", "ink_pixel_f1"),
        "historical_edge_f1": ("historical", "edge_f1"),
    }

    def value(report: Mapping[str, Any], path: Sequence[str]) -> float:
        node: Any = report
        for key in path:
            node = node[key]
        return float(node)

    ratios: dict[str, Any] = {}
    gates: dict[str, bool] = {}
    for name, path in paths.items():
        development_value = value(development, path)
        sealed_value = value(sealed, path)
        ratio = sealed_value / development_value if development_value > 0.0 else 0.0
        ratios[name] = {
            "development": development_value,
            "sealed": sealed_value,
            "ratio": ratio,
        }
        gates[f"{name}_retains_at_least_0_97"] = ratio >= 0.97
    gates["sealed_finite"] = bool(sealed["finite"])
    return {"ratios": ratios, "gates": gates, "pass": all(gates.values())}


def evaluate_continuous_glyph_codec(
    model: ContinuousGlyphCodec,
    rendered_dataset: Dataset[dict[str, Any]],
    historical_dataset: Dataset[dict[str, Any]],
    *,
    split: str,
    device: torch.device,
    precision: str,
    rendered_minimum_patches: int = 4_096,
    historical_minimum_patches: int = 4_096,
    rendered_batch_size: int = 8,
    historical_batch_size: int = 512,
    num_workers: int = 0,
    gallery_directory: str | Path | None = None,
) -> dict[str, Any]:
    if split not in {"development", "sealed"}:
        raise ValueError("V34 evaluator split must be development or sealed")
    directory = Path(gallery_directory) if gallery_directory is not None else None
    rendered, rendered_moments = evaluate_rendered_codec(
        model,
        rendered_dataset,
        device=device,
        precision=precision,
        minimum_patches=rendered_minimum_patches,
        strip_batch_size=rendered_batch_size,
        num_workers=num_workers,
        gallery_path=(directory / f"{split}_rendered_gallery.png" if directory else None),
        gallery_title=f"V34 {split} rendered codec: target / clean / latent noise",
    )
    historical, historical_moments = evaluate_historic_codec(
        model,
        historical_dataset,
        device=device,
        precision=precision,
        minimum_patches=historical_minimum_patches,
        batch_size=historical_batch_size,
        num_workers=num_workers,
        gallery_path=(directory / f"{split}_historic_gallery.png" if directory else None),
        gallery_title=f"V34 {split} historical codec: target / reconstruction",
    )
    blank = evaluate_blank_codec(
        model,
        device=device,
        precision=precision,
    )
    latent = merge_latent_moments(rendered_moments, historical_moments).report()
    return {
        "split": split,
        "finite": bool(rendered["finite"])
        and bool(historical["finite"])
        and bool(blank["finite"])
        and bool(latent["finite"]),
        "rendered": rendered,
        "historical": historical,
        "blank": blank,
        "latent": latent,
    }


def write_v34_evaluation(payload: Mapping[str, Any], path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

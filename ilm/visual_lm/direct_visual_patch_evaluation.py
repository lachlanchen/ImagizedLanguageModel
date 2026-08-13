from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
import tempfile
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import DataLoader, Dataset

from .direct_visual_patch_data import direct_patch_collate
from .direct_visual_patch_lm import DirectVisualPatchLM
from .direct_visual_patch_training import sobel_edges, strip_to_patches
from .visual_semantic_raster_data import normalize_visible_text


def _autocast(device: torch.device, precision: str):
    if device.type != "cuda" or precision == "fp32":
        return nullcontext()
    dtype = torch.float16 if precision == "fp16" else torch.bfloat16
    return torch.autocast(device_type="cuda", dtype=dtype)


def _edit_distance(left: str, right: str) -> int:
    if len(left) < len(right):
        left, right = right, left
    previous = list(range(len(right) + 1))
    for left_index, left_character in enumerate(left, start=1):
        current = [left_index]
        for right_index, right_character in enumerate(right, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[right_index] + 1,
                    previous[right_index - 1] + (left_character != right_character),
                )
            )
        previous = current
    return previous[-1]


def _ocr_character_accuracy(expected: str, observed: str) -> float:
    expected = normalize_visible_text(expected)
    observed = normalize_visible_text(observed)
    if not expected:
        return float(not observed)
    return max(0.0, 1.0 - _edit_distance(expected, observed) / len(expected))


def _tesseract_text(image: Image.Image, *, language: str = "chi_sim") -> str:
    scaled = image.resize((image.width * 2, image.height * 2), Image.Resampling.NEAREST)
    with tempfile.TemporaryDirectory(prefix="ilm-v33-ocr-") as directory:
        path = Path(directory) / "strip.png"
        scaled.save(path)
        result = subprocess.run(
            [
                "tesseract",
                str(path),
                "stdout",
                "-l",
                language,
                "--psm",
                "7",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
    if result.returncode:
        raise RuntimeError(f"tesseract failed: {result.stderr.strip()}")
    return normalize_visible_text(result.stdout)


def tesseract_identity(language: str) -> dict[str, str]:
    executable = shutil.which("tesseract")
    if executable is None:
        raise FileNotFoundError("V33 evaluator requires the tesseract executable")
    version_result = subprocess.run(
        [executable, "--version"],
        check=True,
        capture_output=True,
        text=True,
    )
    list_result = subprocess.run(
        [executable, "--list-langs"],
        check=True,
        capture_output=True,
        text=True,
    )
    listing = f"{list_result.stdout}\n{list_result.stderr}"
    match = re.search(r'List of available languages in "([^"]+)"', listing)
    if match is None:
        raise RuntimeError("V33 evaluator could not locate Tesseract traineddata")
    traineddata = Path(match.group(1)) / f"{language}.traineddata"
    if not traineddata.is_file():
        raise FileNotFoundError(f"V33 evaluator lacks {traineddata}")
    digest = hashlib.sha256()
    with traineddata.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return {
        "executable": executable,
        "version": version_result.stdout.splitlines()[0],
        "language": language,
        "traineddata": str(traineddata.resolve()),
        "traineddata_sha256": digest.hexdigest(),
        "page_segmentation_mode": "7",
        "scale": "2x-nearest",
    }


def _patches_to_image(patches: torch.Tensor, count: int) -> Image.Image:
    if patches.ndim != 4 or patches.shape[1] != 1:
        raise ValueError("V33 image patches must be [L,1,H,W]")
    selected = patches[:count].permute(1, 2, 0, 3).reshape(
        1,
        patches.shape[-2],
        count * patches.shape[-1],
    )
    array = selected[0].detach().float().cpu().clamp(0, 1).numpy()
    return Image.fromarray((array * 255).astype(np.uint8), mode="L")


def _binary_f1(predicted: torch.Tensor, target: torch.Tensor) -> float:
    predicted = predicted.bool()
    target = target.bool()
    true_positive = (predicted & target).sum().float()
    denominator = predicted.sum().float() + target.sum().float()
    if not bool(denominator > 0):
        return 1.0
    return float((2.0 * true_positive / denominator).cpu())


def save_calibration_gallery(
    rows: Sequence[dict[str, Any]],
    path: str | Path,
    *,
    maximum_rows: int = 12,
    title: str = "V33 Stage-A held-out visual calibration",
) -> None:
    selected = list(rows[:maximum_rows])
    if not selected:
        return
    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    font = ImageFont.truetype(font_path, 14) if Path(font_path).exists() else None
    display_width = 1400
    label_width = 150
    strip_height = 32
    row_height = 92
    canvas = Image.new("RGB", (display_width, 36 + len(selected) * row_height), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 8), title, fill="black", font=font)
    for index, row in enumerate(selected):
        top = 36 + index * row_height
        draw.text((8, top + 5), f"target {index + 1}", fill="black", font=font)
        draw.text((8, top + 47), "reconstruction", fill="black", font=font)
        for offset, key in ((4, "target_image"), (44, "reconstruction_image")):
            image = row[key]
            if image.width > display_width - label_width - 8:
                image = image.crop((0, 0, display_width - label_width - 8, strip_height))
            canvas.paste(image.convert("RGB"), (label_width, top + offset))
        draw.text(
            (display_width - 255, top + 5),
            f"OCR {row['character_accuracy']:.3f}",
            fill="black",
            font=font,
        )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


@torch.no_grad()
def _collect_visual_calibration(
    model: DirectVisualPatchLM,
    dataset: Dataset[dict[str, Any]],
    *,
    device: torch.device,
    precision: str,
    minimum_patches: int = 2_048,
    batch_size: int = 8,
    num_workers: int = 0,
    gallery_path: str | Path | None = None,
    ocr_language: str = "chi_sim",
    include_target_ocr: bool = False,
    gallery_title: str = "V33 Stage-A held-out visual calibration",
) -> dict[str, Any]:
    if minimum_patches < 1 or batch_size < 1:
        raise ValueError("V33 calibration audit sizes must be positive")
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        collate_fn=direct_patch_collate,
    )
    was_training = model.training
    model.eval()
    ink_f1_values: list[float] = []
    edge_f1_values: list[float] = []
    character_accuracies: list[float] = []
    target_character_accuracies: list[float] = []
    paired_ocr_agreements: list[float] = []
    blank_false_ink: list[float] = []
    rows: list[dict[str, Any]] = []
    patch_total = 0
    finite = True
    try:
        for batch in loader:
            lengths = batch["patch_mask"].sum(dim=1).long()
            visible = int(lengths.max().item())
            pixels = batch["pixels"][..., : visible * model.config.patch_size].to(device)
            mask = batch["patch_mask"][:, :visible].to(device)
            with _autocast(device, precision):
                output = model(pixels, mask)
            probability = output.patch_logits.sigmoid().float().cpu()
            finite = finite and bool(torch.isfinite(probability).all())
            predicted_white = probability >= 0.5
            target = strip_to_patches(pixels.float().cpu(), model.config.patch_size)
            target_white = target >= 0.5
            for index, length_tensor in enumerate(lengths):
                length = int(length_tensor.item())
                active_predicted_ink = ~predicted_white[index, :length]
                active_target_ink = ~target_white[index, :length]
                ink_f1_values.append(_binary_f1(active_predicted_ink, active_target_ink))
                predicted_edges = sobel_edges(
                    predicted_white[index : index + 1, :length].float()
                ).abs() > 0.05
                target_edges = sobel_edges(
                    target_white[index : index + 1, :length].float()
                ).abs() > 0.05
                edge_f1_values.append(_binary_f1(predicted_edges, target_edges))
                inactive_target = ~active_target_ink.flatten(1).any(dim=1)
                if bool(inactive_target.any()):
                    false_ink = active_predicted_ink[inactive_target].float().mean()
                    blank_false_ink.append(float(false_ink))
                target_image = _patches_to_image(target[index], length)
                reconstruction_image = _patches_to_image(
                    predicted_white[index].float(), length
                )
                expected = str(batch["metadata"][index].get("text", ""))
                observed = _tesseract_text(
                    reconstruction_image,
                    language=ocr_language,
                )
                accuracy = _ocr_character_accuracy(expected, observed)
                character_accuracies.append(accuracy)
                target_observed = ""
                target_accuracy = 0.0
                paired_agreement = 0.0
                if include_target_ocr:
                    target_observed = _tesseract_text(
                        target_image,
                        language=ocr_language,
                    )
                    target_accuracy = _ocr_character_accuracy(expected, target_observed)
                    paired_agreement = _ocr_character_accuracy(
                        target_observed,
                        observed,
                    )
                    target_character_accuracies.append(target_accuracy)
                    paired_ocr_agreements.append(paired_agreement)
                if len(rows) < 12:
                    row = {
                        "identifier": batch["metadata"][index]["identifier"],
                        "expected": expected,
                        "observed": observed,
                        "character_accuracy": accuracy,
                        "target_image": target_image,
                        "reconstruction_image": reconstruction_image,
                    }
                    if include_target_ocr:
                        row.update(
                            {
                                "target_observed": target_observed,
                                "target_character_accuracy": target_accuracy,
                                "paired_ocr_agreement": paired_agreement,
                            }
                        )
                    rows.append(row)
                patch_total += length
                if patch_total >= minimum_patches:
                    break
            if patch_total >= minimum_patches:
                break
    finally:
        model.train(was_training)
    if patch_total < minimum_patches:
        raise ValueError(
            f"V33 calibration dataset supplied {patch_total} patches, need {minimum_patches}"
        )
    if gallery_path is not None:
        save_calibration_gallery(rows, gallery_path, title=gallery_title)

    def mean(values: Sequence[float]) -> float:
        return float(sum(values) / len(values)) if values else 0.0

    report = {
        "samples": len(character_accuracies),
        "patches": patch_total,
        "finite": finite,
        "ink_pixel_f1": mean(ink_f1_values),
        "edge_f1": mean(edge_f1_values),
        "ocr_character_accuracy": mean(character_accuracies),
        "blank_patch_false_ink_rate": mean(blank_false_ink),
        "examples": [
            {
                key: value
                for key, value in row.items()
                if key not in {"target_image", "reconstruction_image"}
            }
            for row in rows
        ],
    }
    if include_target_ocr:
        target_accuracy = mean(target_character_accuracies)
        reconstruction_accuracy = report["ocr_character_accuracy"]
        report.update(
            {
                "target_ocr_character_accuracy": target_accuracy,
                "reconstruction_ocr_character_accuracy": reconstruction_accuracy,
                "ocr_retention": (
                    reconstruction_accuracy / target_accuracy
                    if target_accuracy > 0.0
                    else 0.0
                ),
                "paired_ocr_agreement": mean(paired_ocr_agreements),
            }
        )
    return report


def evaluate_visual_calibration(
    model: DirectVisualPatchLM,
    dataset: Dataset[dict[str, Any]],
    *,
    device: torch.device,
    precision: str,
    minimum_patches: int = 2_048,
    batch_size: int = 8,
    num_workers: int = 0,
    gallery_path: str | Path | None = None,
) -> dict[str, Any]:
    report = _collect_visual_calibration(
        model,
        dataset,
        device=device,
        precision=precision,
        minimum_patches=minimum_patches,
        batch_size=batch_size,
        num_workers=num_workers,
        gallery_path=gallery_path,
    )
    report["gates"] = {
        "finite": report["finite"],
        "ink_pixel_f1_at_least_0_90": report["ink_pixel_f1"] >= 0.90,
        "edge_f1_at_least_0_90": report["edge_f1"] >= 0.90,
        "ocr_character_accuracy_at_least_0_95": (
            report["ocr_character_accuracy"] >= 0.95
        ),
        "blank_false_ink_below_0_01": report["blank_patch_false_ink_rate"] < 0.01,
    }
    report["pass"] = all(report["gates"].values())
    return report


def evaluate_visual_calibration_v331(
    model: DirectVisualPatchLM,
    dataset: Dataset[dict[str, Any]],
    *,
    device: torch.device,
    precision: str,
    minimum_patches: int = 2_048,
    batch_size: int = 8,
    num_workers: int = 0,
    gallery_path: str | Path | None = None,
    ocr_language: str = "chi_tra",
) -> dict[str, Any]:
    identity = tesseract_identity(ocr_language)
    report = _collect_visual_calibration(
        model,
        dataset,
        device=device,
        precision=precision,
        minimum_patches=minimum_patches,
        batch_size=batch_size,
        num_workers=num_workers,
        gallery_path=gallery_path,
        ocr_language=ocr_language,
        include_target_ocr=True,
        gallery_title="V33.1 held-out direct-raster calibration",
    )
    report["ocr_evaluator"] = identity
    report["pixel_threshold"] = 0.5
    report["gates"] = {
        "finite": report["finite"],
        "ink_pixel_f1_at_least_0_90": report["ink_pixel_f1"] >= 0.90,
        "edge_f1_at_least_0_90": report["edge_f1"] >= 0.90,
        "target_ocr_accuracy_at_least_0_60": (
            report["target_ocr_character_accuracy"] >= 0.60
        ),
        "ocr_retention_at_least_0_90": report["ocr_retention"] >= 0.90,
        "paired_ocr_agreement_at_least_0_80": (
            report["paired_ocr_agreement"] >= 0.80
        ),
        "blank_false_ink_below_0_01": report["blank_patch_false_ink_rate"] < 0.01,
    }
    report["pass"] = all(report["gates"].values())
    return report


def write_evaluation_json(payload: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

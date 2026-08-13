from __future__ import annotations

import hashlib
import math
import re
import shutil
import subprocess
import tempfile
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import DataLoader, Dataset, Subset

from .causal_glyph_flow import (
    V35_ARCHITECTURE,
    CausalGlyphFlowConfig,
    CausalGlyphFlowLM,
    causal_glyph_flow_boundary_receipt,
    file_sha256,
)
from .causal_glyph_flow_data import causal_glyph_flow_collate
from .continuous_glyph_codec_training import glyph_sobel_edges
from .visual_semantic_raster_data import normalize_visible_text


V35_DEVELOPMENT_ARCHITECTURE = "causal-glyph-flow-v35-development-audit"
V35_OCR_LANGUAGE = "chi_sim+chi_tra"
V35_EVALUATION_SEED = 20_263_535
V35_STATUS = (
    "not-qualified",
    "visual-causal-qualified",
    "semantic-raster-qualified",
)


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


def ocr_character_accuracy(expected: str, observed: str) -> float:
    expected = normalize_visible_text(expected)
    observed = normalize_visible_text(observed)
    if not expected:
        return float(not observed)
    return max(0.0, 1.0 - _edit_distance(expected, observed) / len(expected))


def text_is_readable(text: str) -> bool:
    return any(character.isalnum() for character in normalize_visible_text(text))


class TesseractStripOCR:
    """Deterministic evaluator-only OCR; never passed to the student."""

    def __init__(self, language: str = V35_OCR_LANGUAGE) -> None:
        self.language = language
        self._identity = self._build_identity()
        self._cache: dict[str, str] = {}

    def _build_identity(self) -> dict[str, Any]:
        executable = shutil.which("tesseract")
        if executable is None:
            raise FileNotFoundError("V35 evaluation requires the tesseract executable")
        version = subprocess.run(
            [executable, "--version"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.splitlines()[0]
        listing = subprocess.run(
            [executable, "--list-langs"],
            check=True,
            capture_output=True,
            text=True,
        )
        text = f"{listing.stdout}\n{listing.stderr}"
        match = re.search(r'List of available languages in "([^"]+)"', text)
        if match is None:
            raise RuntimeError("V35 evaluator could not locate Tesseract traineddata")
        directory = Path(match.group(1))
        traineddata: dict[str, str] = {}
        for language in self.language.split("+"):
            path = directory / f"{language}.traineddata"
            if not path.is_file():
                raise FileNotFoundError(f"V35 evaluator lacks {path}")
            traineddata[language] = file_sha256(path)
        return {
            "executable": executable,
            "version": version,
            "language": self.language,
            "traineddata_directory": str(directory.resolve()),
            "traineddata_sha256": traineddata,
            "page_segmentation_mode": 7,
            "scale": "2x-nearest",
            "cache": "in-memory SHA-256 of normalized image bytes",
        }

    @property
    def identity(self) -> dict[str, Any]:
        return dict(self._identity)

    def __call__(self, image: Image.Image) -> str:
        scaled = image.convert("L").resize(
            (image.width * 2, image.height * 2),
            Image.Resampling.NEAREST,
        )
        cache_key = hashlib.sha256(
            f"{scaled.size}:{self.language}".encode("ascii") + scaled.tobytes()
        ).hexdigest()
        if cache_key in self._cache:
            return self._cache[cache_key]
        with tempfile.TemporaryDirectory(prefix="ilm-v35-ocr-") as directory:
            path = Path(directory) / "strip.png"
            scaled.save(path)
            result = subprocess.run(
                [
                    str(self._identity["executable"]),
                    str(path),
                    "stdout",
                    "-l",
                    self.language,
                    "--psm",
                    "7",
                ],
                check=False,
                capture_output=True,
                text=True,
            )
        if result.returncode:
            raise RuntimeError(f"tesseract failed: {result.stderr.strip()}")
        observed = normalize_visible_text(result.stdout)
        self._cache[cache_key] = observed
        return observed


@dataclass(frozen=True)
class V35RasterCase:
    identifier: str
    stream: str
    expected: str
    prompt_pixels: torch.Tensor
    prompt_mask: torch.Tensor
    target_patches: torch.Tensor
    metadata: Mapping[str, Any]

    def __post_init__(self) -> None:
        if self.prompt_pixels.ndim != 3 or self.prompt_pixels.shape[:2] != (1, 32):
            raise ValueError("V35 case prompt must have shape [1,32,32*L]")
        if self.prompt_pixels.shape[-1] != 32 * len(self.prompt_mask):
            raise ValueError("V35 case prompt pixels and mask do not align")
        if not bool(self.prompt_mask.eq(1).all()):
            raise ValueError("V35 case prompt mask must be an active prefix")
        if self.target_patches.ndim != 4 or self.target_patches.shape[1:] != (
            1,
            32,
            32,
        ):
            raise ValueError("V35 case targets must have shape [L,1,32,32]")
        if len(self.target_patches) < 1:
            raise ValueError("V35 case requires at least one target patch")

    @property
    def prompt_length(self) -> int:
        return len(self.prompt_mask)

    @property
    def target_length(self) -> int:
        return len(self.target_patches)


@dataclass
class V35GeneratedCase:
    identifier: str
    stream: str
    expected: str
    observed: str
    condition: str
    writer: str
    patches: torch.Tensor
    feedback_latents: torch.Tensor
    length: int
    stop_probabilities: torch.Tensor
    character_accuracy: float
    exact_match: bool
    readable: bool
    nonblank: bool
    target_ink_f1: float
    target_edge_f1: float
    target_pixel_disagreement: float

    def report_row(self) -> dict[str, Any]:
        return {
            "identifier": self.identifier,
            "stream": self.stream,
            "expected": self.expected,
            "observed": self.observed,
            "condition": self.condition,
            "writer": self.writer,
            "generated_patches": self.length,
            "character_accuracy": self.character_accuracy,
            "exact_match": self.exact_match,
            "readable": self.readable,
            "nonblank": self.nonblank,
            "target_ink_f1": self.target_ink_f1,
            "target_edge_f1": self.target_edge_f1,
            "target_pixel_disagreement": self.target_pixel_disagreement,
            "stop_probabilities": self.stop_probabilities.tolist(),
        }


def _state_is_finite(value: Any) -> bool:
    if isinstance(value, torch.Tensor):
        return not value.is_floating_point() or bool(torch.isfinite(value).all())
    if isinstance(value, Mapping):
        return all(_state_is_finite(item) for item in value.values())
    if isinstance(value, (tuple, list)):
        return all(_state_is_finite(item) for item in value)
    return True


def load_v35_checkpoint_model(
    path: str | Path,
    *,
    device: torch.device,
    state: str = "ema",
) -> tuple[CausalGlyphFlowLM, dict[str, Any], dict[str, Any]]:
    if state not in {"raw", "ema"}:
        raise ValueError("V35 checkpoint state must be raw or ema")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("architecture") != V35_ARCHITECTURE:
        raise ValueError("evaluation checkpoint is not V35")
    config = CausalGlyphFlowConfig(**checkpoint["model_config"])
    model = CausalGlyphFlowLM(config)
    model.load_state_dict(checkpoint["model"], strict=True)
    if state == "ema":
        ema = checkpoint.get("ema")
        if not isinstance(ema, Mapping) or not isinstance(ema.get("shadow"), Mapping):
            raise ValueError("V35 checkpoint lacks EMA parameters")
        parameters = dict(model.named_parameters())
        if set(ema["shadow"]) != set(parameters):
            raise ValueError("V35 EMA parameter names do not match the model")
        for name, value in ema["shadow"].items():
            if parameters[name].shape != value.shape:
                raise ValueError(f"V35 EMA parameter shape mismatch: {name}")
            parameters[name].data.copy_(value.to(parameters[name]))
    model.requires_grad_(False).eval().to(device)
    receipt = {
        "checkpoint": str(Path(path).resolve()),
        "checkpoint_sha256": file_sha256(path),
        "state": state,
        "global_update": int(checkpoint.get("global_update", -1)),
        "model_config": asdict(config),
        "model_state_finite": _state_is_finite(model.state_dict()),
    }
    return model, checkpoint, receipt


def v35_checkpoint_audit(
    model: CausalGlyphFlowLM,
    checkpoint: Mapping[str, Any],
) -> dict[str, Any]:
    boundary = causal_glyph_flow_boundary_receipt(model)
    planned = sum(
        int(stage.get("updates", 0))
        for stage in checkpoint.get("run_receipt", {}).get("stages", [])
    )
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
        "architecture": checkpoint.get("architecture") == V35_ARCHITECTURE,
        "complete_updates": int(checkpoint.get("global_update", -1)) == planned
        and planned > 0,
        "checkpoint_finite": bool(checkpoint.get("finite", False)),
        "model_finite": _state_is_finite(model.state_dict()),
        "optimizer_finite": _state_is_finite(checkpoint.get("optimizer", {})),
        "stage_a_passed": bool(
            checkpoint.get("stage_a_report", {}).get("passed", False)
        ),
        "boundary_clean": not boundary.get(
            "parameter_names_with_forbidden_fragments",
            ["missing"],
        )
        and not any(bool(boundary.get(name, True)) for name in forbidden_flags),
        "runtime_teacher_absent": checkpoint.get("run_receipt", {}).get(
            "runtime_teacher_retained"
        )
        is False,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "boundary": boundary,
        "global_update": int(checkpoint.get("global_update", -1)),
        "planned_updates": planned,
        "peak_allocated_vram_bytes": int(
            checkpoint.get("peak_allocated_vram_bytes", 0)
        ),
    }


def _strip_to_patches(strip: torch.Tensor) -> torch.Tensor:
    if strip.ndim != 3 or strip.shape[:2] != (1, 32) or strip.shape[-1] % 32:
        raise ValueError("V35 strip must have shape [1,32,32*L]")
    return strip.unfold(-1, 32, 32).permute(2, 0, 1, 3).contiguous()


def patches_to_image(patches: torch.Tensor, length: int | None = None) -> Image.Image:
    if patches.ndim != 4 or patches.shape[1:] != (1, 32, 32):
        raise ValueError("V35 image patches must have shape [L,1,32,32]")
    count = len(patches) if length is None else max(1, min(int(length), len(patches)))
    strip = patches[:count].permute(1, 2, 0, 3).reshape(1, 32, count * 32)
    array = strip[0].detach().float().cpu().clamp(0, 1).numpy()
    return Image.fromarray((array * 255).astype(np.uint8), mode="L")


def instruction_case(sample: Mapping[str, Any], *, stream: str) -> V35RasterCase:
    metadata = sample["metadata"]
    prompt_length = int(metadata["prompt_patches"])
    answer_length = int(metadata["answer_patches"])
    patches = _strip_to_patches(sample["pixels"])
    return V35RasterCase(
        identifier=str(metadata["identifier"]),
        stream=stream,
        expected=normalize_visible_text(str(metadata["answer_text"])),
        prompt_pixels=sample["pixels"][..., : prompt_length * 32].clone(),
        prompt_mask=torch.ones(prompt_length),
        target_patches=patches[prompt_length : prompt_length + answer_length].clone(),
        metadata=dict(metadata),
    )


def continuation_case(
    sample: Mapping[str, Any],
    *,
    maximum_target_patches: int = 8,
) -> V35RasterCase:
    metadata = sample["metadata"]
    occupied = int(metadata["occupied_patches"])
    target_length = min(maximum_target_patches, occupied - 1)
    prompt_length = occupied - target_length
    patches = _strip_to_patches(sample["pixels"])
    return V35RasterCase(
        identifier=str(metadata["identifier"]),
        stream="public",
        expected="",
        prompt_pixels=sample["pixels"][..., : prompt_length * 32].clone(),
        prompt_mask=torch.ones(prompt_length),
        target_patches=patches[prompt_length:occupied].clone(),
        metadata=dict(metadata),
    )


def cases_from_dataset(
    dataset: Dataset[dict[str, Any]],
    *,
    stream: str,
    limit: int,
    maximum_target_patches: int = 8,
) -> list[V35RasterCase]:
    if limit < 1:
        raise ValueError("V35 evaluation case limit must be positive")
    cases: list[V35RasterCase] = []
    for index in range(min(limit, len(dataset))):
        sample = dataset[index]
        case = (
            continuation_case(
                sample,
                maximum_target_patches=maximum_target_patches,
            )
            if stream == "public"
            else instruction_case(sample, stream=stream)
        )
        cases.append(case)
    if not cases:
        raise ValueError(f"V35 {stream} evaluation selected no cases")
    return cases


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


def _padded_patches(
    patches: torch.Tensor,
    length: int,
    maximum: int,
) -> torch.Tensor:
    result = torch.ones(maximum, 1, 32, 32, dtype=torch.float32)
    active = min(int(length), maximum, len(patches))
    if active:
        result[:active] = patches[:active].detach().float().cpu()
    return result


def raster_pair_metrics(
    predicted: torch.Tensor,
    predicted_length: int,
    target: torch.Tensor,
    target_length: int,
) -> dict[str, float]:
    maximum = max(1, int(predicted_length), int(target_length))
    predicted = _padded_patches(predicted, predicted_length, maximum)
    target = _padded_patches(target, target_length, maximum)
    predicted_binary = predicted >= 0.5
    target_binary = target >= 0.5
    ink = _binary_counts(~predicted_binary, ~target_binary)
    predicted_edges = glyph_sobel_edges(predicted).abs() > 0.05
    target_edges = glyph_sobel_edges(target).abs() > 0.05
    edge = _binary_counts(predicted_edges, target_edges)
    return {
        "ink_f1": _f1(*ink),
        "edge_f1": _f1(*edge),
        "pixel_disagreement": float(
            predicted_binary.ne(target_binary).float().mean()
        ),
    }


@torch.no_grad()
def teacher_forced_diagnostics(
    model: CausalGlyphFlowLM,
    dataset: Dataset[dict[str, Any]],
    *,
    device: torch.device,
    precision: str,
    batch_size: int = 4,
    maximum_examples: int = 0,
    flow_seed: int = V35_EVALUATION_SEED,
) -> dict[str, Any]:
    if batch_size < 1 or maximum_examples < 0:
        raise ValueError("V35 teacher-forced audit settings are invalid")
    selected: Dataset[dict[str, Any]] = dataset
    if maximum_examples:
        selected = Subset(dataset, range(min(maximum_examples, len(dataset))))
    loader = DataLoader(
        selected,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0,
        collate_fn=causal_glyph_flow_collate,
    )
    generator = torch.Generator(device=device).manual_seed(flow_seed)
    patches = 0
    cosine_sum = 0.0
    squared_error_sum = 0.0
    flow_squared_error_sum = 0.0
    flow_elements = 0
    ink_counts = [0, 0, 0]
    edge_counts = [0, 0, 0]
    stop_loss_sum = 0.0
    stop_count = 0
    stop_correct = 0
    finite = True
    examples = 0
    started = time.perf_counter()
    for batch in loader:
        mask = batch["patch_mask"]
        visible = int(mask.sum(dim=1).max())
        pixels = batch["pixels"][..., : visible * 32].to(device)
        mask = mask[:, :visible].to(device)
        next_mask = batch["next_patch_mask"][:, :visible].to(device)
        stop_mask = batch["stop_mask"][:, :visible].to(device)
        stop_targets = batch["stop_targets"][:, :visible].to(device)
        with _autocast(device, precision):
            output = model(pixels, mask)
            active = next_mask[:, :-1].bool()
            anchors = output.anchor_latents[:, :-1]
            targets = output.latents[:, 1:]
            hidden = output.hidden_states[:, :-1]
            selected_anchor = anchors[active]
            selected_target = targets[active]
            selected_hidden = hidden[active]
            decoded = model.decode_latents(selected_anchor).float().sigmoid()
            target_patches = model.patchify(pixels)[:, 1:][active].float()
            noise = torch.randn(
                selected_target.shape,
                device=device,
                dtype=selected_target.dtype,
                generator=generator,
            )
            times = torch.rand(
                len(selected_target),
                device=device,
                dtype=selected_target.dtype,
                generator=generator,
            )
            noisy = (1.0 - times[:, None]) * noise + times[:, None] * selected_target
            velocity = model.flow_velocity(noisy, times, selected_hidden)
            target_velocity = selected_target - noise
        count = len(selected_target)
        if count:
            cosine_sum += float(
                F.cosine_similarity(
                    selected_anchor.float(),
                    selected_target.float(),
                    dim=-1,
                ).sum()
            )
            squared_error_sum += float(
                (selected_anchor.float() - selected_target.float())
                .square()
                .mean(dim=-1)
                .sum()
            )
            flow_squared_error_sum += float(
                (velocity.float() - target_velocity.float()).square().sum()
            )
            flow_elements += velocity.numel()
            predicted_binary = decoded >= 0.5
            target_binary = target_patches >= 0.5
            for output_counts, observed in (
                (ink_counts, _binary_counts(~predicted_binary, ~target_binary)),
                (
                    edge_counts,
                    _binary_counts(
                        glyph_sobel_edges(predicted_binary.float()).abs() > 0.05,
                        glyph_sobel_edges(target_binary.float()).abs() > 0.05,
                    ),
                ),
            ):
                for index, value in enumerate(observed):
                    output_counts[index] += value
            patches += count
        if bool(stop_mask.any()):
            losses = F.binary_cross_entropy_with_logits(
                output.stop_logits.float(),
                stop_targets.float(),
                reduction="none",
            )
            stop_loss_sum += float(losses[stop_mask.bool()].sum())
            stop_count += int(stop_mask.sum())
            stop_correct += int(
                output.stop_logits[stop_mask.bool()]
                .ge(0)
                .eq(stop_targets[stop_mask.bool()].bool())
                .sum()
            )
        finite = finite and all(
            bool(torch.isfinite(value).all())
            for value in (
                output.latents,
                output.hidden_states,
                output.anchor_latents,
                output.stop_logits,
                decoded,
                velocity,
            )
        )
        examples += len(batch["metadata"])
    elapsed = time.perf_counter() - started
    return {
        "examples": examples,
        "next_patches": patches,
        "finite": finite,
        "anchor_cosine_similarity": cosine_sum / max(1, patches),
        "anchor_mse": squared_error_sum / max(1, patches),
        "flow_velocity_mse": flow_squared_error_sum / max(1, flow_elements),
        "decoded_ink_f1": _f1(*ink_counts),
        "decoded_edge_f1": _f1(*edge_counts),
        "stop_bce": stop_loss_sum / max(1, stop_count),
        "stop_accuracy": stop_correct / max(1, stop_count),
        "stop_positions": stop_count,
        "elapsed_seconds": elapsed,
    }


def controlled_prompt(
    cases: Sequence[V35RasterCase],
    index: int,
    condition: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    if condition not in {"correct", "shuffled", "blank", "final-quarter"}:
        raise ValueError(f"unknown V35 prompt condition: {condition}")
    case = cases[index]
    pixels = case.prompt_pixels.clone()
    if condition == "correct":
        return pixels, case.prompt_mask.clone()
    if condition == "blank":
        return torch.ones_like(pixels), case.prompt_mask.clone()
    if condition == "final-quarter":
        output = torch.ones_like(pixels)
        count = max(1, math.ceil(case.prompt_length / 4))
        output[..., -count * 32 :] = pixels[..., -count * 32 :]
        return output, case.prompt_mask.clone()
    other = cases[(index + 1) % len(cases)]
    output = torch.ones_like(pixels)
    copy = min(case.prompt_length, other.prompt_length)
    output[..., : copy * 32] = other.prompt_pixels[..., : copy * 32]
    return output, case.prompt_mask.clone()


@torch.no_grad()
def _codec_target(
    model: CausalGlyphFlowLM,
    target: torch.Tensor,
    *,
    device: torch.device,
    precision: str,
) -> torch.Tensor:
    with _autocast(device, precision):
        latents = model.codec.encode(target.to(device))
        logits = model.decode_latents(latents)
    return logits.float().sigmoid().ge(0.5).float().cpu()


@torch.no_grad()
def generate_case(
    model: CausalGlyphFlowLM,
    case: V35RasterCase,
    prompt_pixels: torch.Tensor,
    prompt_mask: torch.Tensor,
    *,
    condition: str,
    writer: str,
    seed: int,
    device: torch.device,
    precision: str,
    ocr: Callable[[Image.Image], str],
) -> V35GeneratedCase:
    if writer not in {"anchor", "flow"}:
        raise ValueError("V35 writer must be anchor or flow")
    maximum = min(31, model.config.maximum_patches - len(prompt_mask))
    if maximum < 1:
        raise ValueError("V35 evaluation prompt leaves no generation context")
    with _autocast(device, precision):
        generated = model.generate(
            prompt_pixels.unsqueeze(0).to(device),
            prompt_mask.unsqueeze(0).to(device),
            maximum_new_patches=maximum,
            minimum_new_patches=1,
            stop_threshold=0.5,
            raster_threshold=0.5,
            flow_steps=8,
            seed=seed,
            use_flow=writer == "flow",
        )
    length = int(generated.lengths[0])
    patches = generated.patches[0].float().cpu()
    feedback = generated.feedback_latents[0].float().cpu()
    observed = ocr(patches_to_image(patches, length))
    expected = case.expected
    if not expected:
        expected = ocr(patches_to_image(case.target_patches))
    raster = raster_pair_metrics(
        patches,
        length,
        case.target_patches,
        case.target_length,
    )
    active = patches[: max(1, length)]
    nonblank = bool((active < 0.5).float().mean() >= 0.001)
    return V35GeneratedCase(
        identifier=case.identifier,
        stream=case.stream,
        expected=expected,
        observed=observed,
        condition=condition,
        writer=writer,
        patches=patches,
        feedback_latents=feedback,
        length=length,
        stop_probabilities=generated.stop_probabilities[0].float().cpu(),
        character_accuracy=ocr_character_accuracy(expected, observed),
        exact_match=normalize_visible_text(expected) == normalize_visible_text(observed),
        readable=text_is_readable(observed),
        nonblank=nonblank,
        target_ink_f1=raster["ink_f1"],
        target_edge_f1=raster["edge_f1"],
        target_pixel_disagreement=raster["pixel_disagreement"],
    )


def summarize_generated(cases: Sequence[V35GeneratedCase]) -> dict[str, Any]:
    if not cases:
        raise ValueError("V35 cannot summarize an empty generation set")

    def mean(values: Sequence[float]) -> float:
        return sum(values) / len(values)

    return {
        "examples": len(cases),
        "finite": all(
            bool(torch.isfinite(case.patches).all())
            and bool(torch.isfinite(case.feedback_latents).all())
            and bool(torch.isfinite(case.stop_probabilities).all())
            for case in cases
        ),
        "ocr_character_accuracy": mean(
            [case.character_accuracy for case in cases]
        ),
        "exact_match_rate": mean([float(case.exact_match) for case in cases]),
        "readable_rate": mean([float(case.readable) for case in cases]),
        "nonblank_rate": mean([float(case.nonblank) for case in cases]),
        "mean_generated_patches": mean([float(case.length) for case in cases]),
        "target_ink_f1": mean([case.target_ink_f1 for case in cases]),
        "target_edge_f1": mean([case.target_edge_f1 for case in cases]),
        "target_pixel_disagreement": mean(
            [case.target_pixel_disagreement for case in cases]
        ),
    }


def output_pair_metrics(
    first: V35GeneratedCase,
    second: V35GeneratedCase,
) -> dict[str, float]:
    maximum = max(1, first.length, second.length)
    first_pixels = _padded_patches(first.patches, first.length, maximum)
    second_pixels = _padded_patches(second.patches, second.length, maximum)
    common = min(first.length, second.length)
    cosine = 1.0
    if common:
        cosine = float(
            F.cosine_similarity(
                first.feedback_latents[:common],
                second.feedback_latents[:common],
                dim=-1,
            ).mean()
        )
    return {
        "pixel_disagreement": float(
            first_pixels.ge(0.5).ne(second_pixels.ge(0.5)).float().mean()
        ),
        "latent_cosine": cosine,
    }


def _target_ocr_report(
    model: CausalGlyphFlowLM,
    cases: Sequence[V35RasterCase],
    *,
    device: torch.device,
    precision: str,
    ocr: Callable[[Image.Image], str],
) -> dict[str, Any]:
    rows = []
    for case in cases:
        raw_observed = ocr(patches_to_image(case.target_patches))
        reconstructed = _codec_target(
            model,
            case.target_patches,
            device=device,
            precision=precision,
        )
        codec_observed = ocr(patches_to_image(reconstructed))
        expected = case.expected or raw_observed
        rows.append(
            {
                "identifier": case.identifier,
                "expected": expected,
                "raw_observed": raw_observed,
                "codec_observed": codec_observed,
                "raw_character_accuracy": ocr_character_accuracy(
                    expected,
                    raw_observed,
                ),
                "codec_character_accuracy": ocr_character_accuracy(
                    expected,
                    codec_observed,
                ),
            }
        )
    return {
        "examples": len(rows),
        "raw_character_accuracy": sum(
            row["raw_character_accuracy"] for row in rows
        )
        / len(rows),
        "codec_character_accuracy": sum(
            row["codec_character_accuracy"] for row in rows
        )
        / len(rows),
        "rows": rows,
    }


def save_autonomous_gallery(
    cases: Sequence[V35RasterCase],
    generated: Mapping[str, Sequence[V35GeneratedCase]],
    path: str | Path,
    *,
    title: str,
    maximum_rows: int = 8,
) -> None:
    conditions = tuple(generated)
    rows = min(maximum_rows, len(cases))
    if rows < 1:
        return
    label_width = 132
    column_width = 390
    row_height = 58 + 34 * (1 + len(conditions))
    width = label_width + column_width
    height = 42 + rows * row_height
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    font = ImageFont.truetype(font_path, 12) if Path(font_path).is_file() else None
    draw.text((8, 10), title, fill="black", font=font)
    for row in range(rows):
        top = 42 + row * row_height
        draw.line((0, top, width, top), fill=(205, 205, 205))
        draw.text((8, top + 8), cases[row].identifier[:18], fill="black", font=font)
        target = patches_to_image(cases[row].target_patches)
        target.thumbnail((column_width - 12, 32), Image.Resampling.NEAREST)
        draw.text((8, top + 30), "target", fill=(55, 55, 55), font=font)
        canvas.paste(target.convert("RGB"), (label_width, top + 26))
        for condition_index, condition in enumerate(conditions):
            result = generated[condition][row]
            image = patches_to_image(result.patches, result.length)
            image.thumbnail((column_width - 12, 32), Image.Resampling.NEAREST)
            y = top + 60 + condition_index * 34
            draw.text((8, y + 7), condition, fill=(55, 55, 55), font=font)
            canvas.paste(image.convert("RGB"), (label_width, y))
            draw.text(
                (width - 82, y + 7),
                f"OCR {result.character_accuracy:.2f}",
                fill=(55, 55, 55),
                font=font,
            )
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)


@torch.no_grad()
def autonomous_case_audit(
    model: CausalGlyphFlowLM,
    cases: Sequence[V35RasterCase],
    *,
    writer: str,
    conditions: Sequence[str],
    device: torch.device,
    precision: str,
    ocr: Callable[[Image.Image], str],
    seed: int = V35_EVALUATION_SEED,
    gallery_path: str | Path | None = None,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    if not cases or not conditions or "correct" not in conditions:
        raise ValueError("V35 autonomous audit requires cases and correct prompts")
    generated: dict[str, list[V35GeneratedCase]] = {}
    started = time.perf_counter()
    for condition in conditions:
        condition_started = time.perf_counter()
        rows = []
        for index, case in enumerate(cases):
            pixels, mask = controlled_prompt(cases, index, condition)
            rows.append(
                generate_case(
                    model,
                    case,
                    pixels,
                    mask,
                    condition=condition,
                    writer=writer,
                    seed=seed + index * 104_729,
                    device=device,
                    precision=precision,
                    ocr=ocr,
                )
            )
        generated[condition] = rows
        if progress is not None:
            progress(
                f"{cases[0].stream}/{writer}/{condition}: "
                f"{len(rows)} cases in {time.perf_counter() - condition_started:.1f}s"
            )
    comparisons: dict[str, Any] = {}
    for condition, rows in generated.items():
        if condition == "correct":
            continue
        metrics = [
            output_pair_metrics(first, second)
            for first, second in zip(generated["correct"], rows)
        ]
        comparisons[condition] = {
            "mean_pixel_disagreement": sum(
                item["pixel_disagreement"] for item in metrics
            )
            / len(metrics),
            "mean_latent_cosine": sum(item["latent_cosine"] for item in metrics)
            / len(metrics),
        }
    if gallery_path is not None:
        save_autonomous_gallery(
            cases,
            generated,
            gallery_path,
            title=f"V35 {writer} {cases[0].stream} closed-raster generation",
        )
    return {
        "writer": writer,
        "conditions": {
            condition: summarize_generated(rows)
            for condition, rows in generated.items()
        },
        "control_comparisons": comparisons,
        "target_ocr": _target_ocr_report(
            model,
            cases,
            device=device,
            precision=precision,
            ocr=ocr,
        ),
        "rows": {
            condition: [row.report_row() for row in rows]
            for condition, rows in generated.items()
        },
        "elapsed_seconds": time.perf_counter() - started,
        "gallery": str(gallery_path) if gallery_path is not None else None,
    }


def select_v35_writer(
    anchor: Mapping[str, Any],
    flow: Mapping[str, Any],
) -> dict[str, Any]:
    anchor_metrics = anchor["conditions"]["correct"]
    flow_metrics = flow["conditions"]["correct"]
    epsilon = 1e-12
    flow_promoted = (
        float(flow_metrics["ocr_character_accuracy"]) + epsilon
        >= float(anchor_metrics["ocr_character_accuracy"]) + 0.03
        and float(flow_metrics["readable_rate"]) + epsilon
        >= float(anchor_metrics["readable_rate"]) - 0.02
    )
    return {
        "selected": "flow" if flow_promoted else "anchor",
        "flow_promoted": flow_promoted,
        "anchor_character_accuracy": anchor_metrics["ocr_character_accuracy"],
        "flow_character_accuracy": flow_metrics["ocr_character_accuracy"],
        "anchor_readable_rate": anchor_metrics["readable_rate"],
        "flow_readable_rate": flow_metrics["readable_rate"],
        "rule": "flow accuracy >= anchor + 0.03 and readability >= anchor - 0.02",
    }


@torch.no_grad()
def copy_counterfactual_audit(
    model: CausalGlyphFlowLM,
    pairs: Sequence[tuple[V35RasterCase, V35RasterCase]],
    *,
    writer: str,
    device: torch.device,
    precision: str,
    ocr: Callable[[Image.Image], str],
    seed: int = V35_EVALUATION_SEED + 1_000_000,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    rows = []
    for index, (first, second) in enumerate(pairs):
        if first.prompt_length != second.prompt_length:
            raise ValueError("V35 counterfactual prompts must have equal visual length")
        if first.target_length != second.target_length:
            raise ValueError("V35 counterfactual targets must have equal visual length")
        pair_seed = seed + index * 104_729
        first_output = generate_case(
            model,
            first,
            first.prompt_pixels,
            first.prompt_mask,
            condition="correct",
            writer=writer,
            seed=pair_seed,
            device=device,
            precision=precision,
            ocr=ocr,
        )
        second_output = generate_case(
            model,
            second,
            second.prompt_pixels,
            second.prompt_mask,
            condition="correct",
            writer=writer,
            seed=pair_seed,
            device=device,
            precision=precision,
            ocr=ocr,
        )
        direct = (
            raster_pair_metrics(
                first_output.patches,
                first_output.length,
                first.target_patches,
                first.target_length,
            )["pixel_disagreement"]
            + raster_pair_metrics(
                second_output.patches,
                second_output.length,
                second.target_patches,
                second.target_length,
            )["pixel_disagreement"]
        )
        swapped = (
            raster_pair_metrics(
                first_output.patches,
                first_output.length,
                second.target_patches,
                second.target_length,
            )["pixel_disagreement"]
            + raster_pair_metrics(
                second_output.patches,
                second_output.length,
                first.target_patches,
                first.target_length,
            )["pixel_disagreement"]
        )
        difference = output_pair_metrics(first_output, second_output)
        rows.append(
            {
                "first": first.identifier,
                "second": second.identifier,
                "direct_cost": direct,
                "swapped_cost": swapped,
                "target_preferred": direct < swapped,
                "output_pixel_disagreement": difference["pixel_disagreement"],
                "first_character_accuracy": first_output.character_accuracy,
                "second_character_accuracy": second_output.character_accuracy,
            }
        )
        if progress is not None and (
            len(rows) == len(pairs) or len(rows) % max(1, min(4, len(pairs))) == 0
        ):
            progress(f"copy/{writer}/counterfactual: {len(rows)}/{len(pairs)} pairs")
    if not rows:
        raise ValueError("V35 counterfactual audit requires at least one pair")
    return {
        "pairs": len(rows),
        "target_preference_rate": sum(
            float(row["target_preferred"]) for row in rows
        )
        / len(rows),
        "mean_output_pixel_disagreement": sum(
            row["output_pixel_disagreement"] for row in rows
        )
        / len(rows),
        "mean_character_accuracy": sum(
            (row["first_character_accuracy"] + row["second_character_accuracy"])
            / 2
            for row in rows
        )
        / len(rows),
        "rows": rows,
    }


def v35_development_gate(report: Mapping[str, Any]) -> dict[str, Any]:
    primary = report["states"]["ema"]
    selected = primary["writer_selection"]["selected"]
    autonomous = primary["autonomous"]
    copy = autonomous["copy"][selected]
    public = autonomous["public"][selected]
    instruction = autonomous["instruction"][selected]
    paraphrase = autonomous["paraphrase"][selected]
    counterfactual = autonomous["copy_counterfactual"][selected]
    copy_correct = copy["conditions"]["correct"]
    copy_shuffled = copy["conditions"]["shuffled"]
    copy_blank = copy["conditions"]["blank"]
    instruction_correct = instruction["conditions"]["correct"]
    instruction_shuffled = instruction["conditions"]["shuffled"]
    instruction_blank = instruction["conditions"]["blank"]
    paraphrase_correct_rows = paraphrase["rows"]["correct"]
    paraphrase_blank_rows = paraphrase["rows"]["blank"]
    wording_shift = any(
        bool(correct["readable"])
        and float(correct["character_accuracy"])
        > float(blank["character_accuracy"])
        for correct, blank in zip(paraphrase_correct_rows, paraphrase_blank_rows)
    )
    copy_ceiling = float(copy["target_ocr"]["codec_character_accuracy"])
    visual_checks = {
        "stage_a": bool(report["checkpoint_audit"]["checks"]["stage_a_passed"]),
        "finite_audit": bool(report["checkpoint_audit"]["passed"])
        and all(
            stream_report[selected]["conditions"]["correct"]["finite"]
            for stream_report in (
                autonomous["copy"],
                autonomous["public"],
                autonomous["instruction"],
            )
        ),
        "closed_loop_receipt": bool(report.get("closed_loop_receipt", {}).get("passed")),
        "copy_target_ocr_ceiling": copy_ceiling >= 0.70,
        "copy_ocr_retention": float(copy_correct["ocr_character_accuracy"])
        >= 0.60 * copy_ceiling,
        "copy_correct_minus_shuffled": float(copy_correct["ocr_character_accuracy"])
        >= float(copy_shuffled["ocr_character_accuracy"]) + 0.20,
        "copy_correct_minus_blank": float(copy_correct["ocr_character_accuracy"])
        >= float(copy_blank["ocr_character_accuracy"]) + 0.25,
        "copy_counterfactual_target_preference": float(
            counterfactual["target_preference_rate"]
        )
        >= 0.75,
        "public_teacher_ink_f1": float(
            primary["teacher_forced"]["public"]["decoded_ink_f1"]
        )
        >= 0.70,
        "public_teacher_edge_f1": float(
            primary["teacher_forced"]["public"]["decoded_edge_f1"]
        )
        >= 0.70,
        "public_autonomous_readable_nonblank": float(
            public["conditions"]["correct"]["readable_rate"]
        )
        >= 0.50
        and float(public["conditions"]["correct"]["nonblank_rate"]) >= 0.50,
        "single_gpu_vram": int(
            report["checkpoint_audit"]["peak_allocated_vram_bytes"]
        )
        < 20 * 1024**3,
    }
    semantic_checks = {
        "visual_causal_gate": all(visual_checks.values()),
        "instruction_target_ocr_ceiling": float(
            instruction["target_ocr"]["codec_character_accuracy"]
        )
        >= 0.60,
        "instruction_correct_accuracy": float(
            instruction_correct["ocr_character_accuracy"]
        )
        >= 0.08,
        "instruction_correct_minus_shuffled": float(
            instruction_correct["ocr_character_accuracy"]
        )
        >= float(instruction_shuffled["ocr_character_accuracy"]) + 0.02,
        "instruction_correct_minus_blank": float(
            instruction_correct["ocr_character_accuracy"]
        )
        >= float(instruction_blank["ocr_character_accuracy"]) + 0.03,
        "instruction_readable": float(instruction_correct["readable_rate"]) >= 0.35,
        "instruction_pixel_response_shuffled": float(
            instruction["control_comparisons"]["shuffled"][
                "mean_pixel_disagreement"
            ]
        )
        >= 0.01,
        "instruction_pixel_response_blank": float(
            instruction["control_comparisons"]["blank"]["mean_pixel_disagreement"]
        )
        >= 0.01,
        "wording_shift": wording_shift,
    }
    if all(semantic_checks.values()):
        status = "semantic-raster-qualified"
    elif all(visual_checks.values()):
        status = "visual-causal-qualified"
    else:
        status = "not-qualified"
    return {
        "status": status,
        "selected_writer": selected,
        "visual_causal": {
            "passed": all(visual_checks.values()),
            "checks": visual_checks,
        },
        "semantic_raster": {
            "passed": all(semantic_checks.values()),
            "checks": semantic_checks,
        },
    }


def _primary_metrics(report: Mapping[str, Any]) -> dict[str, float]:
    primary = report["states"]["ema"]
    selected = primary["writer_selection"]["selected"]
    autonomous = primary["autonomous"]
    copy = autonomous["copy"][selected]
    public = autonomous["public"][selected]
    instruction = autonomous["instruction"][selected]
    copy_correct = copy["conditions"]["correct"]
    copy_shuffled = copy["conditions"]["shuffled"]
    copy_blank = copy["conditions"]["blank"]
    instruction_correct = instruction["conditions"]["correct"]
    instruction_shuffled = instruction["conditions"]["shuffled"]
    instruction_blank = instruction["conditions"]["blank"]
    return {
        "copy_target_ocr_ceiling": float(
            copy["target_ocr"]["codec_character_accuracy"]
        ),
        "copy_correct_ocr": float(copy_correct["ocr_character_accuracy"]),
        "copy_correct_minus_shuffled": float(
            copy_correct["ocr_character_accuracy"]
        )
        - float(copy_shuffled["ocr_character_accuracy"]),
        "copy_correct_minus_blank": float(copy_correct["ocr_character_accuracy"])
        - float(copy_blank["ocr_character_accuracy"]),
        "copy_counterfactual_target_preference": float(
            autonomous["copy_counterfactual"][selected]["target_preference_rate"]
        ),
        "public_teacher_ink_f1": float(
            primary["teacher_forced"]["public"]["decoded_ink_f1"]
        ),
        "public_teacher_edge_f1": float(
            primary["teacher_forced"]["public"]["decoded_edge_f1"]
        ),
        "public_readable": float(
            public["conditions"]["correct"]["readable_rate"]
        ),
        "public_nonblank": float(
            public["conditions"]["correct"]["nonblank_rate"]
        ),
        "instruction_target_ocr_ceiling": float(
            instruction["target_ocr"]["codec_character_accuracy"]
        ),
        "instruction_correct_ocr": float(
            instruction_correct["ocr_character_accuracy"]
        ),
        "instruction_correct_minus_shuffled": float(
            instruction_correct["ocr_character_accuracy"]
        )
        - float(instruction_shuffled["ocr_character_accuracy"]),
        "instruction_correct_minus_blank": float(
            instruction_correct["ocr_character_accuracy"]
        )
        - float(instruction_blank["ocr_character_accuracy"]),
        "instruction_readable": float(instruction_correct["readable_rate"]),
        "instruction_pixel_response_shuffled": float(
            instruction["control_comparisons"]["shuffled"][
                "mean_pixel_disagreement"
            ]
        ),
        "instruction_pixel_response_blank": float(
            instruction["control_comparisons"]["blank"]["mean_pixel_disagreement"]
        ),
    }


def v35_sealed_transfer_gate(
    development: Mapping[str, Any],
    sealed: Mapping[str, Any],
) -> dict[str, Any]:
    development_status = str(
        development.get("decision", {}).get("status", "not-qualified")
    )
    if development_status not in {
        "visual-causal-qualified",
        "semantic-raster-qualified",
    }:
        raise ValueError("V35 sealed data cannot open after an unqualified development run")
    development_writer = str(
        development["states"]["ema"]["writer_selection"]["selected"]
    )
    sealed_writer = str(sealed["states"]["ema"]["writer_selection"]["selected"])
    if sealed_writer != development_writer:
        raise ValueError("V35 sealed evaluation changed the selected writer")
    metrics = _primary_metrics(sealed)
    development_metrics = _primary_metrics(development)
    visual_metric_names = {
        "copy_target_ocr_ceiling",
        "copy_correct_ocr",
        "copy_correct_minus_shuffled",
        "copy_correct_minus_blank",
        "copy_counterfactual_target_preference",
        "public_teacher_ink_f1",
        "public_teacher_edge_f1",
        "public_readable",
        "public_nonblank",
    }
    semantic_metric_names = {
        "instruction_target_ocr_ceiling",
        "instruction_correct_ocr",
        "instruction_correct_minus_shuffled",
        "instruction_correct_minus_blank",
        "instruction_readable",
        "instruction_pixel_response_shuffled",
        "instruction_pixel_response_blank",
    }
    applicable_metric_names = visual_metric_names | (
        semantic_metric_names
        if development_status == "semantic-raster-qualified"
        else set()
    )
    ratios = {
        name: (
            value / development_metrics[name]
            if development_metrics[name] != 0.0
            else None
        )
        for name, value in metrics.items()
    }
    ratio_checks = {
        name: metrics[name] + 1e-12 >= 0.90 * development_metrics[name]
        for name in sorted(applicable_metric_names)
    }
    primary = sealed["states"]["ema"]
    autonomous = primary["autonomous"]
    copy_ceiling = metrics["copy_target_ocr_ceiling"]
    visual_checks = {
        "checkpoint_audit": bool(sealed["checkpoint_audit"]["passed"]),
        "closed_loop_receipt": bool(
            sealed.get("closed_loop_receipt", {}).get("passed")
        ),
        "copy_target_ocr_ceiling": copy_ceiling >= 0.70,
        "copy_ocr_retention": metrics["copy_correct_ocr"] >= 0.60 * copy_ceiling,
        "copy_correct_minus_shuffled": metrics["copy_correct_minus_shuffled"]
        >= 0.20,
        "copy_correct_minus_blank": metrics["copy_correct_minus_blank"] >= 0.25,
        "copy_counterfactual_target_preference": metrics[
            "copy_counterfactual_target_preference"
        ]
        >= 0.75,
        "public_teacher_ink_f1": metrics["public_teacher_ink_f1"] >= 0.70,
        "public_teacher_edge_f1": metrics["public_teacher_edge_f1"] >= 0.70,
        "public_readable_nonblank": metrics["public_readable"] >= 0.50
        and metrics["public_nonblank"] >= 0.50,
        "finite": all(
            stream[sealed_writer]["conditions"]["correct"]["finite"]
            for stream in (
                autonomous["copy"],
                autonomous["public"],
                autonomous["instruction"],
            )
        ),
        "single_gpu_vram": int(
            sealed["checkpoint_audit"]["peak_allocated_vram_bytes"]
        )
        < 20 * 1024**3,
    }
    semantic_checks = {
        "visual_causal": all(visual_checks.values()),
        "instruction_target_ocr_ceiling": metrics[
            "instruction_target_ocr_ceiling"
        ]
        >= 0.60,
        "instruction_correct_ocr": metrics["instruction_correct_ocr"] >= 0.08,
        "instruction_correct_minus_shuffled": metrics[
            "instruction_correct_minus_shuffled"
        ]
        >= 0.02,
        "instruction_correct_minus_blank": metrics[
            "instruction_correct_minus_blank"
        ]
        >= 0.03,
        "instruction_readable": metrics["instruction_readable"] >= 0.35,
        "instruction_pixel_response_shuffled": metrics[
            "instruction_pixel_response_shuffled"
        ]
        >= 0.01,
        "instruction_pixel_response_blank": metrics[
            "instruction_pixel_response_blank"
        ]
        >= 0.01,
        "development_wording_shift": bool(
            development["decision"]["semantic_raster"]["checks"]["wording_shift"]
        ),
    }
    absolute_pass = (
        all(semantic_checks.values())
        if development_status == "semantic-raster-qualified"
        else all(visual_checks.values())
    )
    transfer_pass = absolute_pass and all(ratio_checks.values())
    return {
        "passed": transfer_pass,
        "development_status": development_status,
        "sealed_status": (
            development_status if absolute_pass else "not-qualified"
        ),
        "selected_writer": sealed_writer,
        "absolute_checks": {
            "visual_causal": visual_checks,
            "semantic_raster": semantic_checks,
        },
        "primary_metrics": metrics,
        "development_primary_metrics": development_metrics,
        "applicable_ratio_metrics": sorted(applicable_metric_names),
        "sealed_to_development_ratio": ratios,
        "ratio_at_least_0_90": ratio_checks,
    }


def report_sha256(report: Mapping[str, Any]) -> str:
    import json

    payload = json.dumps(
        report,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()

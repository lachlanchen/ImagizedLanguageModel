from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Callable, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from .visual_semantic_raster_data import (
    VisualRasterRenderConfig,
    normalize_visible_text,
    render_answer_cells,
)
from .visual_semantic_raster_training import sobel_edges
from .visual_semantic_raster_transducer import VisualSemanticRasterTransducer


@dataclass(frozen=True)
class RasterCharacterBank:
    characters: tuple[str, ...]
    prototypes: torch.Tensor
    font_paths: tuple[str, ...]
    variants_per_character: int

    def __post_init__(self) -> None:
        if not self.characters or len(self.characters) != self.prototypes.shape[0]:
            raise ValueError("V32 evaluator bank characters and prototypes do not align")
        if self.prototypes.ndim != 2:
            raise ValueError("V32 evaluator bank prototypes must be [K,D]")

    @property
    def character_to_index(self) -> dict[str, int]:
        return {character: index for index, character in enumerate(self.characters)}

    def receipt(self) -> dict[str, Any]:
        digest = hashlib.sha256()
        digest.update(json.dumps(self.characters, ensure_ascii=False).encode("utf-8"))
        digest.update(self.prototypes.detach().float().cpu().numpy().tobytes())
        return {
            "evaluator_only": True,
            "absent_from_student_generate": True,
            "characters": len(self.characters),
            "embedding_dimension": self.prototypes.shape[1],
            "font_paths": list(self.font_paths),
            "variants_per_character": self.variants_per_character,
            "sha256": digest.hexdigest(),
        }


def render_character_bank_cells(
    characters: Sequence[str],
    *,
    render_config: VisualRasterRenderConfig,
    font_paths: Sequence[str],
    variants_per_font: int = 1,
) -> tuple[tuple[str, ...], torch.Tensor]:
    ordered = tuple(sorted(set(characters)))
    if not ordered or not font_paths or variants_per_font < 1:
        raise ValueError("V32 evaluator bank request is empty")
    views: list[torch.Tensor] = []
    for character_index, character in enumerate(ordered):
        character_views = []
        for font_index, font_path in enumerate(font_paths):
            for variant_index in range(variants_per_font):
                if character == " ":
                    cell = torch.zeros(
                        1,
                        render_config.answer_cell_size,
                        render_config.answer_cell_size,
                    )
                else:
                    cells, _, _, _, _ = render_answer_cells(
                        character,
                        config=render_config,
                        font_path=font_path,
                        variant=(
                            32_000_003 * character_index
                            + 503 * font_index
                            + variant_index
                        ),
                    )
                    cell = cells[0]
                character_views.append(cell)
        views.append(torch.stack(character_views))
    return ordered, torch.stack(views)


@torch.no_grad()
def encode_evaluator_cells(
    model: VisualSemanticRasterTransducer,
    cells: torch.Tensor,
    *,
    device: torch.device,
    batch_size: int = 256,
    autocast: Callable[[], Any] | None = None,
) -> torch.Tensor:
    if cells.ndim != 4 or tuple(cells.shape[-3:]) != (1, 24, 24):
        raise ValueError("V32 evaluator cells must be [N,1,24,24]")
    if batch_size < 1:
        raise ValueError("V32 evaluator batch size must be positive")
    embeddings = []
    for start in range(0, len(cells), batch_size):
        batch = cells[start : start + batch_size].to(device)
        context = autocast() if autocast is not None else torch.no_grad()
        with context:
            encoded = model.cell_retina(batch[:, None])[:, 0]
        embeddings.append(F.normalize(encoded.float(), dim=-1).cpu())
    return torch.cat(embeddings)


def build_raster_character_bank(
    model: VisualSemanticRasterTransducer,
    characters: Sequence[str],
    *,
    render_config: VisualRasterRenderConfig,
    font_paths: Sequence[str],
    device: torch.device,
    variants_per_font: int = 1,
    batch_size: int = 256,
    autocast: Callable[[], Any] | None = None,
) -> RasterCharacterBank:
    ordered, views = render_character_bank_cells(
        characters,
        render_config=render_config,
        font_paths=font_paths,
        variants_per_font=variants_per_font,
    )
    characters_count, views_count = views.shape[:2]
    encoded = encode_evaluator_cells(
        model,
        views.reshape(characters_count * views_count, 1, 24, 24),
        device=device,
        batch_size=batch_size,
        autocast=autocast,
    ).reshape(characters_count, views_count, -1)
    prototypes = F.normalize(encoded.mean(dim=1), dim=-1)
    return RasterCharacterBank(
        characters=ordered,
        prototypes=prototypes,
        font_paths=tuple(font_paths),
        variants_per_character=views_count,
    )


@torch.no_grad()
def decode_raster_cells(
    model: VisualSemanticRasterTransducer,
    cells: torch.Tensor,
    bank: RasterCharacterBank,
    *,
    device: torch.device,
    batch_size: int = 256,
    temperature: float = 0.10,
    autocast: Callable[[], Any] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if cells.ndim != 5 or tuple(cells.shape[-3:]) != (1, 24, 24):
        raise ValueError("V32 decoded cells must be [B,A,1,24,24]")
    if temperature <= 0.0:
        raise ValueError("V32 evaluator temperature must be positive")
    batch, positions = cells.shape[:2]
    embeddings = encode_evaluator_cells(
        model,
        cells.reshape(batch * positions, 1, 24, 24),
        device=device,
        batch_size=batch_size,
        autocast=autocast,
    ).reshape(batch, positions, -1)
    similarities = embeddings @ bank.prototypes.float().transpose(0, 1)
    log_probabilities = F.log_softmax(similarities / temperature, dim=-1)
    return similarities.argmax(dim=-1), log_probabilities


def levenshtein_distance(first: Sequence[Any], second: Sequence[Any]) -> int:
    if len(first) < len(second):
        first, second = second, first
    prior = list(range(len(second) + 1))
    for row, first_value in enumerate(first, start=1):
        current = [row]
        for column, second_value in enumerate(second, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    prior[column] + 1,
                    prior[column - 1] + (first_value != second_value),
                )
            )
        prior = current
    return prior[-1]


def bootstrap_mean_summary(
    values: Sequence[float],
    *,
    seed: int = 20_263_271,
    samples: int = 2_000,
) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or not len(array) or not np.isfinite(array).all():
        raise ValueError("V32 bootstrap values must be a finite nonempty vector")
    output: dict[str, Any] = {
        "mean": float(array.mean()),
        "count": int(len(array)),
        "ci95": None,
    }
    if len(array) >= 100:
        rng = np.random.default_rng(seed)
        means = np.empty(samples, dtype=np.float64)
        for start in range(0, samples, 100):
            count = min(100, samples - start)
            indices = rng.integers(0, len(array), size=(count, len(array)))
            means[start : start + count] = array[indices].mean(axis=1)
        output["ci95"] = [
            float(np.quantile(means, 0.025)),
            float(np.quantile(means, 0.975)),
        ]
    return output


def sequence_evaluation(
    predicted_indices: torch.Tensor,
    predicted_lengths: torch.Tensor,
    target_sequences: Sequence[str],
    bank: RasterCharacterBank,
    *,
    log_probabilities: torch.Tensor | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if predicted_indices.ndim != 2 or predicted_lengths.shape != predicted_indices.shape[:1]:
        raise ValueError("V32 predicted sequences and lengths do not align")
    if len(target_sequences) != predicted_indices.shape[0]:
        raise ValueError("V32 predicted and target sequence counts do not align")
    if log_probabilities is not None and log_probabilities.shape[:2] != predicted_indices.shape:
        raise ValueError("V32 evaluator log probabilities do not align")
    inverse = bank.characters
    index = bank.character_to_index
    rows = []
    for sample_index, raw_target in enumerate(target_sequences):
        target = normalize_visible_text(raw_target)
        length = int(predicted_lengths[sample_index])
        predicted = "".join(
            inverse[int(value)] for value in predicted_indices[sample_index, :length]
        )
        position_matches = sum(
            left == right for left, right in zip(predicted, target)
        )
        log_similarity = None
        if log_probabilities is not None and target:
            values = []
            for position, character in enumerate(target):
                if character not in index or position >= log_probabilities.shape[1]:
                    continue
                values.append(
                    float(log_probabilities[sample_index, position, index[character]])
                )
            if values:
                log_similarity = float(np.mean(values))
        rows.append(
            {
                "predicted": predicted,
                "target": target,
                "character_accuracy": position_matches / max(1, len(target)),
                "character_error_rate": levenshtein_distance(predicted, target)
                / max(1, len(target)),
                "exact": float(predicted == target),
                "length_exact": float(length == len(target)),
                "target_log_similarity": log_similarity,
            }
        )
    names = (
        "character_accuracy",
        "character_error_rate",
        "exact",
        "length_exact",
    )
    metrics = {
        name: bootstrap_mean_summary([float(row[name]) for row in rows])
        for name in names
    }
    log_values = [
        float(row["target_log_similarity"])
        for row in rows
        if row["target_log_similarity"] is not None
    ]
    metrics["target_log_similarity"] = (
        bootstrap_mean_summary(log_values) if log_values else None
    )
    return metrics, rows


def _binary_f1(predicted: torch.Tensor, target: torch.Tensor) -> float:
    predicted = predicted.bool()
    target = target.bool()
    true_positive = float((predicted & target).sum())
    false_positive = float((predicted & ~target).sum())
    false_negative = float((~predicted & target).sum())
    denominator = 2.0 * true_positive + false_positive + false_negative
    return 1.0 if denominator == 0.0 else 2.0 * true_positive / denominator


def raster_quality_evaluation(
    generated_cells: torch.Tensor,
    generated_lengths: torch.Tensor,
    target_cells: torch.Tensor,
    target_mask: torch.Tensor,
    *,
    maximum_cells: int,
    overflow_flags: torch.Tensor | None = None,
) -> tuple[dict[str, Any], list[dict[str, float]]]:
    if generated_cells.shape != target_cells.shape:
        raise ValueError("V32 generated and target raster shapes differ")
    if target_mask.shape != generated_cells.shape[:2]:
        raise ValueError("V32 target mask does not align with rasters")
    if generated_lengths.shape != generated_cells.shape[:1]:
        raise ValueError("V32 generated lengths do not align with rasters")
    rows = []
    predicted_binary = generated_cells.float() >= 0.5
    target_binary = target_cells.float() >= 0.5
    predicted_edges = sobel_edges(generated_cells.float()).square().sum(dim=2).sqrt() >= 0.05
    target_edges = sobel_edges(target_cells.float()).square().sum(dim=2).sqrt() >= 0.05
    for index in range(generated_cells.shape[0]):
        length = int(generated_lengths[index])
        active_cells = generated_cells[index, :length]
        target_ink = float(target_binary[index].sum())
        blank_cells = (
            (active_cells >= 0.5).flatten(1).sum(dim=1) == 0
            if length
            else torch.ones(1, dtype=torch.bool)
        )
        overflow = (
            bool(overflow_flags[index])
            if overflow_flags is not None
            else length >= maximum_cells
        )
        rows.append(
            {
                "pixel_f1": _binary_f1(predicted_binary[index], target_binary[index]),
                "edge_f1": _binary_f1(predicted_edges[index], target_edges[index]),
                "ink_coverage": float(predicted_binary[index].sum()) / max(1.0, target_ink),
                "blank_cell_rate": float(blank_cells.float().mean()),
                "nonblank_answer": float(bool((active_cells >= 0.5).any())),
                "overflow": float(overflow),
                "length_exact": float(length == int(target_mask[index].sum())),
            }
        )
    metrics = {
        name: bootstrap_mean_summary([row[name] for row in rows])
        for name in rows[0]
    }
    return metrics, rows


__all__ = [
    "RasterCharacterBank",
    "bootstrap_mean_summary",
    "build_raster_character_bank",
    "decode_raster_cells",
    "encode_evaluator_cells",
    "levenshtein_distance",
    "raster_quality_evaluation",
    "render_character_bank_cells",
    "sequence_evaluation",
]

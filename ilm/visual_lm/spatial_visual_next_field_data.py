from __future__ import annotations

from typing import Any

from .conditional_visual_density_ratio_data import (
    ConditionalVisualCandidateBank,
    ConditionalVisualNaturalDataset,
    ConditionalVisualRenderConfig,
    build_v29_candidate_bank,
    build_v29_candidate_statistics,
    canonical_target_indices,
    conditional_visual_candidate_bank_receipt,
    conditional_visual_natural_collate,
    conditional_visual_natural_student_batch,
    conditional_visual_render_config_payload,
)
from .spatial_visual_next_field import V30_ARCHITECTURE


V30_CONTEXT_CELLS = 64
V30_SEQUENCE_CELLS = 65
V30_BANK_VIEWS = 2
V30_NATURAL_IMAGE_KEYS = (
    "first_context",
    "second_context",
    "canonical_target",
)
V30_NATURAL_STUDENT_KEYS = ("first_context", "second_context")


def build_v30_candidate_statistics(*args: Any, **kwargs: Any) -> Any:
    return build_v29_candidate_statistics(*args, **kwargs)


def build_v30_candidate_bank(*args: Any, **kwargs: Any) -> ConditionalVisualCandidateBank:
    return build_v29_candidate_bank(*args, **kwargs)


def spatial_visual_data_boundary_receipt() -> dict[str, Any]:
    return {
        "architecture": V30_ARCHITECTURE,
        "natural_context_shape": [V30_CONTEXT_CELLS, 1, 32, 32],
        "candidate_shape": [1, 32, 32],
        "predicted_field_shape": [16, 192],
        "student_natural_keys": list(V30_NATURAL_STUDENT_KEYS),
        "input_is_continuous_image_stream": True,
        "output_is_candidate_independent_continuous_field": True,
        "canonical_identity_derived_from_exact_pixels": True,
        "canonical_indices_are_temporary_loss_only": True,
        "pair_assignment_labels_are_positions": True,
        "pair_candidate_order_is_randomized": True,
        "pair_suffix_pixels_identical": True,
        "training_bank_is_host_only": True,
        "candidate_bank_deployed": False,
        "uses_strings": False,
        "uses_token_ids": False,
        "uses_unicode_ids": False,
        "uses_character_ids": False,
        "uses_ocr": False,
        "uses_visual_codebook": False,
        "uses_external_language_model": False,
    }


__all__ = [
    "ConditionalVisualCandidateBank",
    "ConditionalVisualNaturalDataset",
    "ConditionalVisualRenderConfig",
    "V30_BANK_VIEWS",
    "V30_CONTEXT_CELLS",
    "V30_NATURAL_IMAGE_KEYS",
    "V30_NATURAL_STUDENT_KEYS",
    "V30_SEQUENCE_CELLS",
    "build_v30_candidate_bank",
    "build_v30_candidate_statistics",
    "canonical_target_indices",
    "conditional_visual_candidate_bank_receipt",
    "conditional_visual_natural_collate",
    "conditional_visual_natural_student_batch",
    "conditional_visual_render_config_payload",
    "spatial_visual_data_boundary_receipt",
]

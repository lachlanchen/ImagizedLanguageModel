from __future__ import annotations

from .canonical_glyph_binding_v44 import (
    CanonicalGlyphBindingV44,
    V44_ARCHITECTURE,
    canonical_glyph_binding_v44_boundary_receipt,
)


V44_AUDIT_SEED = 20264440
V44_GATE_EPSILON = 1e-12


def canonical_glyph_binding_v44_boundary_is_clean(
    model: CanonicalGlyphBindingV44,
) -> bool:
    receipt = canonical_glyph_binding_v44_boundary_receipt(model)
    required_true = (
        "input_is_continuous_image_stream",
        "output_is_continuous_image_field",
        "output_is_direct_raster",
        "candidate_independent_residual",
        "long_history_excludes_shared_suffix",
        "tangent_field_update",
        "causal_over_visual_time",
        "rereads_generated_pixels",
        "base_parameters_frozen",
    )
    required_false = (
        "uses_strings",
        "uses_token_ids",
        "uses_unicode_ids",
        "uses_character_ids",
        "uses_vocabulary_embedding",
        "uses_vocabulary_output",
        "uses_ocr",
        "uses_visual_codebook",
        "uses_quantization",
        "uses_glyph_lookup",
        "uses_external_language_model",
        "candidate_bank_deployed",
    )
    return (
        receipt["architecture"] == V44_ARCHITECTURE
        and not receipt["parameter_names_with_forbidden_fragments"]
        and receipt["adapter_parameters"] == receipt["trainable_parameters"]
        and all(receipt.get(key) is True for key in required_true)
        and all(receipt.get(key) is False for key in required_false)
    )


def canonical_glyph_binding_v44_gate_report(
    base_language: dict[str, float],
    language: dict[str, float],
    base_development_pairs: dict[str, float],
    development_pairs: dict[str, float],
    consumed_pairs: dict[str, float],
    unseen_pairs: dict[str, float],
    *,
    boundary_clean: bool,
    base_state_exact: bool,
    adapter_parameters: int,
    peak_allocated_vram_gib: float,
) -> dict[str, bool]:
    def above(value: float, threshold: float) -> bool:
        return value - threshold > V44_GATE_EPSILON

    def below(value: float, threshold: float) -> bool:
        return threshold - value > V44_GATE_EPSILON

    return {
        "full_top1_gain_over_unigram": above(
            language["full_top1"] - language["unigram_top1"], 0.03
        ),
        "full_top1_gain_over_bigram": above(
            language["full_top1"] - language["bigram_top1"], 0.01
        ),
        "ordered_log_probability_gain_over_shuffled": above(
            language["full_target_log_probability"]
            - language["shuffled_target_log_probability"],
            0.05,
        ),
        "ordered_top1_gain_over_shuffled": above(
            language["full_top1"] - language["shuffled_top1"], 0.015
        ),
        "matched_base_top1_retention": (
            language["full_top1"]
            >= base_language["full_top1"] - 0.015 - V44_GATE_EPSILON
        ),
        "matched_base_log_probability_retention": (
            language["full_target_log_probability"]
            >= base_language["full_target_log_probability"]
            - 0.10
            - V44_GATE_EPSILON
        ),
        "development_arm_accuracy": above(
            development_pairs["full_arm_accuracy"], 0.60
        ),
        "development_gain_over_matched_base": above(
            development_pairs["full_arm_accuracy"]
            - base_development_pairs["full_arm_accuracy"],
            0.05,
        ),
        "development_gain_over_shuffled": above(
            development_pairs["full_arm_accuracy"]
            - development_pairs["shuffled_arm_accuracy"],
            0.04,
        ),
        "unseen_train_arm_accuracy": above(
            unseen_pairs["full_arm_accuracy"], 0.60
        ),
        "consumed_to_unseen_gap": below(
            consumed_pairs["full_arm_accuracy"]
            - unseen_pairs["full_arm_accuracy"],
            0.10,
        ),
        "student_boundary_clean": boundary_clean,
        "base_frozen_and_adapter_below_2m": (
            base_state_exact and adapter_parameters < 2_000_000
        ),
        "peak_allocated_vram_below_18_gib": below(
            peak_allocated_vram_gib,
            18.0,
        ),
    }

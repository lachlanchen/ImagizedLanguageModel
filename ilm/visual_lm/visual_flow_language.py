from __future__ import annotations

from typing import Any

import torch

from .canonical_glyph_flow_v43 import (
    CanonicalGlyphFlowV43,
    CanonicalGlyphFlowV43Config,
    canonical_glyph_flow_v43_boundary_receipt,
)
from .canonical_glyph_language import CanonicalGlyphLanguageConfig


VISUAL_FLOW_LANGUAGE_ARCHITECTURE = "visual-flow-language"


class VisualFlowLanguageModel(CanonicalGlyphFlowV43):
    """Train the causal visual reader and raster flow as one language model."""

    def __init__(
        self,
        language_config: CanonicalGlyphLanguageConfig,
        flow_config: CanonicalGlyphFlowV43Config = CanonicalGlyphFlowV43Config(),
    ) -> None:
        super().__init__(language_config, flow_config)

    def enable_end_to_end_training(self) -> None:
        self.unfreeze_language_core()
        self.unfreeze_writer()

    def flow_inputs(
        self,
        context: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        output = self.language_model(context)
        hidden = output["hidden_states"][:, -1]
        anchor = output["anchor_fields"][:, -1]
        size = self.language_model.config.cell_size
        plan = torch.zeros(
            (len(context), 1, size, size),
            device=hidden.device,
            dtype=hidden.dtype,
        )
        return hidden, anchor, plan


def visual_flow_language_boundary_receipt(
    model: VisualFlowLanguageModel,
) -> dict[str, Any]:
    receipt = canonical_glyph_flow_v43_boundary_receipt(model)
    return {
        **receipt,
        "architecture": VISUAL_FLOW_LANGUAGE_ARCHITECTURE,
        "end_to_end_reader_writer": True,
        "deterministic_ink_plan_used": False,
        "noise_is_writer_choice_variable": True,
        "old_field_generator_deployed": False,
    }


def visual_flow_language_boundary_is_clean(
    model: VisualFlowLanguageModel,
) -> bool:
    receipt = visual_flow_language_boundary_receipt(model)
    return (
        receipt["architecture"] == VISUAL_FLOW_LANGUAGE_ARCHITECTURE
        and receipt["end_to_end_reader_writer"] is True
        and receipt["deterministic_ink_plan_used"] is False
        and receipt["noise_is_writer_choice_variable"] is True
        and receipt["old_field_generator_deployed"] is False
        and not receipt["parameter_names_with_forbidden_fragments"]
        and receipt["input_is_continuous_image_stream"] is True
        and receipt["output_is_direct_raster"] is True
        and receipt["uses_token_ids"] is False
        and receipt["uses_unicode_ids"] is False
        and receipt["uses_character_ids"] is False
        and receipt["uses_ocr"] is False
        and receipt["uses_visual_codebook"] is False
        and receipt["uses_glyph_lookup"] is False
        and receipt["uses_external_language_model"] is False
        and receipt["candidate_bank_deployed"] is False
    )

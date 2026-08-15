from __future__ import annotations

import torch
import torch.nn as nn

from ilm.visual_lm.canonical_glyph_language import OrthonormalGlyphField
from ilm.visual_lm.ink_jepa_data import VisualGrammarRecord
from ilm.visual_lm.visual_cell_data import visual_cell_partition
from ilm.visual_lm.visual_cell_eval_data import VisualCharacterStatistics
from ilm.visual_lm.visual_future_block_language_v48 import (
    VisualFutureBlockLanguageConfigV48,
    VisualFutureBlockLanguageModelV48,
)
from ilm.visual_lm.visual_future_block_language_v48_evaluation import (
    build_offset_conditional_counts_v48,
    evaluate_closed_loop_generation_v48,
    evaluate_four_future_fields_v48,
    evaluate_offset_conditional_control_v48,
    finite_metric_tree_v48,
    visual_future_block_gate_report_v48,
    visual_future_block_language_boundary_is_clean_v48,
)


def _identifier_for(split: str) -> str:
    for index in range(100_000):
        identifier = f"v48-evaluation-{split}-{index}"
        if visual_cell_partition(identifier) == split:
            return identifier
    raise AssertionError(f"could not find a V48 {split} identifier")


def _statistics() -> VisualCharacterStatistics:
    return VisualCharacterStatistics(
        characters=("天", "地"),
        counts=(80, 80),
        bigram_rows={},
        visible_character_count=160,
        han_character_count=160,
    )


def _glyph(index: int) -> torch.Tensor:
    image = torch.zeros(1, 32, 32)
    image.reshape(-1)[index : index + 3] = 1.0
    return image


class _RepeatLastImageModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.field = OrthonormalGlyphField()
        self.register_buffer("contrastive_scale", torch.tensor(10.0))

    def language(self, context: torch.Tensor) -> dict[str, torch.Tensor]:
        final = self.field.encode_unit(context[:, -1])
        future = final[:, None, None].expand(-1, context.shape[1], 4, -1)
        return {
            "anchor_fields": future[:, :, 0],
            "future_anchor_fields": future,
        }

    def generate(
        self,
        prefix: torch.Tensor,
        *,
        new_cells: int,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        generated = prefix[:, -1:, ...].expand(-1, new_cells, -1, -1, -1)
        return torch.cat((prefix, generated), dim=1), {
            "generated_cells": generated,
            "rereads_generated_pixels": torch.tensor(True),
        }


def test_offset_conditional_control_is_training_only_and_horizon_specific() -> None:
    record = VisualGrammarRecord(
        identifier=_identifier_for("train"),
        text="天地" * 80,
        language="zh-Hant",
        source="unit-test",
        rights="test-only",
    )
    statistics = _statistics()
    counts = build_offset_conditional_counts_v48(
        [record],
        statistics,
        script_views_mode="original",
    )
    loader = [
        {
            "last_character": ["天", "地"],
            "target_indices": torch.tensor(
                [[1, 0, 1, 0], [0, 1, 0, 1]], dtype=torch.long
            ),
        }
    ]
    result = evaluate_offset_conditional_control_v48(
        counts,
        statistics,
        loader,
    )
    assert set(result) == {"1", "2", "3", "4"}
    assert all(row["top1"] == 1.0 for row in result.values())
    assert finite_metric_tree_v48(result)


def test_future_and_closed_loop_evaluators_score_emitted_images() -> None:
    model = _RepeatLastImageModel()
    bank = torch.stack((_glyph(0), _glyph(8)))
    context = torch.zeros(2, 64, 1, 32, 32)
    context[0, -1] = bank[0]
    context[1, -1] = bank[1]
    future = torch.stack(
        (
            bank[0, None].expand(4, -1, -1, -1),
            bank[1, None].expand(4, -1, -1, -1),
        )
    )
    targets = torch.tensor([[0, 0, 0, 0], [1, 1, 1, 1]])
    loader = [
        {
            "context": context,
            "future_pixels": future,
            "target_indices": targets,
        }
    ]
    predicted = evaluate_four_future_fields_v48(
        model,
        loader,
        bank,
        device=torch.device("cpu"),
        precision="fp32",
    )
    assert all(
        predicted["horizons"][str(index)]["top1"] == 1.0
        for index in range(1, 5)
    )
    closed = evaluate_closed_loop_generation_v48(
        model,
        loader,
        bank,
        device=torch.device("cpu"),
        precision="fp32",
        maximum_examples=2,
    )
    assert closed["mean_identity_top1"] == 1.0
    assert closed["mean_pixel_f1"] == 1.0
    assert closed["blank_outputs"] == 0.0
    assert closed["nonfinite_outputs"] == 0.0
    assert closed["generated_before_candidate_bank_scoring"] is True
    assert closed["recurrent_cells_are_visible_rasters"] is True


def test_boundary_and_frozen_gate_report_are_complete() -> None:
    model = VisualFutureBlockLanguageModelV48(
        VisualFutureBlockLanguageConfigV48(
            model_dim=128,
            layers=1,
            heads=4,
            mlp_ratio=2.0,
            dropout=0.0,
        )
    )
    assert visual_future_block_language_boundary_is_clean_v48(model)
    language = {
        "full_top1": 0.25,
        "bigram_top1": 0.10,
        "unigram_top1": 0.01,
        "shuffled_top1": 0.20,
        "full_target_log_probability": -2.0,
        "shuffled_target_log_probability": -2.2,
    }
    future = {
        "horizons": {
            str(index): {
                "top1": 0.20,
                "distinct_top1_predictions": 128.0,
                "most_common_top1_fraction": 0.15,
            }
            for index in range(1, 5)
        },
        "offset_conditional_control": {
            str(index): {"top1": 0.10} for index in range(1, 5)
        },
    }
    terminal = {
        "63": {"top1": 0.20},
        "64": {"top1": 0.19, "most_common_top1_fraction": 0.09},
    }
    direct = {
        "anchor_identity_top1": 0.20,
        "visible_identity_top1": 0.18,
        "visible_pixel_f1": 0.50,
        "visible_blank_rate": 0.0,
        "proposal_visible_reread_cosine": 0.90,
    }
    gates = visual_future_block_gate_report_v48(
        language,
        future,
        {"full_arm_accuracy": 0.60},
        terminal,
        direct,
        {
            "mean_identity_top1": 0.10,
            "nonfinite_outputs": 0.0,
            "blank_outputs": 0.0,
            "recurrent_cells_are_visible_rasters": True,
        },
        boundary_clean=True,
        trainable_parameters=16_000_000,
        peak_allocated_vram_gib=10.0,
        training_elapsed_seconds=3_600.0,
        matched_v42_full_top1=0.20,
    )
    assert len(gates) == 16
    assert all(gates.values())


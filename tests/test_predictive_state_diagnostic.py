from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ilm.visual_lm.ink_jepa_data import VisualGrammarRecord
from ilm.visual_lm.predictive_state_diagnostic import (
    audit_window_digest,
    build_partition_audit_windows,
    evaluate_context_length_curve,
    evaluate_predictive_state,
    field_geometry,
    partition_generalization_gaps,
    shuffle_prefix_preserving_suffix,
)
from ilm.visual_lm.visual_cell_data import visual_cell_partition
from ilm.visual_lm.visual_cell_eval_data import VisualCharacterStatistics


def _identifier_for(split: str) -> str:
    for index in range(100_000):
        identifier = f"predictive-diagnostic-{split}-{index}"
        if visual_cell_partition(identifier) == split:
            return identifier
    raise AssertionError(f"could not find a {split} record identifier")


def _record(split: str) -> VisualGrammarRecord:
    return VisualGrammarRecord(
        identifier=_identifier_for(split),
        text=("天地" * 80),
        language="zh-Hant",
        source="unit-test",
        rights="test-only",
    )


def _statistics() -> VisualCharacterStatistics:
    return VisualCharacterStatistics(
        characters=("天", "地"),
        counts=(80, 80),
        bigram_rows={},
        visible_character_count=160,
        han_character_count=160,
    )


def test_partition_windows_and_digest_are_split_specific_and_deterministic() -> None:
    records = [_record("train"), _record("development")]
    train = build_partition_audit_windows(
        records,
        _statistics(),
        split="train",
        count=8,
        seed=101,
        script_views_mode="original",
    )
    repeated = build_partition_audit_windows(
        records,
        _statistics(),
        split="train",
        count=8,
        seed=101,
        script_views_mode="original",
    )
    development = build_partition_audit_windows(
        records,
        _statistics(),
        split="development",
        count=8,
        seed=102,
        script_views_mode="original",
    )
    assert train == repeated
    assert all(window.identifier == records[0].identifier for window in train)
    assert all(window.identifier == records[1].identifier for window in development)
    assert audit_window_digest(train) == audit_window_digest(repeated)
    assert audit_window_digest(train) != audit_window_digest(development)


def test_shuffle_is_deterministic_and_preserves_the_final_four_images() -> None:
    context = torch.arange(2 * 8, dtype=torch.float32).reshape(2, 8, 1, 1, 1)
    context = context.expand(-1, -1, 1, 32, 32).clone()
    first = shuffle_prefix_preserving_suffix(
        context,
        first_index=7,
        seed=13,
    )
    second = shuffle_prefix_preserving_suffix(
        context,
        first_index=7,
        seed=13,
    )
    assert torch.equal(first, second)
    assert torch.equal(first[:, -4:], context[:, -4:])
    for row in range(2):
        assert sorted(first[row, :-4, 0, 0, 0].tolist()) == sorted(
            context[row, :-4, 0, 0, 0].tolist()
        )


def test_field_geometry_reports_rank_and_exact_self_retrieval() -> None:
    geometry = field_geometry(torch.eye(4))
    assert geometry["bank_size"] == 4.0
    assert geometry["field_dimension"] == 4.0
    assert geometry["self_retrieval_top1"] == 1.0
    assert geometry["strict_self_retrieval_fraction"] == 1.0
    assert geometry["off_diagonal_cosine_mean"] == 0.0
    assert math.isclose(geometry["centered_effective_rank"], 3.0, rel_tol=1e-5)


class _IdentityField(nn.Module):
    def encode_unit(self, pixels: torch.Tensor) -> torch.Tensor:
        leading = pixels.shape[:-3]
        fields = pixels.reshape(-1, 32 * 32)[:, :2]
        return F.normalize(fields, dim=-1).reshape(*leading, 2)


class _LastImageModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.field = _IdentityField()
        self.register_buffer("contrastive_scale", torch.tensor(10.0))

    def language(self, context: torch.Tensor) -> dict[str, torch.Tensor]:
        fields = self.field.encode_unit(context)
        return {"anchor_fields": fields}


def _glyph(first: float, second: float) -> torch.Tensor:
    image = torch.zeros(1, 32, 32)
    image.reshape(-1)[0] = first
    image.reshape(-1)[1] = second
    return image


def test_predictive_state_evaluator_uses_images_and_emits_complete_curves() -> None:
    bank_images = torch.stack((_glyph(1.0, 0.0), _glyph(0.0, 1.0)))
    context = torch.zeros(2, 64, 1, 32, 32)
    context[0, :, :, 0, 0] = 1.0
    context[1, :, :, 0, 1] = 1.0
    continuation = bank_images[:, None]
    loader = [
        {
            "context": context,
            "continuation": continuation,
            "target_index": torch.tensor([0, 1]),
        }
    ]
    model = _LastImageModel()
    bank_fields = model.field.encode_unit(bank_images)
    result = evaluate_predictive_state(
        model,
        loader,
        bank_fields,
        device=torch.device("cpu"),
        precision="fp32",
        shuffle_seed=29,
    )
    assert result["examples"] == 2.0
    assert set(result["context_curve"]) == {"1", "2", "4", "8", "16", "32", "64"}
    assert set(result["shuffled_curve"]) == {"8", "16", "32", "64"}
    assert all(value["top1"] == 1.0 for value in result["context_curve"].values())
    assert result["anchor_interventions"]["64"]["vs_suffix4"]["anchor_cosine"] == 1.0
    dense = evaluate_context_length_curve(
        model,
        loader,
        bank_fields,
        lengths=(3, 5, 64),
        device=torch.device("cpu"),
        precision="fp32",
    )
    assert set(dense) == {"3", "5", "64"}
    assert all(value["top1"] == 1.0 for value in dense.values())


def test_partition_generalization_gap_is_development_minus_train() -> None:
    models = {
        "v": {
            "partitions": {
                "train_partition": {
                    "context_curve": {
                        str(length): {
                            "top1": 0.5,
                            "target_log_probability": -2.0,
                        }
                        for length in (1, 2, 4, 8, 16, 32, 64)
                    }
                },
                "development_partition": {
                    "context_curve": {
                        str(length): {
                            "top1": 0.25,
                            "target_log_probability": -3.0,
                        }
                        for length in (1, 2, 4, 8, 16, 32, 64)
                    }
                },
            }
        }
    }
    gaps = partition_generalization_gaps(models)["v"]
    assert gaps["context_64_top1"] == -0.25
    assert gaps["context_64_target_log_probability"] == -1.0

from __future__ import annotations

import torch

from ilm.visual_lm.canonical_glyph_language_data import CanonicalGlyphRenderConfig
from ilm.visual_lm.ink_jepa_data import VisualGrammarRecord
from ilm.visual_lm.visual_cell_data import visual_cell_partition
from ilm.visual_lm.visual_cell_eval_data import VisualCharacterStatistics
from ilm.visual_lm.visual_future_block_language_v48 import (
    VisualFutureBlockLanguageConfigV48,
    VisualFutureBlockLanguageModelV48,
    visual_future_block_language_boundary_receipt_v48,
)
from ilm.visual_lm.visual_future_block_language_v48_data import (
    VisualFutureBlockLanguageDataset,
    build_four_future_audit_windows_v48,
    visual_future_block_collate,
    visual_future_block_data_boundary_receipt,
    visual_future_block_student_batch,
)
from ilm.visual_lm.visual_future_block_language_v48_training import (
    visual_future_block_language_loss_v48,
)


def _identifier_for(split: str) -> str:
    for index in range(100_000):
        identifier = f"v48-test-{split}-{index}"
        if visual_cell_partition(identifier) == split:
            return identifier
    raise AssertionError(f"could not find a V48 {split} identifier")


def _record(split: str = "train") -> VisualGrammarRecord:
    writing = (
        "天地玄黄宇宙洪荒日月盈昃辰宿列张寒来暑往秋收冬藏"
        "闰余成岁律吕调阳云腾致雨露结为霜金生丽水玉出昆冈"
        "剑号巨阙珠称夜光果珍李柰菜重芥姜海咸河淡鳞潜羽翔"
    )
    return VisualGrammarRecord(
        identifier=_identifier_for(split),
        text=writing,
        language="zh-Hans",
        source="unit-test",
        rights="test-only",
    )


def _small_config() -> VisualFutureBlockLanguageConfigV48:
    return VisualFutureBlockLanguageConfigV48(
        model_dim=128,
        layers=2,
        heads=4,
        mlp_ratio=2.0,
        dropout=0.0,
    )


def test_v48_dataset_exposes_only_dense_future_image_tensors() -> None:
    dataset = VisualFutureBlockLanguageDataset(
        [_record("train")],
        split="train",
        render_config=CanonicalGlyphRenderConfig(script_views="original"),
        seed=48,
        length=2,
        expose_evaluation_labels=True,
    )
    batch = visual_future_block_collate([dataset[0], dataset[1]])
    assert batch["context"].shape == (2, 64, 1, 32, 32)
    assert batch["future_pixels"].shape == (2, 64, 4, 1, 32, 32)
    assert "segment" in batch["metadata"][0]
    student = visual_future_block_student_batch(batch)
    assert set(student) == {"context", "future_pixels"}
    assert all(value.is_floating_point() for value in student.values())
    receipt = visual_future_block_data_boundary_receipt()
    assert receipt["metadata_excluded_from_student"] is True
    assert receipt["target_is_continuous_image_block"] is True
    assert receipt["candidate_bank_deployed"] is False


def test_v48_model_is_causal_and_predicts_four_unit_image_fields() -> None:
    torch.manual_seed(48)
    model = VisualFutureBlockLanguageModelV48(_small_config()).eval()
    context = torch.rand(2, 7, 1, 32, 32)
    changed = context.clone()
    changed[:, -1] = 1.0 - changed[:, -1]
    first = model.language(context)
    second = model.language(changed)
    assert first["hidden_states"].shape == (2, 7, 128)
    assert first["anchor_fields"].shape == (2, 7, 1024)
    assert first["future_anchor_fields"].shape == (2, 7, 4, 1024)
    torch.testing.assert_close(
        first["future_anchor_fields"].norm(dim=-1),
        torch.ones(2, 7, 4),
        atol=2e-6,
        rtol=2e-6,
    )
    torch.testing.assert_close(
        first["future_anchor_fields"][:, :-1],
        second["future_anchor_fields"][:, :-1],
        atol=2e-5,
        rtol=2e-5,
    )
    assert not torch.allclose(
        first["future_anchor_fields"][:, -1],
        second["future_anchor_fields"][:, -1],
    )
    receipt = visual_future_block_language_boundary_receipt_v48(model)
    assert receipt["parameter_names_with_forbidden_fragments"] == []
    assert receipt["uses_stochastic_generator"] is False
    assert receipt["inverse_dct_threshold_is_fixed_zero"] is True


def test_v48_loss_reaches_shared_head_and_all_future_offsets() -> None:
    torch.manual_seed(49)
    model = VisualFutureBlockLanguageModelV48(_small_config())
    context = torch.rand(2, 4, 1, 32, 32)
    future = torch.rand(2, 4, 4, 1, 32, 32)
    measured = visual_future_block_language_loss_v48(
        model,
        model(context),
        future,
        generator=torch.Generator().manual_seed(50),
        maximum_positions=8,
    )
    assert torch.isfinite(measured.loss)
    assert measured.selected_positions == 8
    assert len(measured.horizons) == 4
    measured.loss.backward()
    assert model.visual_head[-1].weight.grad is not None
    assert model.future_offsets.grad is not None
    assert torch.isfinite(model.visual_head[-1].weight.grad).all()
    assert torch.isfinite(model.future_offsets.grad).all()
    assert (model.future_offsets.grad.abs().sum(dim=1) > 0).all()


def test_v48_generation_emits_and_rereads_only_visible_binary_images() -> None:
    torch.manual_seed(51)
    model = VisualFutureBlockLanguageModelV48(_small_config()).eval()
    prefix = torch.rand(1, 5, 1, 32, 32)
    forecast, forecast_trace = model.forecast(prefix)
    assert forecast.shape == (1, 4, 1, 32, 32)
    assert forecast_trace["future_fields"].shape == (1, 4, 1024)
    assert forecast_trace["visible_reread_fields"].shape == (1, 4, 1024)
    sequence, trace = model.generate(prefix, new_cells=3)
    assert sequence.shape == (1, 8, 1, 32, 32)
    assert trace["generated_cells"].shape == (1, 3, 1, 32, 32)
    assert trace["generated_fields"].shape == (1, 3, 1024)
    assert trace["reread_fields"].shape == (1, 3, 1024)
    assert trace["rereads_generated_pixels"].item() is True
    assert set(torch.unique(trace["generated_cells"]).tolist()).issubset(
        {0.0, 1.0}
    )
    torch.testing.assert_close(
        trace["reread_fields"],
        model.field.encode_unit(trace["generated_cells"]),
        atol=2e-6,
        rtol=2e-6,
    )


def test_v48_future_window_reservoir_is_deterministic() -> None:
    record = _record("development")
    statistics = VisualCharacterStatistics(
        characters=("天", "地"),
        counts=(100, 100),
        bigram_rows={},
        visible_character_count=200,
        han_character_count=200,
    )
    alternating = VisualGrammarRecord(
        identifier=record.identifier,
        text="天地" * 100,
        language=record.language,
        source=record.source,
        rights=record.rights,
    )
    first, eligible = build_four_future_audit_windows_v48(
        [alternating],
        statistics,
        count=8,
        seed=52,
        script_views_mode="original",
    )
    second, repeated_eligible = build_four_future_audit_windows_v48(
        [alternating],
        statistics,
        count=8,
        seed=52,
        script_views_mode="original",
    )
    assert first == second
    assert eligible == repeated_eligible
    assert eligible > len(first)
    assert all(len(window.continuation) == 4 for window in first)


def test_v48_full_model_fits_the_frozen_parameter_budget() -> None:
    model = VisualFutureBlockLanguageModelV48(
        VisualFutureBlockLanguageConfigV48()
    )
    receipt = visual_future_block_language_boundary_receipt_v48(model)
    assert receipt["total_parameters"] == 16_278_401
    assert receipt["trainable_parameters"] == 16_278_401
    assert receipt["total_parameters"] < 17_000_000

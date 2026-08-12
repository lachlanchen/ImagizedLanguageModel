from __future__ import annotations

import torch

from ilm.visual_lm.visual_cell_eval_data import VisualCharacterStatistics
from ilm.visual_lm.visual_cell_stream import VisualCellStreamConfig, VisualCellStreamModel
from scripts.eval_visual_cell_stream_development_v25 import (
    encode_visual_bank,
    evaluate_language,
    evaluate_writer,
    language_gate_report,
    student_boundary_is_clean,
    writer_gate_report,
)


def tiny_model() -> VisualCellStreamModel:
    torch.manual_seed(37)
    return VisualCellStreamModel(
        VisualCellStreamConfig(
            maximum_cells=64,
            visual_dim=64,
            model_dim=128,
            layers=1,
            heads=4,
            mlp_ratio=2.0,
            dropout=0.0,
            retina_base_channels=8,
            writer_base_channels=8,
            time_dim=16,
        )
    ).eval()


def statistics() -> VisualCharacterStatistics:
    return VisualCharacterStatistics(
        characters=("天", "地", "玄", "黄"),
        counts=(40, 30, 20, 10),
        bigram_rows={
            "天": ((0, 2), (1, 8)),
            "地": ((0, 7), (1, 1)),
        },
        visible_character_count=100,
        han_character_count=100,
    )


def audit_batch() -> dict[str, object]:
    return {
        "context": torch.rand(4, 64, 1, 32, 32),
        "continuation": torch.rand(4, 16, 1, 32, 32),
        "reference_continuation": torch.rand(4, 16, 1, 32, 32),
        "target_index": torch.tensor([0, 1, 1, 0]),
        "target_character": ["天", "地", "地", "天"],
        "last_character": ["天", "天", "地", "地"],
        "identifier": ["a", "b", "c", "d"],
        "script_view": ["original"] * 4,
    }


def test_language_audit_reports_all_causal_and_symbolic_controls() -> None:
    model = tiny_model()
    bank_images = torch.rand(4, 2, 1, 32, 32)
    bank_visual = encode_visual_bank(
        model,
        bank_images,
        device=torch.device("cpu"),
        precision="fp32",
        batch_size=2,
    )
    metrics = evaluate_language(
        model,
        [audit_batch()],
        statistics(),
        bank_visual,
        device=torch.device("cpu"),
        precision="fp32",
    )
    assert metrics["examples"] == 4
    assert metrics["counterfactual_pairs"] >= 1
    assert metrics["student_boundary_clean"] == 1.0
    for route in ("full", "last", "shuffled", "blank", "unigram", "bigram"):
        assert 0.0 <= metrics[f"{route}_top1"] <= 1.0
        assert 0.0 <= metrics[f"{route}_top5"] <= 1.0
        assert torch.isfinite(torch.tensor(metrics[f"{route}_target_log_probability"]))
    assert student_boundary_is_clean(model)


def test_language_gates_are_strict_at_preregistered_thresholds() -> None:
    metrics = {
        "full_top1": 0.50,
        "last_top1": 0.47,
        "unigram_top1": 0.46,
        "full_target_log_probability": -2.0,
        "last_target_log_probability": -2.06,
        "shuffled_top1": 0.48,
        "counterfactual_switch_accuracy": 0.56,
        "full_target_cosine": 0.56,
        "student_boundary_clean": 1.0,
        "peak_allocated_vram_gib": 17.0,
    }
    gates = language_gate_report(metrics)
    assert gates["full_history_top1_gain_over_last"] is False
    assert gates["full_history_top1_gain_over_unigram"] is True
    assert gates["full_history_log_probability_gain_over_last"] is True
    assert gates["ordered_history_top1_gain_over_shuffled"] is True


def test_writer_audit_uses_free_pixels_and_reports_rereading() -> None:
    model = tiny_model()
    bank_visual = encode_visual_bank(
        model,
        torch.rand(4, 2, 1, 32, 32),
        device=torch.device("cpu"),
        precision="fp32",
    )
    metrics, sample = evaluate_writer(
        model,
        [audit_batch()],
        statistics(),
        bank_visual,
        device=torch.device("cpu"),
        precision="fp32",
        samples=2,
        autonomous_samples=1,
        candidates=1,
        flow_steps=1,
    )
    assert metrics["examples"] == 2
    assert metrics["autonomous_examples"] == 1
    assert metrics["rereads_generated_pixels"] == 1.0
    assert sample is not None
    assert sample["generated"].shape == (1, 1, 32, 32)
    assert sample["autonomous"].shape == (16, 1, 32, 32)
    assert set(writer_gate_report(metrics)) == {
        "generated_identity_top1",
        "reread_target_cosine",
        "generated_pixel_f1",
        "blank_rate",
        "position16_density_ratio_lower",
        "position16_density_ratio_upper",
        "rereads_generated_pixels",
    }

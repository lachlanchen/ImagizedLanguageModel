from __future__ import annotations

import pytest
import torch

from ilm.visual_lm.visual_semantic_raster_data import (
    V32_DEVELOPMENT_FONTS,
    VisualRasterRenderConfig,
)
from ilm.visual_lm.visual_semantic_raster_evaluation import (
    RasterCharacterBank,
    bootstrap_mean_summary,
    build_raster_character_bank,
    decode_raster_cells,
    levenshtein_distance,
    raster_quality_evaluation,
    render_character_bank_cells,
    sequence_evaluation,
)
from ilm.visual_lm.visual_semantic_raster_transducer import (
    VisualSemanticRasterConfig,
    VisualSemanticRasterTransducer,
)


def _model() -> VisualSemanticRasterTransducer:
    config = VisualSemanticRasterConfig(
        maximum_prompt_patches=8,
        maximum_answer_cells=4,
        reader_hidden_size=32,
        reader_layers=1,
        reader_heads=4,
        reader_intermediate_size=64,
        reader_dropout=0.0,
        planner_dim=32,
        planner_layers=1,
        planner_heads=4,
        planner_mlp_dim=64,
        planner_dropout=0.0,
        cell_retina_channels=8,
        target_width=32,
        target_blocks=1,
        latent_dim=8,
        decoder_width=32,
        decoder_layers=1,
        decoder_heads=4,
        decoder_mlp_dim=64,
        decoder_dropout=0.0,
    )
    return VisualSemanticRasterTransducer(config).eval()


def test_v32_levenshtein_handles_insertions_and_substitutions() -> None:
    assert levenshtein_distance("天地", "天地") == 0
    assert levenshtein_distance("天地", "天玄地") == 1
    assert levenshtein_distance("天地", "天人") == 1


def test_v32_bootstrap_ci_is_only_reported_at_one_hundred_samples() -> None:
    assert bootstrap_mean_summary([1.0] * 99)["ci95"] is None
    summary = bootstrap_mean_summary([0.0, 1.0] * 50, samples=200)
    assert summary["mean"] == 0.5
    assert summary["count"] == 100
    assert summary["ci95"] is not None


def test_v32_candidate_bank_is_evaluator_only_and_decodes_its_views() -> None:
    model = _model()
    config = VisualRasterRenderConfig(
        maximum_prompt_patches=8,
        maximum_answer_cells=4,
        augment=False,
    )
    characters, views = render_character_bank_cells(
        "天地",
        render_config=config,
        font_paths=V32_DEVELOPMENT_FONTS,
    )
    bank = build_raster_character_bank(
        model,
        characters,
        render_config=config,
        font_paths=V32_DEVELOPMENT_FONTS,
        device=torch.device("cpu"),
    )
    indices, log_probabilities = decode_raster_cells(
        model,
        views[:, 0][:, None],
        bank,
        device=torch.device("cpu"),
    )
    assert indices[:, 0].tolist() == [0, 1]
    assert log_probabilities.shape == (2, 1, 2)
    assert bank.receipt()["absent_from_student_generate"] is True


def test_v32_sequence_metrics_include_errors_lengths_and_log_similarity() -> None:
    bank = RasterCharacterBank(
        characters=("人", "地", "天"),
        prototypes=torch.eye(3),
        font_paths=("test",),
        variants_per_character=1,
    )
    predicted = torch.tensor([[2, 1, 0], [2, 0, 1]])
    lengths = torch.tensor([2, 2])
    log_probabilities = torch.full((2, 3, 3), -10.0)
    log_probabilities[0, 0, 2] = 0.0
    log_probabilities[0, 1, 1] = 0.0
    metrics, rows = sequence_evaluation(
        predicted,
        lengths,
        ["天地", "天人地"],
        bank,
        log_probabilities=log_probabilities,
    )
    assert metrics["exact"]["mean"] == 0.5
    assert metrics["length_exact"]["mean"] == 0.5
    assert rows[0]["target_log_similarity"] == 0.0
    assert rows[1]["character_error_rate"] == pytest.approx(1 / 3)


def test_v32_raster_metrics_reward_exact_images_and_detect_overflow() -> None:
    target = torch.zeros(2, 3, 1, 24, 24)
    target[:, :2, :, 5:19, 9:15] = 1.0
    target_mask = torch.tensor([[1.0, 1.0, 0.0], [1.0, 1.0, 0.0]])
    lengths = torch.tensor([2, 3])
    metrics, rows = raster_quality_evaluation(
        target.clone(),
        lengths,
        target,
        target_mask,
        maximum_cells=3,
        overflow_flags=torch.tensor([False, True]),
    )
    assert rows[0]["pixel_f1"] == 1.0
    assert rows[0]["edge_f1"] == 1.0
    assert metrics["overflow"]["mean"] == 0.5
    assert metrics["length_exact"]["mean"] == 0.5

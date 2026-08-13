from __future__ import annotations

import pytest
import torch

from ilm.visual_lm.visual_semantic_raster_transducer import (
    VisualSemanticRasterConfig,
    VisualSemanticRasterTransducer,
    visual_semantic_raster_boundary_receipt,
)


def _config() -> VisualSemanticRasterConfig:
    return VisualSemanticRasterConfig(
        maximum_prompt_patches=12,
        maximum_answer_cells=5,
        reader_hidden_size=48,
        reader_layers=1,
        reader_heads=4,
        reader_intermediate_size=96,
        reader_dropout=0.0,
        planner_dim=64,
        planner_layers=2,
        planner_heads=4,
        planner_mlp_dim=128,
        planner_dropout=0.0,
        cell_retina_channels=16,
        target_width=48,
        target_blocks=2,
        latent_dim=12,
        decoder_width=48,
        decoder_layers=2,
        decoder_heads=4,
        decoder_mlp_dim=96,
        decoder_dropout=0.0,
        feedback_noise_probability=0.0,
        feedback_ground_truth_probability=0.0,
    )


def _batch(config: VisualSemanticRasterConfig, batch: int = 2) -> dict[str, torch.Tensor]:
    answer_mask = torch.tensor([[1, 1, 1, 0, 0]], dtype=torch.float32).expand(batch, -1).clone()
    return {
        "prompt_pixels": torch.rand(batch, 3, 16, config.prompt_width),
        "prompt_mask": torch.ones(batch, config.maximum_prompt_patches),
        "answer_cells": torch.rand(batch, config.maximum_answer_cells, 1, 24, 24),
        "answer_mask": answer_mask,
    }


def test_v32_forward_is_pixel_to_continuous_state_to_pixels() -> None:
    config = _config()
    model = VisualSemanticRasterTransducer(config).eval()
    batch = _batch(config)
    with torch.no_grad():
        output = model(**batch, feedback_mode="decoded")
    assert output.target_states.shape == (2, 5, 12)
    assert output.state_mean.shape == (2, 5, 12)
    assert output.state_log_scale.shape == (2, 5, 12)
    assert output.raster_logits.shape == (2, 5, 1, 24, 24)
    assert output.stop_logits.shape == (2, 6)
    assert output.clean_hidden.shape == (2, 6, 64)
    assert not output.clean_hidden.requires_grad
    assert torch.isfinite(output.raster_logits).all()


def test_v32_causal_planner_hides_future_answer_cells() -> None:
    config = _config()
    model = VisualSemanticRasterTransducer(config).eval()
    batch = _batch(config, batch=1)
    altered = batch["answer_cells"].clone()
    altered[:, 3:] = 1.0 - altered[:, 3:]
    with torch.no_grad():
        first = model.plan(
            batch["prompt_pixels"], batch["prompt_mask"], batch["answer_cells"]
        )
        second = model.plan(batch["prompt_pixels"], batch["prompt_mask"], altered)
    assert torch.equal(first[:, :4], second[:, :4])


def test_v32_decoded_feedback_is_detached_from_state_planner() -> None:
    config = _config()
    model = VisualSemanticRasterTransducer(config).train()
    batch = _batch(config)
    output = model(**batch, feedback_mode="decoded")
    loss = output.state_mean.square().mean() + output.stop_logits.square().mean()
    loss.backward()
    assert any(parameter.grad is not None for parameter in model.planner.parameters())
    assert all(parameter.grad is None for parameter in model.raster_decoder.parameters())


def test_v32_raster_loss_path_reaches_target_encoder_and_decoder() -> None:
    config = _config()
    model = VisualSemanticRasterTransducer(config).train()
    output = model(**_batch(config), feedback_mode="decoded")
    output.raster_logits.square().mean().backward()
    assert any(parameter.grad is not None for parameter in model.target_encoder.parameters())
    assert any(parameter.grad is not None for parameter in model.raster_decoder.parameters())


def test_v32_autonomous_generation_returns_primary_raster_without_candidates() -> None:
    config = _config()
    model = VisualSemanticRasterTransducer(config).eval()
    batch = _batch(config)
    with torch.no_grad():
        model.stop_head.weight.zero_()
        model.stop_head.bias.fill_(-20.0)
        generation = model.generate(
            batch["prompt_pixels"],
            batch["prompt_mask"],
            maximum_cells=3,
        )
    assert generation.cells.shape == (2, 3, 1, 24, 24)
    assert generation.cell_mask.shape == (2, 3)
    assert generation.strips().shape == (2, 1, 24, 72)
    assert generation.lengths.tolist() == [3, 3]
    assert torch.isfinite(generation.cells).all()


def test_v32_stop_head_can_end_after_the_required_first_cell() -> None:
    config = _config()
    model = VisualSemanticRasterTransducer(config).eval()
    batch = _batch(config)
    with torch.no_grad():
        model.stop_head.weight.zero_()
        model.stop_head.bias.fill_(20.0)
        generation = model.generate(
            batch["prompt_pixels"],
            batch["prompt_mask"],
            maximum_cells=4,
            minimum_cells=1,
        )
    assert generation.lengths.tolist() == [1, 1]
    assert generation.cell_mask.sum(dim=1).tolist() == [1.0, 1.0]


def test_v32_boundary_has_no_symbolic_or_candidate_path() -> None:
    config = _config()
    model = VisualSemanticRasterTransducer(config)
    receipt = visual_semantic_raster_boundary_receipt(model)
    assert receipt["primary_output"] == "generated answer raster"
    assert receipt["continuous_glyph_state_dimension"] == 12
    assert receipt["forbidden_parameter_names"] == []
    assert receipt["generation_is_autoregressive_raster_feedback"]
    for key in (
        "uses_strings",
        "uses_token_ids",
        "uses_unicode_ids",
        "uses_character_ids",
        "uses_vocabulary_logits",
        "uses_ocr",
        "uses_visual_codebook",
        "uses_external_language_model",
        "candidate_bank_deployed",
    ):
        assert receipt[key] is False


def test_v32_frozen_reader_stays_in_evaluation_mode() -> None:
    model = VisualSemanticRasterTransducer(_config())
    model.freeze_reader()
    model.train()
    assert not model.reader.training
    assert all(not parameter.requires_grad for parameter in model.reader.parameters())


def test_v32_partial_reader_training_only_activates_selected_blocks() -> None:
    config = VisualSemanticRasterConfig(
        **{
            **_config().__dict__,
            "reader_layers": 3,
        }
    )
    model = VisualSemanticRasterTransducer(config)
    model.unfreeze_reader_final_blocks(1)
    model.train()
    assert not model.reader.embeddings.training
    assert not model.reader.encoder.layer[0].training
    assert model.reader.encoder.layer[-1].training
    assert model.reader.layernorm.training


def test_v32_rejects_integer_pixels() -> None:
    config = _config()
    model = VisualSemanticRasterTransducer(config)
    batch = _batch(config)
    batch["prompt_pixels"] = batch["prompt_pixels"].long()
    with pytest.raises(TypeError, match="floating tensor"):
        model(**batch)

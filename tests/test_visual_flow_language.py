from __future__ import annotations

import torch

from ilm.visual_lm.canonical_glyph_language import CanonicalGlyphLanguageConfig
from ilm.visual_lm.visual_flow_language import (
    VISUAL_FLOW_LANGUAGE_ARCHITECTURE,
    VisualFlowLanguageModel,
    visual_flow_language_boundary_is_clean,
    visual_flow_language_boundary_receipt,
)
from ilm.visual_lm.visual_flow_language_evaluation import _pixel_f1
from ilm.visual_lm.visual_flow_language_training import visual_flow_language_loss


def _tiny_model() -> VisualFlowLanguageModel:
    return VisualFlowLanguageModel(
        CanonicalGlyphLanguageConfig(
            model_dim=128,
            layers=1,
            heads=4,
            mlp_ratio=2.0,
            dropout=0.0,
            noise_dim=16,
            generator_layers=1,
        )
    )


def test_visual_flow_uses_hidden_state_without_deterministic_ink_plan() -> None:
    torch.manual_seed(49)
    model = _tiny_model()
    context = torch.rand(2, 8, 1, 32, 32)
    hidden, anchor, plan = model.flow_inputs(context)
    assert hidden.shape == (2, 128)
    assert anchor.shape == (2, 1024)
    assert plan.shape == (2, 1, 32, 32)
    assert torch.count_nonzero(plan) == 0


def test_end_to_end_loss_trains_reader_and_writer() -> None:
    torch.manual_seed(50)
    model = _tiny_model()
    model.enable_end_to_end_training()
    context = torch.rand(1, 8, 1, 32, 32)
    target = (torch.rand(1, 8, 1, 32, 32) > 0.82).float()
    output = model.language_model(context)
    loss = visual_flow_language_loss(
        model,
        output,
        target,
        positions_per_stream=2,
        generator=torch.Generator().manual_seed(51),
    )
    assert torch.isfinite(loss.loss)
    loss.loss.backward()
    reader_gradient = model.language_model.blocks[0].attention.output.weight.grad
    writer_gradient = model.writer.output[-1].weight.grad
    assert reader_gradient is not None and torch.isfinite(reader_gradient).all()
    assert writer_gradient is not None and torch.isfinite(writer_gradient).all()
    assert not any(
        parameter.requires_grad
        for parameter in model.language_model.generator.parameters()
    )


def test_visual_flow_samples_binary_pixels_and_rereads_them() -> None:
    torch.manual_seed(52)
    model = _tiny_model().eval()
    context = torch.rand(1, 8, 1, 32, 32)
    pixels, trace = model.sample_next(
        context,
        generator=torch.Generator().manual_seed(53),
        samples=2,
        steps=1,
        guidance_scale=1.0,
    )
    assert pixels.shape == (1, 1, 32, 32)
    assert trace["candidate_pixels"].shape == (1, 2, 1, 32, 32)
    assert set(torch.unique(pixels).tolist()).issubset({0.0, 1.0})
    sequence, generated = model.generate(
        context,
        new_cells=1,
        generator=torch.Generator().manual_seed(54),
        samples=1,
        steps=1,
        guidance_scale=1.0,
    )
    assert sequence.shape == (1, 9, 1, 32, 32)
    assert generated["rereads_generated_pixels"].item() is True


def test_visual_flow_boundary_is_image_only() -> None:
    model = _tiny_model()
    receipt = visual_flow_language_boundary_receipt(model)
    assert receipt["architecture"] == VISUAL_FLOW_LANGUAGE_ARCHITECTURE
    assert receipt["end_to_end_reader_writer"] is True
    assert receipt["deterministic_ink_plan_used"] is False
    assert receipt["candidate_bank_deployed"] is False
    assert visual_flow_language_boundary_is_clean(model)


def test_visual_flow_pixel_f1_is_per_example() -> None:
    target = torch.zeros(2, 1, 4, 4)
    target[0, 0, :2, :2] = 1
    target[1, 0, 2:, 2:] = 1
    prediction = target.clone()
    prediction[1].zero_()
    measured = _pixel_f1(prediction, target)
    assert torch.allclose(measured, torch.tensor([1.0, 0.0]))

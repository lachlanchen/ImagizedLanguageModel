from __future__ import annotations

import torch

from ilm.visual_lm.glyph_content_form import (
    GlyphContentFormConfig,
    GlyphContentFormModel,
    glyph_content_form_boundary_receipt,
)
from ilm.visual_lm.glyph_content_form_evaluation import binary_glyph_metrics
from ilm.visual_lm.glyph_content_form_training import (
    WarmStartTrainableEMA,
    glyph_content_form_loss,
    supervised_contrastive_loss,
    symmetric_paired_contrastive_loss,
)


def _model() -> GlyphContentFormModel:
    return GlyphContentFormModel(
        GlyphContentFormConfig(
            content_width=32,
            form_width=8,
            encoder_width=64,
            form_encoder_width=32,
            synthesis_width=64,
            synthesis_depth=1,
        )
    )


def _batch(batch_size: int = 2) -> dict[str, torch.Tensor]:
    return {
        key: torch.randint(0, 2, (batch_size, 1, 32, 32)).float()
        for key in (
            "anchor_pixels",
            "positive_pixels",
            "anchor_style_pixels",
            "positive_style_pixels",
        )
    }


def test_v40_forward_is_image_only_and_decodes_raster_logits() -> None:
    model = _model()
    batch = _batch()
    output = model(**batch)
    rendered = model.render(output.anchor_content, output.positive_form)
    receipt = glyph_content_form_boundary_receipt(model)

    assert output.anchor_content.shape == (2, 32)
    assert output.anchor_form.shape == (2, 8)
    assert output.anchor_reference_surface.shape == (2, 768)
    assert rendered.shape == (2, 1, 32, 32)
    assert receipt["codec_trainable_parameters"] == 0
    assert receipt["uses_character_ids"] is False
    assert receipt["uses_unicode_ids"] is False
    assert receipt["uses_visual_codebook"] is False


def test_v40_loss_is_finite_and_keeps_codec_frozen() -> None:
    torch.manual_seed(40)
    model = _model().train()
    batch = _batch()
    output = model(**batch)
    stage_ids = torch.tensor([0, 1, 1, 0, 0, 1, 1, 0])
    loss = glyph_content_form_loss(
        model,
        output,
        batch,
        stage_ids=stage_ids,
    )

    loss.loss.backward()

    assert torch.isfinite(loss.loss)
    assert 0.0 <= float(loss.content_top1) <= 1.0
    assert 0.0 <= float(loss.form_top1) <= 1.0
    assert all(parameter.grad is None for parameter in model.codec.parameters())
    assert any(
        parameter.grad is not None
        for name, parameter in model.named_parameters()
        if not name.startswith("codec.")
    )


def test_v40_contrastive_objectives_reward_exact_pairing() -> None:
    states = torch.eye(4)
    paired, paired_top1 = symmetric_paired_contrastive_loss(states @ states.T * 10)
    labels = torch.tensor([0, 1, 0, 1])
    forms = torch.tensor(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [0.0, 1.0],
        ]
    )
    form, form_top1 = supervised_contrastive_loss(forms, labels)

    assert paired < 0.001
    assert paired_top1 == 1.0
    assert form < 0.001
    assert form_top1 == 1.0


def test_v40_binary_visual_metrics_reward_exact_ink() -> None:
    targets = torch.ones(2, 1, 32, 32)
    targets[:, :, 8:24, 15:17] = 0.0
    logits = torch.where(targets > 0.5, 10.0, -10.0)

    metrics = binary_glyph_metrics(logits, targets)

    assert metrics.pixel_accuracy == 1.0
    assert metrics.ink_iou == 1.0
    assert metrics.ink_f1 == 1.0


def test_v40_ema_warm_start_does_not_preserve_random_initialization() -> None:
    layer = torch.nn.Linear(2, 1, bias=False)
    torch.nn.init.zeros_(layer.weight)
    ema = WarmStartTrainableEMA(layer, decay=0.999)
    layer.weight.data.fill_(1.0)

    effective_decay = ema.update(layer)
    ema.copy_to(layer)

    assert effective_decay == 0.1
    assert torch.allclose(layer.weight, torch.full_like(layer.weight, 0.9))

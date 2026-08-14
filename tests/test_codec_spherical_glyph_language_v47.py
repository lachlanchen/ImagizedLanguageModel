from __future__ import annotations

from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from ilm.visual_lm.codec_spherical_glyph_language_v47 import (
    V47_REQUIRED_CODEC_CHECKPOINT_SHA256,
    V47_REQUIRED_CODEC_STATE_SHA256,
    CodecSphericalGlyphFieldV47,
    CodecSphericalGlyphLanguageModelV47,
    CodecSphericalGlyphLanguageV47Config,
    codec_spherical_glyph_language_v47_boundary_receipt,
    load_verified_v34_codec,
)
from ilm.visual_lm.codec_spherical_glyph_language_v47_evaluation import (
    codec_spherical_language_v47_boundary_is_clean,
    codec_spherical_language_v47_gate_report,
)
from ilm.visual_lm.codec_spherical_glyph_language_v47_training import (
    codec_spherical_glyph_language_v47_loss,
    codec_spherical_glyph_language_v47_pair_loss,
    dynamic_codec_spherical_contrastive_loss_v47,
    exact_raster_positive_mask_v47,
)
from ilm.visual_lm.continuous_glyph_codec import (
    ContinuousGlyphCodec,
    ContinuousGlyphCodecConfig,
)


ROOT = Path(__file__).resolve().parents[1]
V34_CHECKPOINT = (
    ROOT / "artifacts/continuous_glyph_codec_v34_20260814/checkpoint_latest.pt"
)


def _codec() -> ContinuousGlyphCodec:
    codec = ContinuousGlyphCodec(ContinuousGlyphCodecConfig())
    codec.requires_grad_(False).eval()
    return codec


def _model(
    config: CodecSphericalGlyphLanguageV47Config | None = None,
) -> CodecSphericalGlyphLanguageModelV47:
    if config is None:
        config = CodecSphericalGlyphLanguageV47Config(
            model_dim=128,
            layers=2,
            heads=4,
            generator_layers=2,
        )
    return CodecSphericalGlyphLanguageModelV47(
        config,
        _codec(),
        codec_checkpoint_sha256=V47_REQUIRED_CODEC_CHECKPOINT_SHA256,
        codec_state_sha256=V47_REQUIRED_CODEC_STATE_SHA256,
    )


def test_v47_verified_codec_loader_selects_the_frozen_ema() -> None:
    if not V34_CHECKPOINT.is_file():
        pytest.skip("local V34 checkpoint is not installed")
    codec, receipt = load_verified_v34_codec(V34_CHECKPOINT)
    assert receipt["checkpoint_sha256"] == V47_REQUIRED_CODEC_CHECKPOINT_SHA256
    assert receipt["ema_tensor_state_sha256"] == V47_REQUIRED_CODEC_STATE_SHA256
    assert receipt["selection"] == "ema-shadow"
    assert receipt["trainable_parameters"] == 0
    assert not any(parameter.requires_grad for parameter in codec.parameters())


def test_v47_field_is_unit_spherical_and_soft_cycle_is_differentiable() -> None:
    torch.manual_seed(47)
    field = CodecSphericalGlyphFieldV47(_codec())
    pixels = torch.rand(2, 3, 1, 32, 32)
    encoded = field.encode(pixels)
    assert encoded.shape == (2, 3, 768)
    torch.testing.assert_close(
        encoded.norm(dim=-1),
        torch.ones(2, 3),
        atol=2e-6,
        rtol=2e-6,
    )
    proposals = F.normalize(torch.randn(4, 2, 768), dim=-1).requires_grad_()
    reread = field.soft_reread(proposals)
    cycle = (1.0 - (proposals * reread).sum(dim=-1)).mean()
    cycle.backward()
    assert proposals.grad is not None
    assert torch.isfinite(proposals.grad).all()
    assert not any(parameter.grad is not None for parameter in field.parameters())
    visible, visible_reread = field.visible_reread(proposals.detach())
    assert visible.shape == (4, 2, 1, 32, 32)
    assert visible_reread.shape == proposals.shape
    assert set(torch.unique(visible).tolist()).issubset({0.0, 1.0})


def test_v47_model_is_causal_and_image_only() -> None:
    torch.manual_seed(20264700)
    model = _model().eval()
    context = torch.rand(2, 7, 1, 32, 32)
    changed = context.clone()
    changed[:, -1] = 1.0 - changed[:, -1]
    first = model.language(context)
    second = model.language(changed)
    assert first["hidden_states"].shape == (2, 7, 128)
    assert first["anchor_fields"].shape == (2, 7, 768)
    torch.testing.assert_close(
        first["anchor_fields"][:, :-1],
        second["anchor_fields"][:, :-1],
        atol=2e-5,
        rtol=2e-5,
    )
    assert not torch.allclose(
        first["anchor_fields"][:, -1],
        second["anchor_fields"][:, -1],
    )
    torch.testing.assert_close(
        first["anchor_fields"].norm(dim=-1),
        torch.ones(2, 7),
        atol=2e-6,
        rtol=2e-6,
    )
    receipt = codec_spherical_glyph_language_v47_boundary_receipt(model)
    assert receipt["field_trainable_parameters"] == 0
    assert receipt["parameter_names_with_forbidden_fragments"] == []
    assert receipt["selection_uses_visible_reread"] is True
    assert receipt["candidate_bank_deployed"] is False
    assert codec_spherical_language_v47_boundary_is_clean(model)


def test_v47_natural_and_pair_losses_reach_reader_generator_and_pixels() -> None:
    torch.manual_seed(11)
    model = _model()
    context = torch.rand(2, 4, 1, 32, 32)
    target = torch.rand(2, 4, 1, 32, 32)
    measured = codec_spherical_glyph_language_v47_loss(
        model,
        model(context),
        target,
        generator=torch.Generator().manual_seed(17),
        maximum_contrastive_positions=8,
        maximum_energy_positions=4,
        energy_samples=2,
    )
    pair_contexts = torch.rand(1, 2, 64, 1, 32, 32)
    pair_candidates = torch.rand(1, 2, 1, 32, 32)
    pair_assignment = torch.tensor([[1, 0]], dtype=torch.long)
    paired = codec_spherical_glyph_language_v47_pair_loss(
        model,
        pair_contexts,
        pair_candidates,
        pair_assignment,
    )
    combined = measured.loss + paired.loss
    assert torch.isfinite(combined)
    assert measured.contrastive_positions == 8
    assert measured.energy_positions == 4
    assert measured.cycle >= 0.0
    combined.backward()
    assert model.anchor_head[-1].weight.grad is not None
    assert model.generator.output.weight.grad is not None
    assert torch.isfinite(model.anchor_head[-1].weight.grad).all()
    assert torch.isfinite(model.generator.output.weight.grad).all()
    assert not any(parameter.grad is not None for parameter in model.field.parameters())


def test_v47_exact_visual_positives_and_contrastive_rows() -> None:
    pixels = torch.zeros(3, 1, 32, 32)
    pixels[0, :, 4:12, 5:9] = 1.0
    pixels[1] = pixels[0]
    pixels[2, :, 16:25, 17:21] = 1.0
    positives = exact_raster_positive_mask_v47(pixels)
    assert positives.tolist() == [
        [True, True, False],
        [True, True, False],
        [False, False, True],
    ]
    fields = F.normalize(torch.randn(3, 768), dim=-1)
    fields[1] = fields[0]
    contrastive, accuracy = dynamic_codec_spherical_contrastive_loss_v47(
        fields,
        fields,
        scale=torch.tensor(20.0),
        positive_mask=positives,
    )
    assert torch.isfinite(contrastive)
    assert accuracy == 1.0


def test_v47_generation_selects_only_after_visible_reread() -> None:
    torch.manual_seed(23)
    model = _model().eval()
    prefix = torch.rand(1, 5, 1, 32, 32)
    pixels, trace = model.sample_next(
        prefix,
        samples=2,
        generator=torch.Generator().manual_seed(29),
    )
    assert pixels.shape == (1, 1, 32, 32)
    assert trace["reread_fields"].shape == (1, 2, 768)
    assert trace["selected_proposal_fields"].shape == (1, 768)
    assert trace["selected_reread_fields"].shape == (1, 768)
    torch.testing.assert_close(
        trace["selected_reread_fields"],
        model.field.encode(pixels),
        atol=2e-6,
        rtol=2e-6,
    )
    sequence, generated = model.generate(
        prefix,
        new_cells=2,
        samples=2,
        generator=torch.Generator().manual_seed(31),
    )
    assert sequence.shape == (1, 7, 1, 32, 32)
    assert generated["generated_cells"].shape == (1, 2, 1, 32, 32)
    assert generated["generated_fields"].shape == (1, 2, 768)
    assert generated["rereads_generated_pixels"].item() is True


def test_v47_full_model_fits_frozen_parameter_budget_and_gates_conjoin() -> None:
    model = _model(CodecSphericalGlyphLanguageV47Config())
    receipt = codec_spherical_glyph_language_v47_boundary_receipt(model)
    assert receipt["total_parameters"] == 31_376_130
    assert receipt["trainable_parameters"] == 23_952_769
    assert receipt["total_parameters"] < 32_000_000
    assert receipt["trainable_parameters"] < 25_000_000

    preflight = {"pass": True}
    language = {
        "full_top1": 0.25,
        "unigram_top1": 0.01,
        "bigram_top1": 0.12,
        "full_target_log_probability": -5.0,
        "shuffled_target_log_probability": -5.2,
        "shuffled_top1": 0.20,
        "peak_allocated_vram_gib": 10.0,
    }
    pairs = {"full_arm_accuracy": 0.61}
    generated = {
        "generated_identity_top1": 0.10,
        "generated_pixel_f1": 0.60,
        "generated_blank_rate": 0.0,
        "mean_selected_proposal_to_visible_reread_cosine": 0.91,
    }
    gates = codec_spherical_language_v47_gate_report(
        preflight,
        language,
        pairs,
        generated,
        boundary_clean=True,
        protocol_integrity_clean=True,
        updates_complete=True,
        pair_rows_consumed=80_000,
        total_parameters=receipt["total_parameters"],
        trainable_parameters=receipt["trainable_parameters"],
        total_elapsed_seconds=1_000.0,
    )
    assert len(gates) == 16
    assert all(gates.values())
    generated["generated_pixel_f1"] = 0.54
    failed = codec_spherical_language_v47_gate_report(
        preflight,
        language,
        pairs,
        generated,
        boundary_clean=True,
        protocol_integrity_clean=True,
        updates_complete=True,
        pair_rows_consumed=80_000,
        total_parameters=receipt["total_parameters"],
        trainable_parameters=receipt["trainable_parameters"],
        total_elapsed_seconds=1_000.0,
    )
    assert failed["generated_pixel_f1"] is False
    assert not all(failed.values())

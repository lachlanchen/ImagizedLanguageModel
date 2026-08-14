from __future__ import annotations

import torch

from ilm.visual_lm.canonical_glyph_language import (
    CanonicalGlyphLanguageConfig,
    CanonicalGlyphLanguageModel,
)
from ilm.visual_lm.noise_limited_retinal_field_v45 import (
    NoiseLimitedRetinalFieldV45,
    NoiseLimitedRetinalFieldV45Config,
)
from ilm.visual_lm.scaled_retinal_glyph_language_v46 import (
    V46_REFERENCE_RADIUS,
    ScaledNoiseLimitedRetinalFieldV46,
    ScaledRetinalGlyphLanguageModelV46,
    ScaledRetinalGlyphLanguageV46Config,
    scaled_retinal_glyph_language_v46_boundary_receipt,
)
from ilm.visual_lm.scaled_retinal_glyph_language_v46_evaluation import (
    V46_REQUIRED_TRAINABLE_PARAMETERS,
    scaled_retinal_field_roundtrip_receipt,
    scaled_retinal_language_v46_boundary_is_clean,
    scaled_retinal_language_v46_gate_report,
)
from ilm.visual_lm.scaled_retinal_glyph_language_v46_training import (
    dynamic_scaled_retinal_contrastive_loss,
    exact_raster_positive_mask,
    scaled_retinal_glyph_language_v46_loss,
)


def _retinal_field() -> NoiseLimitedRetinalFieldV45:
    dimension = 1024
    return NoiseLimitedRetinalFieldV45(
        NoiseLimitedRetinalFieldV45Config(),
        mean_dct=torch.linspace(-0.2, 0.2, dimension, dtype=torch.float64),
        eigenvectors=torch.eye(dimension, dtype=torch.float64),
        eigenvalues=torch.linspace(0.5, 1.5, dimension, dtype=torch.float64),
    )


def _small_config() -> ScaledRetinalGlyphLanguageV46Config:
    return ScaledRetinalGlyphLanguageV46Config(
        model_dim=128,
        layers=2,
        heads=4,
        mlp_ratio=2.0,
        dropout=0.0,
        noise_dim=32,
        generator_layers=2,
    )


def _model(
    config: ScaledRetinalGlyphLanguageV46Config | None = None,
) -> ScaledRetinalGlyphLanguageModelV46:
    return ScaledRetinalGlyphLanguageModelV46(
        config or _small_config(),
        _retinal_field(),
        v45_checkpoint_sha256="0" * 64,
    )


def _pixels(count: int = 4) -> torch.Tensor:
    pixels = torch.zeros(count, 1, 32, 32)
    for index in range(count):
        pixels[index, 0, 3 + index : 18 + index, 4:7] = 1.0
        pixels[index, 0, 14:17, 5 + index : 24 + index] = 1.0
    return pixels


def test_v46_scaled_field_is_exact_and_preserves_radius() -> None:
    pixels = _pixels()
    field = ScaledNoiseLimitedRetinalFieldV46(_retinal_field())
    encoded = field.encode(pixels, exact=True)
    decoded = field.decode_dct(encoded, exact=True)
    source = field.retinal.dct.encode(pixels).double()
    torch.testing.assert_close(decoded, source, atol=2e-8, rtol=0.0)
    assert torch.equal(field.binary(encoded, exact=True), pixels)
    assert bool((encoded.norm(dim=-1) > 0.0).all())
    assert float(field.reference_radius) == V46_REFERENCE_RADIUS

    model = _model()
    receipt = scaled_retinal_field_roundtrip_receipt(model, [pixels])
    assert receipt["all_finite"] is True
    assert receipt["maximum_dct_absolute_error"] < 2e-8
    assert receipt["binary_pixel_accuracy"] == 1.0


def test_v46_model_is_causal_and_boundary_is_image_only() -> None:
    torch.manual_seed(20264200)
    model = _model().eval()
    context = torch.rand(2, 7, 1, 32, 32)
    changed = context.clone()
    changed[:, -1] = 1.0 - changed[:, -1]
    first = model.language(context)
    second = model.language(changed)
    assert first["hidden_states"].shape == (2, 7, 128)
    assert first["anchor_fields"].shape == (2, 7, 1024)
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
    norms = first["anchor_fields"].norm(dim=-1)
    assert bool(torch.isfinite(norms).all())
    assert 0.1 < float(norms.detach().mean()) < 1.5

    receipt = scaled_retinal_glyph_language_v46_boundary_receipt(model)
    assert receipt["field_trainable_parameters"] == 0
    assert receipt["parameter_names_with_forbidden_fragments"] == []
    assert receipt["field_preserves_direction_and_radius"] is True
    assert receipt["candidate_bank_deployed"] is False
    assert scaled_retinal_language_v46_boundary_is_clean(model)


def test_v46_exact_positives_and_full_loss_are_differentiable() -> None:
    torch.manual_seed(11)
    model = _model()
    context = torch.rand(2, 4, 1, 32, 32)
    target = torch.rand(2, 4, 1, 32, 32)
    output = model(context)
    measured = scaled_retinal_glyph_language_v46_loss(
        model,
        output,
        target,
        generator=torch.Generator().manual_seed(17),
        maximum_contrastive_positions=8,
        maximum_energy_positions=4,
        energy_samples=2,
    )
    assert torch.isfinite(measured.loss)
    assert measured.contrastive_positions == 8
    assert measured.energy_positions == 4
    assert measured.anchor_radius_mae >= 0.0
    measured.loss.backward()
    assert model.anchor_head[-1].weight.grad is not None
    assert model.generator.output.weight.grad is not None
    assert torch.isfinite(model.anchor_head[-1].weight.grad).all()
    assert torch.isfinite(model.generator.output.weight.grad).all()

    pixels = _pixels(3)
    pixels[1] = pixels[0]
    fields = model.field.encode(pixels)
    positives = exact_raster_positive_mask(pixels)
    assert positives.tolist() == [
        [True, True, False],
        [True, True, False],
        [False, False, True],
    ]
    contrastive, accuracy = dynamic_scaled_retinal_contrastive_loss(
        fields,
        fields,
        scale=torch.tensor(20.0),
        positive_mask=positives,
    )
    assert torch.isfinite(contrastive)
    assert accuracy == 1.0


def test_v46_bank_free_generation_rereads_binary_rasters() -> None:
    torch.manual_seed(23)
    model = _model().eval()
    prefix = torch.rand(1, 5, 1, 32, 32)
    sequence, trace = model.generate(
        prefix,
        new_cells=3,
        samples=2,
        generator=torch.Generator().manual_seed(29),
    )
    assert sequence.shape == (1, 8, 1, 32, 32)
    assert trace["generated_cells"].shape == (1, 3, 1, 32, 32)
    assert trace["generated_fields"].shape == (1, 3, 1024)
    assert trace["rereads_generated_pixels"].item() is True
    assert set(torch.unique(trace["generated_cells"]).tolist()).issubset({0.0, 1.0})


def test_v46_matches_v42_parameter_count_and_gate_report_is_conjunctive() -> None:
    v42 = CanonicalGlyphLanguageModel(CanonicalGlyphLanguageConfig())
    v42_parameters = sum(parameter.numel() for parameter in v42.parameters())
    del v42
    model = _model(ScaledRetinalGlyphLanguageV46Config())
    v46_parameters = sum(parameter.numel() for parameter in model.parameters())
    assert v42_parameters == V46_REQUIRED_TRAINABLE_PARAMETERS
    assert v46_parameters == v42_parameters

    language = {
        "full_top1": 0.25,
        "unigram_top1": 0.01,
        "bigram_top1": 0.12,
        "full_target_log_probability": -5.0,
        "shuffled_target_log_probability": -5.2,
        "shuffled_top1": 0.20,
        "peak_allocated_vram_gib": 1.0,
    }
    pairs = {"full_arm_accuracy": 0.61}
    generated = {
        "generated_identity_top1": 0.10,
        "generated_pixel_f1": 0.60,
        "generated_blank_rate": 0.0,
    }
    gates = scaled_retinal_language_v46_gate_report(
        language,
        pairs,
        generated,
        boundary_clean=True,
        protocol_integrity_clean=True,
    )
    assert len(gates) == 14
    assert all(gates.values())
    generated["generated_identity_top1"] = 0.09
    failed = scaled_retinal_language_v46_gate_report(
        language,
        pairs,
        generated,
        boundary_clean=True,
        protocol_integrity_clean=True,
    )
    assert failed["generated_identity_beats_v42_by_0_01"] is False
    assert not all(failed.values())

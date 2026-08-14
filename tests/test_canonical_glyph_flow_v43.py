from __future__ import annotations

import torch

from ilm.visual_lm.canonical_glyph_flow_v43 import (
    CanonicalGlyphFlowV43,
    canonical_glyph_flow_v43_boundary_receipt,
)
from ilm.visual_lm.canonical_glyph_flow_v43_data import (
    CanonicalGlyphPairTrainingDataset,
    canonical_glyph_pair_student_batch,
    canonical_glyph_pair_training_collate,
)
from ilm.visual_lm.canonical_glyph_flow_v43_evaluation import (
    canonical_glyph_flow_v43_boundary_is_clean,
)
from ilm.visual_lm.canonical_glyph_flow_v43_training import (
    canonical_glyph_flow_v43_language_loss,
    canonical_glyph_flow_v43_writer_loss,
)
from ilm.visual_lm.canonical_glyph_language import CanonicalGlyphLanguageConfig
from ilm.visual_lm.canonical_glyph_language_data import CanonicalGlyphRenderConfig
from ilm.visual_lm.factorized_visual_context_data import (
    FactorizedVisualSuffixPair,
)


def _tiny_model() -> CanonicalGlyphFlowV43:
    return CanonicalGlyphFlowV43(
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


def _pair() -> FactorizedVisualSuffixPair:
    suffix = "天地人心"
    return FactorizedVisualSuffixPair(
        suffix_cells=4,
        identifier_a="train-a",
        script_view_a="original",
        context_a="中" * 60 + suffix,
        target_a="文",
        identifier_b="train-b",
        script_view_b="simplified",
        context_b="国" * 60 + suffix,
        target_b="学",
    )


def test_pair_dataset_hides_metadata_and_permutes_candidates() -> None:
    pair = _pair()
    dataset = CanonicalGlyphPairTrainingDataset(
        [pair],
        render_config=CanonicalGlyphRenderConfig(),
        seed=20264302,
        length=2,
    )
    raw = canonical_glyph_pair_training_collate([dataset[0], dataset[1]])
    student = canonical_glyph_pair_student_batch(raw)
    assert set(student) == {"contexts", "candidates", "assignment"}
    assert student["contexts"].shape == (2, 2, 64, 1, 32, 32)
    assert student["candidates"].shape == (2, 2, 1, 32, 32)
    assert torch.equal(
        student["contexts"][:, 0, -4:],
        student["contexts"][:, 1, -4:],
    )
    assert all(sorted(row.tolist()) == [0, 1] for row in student["assignment"])
    expected_a = dataset._render(pair.context_a + pair.target_a)[64]
    expected_b = dataset._render(pair.context_b + pair.target_b)[64]
    for item in (dataset[0], dataset[1]):
        assignment = item["assignment"]
        assert torch.equal(item["candidates"][assignment[0]], expected_a)
        assert torch.equal(item["candidates"][assignment[1]], expected_b)


def test_v43_stage_freezing_keeps_only_the_intended_module_trainable() -> None:
    model = _tiny_model()
    model.freeze_writer()
    model.unfreeze_language_core()
    assert not any(parameter.requires_grad for parameter in model.writer.parameters())
    assert not any(
        parameter.requires_grad
        for parameter in model.language_model.generator.parameters()
    )
    assert any(
        parameter.requires_grad
        for name, parameter in model.language_model.named_parameters()
        if not name.startswith("generator.")
    )

    model.freeze_language()
    model.unfreeze_writer()
    assert not any(
        parameter.requires_grad for parameter in model.language_model.parameters()
    )
    assert all(parameter.requires_grad for parameter in model.writer.parameters())


def test_language_pair_loss_is_finite_and_trains_reader() -> None:
    torch.manual_seed(43)
    model = _tiny_model()
    natural_context = torch.rand(1, 64, 1, 32, 32)
    natural_target = torch.rand(1, 64, 1, 32, 32)
    pair_contexts = torch.rand(1, 2, 64, 1, 32, 32)
    pair_candidates = torch.rand(1, 2, 1, 32, 32)
    assignment = torch.tensor([[1, 0]], dtype=torch.long)
    output = model.language_model(natural_context)
    loss = canonical_glyph_flow_v43_language_loss(
        model,
        output,
        natural_target,
        pair_contexts,
        pair_candidates,
        assignment,
    )
    assert torch.isfinite(loss.loss)
    assert 0.0 <= float(loss.pair_arm_accuracy) <= 1.0
    loss.loss.backward()
    assert model.language_model.anchor_head[-1].weight.grad is not None


def test_writer_loss_is_finite_and_trains_spatial_flow() -> None:
    torch.manual_seed(44)
    model = _tiny_model()
    hidden = torch.randn(4, model.language_model.config.model_dim)
    anchors = torch.nn.functional.normalize(torch.randn(4, 1024), dim=-1)
    targets = (torch.rand(4, 1, 32, 32) > 0.8).float()
    generator = torch.Generator().manual_seed(45)
    loss = canonical_glyph_flow_v43_writer_loss(
        model,
        hidden,
        anchors,
        targets,
        generator=generator,
    )
    assert torch.isfinite(loss.loss)
    loss.loss.backward()
    assert model.writer.output[-1].weight.grad is not None


def test_bank_free_writer_samples_and_rereads_pixels() -> None:
    torch.manual_seed(46)
    model = _tiny_model().eval()
    context = torch.rand(1, 64, 1, 32, 32)
    generator = torch.Generator().manual_seed(47)
    pixels, trace = model.sample_next(
        context,
        generator=generator,
        samples=2,
        steps=1,
        guidance_scale=1.0,
    )
    assert pixels.shape == (1, 1, 32, 32)
    assert trace["candidate_pixels"].shape == (1, 2, 1, 32, 32)
    assert trace["candidate_fields"].shape == (1, 2, 1024)
    assert torch.equal(pixels, (pixels >= 0.5).to(pixels.dtype))


def test_v43_boundary_has_no_symbolic_runtime_path() -> None:
    model = _tiny_model()
    receipt = canonical_glyph_flow_v43_boundary_receipt(model)
    assert receipt["output_is_direct_raster"] is True
    assert receipt["conditional_spatial_flow"] is True
    assert receipt["candidate_bank_deployed"] is False
    assert receipt["uses_token_ids"] is False
    assert canonical_glyph_flow_v43_boundary_is_clean(model)

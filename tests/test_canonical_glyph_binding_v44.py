from __future__ import annotations

import torch

from ilm.visual_lm.canonical_glyph_binding_v44 import (
    CanonicalGlyphBindingV44,
    canonical_glyph_binding_v44_boundary_receipt,
)
from ilm.visual_lm.canonical_glyph_binding_v44_evaluation import (
    canonical_glyph_binding_v44_boundary_is_clean,
)
from ilm.visual_lm.canonical_glyph_binding_v44_training import (
    canonical_glyph_binding_v44_loss,
    shuffle_v44_pair_prefixes,
)
from ilm.visual_lm.canonical_glyph_language import CanonicalGlyphLanguageConfig


def _tiny_model() -> CanonicalGlyphBindingV44:
    return CanonicalGlyphBindingV44(
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


def _paired_contexts() -> torch.Tensor:
    first = (torch.rand(60, 1, 32, 32) > 0.75).float()
    second = (torch.rand(60, 1, 32, 32) > 0.75).float()
    suffix = (torch.rand(4, 1, 32, 32) > 0.75).float()
    return torch.stack((torch.cat((first, suffix)), torch.cat((second, suffix))))[
        None
    ]


def test_zero_initialized_v44_is_exactly_the_frozen_v42_reader() -> None:
    torch.manual_seed(440)
    model = _tiny_model().eval()
    context = (torch.rand(2, 64, 1, 32, 32) > 0.8).float()
    base = model.base.language(context)
    output = model.language(context)
    assert torch.allclose(
        output["anchor_fields"],
        base["anchor_fields"],
        atol=1e-6,
        rtol=1e-6,
    )
    assert torch.count_nonzero(output["residual_fields"]) == 0


def test_short_suffix_controls_bypass_the_v44_residual() -> None:
    torch.manual_seed(441)
    model = _tiny_model().eval()
    for length in (1, 4):
        context = (torch.rand(2, length, 1, 32, 32) > 0.8).float()
        output = model.language(context)
        base = model.base.language(context)
        assert torch.equal(output["anchor_fields"], base["anchor_fields"])
        assert torch.count_nonzero(output["residual_fields"]) == 0


def test_v44_residual_is_tangent_to_the_base_image_field() -> None:
    torch.manual_seed(442)
    model = _tiny_model().eval()
    with torch.no_grad():
        model.adapter.output.weight.normal_(std=0.01)
    context = (torch.rand(2, 64, 1, 32, 32) > 0.8).float()
    output = model.language(context)
    tangent = output["residual_fields"][:, -1]
    base = output["base_anchor_fields"][:, -1]
    orthogonality = (tangent.float() * base.float()).sum(dim=-1).abs()
    assert float(orthogonality.max().detach()) < 1e-5
    assert bool((tangent.norm(dim=-1) > 0).all())


def test_v44_prefix_shuffle_preserves_images_and_shared_suffix() -> None:
    context = torch.zeros(1, 2, 64, 1, 32, 32)
    for arm in range(2):
        for position in range(60):
            context[0, arm, position].fill_((position + 1 + arm) / 128.0)
    context[:, :, -4:].fill_(0.75)
    generator = torch.Generator().manual_seed(443)
    shuffled = shuffle_v44_pair_prefixes(
        context,
        suffix_cells=4,
        generator=generator,
    )
    assert torch.equal(shuffled[:, :, -4:], context[:, :, -4:])
    assert not torch.equal(shuffled[:, :, :60], context[:, :, :60])
    original_values = context[:, :, :60, 0, 0, 0].sort(dim=2).values
    shuffled_values = shuffled[:, :, :60, 0, 0, 0].sort(dim=2).values
    assert torch.equal(original_values, shuffled_values)


def test_v44_loss_trains_only_the_residual_adapter() -> None:
    torch.manual_seed(444)
    model = _tiny_model().train()
    natural_context = (torch.rand(2, 64, 1, 32, 32) > 0.8).float()
    natural_target = (torch.rand(2, 1, 32, 32) > 0.8).float()
    pair_contexts = _paired_contexts()
    pair_candidates = (torch.rand(1, 2, 1, 32, 32) > 0.8).float()
    assignment = torch.tensor([[1, 0]], dtype=torch.long)
    generator = torch.Generator().manual_seed(445)
    shuffled = shuffle_v44_pair_prefixes(
        pair_contexts,
        suffix_cells=4,
        generator=generator,
    )
    natural_output = model.language(natural_context)
    loss = canonical_glyph_binding_v44_loss(
        model,
        natural_output,
        natural_target,
        pair_contexts,
        pair_candidates,
        assignment,
        shuffled,
    )
    assert torch.isfinite(loss.loss)
    assert 0.0 <= float(loss.pair_arm_accuracy) <= 1.0
    loss.loss.backward()
    assert model.adapter.output.weight.grad is not None
    assert any(
        parameter.grad is not None for parameter in model.adapter.parameters()
    )
    assert all(parameter.grad is None for parameter in model.base.parameters())
    assert not any(parameter.requires_grad for parameter in model.base.parameters())


def test_v44_boundary_is_image_only_and_below_parameter_limit() -> None:
    model = _tiny_model()
    receipt = canonical_glyph_binding_v44_boundary_receipt(model)
    assert receipt["adapter_parameters"] < 2_000_000
    assert receipt["adapter_parameters"] == receipt["trainable_parameters"]
    assert receipt["base_parameters_frozen"] is True
    assert receipt["candidate_independent_residual"] is True
    assert receipt["uses_token_ids"] is False
    assert receipt["candidate_bank_deployed"] is False
    assert canonical_glyph_binding_v44_boundary_is_clean(model)

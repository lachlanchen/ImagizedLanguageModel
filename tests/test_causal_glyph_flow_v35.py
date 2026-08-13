from __future__ import annotations

import torch

from ilm.visual_lm.causal_glyph_flow import (
    CausalGlyphFlowConfig,
    CausalGlyphFlowLM,
    causal_glyph_flow_boundary_receipt,
)


def tiny_config() -> CausalGlyphFlowConfig:
    return CausalGlyphFlowConfig(
        maximum_patches=12,
        hidden_size=64,
        layers=2,
        attention_heads=4,
        key_value_heads=2,
        intermediate_size=128,
        flow_width=64,
        flow_depth=2,
        codec_channels=(8, 16, 24, 32),
        codec_group_norm_groups=8,
    )


def sample_inputs(length: int = 5) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(35)
    pixels = (torch.rand(2, 1, 32, length * 32, generator=generator) > 0.82).float()
    mask = torch.ones(2, length)
    return pixels, mask


def test_forward_is_continuous_and_causal() -> None:
    torch.manual_seed(35)
    model = CausalGlyphFlowLM(tiny_config()).eval()
    pixels, mask = sample_inputs()
    output = model(pixels, mask)

    assert output.latents.shape == (2, 5, 768)
    assert output.input_states.shape == (2, 5, 64)
    assert output.hidden_states.shape == (2, 5, 64)
    assert output.anchor_latents.shape == (2, 5, 768)
    assert output.stop_logits.shape == (2, 5)
    assert torch.isfinite(output.anchor_latents).all()
    assert torch.allclose(
        output.anchor_latents.float().mean(dim=-1),
        torch.zeros(2, 5),
        atol=2e-5,
    )

    changed = pixels.clone()
    changed[..., -32:] = 1.0 - changed[..., -32:]
    changed_output = model(changed, mask)
    assert torch.allclose(
        output.hidden_states[:, :-1],
        changed_output.hidden_states[:, :-1],
        atol=2e-5,
        rtol=2e-5,
    )


def test_flow_sampling_and_visible_feedback_are_finite() -> None:
    torch.manual_seed(36)
    model = CausalGlyphFlowLM(tiny_config()).eval()
    conditions = torch.randn(3, 64)
    noise = torch.randn(3, 768)
    first = model.sample_flow(conditions, noise.clone(), steps=3)
    second = model.sample_flow(conditions, noise.clone(), steps=3)
    assert first.shape == (3, 768)
    assert torch.equal(first, second)
    assert torch.isfinite(first).all()

    patches, feedback = model.visible_feedback(first)
    assert patches.shape == (3, 1, 32, 32)
    assert feedback.shape == (3, 768)
    assert set(patches.unique().tolist()).issubset({0.0, 1.0})
    expected = model.codec.encode(patches)
    assert torch.equal(feedback, expected)


def test_generation_uses_closed_visible_loop() -> None:
    torch.manual_seed(37)
    model = CausalGlyphFlowLM(tiny_config()).eval()
    pixels, mask = sample_inputs(length=3)
    with torch.no_grad():
        model.stop_head.bias.fill_(10.0)
    generation = model.generate(
        pixels,
        mask,
        maximum_new_patches=2,
        minimum_new_patches=1,
        seed=37,
    )
    assert generation.patches.shape == (2, 2, 1, 32, 32)
    assert generation.feedback_latents.shape == (2, 2, 768)
    assert generation.lengths.tolist() == [1, 1]
    assert generation.patch_mask[:, 0].eq(1).all()
    assert generation.patch_mask[:, 1].eq(0).all()
    assert generation.strips().shape == (2, 1, 32, 64)
    expected = model.codec.encode(generation.patches[:, 0])
    assert torch.equal(generation.feedback_latents[:, 0], expected)


def test_generation_records_bfloat16_stop_probabilities_in_float_buffer() -> None:
    torch.manual_seed(38)
    model = CausalGlyphFlowLM(tiny_config()).eval()
    pixels, mask = sample_inputs(length=3)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        generation = model.generate(
            pixels,
            mask,
            maximum_new_patches=1,
            minimum_new_patches=1,
            seed=38,
        )
    assert generation.stop_probabilities.dtype == torch.float32
    assert torch.isfinite(generation.stop_probabilities).all()


def test_boundary_has_no_symbolic_runtime_interface() -> None:
    model = CausalGlyphFlowLM(tiny_config())
    receipt = causal_glyph_flow_boundary_receipt(model)
    assert receipt["forward_parameters"] == ["pixels", "patch_mask"]
    assert receipt["generate_parameters"][:2] == ["pixels", "patch_mask"]
    assert receipt["parameter_names_with_forbidden_fragments"] == []
    assert receipt["codec_trainable_parameters"] == 0
    assert receipt["uses_token_ids"] is False
    assert receipt["uses_unicode_ids"] is False
    assert receipt["uses_visual_codebook"] is False
    assert receipt["feedback_boundary"] == "decode-threshold-reencode-visible-raster"


def test_production_parameter_count_is_below_protocol_limit() -> None:
    model = CausalGlyphFlowLM(CausalGlyphFlowConfig())
    receipt = causal_glyph_flow_boundary_receipt(model)
    assert receipt["production_shape"] is True
    assert receipt["total_parameters"] < 150_000_000
    assert receipt["codec_parameters"] == 7_423_361

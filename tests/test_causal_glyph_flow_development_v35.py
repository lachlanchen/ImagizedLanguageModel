from __future__ import annotations

import torch

from ilm.visual_lm.causal_glyph_flow import CausalGlyphFlowConfig, CausalGlyphFlowLM
from ilm.visual_lm.causal_glyph_flow_development import (
    V35GeneratedCase,
    V35RasterCase,
    autonomous_case_audit,
    controlled_prompt,
    ocr_character_accuracy,
    output_pair_metrics,
    raster_pair_metrics,
    select_v35_writer,
    text_is_readable,
    v35_development_gate,
)


def _case(identifier: str, value: float = 1.0) -> V35RasterCase:
    prompt = torch.full((1, 32, 4 * 32), value)
    target = torch.ones(2, 1, 32, 32)
    target[:, :, 8:24, 14:18] = 0.0
    return V35RasterCase(
        identifier=identifier,
        stream="copy",
        expected="中",
        prompt_pixels=prompt,
        prompt_mask=torch.ones(4),
        target_patches=target,
        metadata={},
    )


def _tiny_model() -> CausalGlyphFlowLM:
    return CausalGlyphFlowLM(
        CausalGlyphFlowConfig(
            maximum_patches=12,
            hidden_size=64,
            layers=1,
            attention_heads=4,
            key_value_heads=2,
            intermediate_size=128,
            flow_width=64,
            flow_depth=1,
            codec_channels=(8, 16, 24, 32),
            codec_group_norm_groups=8,
        )
    )


def test_text_and_raster_metrics_are_normalized() -> None:
    assert ocr_character_accuracy("天地", "天地") == 1.0
    assert ocr_character_accuracy("天地", "天") == 0.5
    assert text_is_readable("，。") is False
    assert text_is_readable("文") is True

    target = _case("one").target_patches
    exact = raster_pair_metrics(target, 2, target, 2)
    assert exact == {
        "ink_f1": 1.0,
        "edge_f1": 1.0,
        "pixel_disagreement": 0.0,
    }
    blank = raster_pair_metrics(torch.ones_like(target), 2, target, 2)
    assert blank["ink_f1"] == 0.0
    assert blank["pixel_disagreement"] > 0.0


def test_prompt_controls_preserve_visual_length() -> None:
    cases = [_case("one", 0.0), _case("two", 0.5)]
    correct, mask = controlled_prompt(cases, 0, "correct")
    shuffled, shuffled_mask = controlled_prompt(cases, 0, "shuffled")
    blank, blank_mask = controlled_prompt(cases, 0, "blank")
    partial, partial_mask = controlled_prompt(cases, 0, "final-quarter")
    assert correct.shape == shuffled.shape == blank.shape == partial.shape
    assert torch.equal(mask, shuffled_mask)
    assert torch.equal(mask, blank_mask)
    assert torch.equal(mask, partial_mask)
    assert torch.equal(shuffled, cases[1].prompt_pixels)
    assert bool(blank.eq(1).all())
    assert bool(partial[..., : 3 * 32].eq(1).all())
    assert bool(partial[..., -32:].eq(0).all())


def test_tiny_closed_raster_audit_runs_without_text_inside_model() -> None:
    model = _tiny_model().eval()
    report = autonomous_case_audit(
        model,
        [_case("one"), _case("two", 0.5)],
        writer="anchor",
        conditions=("correct", "blank"),
        device=torch.device("cpu"),
        precision="fp32",
        ocr=lambda _image: "",
    )
    assert report["writer"] == "anchor"
    assert report["conditions"]["correct"]["examples"] == 2
    assert report["conditions"]["correct"]["finite"] is True
    assert "mean_pixel_disagreement" in report["control_comparisons"]["blank"]


def _generated(value: float) -> V35GeneratedCase:
    patches = torch.full((2, 1, 32, 32), value)
    return V35GeneratedCase(
        identifier="example",
        stream="copy",
        expected="中",
        observed="中",
        condition="correct",
        writer="anchor",
        patches=patches,
        feedback_latents=torch.ones(2, 768) * value,
        length=2,
        stop_probabilities=torch.zeros(3),
        character_accuracy=1.0,
        exact_match=True,
        readable=True,
        nonblank=value < 0.5,
        target_ink_f1=1.0,
        target_edge_f1=1.0,
        target_pixel_disagreement=0.0,
    )


def test_output_pair_and_writer_selection_follow_frozen_rule() -> None:
    pair = output_pair_metrics(_generated(0.0), _generated(1.0))
    assert pair["pixel_disagreement"] == 1.0

    anchor = {
        "conditions": {
            "correct": {"ocr_character_accuracy": 0.40, "readable_rate": 0.80}
        }
    }
    flow = {
        "conditions": {
            "correct": {"ocr_character_accuracy": 0.43, "readable_rate": 0.78}
        }
    }
    assert select_v35_writer(anchor, flow)["selected"] == "flow"
    flow["conditions"]["correct"]["readable_rate"] = 0.77
    assert select_v35_writer(anchor, flow)["selected"] == "anchor"


def _qualified_report() -> dict:
    def conditions(correct: float, shuffled: float, blank: float) -> dict:
        return {
            "correct": {
                "finite": True,
                "ocr_character_accuracy": correct,
                "readable_rate": 0.8,
                "nonblank_rate": 0.9,
            },
            "shuffled": {
                "finite": True,
                "ocr_character_accuracy": shuffled,
                "readable_rate": 0.5,
                "nonblank_rate": 0.8,
            },
            "blank": {
                "finite": True,
                "ocr_character_accuracy": blank,
                "readable_rate": 0.4,
                "nonblank_rate": 0.7,
            },
        }

    copy = {
        "conditions": conditions(0.60, 0.30, 0.20),
        "target_ocr": {"codec_character_accuracy": 0.80},
        "control_comparisons": {
            "shuffled": {"mean_pixel_disagreement": 0.02},
            "blank": {"mean_pixel_disagreement": 0.03},
        },
        "rows": {"correct": [], "blank": []},
    }
    public = {
        "conditions": conditions(0.50, 0.20, 0.10),
        "target_ocr": {"codec_character_accuracy": 0.80},
        "control_comparisons": {
            "shuffled": {"mean_pixel_disagreement": 0.02},
            "blank": {"mean_pixel_disagreement": 0.03},
        },
        "rows": {"correct": [], "blank": []},
    }
    instruction = {
        "conditions": conditions(0.20, 0.10, 0.05),
        "target_ocr": {"codec_character_accuracy": 0.70},
        "control_comparisons": {
            "shuffled": {"mean_pixel_disagreement": 0.02},
            "blank": {"mean_pixel_disagreement": 0.03},
        },
        "rows": {"correct": [], "blank": []},
    }
    paraphrase = {
        "conditions": conditions(0.20, 0.10, 0.05),
        "target_ocr": {"codec_character_accuracy": 0.70},
        "control_comparisons": {
            "shuffled": {"mean_pixel_disagreement": 0.02},
            "blank": {"mean_pixel_disagreement": 0.03},
        },
        "rows": {
            "correct": [{"readable": True, "character_accuracy": 0.2}],
            "blank": [{"readable": False, "character_accuracy": 0.0}],
        },
    }
    autonomous = {
        "copy": {"anchor": copy},
        "public": {"anchor": public},
        "instruction": {"anchor": instruction},
        "paraphrase": {"anchor": paraphrase},
        "copy_counterfactual": {
            "anchor": {"target_preference_rate": 0.8}
        },
    }
    return {
        "checkpoint_audit": {
            "passed": True,
            "checks": {"stage_a_passed": True},
            "peak_allocated_vram_bytes": 3 * 1024**3,
        },
        "closed_loop_receipt": {"passed": True},
        "states": {
            "ema": {
                "writer_selection": {"selected": "anchor"},
                "teacher_forced": {
                    "public": {"decoded_ink_f1": 0.8, "decoded_edge_f1": 0.8}
                },
                "autonomous": autonomous,
            }
        },
    }


def test_development_gate_uses_exact_status_vocabulary() -> None:
    report = _qualified_report()
    gate = v35_development_gate(report)
    assert gate["status"] == "semantic-raster-qualified"
    assert gate["visual_causal"]["passed"] is True
    assert gate["semantic_raster"]["passed"] is True

    report["states"]["ema"]["autonomous"]["copy"]["anchor"]["conditions"][
        "correct"
    ]["ocr_character_accuracy"] = 0.10
    gate = v35_development_gate(report)
    assert gate["status"] == "not-qualified"
    assert gate["visual_causal"]["passed"] is False

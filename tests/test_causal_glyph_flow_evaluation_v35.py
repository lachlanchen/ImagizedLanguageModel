from __future__ import annotations

import torch
import torch.nn as nn

from ilm.visual_lm.causal_glyph_flow import (
    CausalGlyphFlowConfig,
    CausalGlyphFlowLM,
    causal_glyph_flow_boundary_receipt,
)
from ilm.visual_lm.causal_glyph_flow_evaluation import (
    evaluate_visual_interface_alignment,
    v35_stage_a_gate,
)
from ilm.visual_lm.direct_visual_patch_data import (
    DirectPatchContinuationDataset,
    DirectPatchRenderConfig,
    direct_patch_partition,
)
from ilm.visual_lm.visual_semantic_raster_data import VisualTextRecord


def _development_record() -> VisualTextRecord:
    for index in range(10_000):
        identifier = f"v35-alignment:{index}"
        if direct_patch_partition(identifier, stream="public-domain") == "development":
            return VisualTextRecord(
                identifier=identifier,
                text="天地玄黄宇宙洪荒日月盈昃辰宿列张寒来暑往",
                language="zh",
                source="unit",
                rights="public domain",
            )
    raise AssertionError("could not construct development record")


def _model() -> CausalGlyphFlowLM:
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


def test_alignment_evaluator_is_finite_and_writes_gallery(tmp_path) -> None:
    dataset = DirectPatchContinuationDataset(
        [_development_record()],
        split="development",
        config=DirectPatchRenderConfig(
            maximum_patches=12,
            maximum_prompt_patches=8,
            maximum_answer_patches=4,
            augment=False,
        ),
        variants_per_record=3,
        seed=35,
    )
    model = _model()
    teacher = nn.Conv2d(1, 64, kernel_size=32, stride=32, bias=False)
    gallery = tmp_path / "alignment.png"
    report = evaluate_visual_interface_alignment(
        model,
        teacher,
        dataset,
        device=torch.device("cpu"),
        precision="fp32",
        minimum_patches=12,
        batch_size=1,
        gallery_path=gallery,
    )
    assert report["finite"] is True
    assert report["patches"] >= 12
    assert torch.isfinite(torch.tensor(report["mean_squared_error"]))
    assert -1.0 <= report["mean_cosine_similarity"] <= 1.0
    assert 0.0 <= report["codec_ink_f1"] <= 1.0
    assert gallery.is_file()


def test_stage_a_gate_requires_every_preregistered_condition() -> None:
    model = _model()
    boundary = causal_glyph_flow_boundary_receipt(model)
    alignment = {
        "finite": True,
        "patches": 2_048,
        "mean_cosine_similarity": 0.95,
        "mean_squared_error": 0.035,
    }
    passed = v35_stage_a_gate(
        alignment,
        boundary,
        initial_core_sha256="core",
        observed_core_sha256="core",
        initial_codec_sha256="codec",
        observed_codec_sha256="codec",
    )
    assert passed["passed"] is True
    assert all(passed["checks"].values())

    failed = v35_stage_a_gate(
        alignment | {"mean_cosine_similarity": 0.949},
        boundary,
        initial_core_sha256="core",
        observed_core_sha256="changed",
        initial_codec_sha256="codec",
        observed_codec_sha256="codec",
    )
    assert failed["passed"] is False
    assert failed["checks"]["cosine_at_least_0_95"] is False
    assert failed["checks"]["core_unchanged"] is False


def test_stage_a_gate_rejects_forbidden_runtime_flag() -> None:
    boundary = causal_glyph_flow_boundary_receipt(_model()) | {"uses_ocr": True}
    gate = v35_stage_a_gate(
        {
            "finite": True,
            "patches": 2_048,
            "mean_cosine_similarity": 1.0,
            "mean_squared_error": 0.0,
        },
        boundary,
        initial_core_sha256="same",
        observed_core_sha256="same",
        initial_codec_sha256="same",
        observed_codec_sha256="same",
    )
    assert gate["passed"] is False
    assert gate["checks"]["runtime_boundary_clean"] is False

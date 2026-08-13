from __future__ import annotations

import json
from dataclasses import asdict

import torch
from PIL import Image

from ilm.visual_lm.causal_glyph_flow import (
    V35_ARCHITECTURE,
    CausalGlyphFlowConfig,
    CausalGlyphFlowLM,
    file_sha256,
)
from scripts.export_causal_glyph_flow_v35 import (
    build_standalone_payload,
    standalone_checkpoint_is_clean,
)
from scripts.infer_causal_glyph_flow_v35 import (
    load_image_prompt,
    load_standalone_model,
    render_text_prompt,
)


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


def _training_checkpoint(path) -> None:
    model = _model()
    torch.save(
        {
            "architecture": V35_ARCHITECTURE,
            "model_config": asdict(model.config),
            "model": model.state_dict(),
            "ema": {
                "decay": 0.999,
                "shadow": {
                    name: parameter.detach().float().clone()
                    for name, parameter in model.named_parameters()
                },
            },
            "global_update": 1,
            "run_receipt": {
                "stages": [{"name": "unit", "updates": 1}],
                "initialization": {"route": "unit"},
            },
        },
        path,
    )


def test_standalone_export_and_load_remove_training_state(tmp_path) -> None:
    training = tmp_path / "training.pt"
    report = tmp_path / "report.json"
    standalone = tmp_path / "standalone.pt"
    _training_checkpoint(training)
    digest = file_sha256(training)
    report.write_text(
        json.dumps(
            {
                "checkpoint": {
                    "raw": {"checkpoint_sha256": digest},
                    "ema": {"checkpoint_sha256": digest},
                },
                "decision": {
                    "status": "visual-causal-qualified",
                    "selected_writer": "anchor",
                },
                "evidence_eligible": True,
            }
        ),
        encoding="utf-8",
    )
    payload = build_standalone_payload(training, report)
    assert standalone_checkpoint_is_clean(payload) is True
    assert payload["optimizer"] is None
    assert payload["runtime_teacher"] is None
    assert payload["ocr"] is None
    torch.save(payload, standalone)
    model, loaded = load_standalone_model(standalone, device=torch.device("cpu"))
    assert loaded["writer"] == "anchor"
    assert model.config.maximum_patches == 12


def test_text_and_image_wrappers_emit_only_raster_tensors(tmp_path) -> None:
    font = "/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc"
    text = render_text_prompt(
        "问：中",
        font_path=font,
        font_size=24,
        maximum_patches=8,
    )
    assert text.ndim == 3
    assert text.shape[:2] == (1, 32)
    assert text.shape[-1] % 32 == 0
    assert set(text.unique().tolist()).issubset({0.0, 1.0})

    image_path = tmp_path / "prompt.png"
    Image.new("L", (50, 20), 255).save(image_path)
    image = load_image_prompt(image_path, maximum_patches=8)
    assert image.shape == (1, 32, 96)
    assert bool(image.eq(1).all())

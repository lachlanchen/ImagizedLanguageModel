from pathlib import Path

import pytest
import torch

from ilm.visual_lm.glyph_motor_bridge import (
    binary_ink_f1,
    noise_condition_name,
    render_centered_glyph,
    unit_grayscale_to_pil,
)
from scripts.audit_glyph_motor_bridge_v41 import evaluate_motor_gate


FONT = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")


def test_centered_glyph_render_is_a_nonblank_unit_raster() -> None:
    if not FONT.is_file():
        pytest.skip("Noto CJK font is unavailable")
    pixels = render_centered_glyph(FONT, "中")

    assert pixels.shape == (1, 128, 128)
    assert torch.isfinite(pixels).all()
    assert float(pixels.min()) < 0.5
    assert float(pixels.max()) == 1.0
    assert unit_grayscale_to_pil(pixels).size == (128, 128)


def test_binary_ink_f1_distinguishes_exact_and_blank_images() -> None:
    target = torch.ones(2, 1, 8, 8)
    target[:, :, 2:6, 3:5] = 0.0
    exact = binary_ink_f1(target, target)
    blank = binary_ink_f1(torch.ones_like(target), target)

    assert torch.allclose(exact, torch.ones(2))
    assert torch.all(blank < 1e-5)


def test_noise_condition_names_are_stable() -> None:
    assert noise_condition_name(0.0) == "v34_sigma_0p000"
    assert noise_condition_name(0.03) == "v34_sigma_0p030"
    with pytest.raises(ValueError):
        noise_condition_name(-0.1)


def _passing_metrics() -> dict[str, dict[str, float]]:
    values = {}
    for name, target_f1 in (
        ("exact_source", 0.58),
        ("coarse_32", 0.56),
        ("v34_sigma_0p000", 0.55),
        ("v34_sigma_0p030", 0.54),
        ("v34_sigma_0p050", 0.53),
        ("v34_sigma_0p100", 0.52),
    ):
        values[name] = {
            "target_ink_f1_mean": target_f1,
            "target_ink_f1_min": 0.4,
            "source_before_motor_ink_f1_mean": target_f1 - 0.1,
            "motor_delta_ink_f1": 0.1,
            "pixel_mae": 0.2,
            "nonblank_fraction": 1.0,
        }
    return values


def test_motor_gate_is_conjunctive() -> None:
    metrics = _passing_metrics()
    assert evaluate_motor_gate(metrics)["passed"] is True

    metrics["coarse_32"]["motor_delta_ink_f1"] = -0.01
    result = evaluate_motor_gate(metrics)
    assert result["passed"] is False
    assert result["checks"]["motor_improves_every_condition"] is False


def test_motor_gate_requires_protocol_conditions() -> None:
    metrics = _passing_metrics()
    del metrics["v34_sigma_0p050"]
    with pytest.raises(ValueError, match="required conditions"):
        evaluate_motor_gate(metrics)

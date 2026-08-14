from __future__ import annotations

import torch

from ilm.visual_lm.noise_limited_retinal_field_v45 import (
    V45_ARCHITECTURE,
    NoiseLimitedRetinalFieldV45Config,
    fit_noise_limited_retinal_field_v45,
    noise_limited_retinal_field_v45_boundary_is_clean,
    noise_limited_retinal_field_v45_boundary_receipt,
    noise_limited_retinal_field_v45_from_checkpoint_payload,
    noise_limited_retinal_field_v45_state_sha256,
)
from ilm.visual_lm.noise_limited_retinal_field_v45_evaluation import (
    field_geometry_metrics,
    noise_limited_retinal_field_v45_gate_report,
    pair_displacement_metrics,
    retrieval_metrics,
    roundtrip_metrics,
)


def _pixels(count: int = 16) -> torch.Tensor:
    pixels = torch.zeros(count, 1, 32, 32)
    for index in range(count):
        top = 2 + index % 8
        left = 2 + (3 * index) % 12
        pixels[index, 0, top : top + 8, left : left + 3] = 1.0
        pixels[index, 0, 14:17, 3 + index % 16 : 15 + index % 16] = 1.0
    return pixels


def _field():
    pixels = _pixels()
    config = NoiseLimitedRetinalFieldV45Config(
        fit_bank_size=len(pixels),
        identity_bank_size=8,
    )
    return fit_noise_limited_retinal_field_v45(
        pixels,
        torch.arange(1, len(pixels) + 1),
        config=config,
    )


def test_v45_direction_and_radius_preserve_the_exact_raster_field() -> None:
    pixels = _pixels()
    field = _field()
    source = field.dct.encode(pixels).double()
    encoded = field.encode_dct(source, exact=True)
    decoded = field.decode_dct(
        encoded.direction,
        encoded.log_radius,
        exact=True,
    )
    torch.testing.assert_close(decoded, source, atol=2e-8, rtol=0.0)
    torch.testing.assert_close(
        encoded.direction.norm(dim=-1),
        torch.ones(len(pixels), dtype=torch.float64),
        atol=1e-10,
        rtol=0.0,
    )
    assert bool((encoded.radius > 0).all())
    assert torch.equal(field.binary(encoded.direction, encoded.log_radius, exact=True), pixels)


def test_v45_roundtrip_and_boundary_receipts_are_strict() -> None:
    pixels = _pixels()
    field = _field()
    measured = roundtrip_metrics(field, pixels)
    assert measured["finite"] is True
    assert measured["maximum_dct_absolute_error"] < 2e-8
    assert measured["binary_pixel_accuracy"] == 1.0
    assert measured["ink_f1"] == 1.0
    receipt = noise_limited_retinal_field_v45_boundary_receipt(field)
    assert receipt["trainable_parameters"] == 0
    assert receipt["exact_radial_side_channel"] is True
    assert receipt["uses_token_ids"] is False
    assert receipt["candidate_bank_deployed"] is False
    assert noise_limited_retinal_field_v45_boundary_is_clean(field)
    assert len(noise_limited_retinal_field_v45_state_sha256(field)) == 64


def test_v45_checkpoint_payload_reloads_the_exact_field() -> None:
    field = _field()
    digest = noise_limited_retinal_field_v45_state_sha256(field)
    payload = {
        "architecture": V45_ARCHITECTURE,
        "config": {
            "size": 32,
            "fit_bank_size": 16,
            "identity_bank_size": 8,
            "whitening_power": 0.10,
            "ridge_ratio": 0.50,
            "radius_epsilon": 1e-8,
            "binary_threshold": 0.5,
        },
        "field": field.state_dict(),
        "field_state_sha256": digest,
    }
    restored = noise_limited_retinal_field_v45_from_checkpoint_payload(payload)
    assert noise_limited_retinal_field_v45_state_sha256(restored) == digest
    source = field.dct.encode(_pixels())
    expected = field.encode_dct(source, exact=True)
    measured = restored.encode_dct(source, exact=True)
    torch.testing.assert_close(measured.direction, expected.direction)
    torch.testing.assert_close(measured.log_radius, expected.log_radius)

    payload["field_state_sha256"] = "0" * 64
    try:
        noise_limited_retinal_field_v45_from_checkpoint_payload(payload)
    except ValueError as error:
        assert "digest differs" in str(error)
    else:
        raise AssertionError("V45 accepted a field-state digest mismatch")


def test_v45_geometry_and_retrieval_metrics_use_continuous_fields() -> None:
    field = _field()
    pixels = _pixels()
    dct = field.dct.encode(pixels)
    transformed = field.encode_dct(dct)
    weights = torch.arange(1, len(pixels) + 1, dtype=torch.float32)
    geometry = field_geometry_metrics(
        transformed.direction,
        transformed.radius,
        weights=weights,
    )
    assert 1.0 <= geometry["effective_rank"] <= len(pixels)
    assert geometry["radius_minimum"] > 0.0
    retrieval = retrieval_metrics(
        transformed.direction[:8],
        transformed.direction[:8],
    )
    assert retrieval["top1"] == 1.0
    pair = transformed.direction[:8].reshape(4, 2, -1)
    pair_metrics = pair_displacement_metrics(pair)
    assert pair_metrics["pairs"] == 4.0
    assert pair_metrics["delta_norm_p05"] > 0.0


def test_v45_gate_report_is_conjunctive() -> None:
    roundtrip = {
        "all_finite": True,
        "maximum_blank_rate": 0.0,
        "maximum_dct_absolute_error": 1e-10,
        "minimum_binary_pixel_accuracy": 1.0,
        "minimum_ink_f1": 1.0,
    }
    raw_geometry = {"weighted_resultant_length": 0.70, "effective_rank": 100.0}
    field_geometry = {"weighted_resultant_length": 0.02, "effective_rank": 125.0}
    held = {
        "first": {"raw": {"top1": 0.80}, "v45": {"top1": 0.81}},
        "second": {"raw": {"top1": 0.90}, "v45": {"top1": 0.90}},
    }
    shifts = {
        name: {"raw": {"top1": 0.75}, "v45": {"top1": 0.76}}
        for name in ("left", "right", "up", "down")
    }
    raw_pairs = {
        "candidate_pair_cosine": 0.55,
        "delta_norm_p05": 0.30,
        "delta_effective_rank": 100.0,
        "delta_stable_rank": 50.0,
    }
    field_pairs = {
        "candidate_pair_cosine": 0.10,
        "delta_norm_p05": 0.60,
        "delta_effective_rank": 115.0,
        "delta_stable_rank": 56.0,
    }
    gates = noise_limited_retinal_field_v45_gate_report(
        roundtrip=roundtrip,
        raw_geometry=raw_geometry,
        field_geometry=field_geometry,
        held_fonts=held,
        shifts=shifts,
        raw_pairs=raw_pairs,
        field_pairs=field_pairs,
        fit_boundary_clean=True,
        frozen_partition_opened=False,
        peak_allocated_vram_gib=1.0,
        elapsed_seconds=60.0,
    )
    assert len(gates) == 13
    assert all(gates.values())
    field_pairs["candidate_pair_cosine"] = 0.40
    failed = noise_limited_retinal_field_v45_gate_report(
        roundtrip=roundtrip,
        raw_geometry=raw_geometry,
        field_geometry=field_geometry,
        held_fonts=held,
        shifts=shifts,
        raw_pairs=raw_pairs,
        field_pairs=field_pairs,
        fit_boundary_clean=True,
        frozen_partition_opened=False,
        peak_allocated_vram_gib=1.0,
        elapsed_seconds=60.0,
    )
    assert failed["pair_cosine_reduction"] is False
    assert not all(failed.values())

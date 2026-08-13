from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import Dataset

from ilm.visual_lm import continuous_glyph_codec_evaluation as evaluation
from ilm.visual_lm.continuous_glyph_codec_evaluation import (
    BinaryPatchMetrics,
    LatentMoments,
    merge_latent_moments,
    evaluate_continuous_glyph_codec,
    v34_development_gate,
    v34_sealed_transfer_gate,
)


class WhiteCodec(torch.nn.Module):
    def encode(self, pixels: torch.Tensor) -> torch.Tensor:
        return torch.zeros(pixels.shape[0], 768, device=pixels.device)

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        return torch.full(
            (latents.shape[0], 1, 32, 32),
            12.0,
            device=latents.device,
        )


class BlankRenderedDataset(Dataset[dict]):
    def __len__(self) -> int:
        return 2

    def __getitem__(self, index: int) -> dict:
        patches = 4
        return {
            "pixels": torch.ones(1, 32, patches * 32),
            "patch_mask": torch.ones(patches),
            "next_patch_mask": torch.zeros(patches),
            "reconstruction_mask": torch.ones(patches),
            "stop_targets": torch.zeros(patches),
            "stop_mask": torch.zeros(patches),
            "metadata": {"identifier": f"blank:{index}", "text": ""},
        }


class BlankHistoricDataset(Dataset[dict]):
    def __len__(self) -> int:
        return 3

    def __getitem__(self, index: int) -> dict:
        return {
            "pixels": torch.ones(1, 32, 32),
            "metadata": {
                "identifier": f"historic:{index}",
                "stage": "oracle",
                "label": f"J{index}",
            },
        }


def perfect_report() -> dict:
    return {
        "finite": True,
        "rendered": {
            "clean": {"ink_pixel_f1": 0.99, "edge_f1": 0.99},
            "noisy": {"ink_pixel_f1": 0.98, "edge_f1": 0.97},
            "ocr": {"retention": 0.97},
        },
        "historical": {"ink_pixel_f1": 0.98, "edge_f1": 0.96},
        "blank": {"false_ink_rate": 0.0},
        "latent": {"finite": True, "mean_per_dimension_std": 0.5},
    }


def test_binary_patch_metrics_distinguish_exact_and_inverted() -> None:
    target = torch.ones(2, 1, 32, 32, dtype=torch.bool)
    target[:, :, 8:24, 12:20] = False
    exact = BinaryPatchMetrics()
    exact.update(target, target)
    report = exact.report()
    assert report["ink_pixel_f1"] == 1.0
    assert report["edge_f1"] == 1.0
    assert report["exact_patch_rate"] == 1.0

    wrong = BinaryPatchMetrics()
    wrong.update(~target, target)
    assert wrong.report()["ink_pixel_f1"] == 0.0
    assert wrong.report()["exact_patch_rate"] == 0.0


def test_latent_moments_merge_matches_joint_population_std() -> None:
    first = LatentMoments()
    second = LatentMoments()
    first.update(torch.tensor([[0.0, 2.0], [2.0, 4.0]]))
    second.update(torch.tensor([[4.0, 6.0], [6.0, 8.0]]))
    report = merge_latent_moments(first, second).report()
    expected = torch.tensor([[0.0, 2.0], [2.0, 4.0], [4.0, 6.0], [6.0, 8.0]])
    assert report["samples"] == 4
    assert abs(report["mean_per_dimension_std"] - float(expected.std(0, unbiased=False).mean())) < 1e-7


def test_development_gate_matches_preregistered_thresholds() -> None:
    passed = v34_development_gate(
        perfect_report(),
        updates_complete=True,
        checkpoint_finite=True,
        peak_vram_bytes=8 * 1024**3,
    )
    assert passed["pass"] is True
    failed_report = perfect_report()
    failed_report["historical"]["edge_f1"] = 0.939
    failed = v34_development_gate(
        failed_report,
        updates_complete=True,
        checkpoint_finite=True,
        peak_vram_bytes=8 * 1024**3,
    )
    assert failed["pass"] is False
    assert failed["gates"]["historical_edge_f1_at_least_0_940"] is False


def test_sealed_gate_requires_97_percent_retention_for_each_metric() -> None:
    development = perfect_report()
    sealed = perfect_report()
    passed = v34_sealed_transfer_gate(development, sealed)
    assert passed["pass"] is True
    sealed["rendered"]["ocr"]["retention"] = 0.90
    failed = v34_sealed_transfer_gate(development, sealed)
    assert failed["pass"] is False
    assert failed["gates"]["clean_rendered_ocr_retention_retains_at_least_0_97"] is False


def test_evaluator_runs_pixel_only_path_and_writes_galleries(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(evaluation, "_tesseract_text", lambda image, language: "")
    monkeypatch.setattr(
        evaluation,
        "tesseract_identity",
        lambda language: {"language": language, "traineddata_sha256": "test"},
    )
    report = evaluate_continuous_glyph_codec(
        WhiteCodec(),  # type: ignore[arg-type]
        BlankRenderedDataset(),
        BlankHistoricDataset(),
        split="development",
        device=torch.device("cpu"),
        precision="fp32",
        rendered_minimum_patches=8,
        historical_minimum_patches=3,
        rendered_batch_size=2,
        historical_batch_size=3,
        gallery_directory=tmp_path,
    )
    assert report["finite"] is True
    assert report["rendered"]["clean"]["exact_patch_rate"] == 1.0
    assert report["rendered"]["noisy"]["exact_patch_rate"] == 1.0
    assert report["rendered"]["ocr"]["retention"] == 1.0
    assert report["historical"]["exact_patch_rate"] == 1.0
    assert report["blank"]["false_ink_rate"] == 0.0
    assert (tmp_path / "development_rendered_gallery.png").is_file()
    assert (tmp_path / "development_historic_gallery.png").is_file()

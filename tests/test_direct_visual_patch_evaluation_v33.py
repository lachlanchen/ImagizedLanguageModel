from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import Dataset

from ilm.visual_lm.direct_visual_patch_evaluation import (
    _edit_distance,
    evaluate_visual_calibration,
)
from ilm.visual_lm.direct_visual_patch_lm import (
    DirectVisualPatchConfig,
    DirectVisualPatchLM,
)


class BlankDataset(Dataset[dict]):
    def __len__(self) -> int:
        return 2

    def __getitem__(self, index: int) -> dict:
        patches = 4
        return {
            "pixels": torch.ones(1, 32, patches * 32),
            "patch_mask": torch.ones(patches),
            "next_patch_mask": torch.ones(patches),
            "reconstruction_mask": torch.ones(patches),
            "stop_targets": torch.zeros(patches),
            "stop_mask": torch.zeros(patches),
            "metadata": {
                "identifier": f"blank:{index}",
                "text": "",
            },
        }


def tiny_model() -> DirectVisualPatchLM:
    return DirectVisualPatchLM(
        DirectVisualPatchConfig(
            patch_size=32,
            maximum_patches=8,
            hidden_size=64,
            layers=1,
            attention_heads=4,
            key_value_heads=4,
            intermediate_size=128,
        )
    )


def test_edit_distance() -> None:
    assert _edit_distance("语言", "语言") == 0
    assert _edit_distance("语言", "语") == 1
    assert _edit_distance("kitten", "sitting") == 3


def test_calibration_audit_writes_gallery(tmp_path: Path) -> None:
    model = tiny_model()
    with torch.no_grad():
        model.output_projection.weight.zero_()
    gallery = tmp_path / "gallery.png"
    report = evaluate_visual_calibration(
        model,
        BlankDataset(),
        device=torch.device("cpu"),
        precision="fp32",
        minimum_patches=8,
        batch_size=2,
        gallery_path=gallery,
    )
    assert report["patches"] == 8
    assert report["finite"] is True
    assert report["pass"] is True
    assert gallery.is_file()

from __future__ import annotations

from typing import Any

import torch
from torch.utils.data import Dataset

from ilm.visual_lm.causal_glyph_flow_data import (
    CausalGlyphCopyDataset,
    CausalGlyphStageCMixture,
    causal_glyph_flow_collate,
    causal_glyph_flow_data_boundary_receipt,
    causal_glyph_flow_student_batch,
)
from ilm.visual_lm.direct_visual_patch_data import (
    DirectPatchRenderConfig,
    direct_patch_partition,
    render_direct_patch_continuation,
)
from ilm.visual_lm.visual_semantic_raster_data import VisualTextRecord


def _identifier_for_split(split: str, *, prefix: str) -> str:
    for index in range(10_000):
        identifier = f"{prefix}:{index}"
        if direct_patch_partition(identifier, stream="public-domain") == split:
            return identifier
    raise AssertionError(f"could not construct a {split} identifier")


def _record(split: str, *, prefix: str) -> VisualTextRecord:
    return VisualTextRecord(
        identifier=_identifier_for_split(split, prefix=prefix),
        text="天地玄黄宇宙洪荒日月盈昃辰宿列张",
        language="zh",
        source="unit",
        rights="public domain",
    )


def _config() -> DirectPatchRenderConfig:
    return DirectPatchRenderConfig(
        maximum_patches=24,
        maximum_prompt_patches=16,
        maximum_answer_patches=8,
        minimum_font_size=24,
        maximum_font_size=24,
        augment=False,
    )


class _NamedDataset(Dataset[dict[str, Any]]):
    def __init__(self, name: str, length: int = 128) -> None:
        self.name = name
        self.length = length

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> dict[str, Any]:
        if not 0 <= index < self.length:
            raise IndexError(index)
        patches = 4
        return {
            "pixels": torch.ones(1, 32, patches * 32),
            "patch_mask": torch.ones(patches),
            "next_patch_mask": torch.ones(patches),
            "reconstruction_mask": torch.ones(patches),
            "stop_targets": torch.zeros(patches),
            "stop_mask": torch.ones(patches),
            "metadata": {"name": self.name, "source_index": index},
        }


def test_copy_dataset_is_deterministic_and_inherits_source_split() -> None:
    train = _record("train", prefix="copy-train")
    development = _record("development", prefix="copy-development")
    dataset = CausalGlyphCopyDataset(
        [train, development],
        split="train",
        config=_config(),
        length=12,
        seed=17,
    )
    assert [record.identifier for record in dataset.records] == [train.identifier]
    first = dataset[3]
    repeated = dataset[3]
    assert torch.equal(first["pixels"], repeated["pixels"])
    metadata = first["metadata"]
    assert metadata["source_identifier"] == train.identifier
    assert metadata["source_split"] == "train"
    assert 2 <= len(metadata["copy_span"]) <= 16
    assert metadata["answer_text"] == metadata["copy_span"]
    assert metadata["prompt_text"] == f"照写：{metadata['copy_span']} 答："
    assert metadata["prompt"]["origin"] in range(32)
    assert metadata["answer"]["origin"] in range(32)


def test_stage_c_mixture_has_exact_six_one_one_cycle() -> None:
    mixture = CausalGlyphStageCMixture(
        _NamedDataset("instruction"),
        _NamedDataset("copy"),
        _NamedDataset("public"),
        length=80,
    )
    assert mixture.mixture_counts() == {
        "instruction": 60,
        "copy": 10,
        "public": 10,
    }
    assert [mixture[index]["metadata"]["mixture_stream"] for index in range(8)] == [
        "instruction",
        "instruction",
        "instruction",
        "instruction",
        "instruction",
        "instruction",
        "copy",
        "public",
    ]
    assert mixture[8]["metadata"]["source_index"] == 6
    assert mixture[14]["metadata"]["source_index"] == 1
    assert mixture[15]["metadata"]["source_index"] == 1


def test_v35_student_boundary_excludes_all_text_metadata() -> None:
    sample = render_direct_patch_continuation(
        _record("train", prefix="boundary"),
        split="train",
        config=_config(),
        variant=5,
    )
    batch = causal_glyph_flow_collate([sample, sample])
    student = causal_glyph_flow_student_batch(batch)
    receipt = causal_glyph_flow_data_boundary_receipt(batch)
    assert "metadata" not in student
    assert receipt["metadata_excluded"] is True
    assert receipt["all_student_values_are_tensors"] is True
    assert receipt["student_contains_strings"] is False
    assert student["pixels"].shape == (2, 1, 32, 24 * 32)


def test_copy_dataset_rejects_cross_split_only_input() -> None:
    development = _record("development", prefix="only-development")
    try:
        CausalGlyphCopyDataset(
            [development],
            split="train",
            config=_config(),
            length=1,
        )
    except ValueError as error:
        assert "empty" in str(error)
    else:
        raise AssertionError("V35 accepted a source from another split")

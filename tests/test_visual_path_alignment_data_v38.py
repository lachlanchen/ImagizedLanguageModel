from __future__ import annotations

import hashlib
import json

import torch

from ilm.visual_lm.visual_path_alignment_data import (
    V38_PIXEL_KEYS,
    VisualPathAlignmentParaphrase,
    load_v38_paraphrases,
    render_visual_path_alignment_record,
    visual_path_alignment_collate,
    visual_path_alignment_data_boundary_receipt,
    visual_path_alignment_pixel_batch,
)
from ilm.visual_lm.visual_semantic_distillation_data import (
    VisualSemanticDistillationRenderConfig,
)
from ilm.visual_lm.visual_semantic_raster_data import VisualRasterRecord


def record() -> VisualRasterRecord:
    return VisualRasterRecord(
        identifier="alpaca-zh:unit",
        prompt="问：请解释视觉语言模型。",
        answer="视觉语言模型直接读取文字图像。",
        language="zh",
        source="unit",
        rights="test",
    )


def test_five_paths_use_distinct_fonts_and_paraphrase_pixels() -> None:
    source = record()
    paraphrase = VisualPathAlignmentParaphrase(
        identifier=source.identifier,
        text="问：说明什么是视觉语言模型。",
        source_prompt_sha256=hashlib.sha256(source.prompt.encode()).hexdigest(),
    )
    item = render_visual_path_alignment_record(
        source,
        config=VisualSemanticDistillationRenderConfig(augment=True),
        variant=38,
        paraphrase=paraphrase,
    )

    assert set(V38_PIXEL_KEYS).issubset(item)
    assert item["metadata"]["semantic_view_kind"] == "paraphrase"
    assert item["metadata"]["distinct_font_paths"] == 5
    fonts = {
        value["font_path"] for value in item["metadata"]["views"].values()
    }
    assert len(fonts) == 5
    assert not torch.equal(item["prompt_anchor_pixels"], item["semantic_view_pixels"])


def test_paraphrase_loader_checks_consensus_and_source_hash(tmp_path) -> None:
    source = record()
    path = tmp_path / "paraphrases.jsonl"
    row = {
        "identifier": source.identifier,
        "paraphrase": "问：说明什么是视觉语言模型。",
        "source_prompt_sha256": hashlib.sha256(source.prompt.encode()).hexdigest(),
        "instruction_judge": "pass",
        "constraint_adjudicator": "pass",
        "adversarial_confirmation": "pass",
    }
    path.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")

    loaded = load_v38_paraphrases(path, [source])

    assert loaded[source.identifier].text == row["paraphrase"]


def test_collated_tensor_boundary_excludes_metadata() -> None:
    source = record()
    config = VisualSemanticDistillationRenderConfig(augment=True)
    items = [
        render_visual_path_alignment_record(source, config=config, variant=seed)
        for seed in (10, 20)
    ]
    collated = visual_path_alignment_collate(items)

    pixels = visual_path_alignment_pixel_batch(collated)

    assert set(pixels) == set(V38_PIXEL_KEYS)
    assert all(torch.is_floating_point(value) for value in pixels.values())
    assert "metadata" not in pixels


def test_data_boundary_declares_five_training_only_paths() -> None:
    receipt = visual_path_alignment_data_boundary_receipt()

    assert receipt["paired_paths"] == 5
    assert receipt["distinct_training_font_paths_per_item"] == 5
    assert receipt["deployable_keys"] == ["prompt_pixels", "prompt_mask"]
    assert not receipt["uses_strings"]
    assert not receipt["uses_token_ids"]
    assert not receipt["uses_ocr"]


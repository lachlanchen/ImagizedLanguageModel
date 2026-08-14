from __future__ import annotations

import json

import torch

from ilm.visual_lm.visual_answer_trajectory_data import (
    V39_MAX_SEGMENT_UNITS,
    V39_PIXEL_KEYS,
    VisualAnswerTrajectoryRecord,
    load_v39_instruction_records,
    render_visual_answer_trajectory_record,
    segment_visual_answer,
    select_v39_instruction_records,
    visual_answer_trajectory_collate,
    visual_answer_trajectory_data_boundary_receipt,
    visual_answer_trajectory_tensor_batch,
)
from ilm.visual_lm.visual_semantic_distillation_data import (
    VisualSemanticDistillationRenderConfig,
)


def record() -> VisualAnswerTrajectoryRecord:
    return VisualAnswerTrajectoryRecord(
        identifier="alpaca-zh:unit",
        prompt="问：请解释视觉语言模型。",
        answer="视觉语言模型直接读取文字图像。它以连续状态组织答案。",
        segments=("视觉语言模型直接读取文字图像。", "它以连续状态组织答案。"),
        language="zh",
        source="unit",
        rights="test",
    )


def test_punctuation_segmenter_is_bounded_and_preserves_writing() -> None:
    text = (
        "第一段说明视觉输入为什么重要，它需要保留字形、布局和历史异体。"
        "第二段说明模型如何形成连续语义状态，并按照书写顺序生成后续内容。"
        "第三段包含组合附加符：a\u0301，并以完整句号结束。"
    )

    segments = segment_visual_answer(text)

    assert 2 <= len(segments) <= 16
    assert all(len(segment) <= V39_MAX_SEGMENT_UNITS + 2 for segment in segments)
    assert "".join(segments).replace(" ", "") == text.replace(" ", "")
    assert any("a\u0301" in segment for segment in segments)


def test_loader_admits_long_answers_and_rejects_more_than_sixteen_spans(
    tmp_path,
) -> None:
    long_answer = "。".join("这是第%02d段完整说明" % index for index in range(12)) + "。"
    oversized = "。".join(
        ("第%02d段" % index) + "内容" * 24 for index in range(17)
    ) + "。"
    path = tmp_path / "alpaca.json"
    path.write_text(
        json.dumps(
            [
                {"instruction": "解释方法", "input": "", "output": long_answer},
                {"instruction": "超长方法", "input": "", "output": oversized},
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    records = load_v39_instruction_records(path)

    assert len(records) == 1
    assert len(records[0].answer) > 32
    assert 1 < len(records[0].segments) <= 16


def test_four_paths_use_distinct_fonts_and_a_geometric_segment_index() -> None:
    item = render_visual_answer_trajectory_record(
        record(),
        config=VisualSemanticDistillationRenderConfig(augment=True),
        variant=39,
    )

    assert set(V39_PIXEL_KEYS).issubset(item)
    assert item["segment_index"] in {0, 1}
    assert item["metadata"]["distinct_font_paths"] == 4
    assert item["metadata"]["script_conversion"] in {"original", "s2t", "t2s"}
    assert all(torch.is_floating_point(item[key]) for key in V39_PIXEL_KEYS)


def test_collated_tensor_boundary_excludes_strings_and_metadata() -> None:
    config = VisualSemanticDistillationRenderConfig(augment=True)
    items = [
        render_visual_answer_trajectory_record(record(), config=config, variant=seed)
        for seed in (10, 20)
    ]

    collated = visual_answer_trajectory_collate(items)
    tensors = visual_answer_trajectory_tensor_batch(collated)

    assert set(tensors) == set(V39_PIXEL_KEYS) | {"segment_indices"}
    assert tensors["segment_indices"].dtype == torch.long
    assert "metadata" not in tensors
    assert "identifiers" not in tensors


def test_data_boundary_declares_offline_only_text_operations() -> None:
    receipt = visual_answer_trajectory_data_boundary_receipt()

    assert receipt["maximum_segments"] == 16
    assert receipt["paired_visual_paths"] == 4
    assert receipt["distinct_training_font_paths_per_item"] == 4
    assert receipt["deployable_keys"] == ["prompt_pixels", "prompt_mask"]
    assert not receipt["uses_strings_after_tensor_boundary"]
    assert not receipt["uses_token_ids"]
    assert not receipt["uses_unicode_ids"]
    assert not receipt["uses_ocr"]
    assert not receipt["uses_runtime_script_converter"]


def test_font_fit_selection_uses_bounded_geometry_and_rejects_overflow() -> None:
    short = record()
    long = VisualAnswerTrajectoryRecord(
        identifier="alpaca-zh:long",
        prompt="问：" + "汉" * 150,
        answer="可见回答。",
        segments=("可见回答。",),
        language="zh",
        source="unit",
        rights="test",
    )

    selected, rejected = select_v39_instruction_records(
        (short, long),
        split="train",
        render_config=VisualSemanticDistillationRenderConfig(augment=False),
        include_all_records=True,
    )

    assert tuple(item.identifier for item in selected) == (short.identifier,)
    assert rejected == (long.identifier,)

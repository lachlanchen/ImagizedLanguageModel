from __future__ import annotations

import pytest
import torch

from ilm.visual_lm.factorized_visual_context_data import (
    FactorizedVisualPairDataset,
    FactorizedVisualRenderConfig,
    FactorizedVisualSuffixPair,
    factorized_visual_pair_collate,
    factorized_visual_pair_student_batch,
)


def _pair() -> FactorizedVisualSuffixPair:
    context_a = "天地玄黃" * 15 + "知之為知"
    context_b = "宇宙洪荒" * 15 + "知之為知"
    assert len(context_a) == len(context_b) == 64
    return FactorizedVisualSuffixPair(
        suffix_cells=4,
        identifier_a="a",
        script_view_a="original",
        context_a=context_a,
        target_a="也",
        identifier_b="b",
        script_view_b="original",
        context_b=context_b,
        target_b="矣",
    )


def test_pair_contract_rejects_nonmatching_suffix() -> None:
    pair = _pair()
    with pytest.raises(ValueError, match="suffixes"):
        FactorizedVisualSuffixPair(
            **{
                **pair.__dict__,
                "context_b": pair.context_b[:-1] + "不",
            }
        )


def test_pair_dataset_makes_shared_suffix_pixel_identical() -> None:
    dataset = FactorizedVisualPairDataset(
        [_pair()],
        split="train",
        render_config=FactorizedVisualRenderConfig(augment=True),
        seed=3,
        length=1,
    )
    item = dataset[0]
    assert torch.equal(item["context_a"][-4:], item["context_b"][-4:])
    assert torch.equal(
        item["reference_context_a"][-4:], item["reference_context_b"][-4:]
    )
    raw = factorized_visual_pair_collate([item])
    student = factorized_visual_pair_student_batch(raw)
    assert set(student) == {
        "context_a",
        "target_a",
        "reference_context_a",
        "reference_target_a",
        "context_b",
        "target_b",
        "reference_context_b",
        "reference_target_b",
    }
    assert all(torch.is_floating_point(value) for value in student.values())

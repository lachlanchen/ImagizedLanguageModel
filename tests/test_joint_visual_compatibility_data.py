from __future__ import annotations

import torch

from ilm.visual_lm.factorized_visual_context_data import FactorizedVisualSuffixPair
from ilm.visual_lm.joint_visual_compatibility_data import (
    JointVisualPairDataset,
    JointVisualRenderConfig,
    joint_visual_data_boundary_receipt,
    joint_visual_pair_collate,
    joint_visual_pair_student_batch,
)


def _pair() -> FactorizedVisualSuffixPair:
    context_a = "天地玄黃" * 15 + "知之為知"
    context_b = "宇宙洪荒" * 15 + "知之為知"
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


def test_pair_dataset_preserves_suffix_and_randomizes_only_candidate_positions() -> None:
    dataset = JointVisualPairDataset(
        [_pair()],
        split="train",
        render_config=JointVisualRenderConfig(augment=True),
        seed=20260914,
        length=2,
    )
    item = dataset[0]
    assert item["contexts"].shape == (2, 64, 1, 32, 32)
    assert item["candidates"].shape == (2, 1, 32, 32)
    assert torch.equal(item["contexts"][0, -4:], item["contexts"][1, -4:])
    assert torch.equal(
        item["reference_contexts"][0, -4:],
        item["reference_contexts"][1, -4:],
    )
    assert torch.equal(item["assignment"].sort().values, torch.arange(2))
    assert torch.equal(
        item["reference_assignment"].sort().values, torch.arange(2)
    )

    raw = joint_visual_pair_collate([item, dataset[1]])
    student = joint_visual_pair_student_batch(raw)
    assert set(student) == {
        "contexts",
        "candidates",
        "assignment",
        "reference_contexts",
        "reference_candidates",
        "reference_assignment",
    }
    assert student["contexts"].shape == (2, 2, 64, 1, 32, 32)
    assert student["candidates"].shape == (2, 2, 1, 32, 32)
    assert student["assignment"].dtype == torch.long


def test_v27_data_boundary_distinguishes_positions_from_character_ids() -> None:
    receipt = joint_visual_data_boundary_receipt()
    assert receipt["canonical_identity_derived_from_exact_pixels"] is True
    assert receipt["pair_assignment_labels_are_positions"] is True
    assert receipt["pair_candidate_order_is_randomized"] is True
    assert receipt["uses_character_ids"] is False

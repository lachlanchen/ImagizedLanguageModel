from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader

from ilm.visual_lm.visual_semantic_plan import VisualSentenceImageTeacher
from ilm.visual_lm.visual_semantic_plan_data import (
    VisualSemanticPlanAnswerDataset,
    VisualSemanticPlanRenderConfig,
    visual_semantic_plan_answer_collate,
)
from ilm.visual_lm.visual_semantic_plan_training import VisualSemanticPlanTargetBank
from ilm.visual_lm.visual_semantic_raster_data import VisualRasterRecord
from scripts.build_visual_semantic_plan_targets_v36 import (
    build_target_bank,
    v36_model_config,
)


FONT = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"


def test_tiny_target_builder_preserves_external_identifier_alignment() -> None:
    if not Path(FONT).is_file():
        return
    records = [
        VisualRasterRecord(
            identifier=f"tiny-target:{index}",
            prompt=f"问：第{index}项是什么？",
            answer=f"这是第{index}项。",
            language="zh",
            source="unit-test",
            rights="test-only",
        )
        for index in range(2)
    ]
    dataset = VisualSemanticPlanAnswerDataset(
        records,
        split="train",
        render_config=VisualSemanticPlanRenderConfig(augment=False),
        seed=36,
        include_all_records=True,
    )
    loader = DataLoader(
        dataset,
        batch_size=2,
        collate_fn=visual_semantic_plan_answer_collate,
    )
    config = v36_model_config(tiny=True)
    bank = build_target_bank(
        VisualSentenceImageTeacher(config),
        loader,
        device=torch.device("cpu"),
        precision="fp32",
        receipt={"test": True},
    )
    restored = VisualSemanticPlanTargetBank.from_state_dict(bank.state_dict())

    assert restored.identifiers == ("tiny-target:0", "tiny-target:1")
    assert restored.global_plans.shape == (2, config.plan_dim)
    assert restored.chunk_plans.shape == (2, 4, config.plan_dim)
    assert torch.isfinite(restored.global_plans).all()
    assert restored.receipt == {"test": True}

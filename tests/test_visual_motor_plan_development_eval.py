from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image

from scripts.eval_visual_motor_plan_development import save_review_pages


def test_save_review_pages_chunks_trace_without_dropping_examples(tmp_path: Path) -> None:
    trace = {
        key: torch.rand(10, 1, 32, 32)
        for key in (
            "target_ink",
            "semantic_reference",
            "style_reference",
            "correct_ink",
            "shuffled_ink",
        )
    }

    names = save_review_pages(
        trace,
        tmp_path,
        stem="audit",
        count=10,
        columns=4,
    )

    assert names == ["audit_01.png", "audit_02.png", "audit_03.png"]
    widths = [Image.open(tmp_path / name).width for name in names]
    assert widths[0] == widths[1]
    assert widths[2] < widths[1]

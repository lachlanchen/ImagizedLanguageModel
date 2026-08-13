from __future__ import annotations

import torch

from scripts.eval_visual_semantic_plan_v36 import (
    PromptRasterSet,
    controlled_prompt_rasters,
    indexed_plan_retrieval_metrics,
)


def test_v36_prompt_controls_create_independent_rasters() -> None:
    pixels = torch.arange(4 * 3 * 16 * 1024, dtype=torch.float32).reshape(
        4, 3, 16, 1024
    )
    mask = torch.ones(4, 64)
    prompts = PromptRasterSet(
        identifiers=("a", "b", "c", "d"),
        pixels=pixels,
        mask=mask,
    )

    shuffled_pixels, shuffled_mask = controlled_prompt_rasters(prompts, "shuffled")
    blank_pixels, blank_mask = controlled_prompt_rasters(prompts, "blank")
    quarter_pixels, quarter_mask = controlled_prompt_rasters(
        prompts,
        "final-quarter",
    )

    assert torch.equal(shuffled_pixels[0], pixels[-1])
    assert torch.equal(shuffled_mask, mask)
    assert torch.all(blank_pixels == 1.0)
    assert torch.all(blank_mask == 0.0)
    assert torch.all(quarter_pixels[..., :768] == 1.0)
    assert torch.equal(quarter_pixels[..., 768:], pixels[..., 768:])
    assert torch.all(quarter_mask[:, :48] == 0.0)
    assert torch.all(quarter_mask[:, 48:] == 1.0)
    assert torch.equal(prompts.pixels, pixels)


def test_v36_indexed_retrieval_uses_full_candidate_bank() -> None:
    candidates = torch.eye(6)
    plans = torch.stack((candidates[4], candidates[1], candidates[5]))
    labels = torch.tensor([4, 1, 5])
    metrics = indexed_plan_retrieval_metrics(plans, candidates, labels)

    assert metrics["samples"] == 3
    assert metrics["candidates"] == 6
    assert metrics["top1"] == 1.0
    assert metrics["top5"] == 1.0
    assert metrics["mrr"] == 1.0

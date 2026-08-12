from __future__ import annotations

import pytest
import torch

from scripts.prepare_visual_packet_opaque_review_v24 import (
    REVIEW_CASES,
    REVIEW_HELDOUT_LENGTH_CASES,
    REVIEW_SEEN_LENGTH_CASES,
    render_review_card,
    select_review_items,
)
from scripts.score_visual_packet_opaque_review_v24 import score_review


class ReviewDataset:
    def __init__(self) -> None:
        self.items = [
            {"metadata": {"heldout_length": index % 2 == 0}} for index in range(200)
        ]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict[str, object]:
        return self.items[index]


def review_rows() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    answers: list[dict[str, object]] = []
    responses: list[dict[str, object]] = []
    for index in range(REVIEW_CASES):
        review_id = f"P-{index:010d}"
        choice = "A" if index % 2 == 0 else "B"
        answers.append(
            {
                "review_id": review_id,
                "correct_frame1_source": choice,
                "correct_frame2_label": choice,
                "heldout_length": index < REVIEW_HELDOUT_LENGTH_CASES,
            }
        )
        responses.append(
            {
                "review_id": review_id,
                "frame1_response": choice,
                "frame2_response": choice,
            }
        )
    return answers, responses


def test_v24_review_selection_has_fixed_length_balance() -> None:
    selected = select_review_items(ReviewDataset())  # type: ignore[arg-type]
    heldout = sum(bool(item["metadata"]["heldout_length"]) for _, item in selected)
    assert len(selected) == REVIEW_CASES
    assert heldout == REVIEW_HELDOUT_LENGTH_CASES
    assert len(selected) - heldout == REVIEW_SEEN_LENGTH_CASES


def test_v24_review_card_accepts_only_two_frame_visual_output() -> None:
    prompt = torch.rand(15, 1, 32, 32)
    generated = torch.rand(2, 1, 32, 32)
    card = render_review_card(
        "P-1234567890", prompt, generated, pair_packet_indices=(0, 2)
    )
    assert card.width > card.height
    with pytest.raises(ValueError, match="two image frames"):
        render_review_card(
            "P-1234567890",
            prompt,
            generated[:1],
            pair_packet_indices=(0, 2),
        )


def test_v24_review_scores_both_frames_and_heldout_lengths() -> None:
    answers, responses = review_rows()
    assert score_review(answers, responses)["passed"] is True

    for index in range(5):
        responses[index]["frame1_response"] = (
            "B" if responses[index]["frame1_response"] == "A" else "A"
        )
    result = score_review(answers, responses)
    assert result["frame1_correct"] == 43
    assert result["passed"] is False

    answers, responses = review_rows()
    for index in range(2):
        responses[index]["frame2_response"] = (
            "B" if responses[index]["frame2_response"] == "A" else "A"
        )
    result = score_review(answers, responses)
    assert result["frame2_correct"] == 46
    assert result["heldout_length_frame2_correct"] == 10
    assert result["passed"] is False


def test_v24_review_rejects_incomplete_or_nonvisual_choices() -> None:
    answers, responses = review_rows()
    responses[0]["frame1_response"] = None
    with pytest.raises(ValueError, match="must be A or B"):
        score_review(answers, responses)
    with pytest.raises(ValueError, match="exactly 48"):
        score_review(answers, responses[:-1])

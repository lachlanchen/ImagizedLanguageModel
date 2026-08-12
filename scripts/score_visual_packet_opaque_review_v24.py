#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from scripts.prepare_visual_packet_opaque_review_v24 import (
    ARCHITECTURE as REVIEW_ARCHITECTURE,
    EXPECTED_PAIRED_AUDIT_SHA256,
    REVIEW_CASES,
    REVIEW_HELDOUT_LENGTH_CASES,
)
from scripts.train_visual_state_actuator import file_sha256


ARCHITECTURE = "visual-packet-reread-stream-v24-opaque-review-result"
OVERALL_REQUIRED = 44
HELDOUT_REQUIRED = 11


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score a completed V24 two-frame opaque visual review."
    )
    parser.add_argument("--receipt", required=True)
    parser.add_argument("--answer-key", required=True)
    parser.add_argument("--responses", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--reviewer", required=True)
    return parser.parse_args()


def _validate_response(value: Any, review_id: str, frame: int) -> str:
    response = str(value or "").upper()
    if response not in {"A", "B"}:
        raise ValueError(f"V24 frame-{frame} response for {review_id} must be A or B")
    return response


def score_review(
    answer_rows: Sequence[dict[str, Any]],
    response_rows: Sequence[dict[str, Any]],
) -> dict[str, int | bool]:
    if len(answer_rows) != REVIEW_CASES or len(response_rows) != REVIEW_CASES:
        raise ValueError(f"V24 review requires exactly {REVIEW_CASES} responses")
    answers = {str(row["review_id"]): row for row in answer_rows}
    if len(answers) != REVIEW_CASES:
        raise ValueError("V24 answer key contains duplicate review IDs")
    responses: dict[str, tuple[str, str]] = {}
    for row in response_rows:
        review_id = str(row.get("review_id", ""))
        if review_id in responses:
            raise ValueError(f"duplicate V24 response ID {review_id}")
        if review_id not in answers:
            raise ValueError(f"unknown V24 response ID {review_id}")
        responses[review_id] = (
            _validate_response(row.get("frame1_response"), review_id, 1),
            _validate_response(row.get("frame2_response"), review_id, 2),
        )
    if set(responses) != set(answers):
        raise ValueError("V24 response set does not match the answer key")

    frame1_correct = 0
    frame2_correct = 0
    heldout_frame1_correct = 0
    heldout_frame2_correct = 0
    heldout_total = 0
    for review_id, answer in answers.items():
        frame1_response, frame2_response = responses[review_id]
        frame1_match = frame1_response == answer["correct_frame1_source"]
        frame2_match = frame2_response == answer["correct_frame2_label"]
        frame1_correct += int(frame1_match)
        frame2_correct += int(frame2_match)
        if answer["heldout_length"]:
            heldout_total += 1
            heldout_frame1_correct += int(frame1_match)
            heldout_frame2_correct += int(frame2_match)
    if heldout_total != REVIEW_HELDOUT_LENGTH_CASES:
        raise ValueError("V24 answer key has the wrong held-out-length count")
    passed = (
        frame1_correct >= OVERALL_REQUIRED
        and frame2_correct >= OVERALL_REQUIRED
        and heldout_frame1_correct >= HELDOUT_REQUIRED
        and heldout_frame2_correct >= HELDOUT_REQUIRED
    )
    return {
        "review_cases": REVIEW_CASES,
        "frame1_correct": frame1_correct,
        "frame1_required_correct": OVERALL_REQUIRED,
        "frame2_correct": frame2_correct,
        "frame2_required_correct": OVERALL_REQUIRED,
        "heldout_length_cases": heldout_total,
        "heldout_length_frame1_correct": heldout_frame1_correct,
        "heldout_length_frame1_required_correct": HELDOUT_REQUIRED,
        "heldout_length_frame2_correct": heldout_frame2_correct,
        "heldout_length_frame2_required_correct": HELDOUT_REQUIRED,
        "passed": passed,
    }


def main() -> None:
    args = parse_args()
    receipt = json.loads(Path(args.receipt).read_text(encoding="utf-8"))
    if receipt.get("architecture") != REVIEW_ARCHITECTURE:
        raise ValueError("V24 scorer requires an opaque-review receipt")
    if receipt.get("paired_audit_sha256") != EXPECTED_PAIRED_AUDIT_SHA256:
        raise ValueError("V24 review receipt has the wrong paired audit")
    if receipt.get("answer_key_sha256") != file_sha256(args.answer_key):
        raise ValueError("V24 answer key hash differs from its sealed receipt")
    if receipt.get("frozen_images_instantiated") != 0:
        raise ValueError("V24 review receipt broke the frozen seal")
    if receipt.get("contains_target_images") is not False:
        raise ValueError("V24 review receipt does not establish target-image opacity")
    if receipt.get("contains_correctness_labels") is not False:
        raise ValueError("V24 review receipt contains correctness labels")
    answer_rows = json.loads(Path(args.answer_key).read_text(encoding="utf-8"))
    response_rows = json.loads(Path(args.responses).read_text(encoding="utf-8"))
    scores = score_review(answer_rows, response_rows)
    result = {
        "architecture": ARCHITECTURE,
        "reviewer": args.reviewer,
        "review_receipt_sha256": file_sha256(args.receipt),
        "answer_key_sha256": file_sha256(args.answer_key),
        "responses_sha256": file_sha256(args.responses),
        **scores,
        "paired_gate_passed": True,
        "opaque_review_passed": scores["passed"],
        "frozen_evaluation_permitted": scores["passed"],
        "frozen_images_instantiated": 0,
    }
    output = Path(args.out)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite V24 review result: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    if not scores["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

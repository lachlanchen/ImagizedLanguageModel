#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from scripts.prepare_visual_relation_blinded_review_v23 import (
    ARCHITECTURE as REVIEW_ARCHITECTURE,
    EXPECTED_PAIRED_AUDIT_SHA256,
    REVIEW_CASES,
    REVIEW_HELDOUT_CASES,
)
from scripts.train_visual_state_actuator import file_sha256


ARCHITECTURE = "visual-relation-circuit-v23-blinded-review-result"
OVERALL_REQUIRED = 44
HELDOUT_REQUIRED = 11


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score a completed V23 blinded A/B visual review."
    )
    parser.add_argument("--receipt", required=True)
    parser.add_argument("--answer-key", required=True)
    parser.add_argument("--responses", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--reviewer", required=True)
    return parser.parse_args()


def score_review(
    answer_rows: Sequence[dict[str, Any]],
    response_rows: Sequence[dict[str, Any]],
) -> dict[str, int | bool]:
    if len(answer_rows) != REVIEW_CASES or len(response_rows) != REVIEW_CASES:
        raise ValueError(f"V23 review requires exactly {REVIEW_CASES} responses")
    answers = {row["review_id"]: row for row in answer_rows}
    if len(answers) != REVIEW_CASES:
        raise ValueError("V23 answer key contains duplicate review IDs")
    responses: dict[str, str] = {}
    for row in response_rows:
        review_id = str(row.get("review_id", ""))
        response = str(row.get("response", "")).upper()
        if review_id in responses:
            raise ValueError(f"duplicate V23 response ID {review_id}")
        if review_id not in answers:
            raise ValueError(f"unknown V23 response ID {review_id}")
        if response not in {"A", "B"}:
            raise ValueError(f"V23 response for {review_id} must be A or B")
        responses[review_id] = response
    if set(responses) != set(answers):
        raise ValueError("V23 response set does not match the answer key")

    correct = 0
    heldout_correct = 0
    heldout_total = 0
    for review_id, answer in answers.items():
        is_correct = responses[review_id] == answer["correct_source"]
        correct += int(is_correct)
        if answer["heldout_combination"]:
            heldout_total += 1
            heldout_correct += int(is_correct)
    if heldout_total != REVIEW_HELDOUT_CASES:
        raise ValueError("V23 answer key has the wrong held-out count")
    passed = correct >= OVERALL_REQUIRED and heldout_correct >= HELDOUT_REQUIRED
    return {
        "review_cases": REVIEW_CASES,
        "correct": correct,
        "required_correct": OVERALL_REQUIRED,
        "heldout_cases": heldout_total,
        "heldout_correct": heldout_correct,
        "heldout_required_correct": HELDOUT_REQUIRED,
        "passed": passed,
    }


def main() -> None:
    args = parse_args()
    receipt = json.loads(Path(args.receipt).read_text(encoding="utf-8"))
    if receipt.get("architecture") != REVIEW_ARCHITECTURE:
        raise ValueError("V23 scorer requires a blinded-review receipt")
    if receipt.get("paired_audit_sha256") != EXPECTED_PAIRED_AUDIT_SHA256:
        raise ValueError("V23 review receipt has the wrong paired audit")
    if receipt.get("answer_key_sha256") != file_sha256(args.answer_key):
        raise ValueError("V23 answer key hash differs from its sealed receipt")
    if receipt.get("frozen_images_instantiated") != 0:
        raise ValueError("V23 review receipt broke the frozen seal")
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
        "blinded_review_passed": scores["passed"],
        "frozen_evaluation_permitted": scores["passed"],
        "frozen_images_instantiated": 0,
    }
    output = Path(args.out)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite V23 review result: {output}")
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

from __future__ import annotations

import json

import pytest
import torch

from ilm.visual_lm.visual_semantic_distillation_data import (
    VisualSemanticDistillationRenderConfig,
)
from ilm.visual_lm.visual_semantic_raster_data import VisualRasterRecord
from scripts import build_visual_path_alignment_paraphrases_v38 as builder


def record(identifier: str, prompt: str = "\u95ee\uff1a\u8bf4\u660e\u6c34\u7684\u7528\u9014\u3002") -> VisualRasterRecord:
    return VisualRasterRecord(
        identifier=identifier,
        prompt=prompt,
        answer="\u6c34\u53ef\u4ee5\u996e\u7528\u3002",
        language="zh",
        source="test",
        rights="test-only",
    )


def test_clean_paraphrase_normalizes_label_and_question_prefix() -> None:
    assert builder.clean_paraphrase("\u6539\u5199\uff1a  \u8bf7\u8bf4\u660e\u6c34\u7684\u7528\u9014\u3002 ") == (
        "\u95ee\uff1a\u8bf7\u8bf4\u660e\u6c34\u7684\u7528\u9014\u3002"
    )
    assert builder.clean_paraphrase("\u95ee\uff1a\u6c34\u6709\u4ec0\u4e48\u7528\u9014\uff1f") == (
        "\u95ee\uff1a\u6c34\u6709\u4ec0\u4e48\u7528\u9014\uff1f"
    )


def test_local_endpoint_rejects_nonlocal_or_wrong_paths() -> None:
    assert builder._validate_local_endpoint(
        "http://127.0.0.1:11434/api/chat", path="/api/chat"
    ) == "http://127.0.0.1:11434/api/chat"
    with pytest.raises(ValueError):
        builder._validate_local_endpoint(
            "https://127.0.0.1:11434/api/chat", path="/api/chat"
        )
    with pytest.raises(ValueError):
        builder._validate_local_endpoint(
            "http://example.com:11434/api/chat", path="/api/chat"
        )


def test_holdout_sources_are_converted_to_training_identifiers(tmp_path) -> None:
    path = tmp_path / "holdout.jsonl"
    path.write_text(
        json.dumps({"identifier": "external-source:17", "paraphrase": "x"})
        + "\n"
        + json.dumps({"identifier": "external-source:29", "paraphrase": "y"})
        + "\n",
        encoding="utf-8",
    )
    assert builder._holdout_source_identifiers(path) == {
        "alpaca-zh:17",
        "alpaca-zh:29",
    }


def test_candidate_order_is_deterministic_and_excludes_holdout() -> None:
    records = [record(f"alpaca-zh:{index}") for index in range(12)]
    excluded = {"alpaca-zh:3", "alpaca-zh:8"}
    first = builder.deterministic_candidates(records, excluded=excluded, seed=7)
    second = builder.deterministic_candidates(records, excluded=excluded, seed=7)
    assert [item.identifier for item in first] == [item.identifier for item in second]
    assert not ({item.identifier for item in first} & excluded)
    assert len(first) == 10


def test_validation_accepts_semantic_rewrite_and_rejects_copy(monkeypatch) -> None:
    source = record("alpaca-zh:4")
    rows = [
        {"identifier": source.identifier, "paraphrase": source.prompt},
        {
            "identifier": source.identifier,
            "paraphrase": "\u95ee\uff1a\u6c34\u53ef\u4ee5\u7528\u6765\u505a\u4ec0\u4e48\uff1f",
        },
    ]
    monkeypatch.setattr(builder, "paraphrase_fits", lambda *_args: True)

    original = torch.zeros(1_024)
    original[0] = 1
    paraphrase = torch.zeros(1_024)
    paraphrase[0] = 0.95
    paraphrase[1] = (1 - 0.95**2) ** 0.5
    answer = torch.zeros(1_024)
    answer[2] = 1
    embeddings = torch.stack((original, paraphrase, answer))
    monkeypatch.setattr(
        builder,
        "request_bge_embeddings",
        lambda *_args, **_kwargs: embeddings,
    )

    accepted, reasons = builder.validate_candidates(
        rows,
        {source.identifier: source},
        endpoint=builder.BGE_ENDPOINT,
        model=builder.BGE_MODEL,
        timeout=1,
        batch_size=8,
        minimum_cosine=0.82,
        render_config=VisualSemanticDistillationRenderConfig(),
    )
    assert reasons == {"exact-copy": 1}
    assert len(accepted) == 1
    assert accepted[0]["semantic_cosine"] == pytest.approx(0.95)
    assert accepted[0]["paraphrase_answer_cosine"] == pytest.approx(0.0)


def test_instruction_judge_is_resumable_and_excludes_failed_rows(
    monkeypatch, tmp_path
) -> None:
    records = {
        "alpaca-zh:1": record("alpaca-zh:1"),
        "alpaca-zh:2": record("alpaca-zh:2"),
    }
    rows = [
        {"identifier": "alpaca-zh:1", "paraphrase": "\u95ee\uff1a\u6c34\u80fd\u505a\u4ec0\u4e48\uff1f"},
        {"identifier": "alpaca-zh:2", "paraphrase": "\u95ee\uff1a\u6c34\u53ef\u4ee5\u996e\u7528\u3002"},
    ]
    calls: list[str] = []

    def fake_judgment(source, paraphrase, **_kwargs):
        calls.append(source.identifier)
        accepted = source.identifier.endswith(":1")
        return {
            "original_operation": "ask",
            "candidate_operation": "ask" if accepted else "answer",
            "candidate_is_instruction": accepted,
            "same_requested_operation": accepted,
            "preserves_all_inputs_and_conditions": True,
            "performs_or_answers_task": not accepted,
            "reason": "valid rewrite" if accepted else "answers task",
        }, {
            "prompt_eval_count": 1,
            "eval_count": 1,
            "total_duration_ns": 1,
        }

    monkeypatch.setattr(builder, "request_judgment", fake_judgment)
    journal = tmp_path / "judgments.jsonl"
    first, summary = builder.judge_candidates(
        rows,
        records,
        journal_path=journal,
        endpoint=builder.QWEN_ENDPOINT,
        model=builder.JUDGE_MODEL,
        seed=8,
        timeout=1,
    )
    assert [row["identifier"] for row in first] == ["alpaca-zh:1"]
    assert summary["passed"] == 1
    assert summary["failed"] == 1
    assert len(calls) == 2

    second, second_summary = builder.judge_candidates(
        rows,
        records,
        journal_path=journal,
        endpoint=builder.QWEN_ENDPOINT,
        model=builder.JUDGE_MODEL,
        seed=8,
        timeout=1,
    )
    assert [row["identifier"] for row in second] == ["alpaca-zh:1"]
    assert second_summary["journal_rows"] == 2
    assert len(calls) == 2


@pytest.mark.parametrize(
    ("original", "candidate", "expected_reason"),
    (
        (
            "\u9009\u62e9\u6b63\u786e\u7684\u8bcd\u586b\u7a7a\uff1a\u57ce\u5e02\u7684\u7a7a\u6c14\u6c61\u67d3\u6b63\u5728\u53d8\u5f97____\u3002",
            "\u95ee\uff1a\u57ce\u5e02\u7684\u7a7a\u6c14\u6c61\u67d3\u6b63\u5728\u53d8\u5f97\u8d8a\u6765\u8d8a\u4e25\u91cd\u3002",
            "missing-fill-or-complete-operation",
        ),
        (
            "\u75c5\u4eba\u53d1\u70e7\u3002\u8bf7\u66f4\u52a0\u7cbe\u786e\u5730\u7f16\u8f91\u8fd9\u53e5\u8bdd\u3002",
            "\u95ee\uff1a\u75c5\u4eba\u51fa\u73b0\u53d1\u70ed\u75c7\u72b6\u3002",
            "missing-rewrite-or-edit-operation",
        ),
        (
            "\u5b8c\u6210\u4ee5\u4e0b\u7c7b\u6bd4\uff1a'\u7231\u5f97\u50cf____\u4e00\u6837\u6709\u8010\u5fc3\u3002'",
            "\u95ee\uff1a\u7231\u5f97\u50cf\u732b\u4e00\u6837\u6709\u8010\u5fc3\u3002",
            "missing-fill-or-complete-operation",
        ),
    ),
)
def test_deterministic_operation_gate_rejects_executed_tasks(
    original, candidate, expected_reason
) -> None:
    assert builder.deterministic_operation_gate(original, candidate) == (
        False,
        expected_reason,
    )


def test_deterministic_operation_gate_preserves_valid_rewrite_request() -> None:
    assert builder.deterministic_operation_gate(
        "\u75c5\u4eba\u53d1\u70e7\u3002\u8bf7\u66f4\u52a0\u7cbe\u786e\u5730\u7f16\u8f91\u8fd9\u53e5\u8bdd\u3002",
        "\u95ee\uff1a\u8bf7\u5c06\u201c\u75c5\u4eba\u53d1\u70e7\u201d\u6539\u5199\u5f97\u66f4\u7cbe\u786e\u3002",
    ) == (True, "")


def test_judgment_decision_is_computed_from_all_subdecisions() -> None:
    valid = {
        "candidate_is_instruction": True,
        "same_requested_operation": True,
        "preserves_all_inputs_and_conditions": True,
        "performs_or_answers_task": False,
    }
    assert builder.judgment_passes(valid)
    assert not builder.judgment_passes(valid | {"performs_or_answers_task": True})
    assert not builder.judgment_passes(valid | {"same_requested_operation": False})


def test_numeric_gate_normalizes_arabic_and_chinese_quantities() -> None:
    assert builder.deterministic_numeric_gate(
        "\u95ee\uff1a12/3\u7684\u7ed3\u679c\u662f\u4ec0\u4e48\uff1f",
        "\u95ee\uff1a12\u9664\u4ee53\u7684\u7ed3\u679c\u662f\u591a\u5c11\uff1f",
    ) == (True, "")
    assert builder.deterministic_numeric_gate(
        "\u95ee\uff1a\u5199\u4e00\u9996\u4e94\u884c\u8bd7\uff0c\u6bcf\u884c\u56db\u4e2a\u97f3\u8282\u3002",
        "\u95ee\uff1a\u5199\u4e00\u9996\u4e94\u8a00\u4e94\u884c\u8bd7\u3002",
    ) == (False, "numeric-constraint-changed")


def test_adjudication_decision_rejects_any_non_equal_relation() -> None:
    valid = {
        "candidate_form": "request",
        "operation_relation": "equal",
        "quantity_unit_relation": "equal",
        "category_scope_relation": "equal",
        "named_input_relation": "not_applicable",
        "output_requirement_relation": "equal",
        "task_execution": "not_performed",
    }
    assert builder.adjudication_passes(valid)
    assert not builder.adjudication_passes(
        valid | {"category_scope_relation": "candidate_broader"}
    )
    assert not builder.adjudication_passes(
        valid | {"task_execution": "partly_performed"}
    )


def test_final_adjudicator_is_resumable_and_fails_closed(monkeypatch, tmp_path) -> None:
    records = {
        "alpaca-zh:1": record("alpaca-zh:1"),
        "alpaca-zh:2": record("alpaca-zh:2"),
    }
    rows = [
        {"identifier": "alpaca-zh:1", "paraphrase": "\u95ee\uff1a\u6c34\u80fd\u505a\u4ec0\u4e48\uff1f"},
        {"identifier": "alpaca-zh:2", "paraphrase": "\u95ee\uff1a\u6c34\u7684\u4e3b\u8981\u7528\u9014\u662f\u4ec0\u4e48\uff1f"},
    ]
    calls: list[str] = []

    def fake_adjudication(source, paraphrase, **_kwargs):
        calls.append(source.identifier)
        accepted = source.identifier.endswith(":1")
        relation = "equal" if accepted else "candidate_broader"
        return {
            "candidate_form": "request",
            "operation_relation": "equal",
            "quantity_unit_relation": "not_applicable",
            "category_scope_relation": relation,
            "named_input_relation": "equal",
            "output_requirement_relation": "equal",
            "task_execution": "not_performed",
            "original_requirements": ["ask"],
            "candidate_requirements": ["ask"],
            "reason": "valid" if accepted else "broader",
        }, {
            "prompt_eval_count": 1,
            "eval_count": 1,
            "total_duration_ns": 1,
        }

    monkeypatch.setattr(builder, "request_adjudication", fake_adjudication)
    journal = tmp_path / "adjudications.jsonl"
    first, summary = builder.adjudicate_candidates(
        rows,
        records,
        journal_path=journal,
        endpoint=builder.QWEN_ENDPOINT,
        model=builder.ADJUDICATOR_MODEL,
        seed=9,
        timeout=1,
    )
    assert [row["identifier"] for row in first] == ["alpaca-zh:1"]
    assert summary["passed"] == 1
    assert summary["failed"] == 1
    assert len(calls) == 2

    second, second_summary = builder.adjudicate_candidates(
        rows,
        records,
        journal_path=journal,
        endpoint=builder.QWEN_ENDPOINT,
        model=builder.ADJUDICATOR_MODEL,
        seed=9,
        timeout=1,
    )
    assert [row["identifier"] for row in second] == ["alpaca-zh:1"]
    assert second_summary["journal_rows"] == 2
    assert len(calls) == 2


def test_adversarial_confirmation_is_separate_and_resumable(
    monkeypatch, tmp_path
) -> None:
    assert builder.confirmator_protocol_sha256() != builder.adjudicator_protocol_sha256()
    records = {"alpaca-zh:1": record("alpaca-zh:1")}
    rows = [
        {
            "identifier": "alpaca-zh:1",
            "paraphrase": "\u95ee\uff1a\u6c34\u80fd\u505a\u4ec0\u4e48\uff1f",
            "constraint_adjudicator": "pass",
        }
    ]
    calls = 0

    def fake_confirmation(source, paraphrase, **_kwargs):
        nonlocal calls
        calls += 1
        return {
            "candidate_form": "request",
            "operation_relation": "equal",
            "quantity_unit_relation": "not_applicable",
            "category_scope_relation": "equal",
            "named_input_relation": "equal",
            "output_requirement_relation": "equal",
            "task_execution": "not_performed",
            "original_requirements": ["ask"],
            "candidate_requirements": ["ask"],
            "reason": "no counterexample",
        }, {
            "prompt_eval_count": 1,
            "eval_count": 1,
            "total_duration_ns": 1,
        }

    monkeypatch.setattr(builder, "request_confirmation", fake_confirmation)
    journal = tmp_path / "confirmations.jsonl"
    first, summary = builder.confirm_candidates(
        rows,
        records,
        journal_path=journal,
        endpoint=builder.QWEN_ENDPOINT,
        model=builder.ADJUDICATOR_MODEL,
        seed=10,
        timeout=1,
    )
    assert first[0]["adversarial_confirmation"] == "pass"
    assert summary["passed"] == 1
    assert calls == 1

    second, second_summary = builder.confirm_candidates(
        rows,
        records,
        journal_path=journal,
        endpoint=builder.QWEN_ENDPOINT,
        model=builder.ADJUDICATOR_MODEL,
        seed=10,
        timeout=1,
    )
    assert second[0]["identifier"] == "alpaca-zh:1"
    assert second_summary["journal_rows"] == 1
    assert calls == 1

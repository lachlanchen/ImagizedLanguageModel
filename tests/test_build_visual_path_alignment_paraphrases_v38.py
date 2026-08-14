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

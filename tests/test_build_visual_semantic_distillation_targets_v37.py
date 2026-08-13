from __future__ import annotations

import torch
import torch.nn.functional as F

from scripts.build_visual_semantic_distillation_targets_v37 import (
    retrieval_metrics,
    synthetic_embeddings,
    transform_targets,
    validate_development_target_sanity,
    validate_local_embed_endpoint,
)


def test_endpoint_validation_accepts_only_loopback_ollama() -> None:
    assert (
        validate_local_embed_endpoint("http://127.0.0.1:11434/api/embed")
        == "http://127.0.0.1:11434/api/embed"
    )
    for invalid in (
        "https://127.0.0.1:11434/api/embed",
        "http://localhost:11434/api/embed",
        "http://127.0.0.1:11434/api/embeddings",
        "http://user:secret@127.0.0.1:11434/api/embed",
        "http://127.0.0.1:11434/api/embed?model=bge-m3",
    ):
        try:
            validate_local_embed_endpoint(invalid)
        except ValueError:
            pass
        else:
            raise AssertionError(f"accepted unsafe endpoint: {invalid}")


def test_target_transform_uses_joint_train_mean() -> None:
    generator = torch.Generator().manual_seed(37)
    prompt = F.normalize(torch.randn(32, 1024, generator=generator), dim=-1)
    answer = F.normalize(torch.randn(32, 1024, generator=generator), dim=-1)

    transformed_prompt, transformed_answer, mean = transform_targets(
        prompt,
        answer,
        teacher_mean=None,
    )

    assert torch.allclose(mean, torch.cat((prompt, answer)).mean(dim=0))
    assert torch.allclose(transformed_prompt.norm(dim=-1), torch.ones(32), atol=1e-5)
    assert torch.allclose(transformed_answer.norm(dim=-1), torch.ones(32), atol=1e-5)


def test_development_reuses_train_mean() -> None:
    prompt = synthetic_embeddings(("问题一", "问题二"))
    answer = synthetic_embeddings(("回答一", "回答二"))
    train_mean = torch.linspace(-0.01, 0.01, 1024)

    transformed_prompt, transformed_answer, returned_mean = transform_targets(
        prompt,
        answer,
        teacher_mean=train_mean,
    )

    assert torch.equal(returned_mean, train_mean)
    assert torch.allclose(transformed_prompt.norm(dim=-1), torch.ones(2), atol=1e-5)
    assert torch.allclose(transformed_answer.norm(dim=-1), torch.ones(2), atol=1e-5)


def test_retrieval_metrics_and_sanity_gate() -> None:
    generator = torch.Generator().manual_seed(5)
    answer = F.normalize(torch.randn(96, 1024, generator=generator), dim=-1)
    prompt = F.normalize(
        answer + 0.01 * torch.randn(96, 1024, generator=generator), dim=-1
    )
    metrics = retrieval_metrics(prompt, answer)

    assert metrics["top1"] == 1.0
    assert metrics["top5"] == 1.0
    assert metrics["mrr"] == 1.0
    assert metrics["answer_effective_rank"] > 70
    validate_development_target_sanity(metrics)

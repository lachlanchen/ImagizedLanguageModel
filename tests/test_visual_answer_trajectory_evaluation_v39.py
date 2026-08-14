from __future__ import annotations

from dataclasses import asdict

import torch
import torch.nn.functional as F

from ilm.visual_lm.visual_answer_trajectory import (
    V39_ARCHITECTURE,
    VisualAnswerTrajectoryConfig,
    VisualAnswerTrajectoryModel,
)
from ilm.visual_lm.visual_answer_trajectory_data import VisualAnswerTrajectoryRecord
from ilm.visual_lm.visual_answer_trajectory_evaluation import (
    VisualAnswerTrajectoryEvaluationOutputs,
    active_segment_geometry,
    indexed_retrieval_metrics,
    output_effective_rank,
    stop_and_length_metrics,
    trajectory_consistency_metrics,
    trajectory_content_metrics,
)
from ilm.visual_lm.visual_semantic_distillation_data import (
    VisualSemanticDistillationRenderConfig,
)
from scripts.eval_visual_answer_trajectory_v39 import (
    VIEW_NAMES,
    VisualAnswerTrajectoryEvaluationDataset,
    load_checkpoint_model,
    operation_bucket,
)


DIMENSION = 32


def outputs(records: int = 4) -> VisualAnswerTrajectoryEvaluationOutputs:
    generator = torch.Generator().manual_seed(39)
    read = F.normalize(torch.randn(records, DIMENSION, generator=generator), dim=-1)
    segments = F.normalize(
        torch.randn(records, 16, DIMENSION, generator=generator),
        dim=-1,
    )
    stop = torch.full((records, 16), -4.0)
    stop[:, 1] = 4.0
    active = torch.cat(
        (
            torch.ones(records, 1),
            (1 - stop.sigmoid())[:, :-1].cumprod(dim=1),
        ),
        dim=1,
    )
    return VisualAnswerTrajectoryEvaluationOutputs(
        read_state=read,
        baseline_answer_state=read.clone(),
        answer_state=read.clone(),
        stage1_answer_state=read.clone(),
        segment_states=segments,
        stage1_segment_states=segments.clone(),
        stop_logits=stop,
        active_probabilities=active,
        lengths=torch.full((records, 16), 5.0),
    )


def test_blockwise_retrieval_recovers_exact_labels() -> None:
    candidate = F.normalize(torch.randn(24, DIMENSION), dim=-1)
    labels = torch.tensor([2, 10, 17, 23])

    metrics = indexed_retrieval_metrics(
        candidate[labels],
        candidate,
        labels,
        block_size=2,
    )

    assert metrics["top1"] == 1.0
    assert metrics["mrr"] == 1.0
    assert metrics["positive_cosine"] > 0.999


def test_segment_geometry_and_content_metrics_exclude_padding() -> None:
    offsets = torch.tensor([0, 2, 5, 6, 8], dtype=torch.long)
    count, mask, labels = active_segment_geometry(offsets)
    target = F.normalize(torch.randn(8, DIMENSION), dim=-1)
    predicted = torch.zeros(4, 16, DIMENSION)
    predicted[mask] = target

    metrics = trajectory_content_metrics(predicted, target, offsets, block_size=3)

    assert count.tolist() == [2, 3, 1, 2]
    assert mask.sum() == 8
    assert labels.tolist() == list(range(8))
    assert metrics["retrieval"]["top1"] == 1.0
    assert metrics["paired_cosine"] > 0.999
    assert metrics["paired_beats_permuted"] == 1.0


def test_stop_length_and_consistency_metrics_are_finite() -> None:
    value = outputs()
    offsets = torch.tensor([0, 2, 4, 6, 8], dtype=torch.long)
    target_lengths = torch.full((8,), 5.0)

    stop = stop_and_length_metrics(value, offsets, target_lengths)
    consistency = trajectory_consistency_metrics(value, value, offsets)

    assert stop["count_accuracy"] == 1.0
    assert stop["length_mae"] == 0.0
    assert consistency["read_cosine"] > 0.999
    assert consistency["answer_cosine"] > 0.999
    assert consistency["segment_cosine"] > 0.999


def test_output_effective_rank_uses_only_active_states() -> None:
    value = outputs(records=16)
    offsets = torch.arange(0, 33, 2, dtype=torch.long)

    metrics = output_effective_rank(value, offsets, maximum_samples=24)

    assert metrics["answer_samples"] == 16
    assert metrics["segment_samples"] == 24
    assert metrics["answer_effective_rank"] > 1
    assert metrics["segment_effective_rank"] > 1


def tiny_config() -> VisualAnswerTrajectoryConfig:
    return VisualAnswerTrajectoryConfig(
        reader_hidden_size=64,
        reader_layers=1,
        reader_heads=4,
        reader_intermediate_size=128,
        reader_dropout=0.0,
        projection_hidden_size=96,
        semantic_dim=64,
        projection_dropout=0.0,
        answer_hidden_size=32,
        planner_hidden_size=64,
        planner_layers=1,
        planner_heads=4,
        planner_intermediate_size=128,
        planner_dropout=0.0,
        length_hidden_size=16,
    )


def evaluation_record(identifier: str, prompt: str) -> VisualAnswerTrajectoryRecord:
    return VisualAnswerTrajectoryRecord(
        identifier=identifier,
        prompt=prompt,
        answer="这是一个可见答案。",
        segments=("这是一个可见答案。",),
        language="zh",
        source="unit",
        rights="test",
    )


def test_evaluation_dataset_builds_five_image_only_views() -> None:
    dataset = VisualAnswerTrajectoryEvaluationDataset(
        (
            evaluation_record("one", "问：解释汉字语言。"),
            evaluation_record("two", "问：说明视觉模型。"),
        ),
        render_config=VisualSemanticDistillationRenderConfig(augment=False),
    )

    item = dataset[0]

    assert all(f"{name}_pixels" in item for name in VIEW_NAMES)
    assert all(f"{name}_mask" in item for name in VIEW_NAMES)
    assert item["blank_pixels"].shape == (3, 16, 1024)
    assert item["blank_mask"].sum() == 0
    assert not torch.equal(item["canonical_pixels"], item["shuffled_pixels"])
    assert set(item) == {
        *(f"{name}_pixels" for name in VIEW_NAMES),
        *(f"{name}_mask" for name in VIEW_NAMES),
        "script_changed",
    }


def test_standalone_checkpoint_loader_preserves_ema_route(tmp_path) -> None:
    config = tiny_config()
    model = VisualAnswerTrajectoryModel(config)
    checkpoint = tmp_path / "student_ema.pt"
    torch.save(
        {
            "architecture": V39_ARCHITECTURE,
            "weight_route": "all-parameter-ema",
            "model_config": asdict(config),
            "model": model.state_dict(),
            "contains_target_tensors": False,
            "contains_teacher_model": False,
            "contains_candidate_tensors": False,
            "contains_source_language_strings": False,
        },
        checkpoint,
    )

    loaded, _payload, receipt = load_checkpoint_model(
        checkpoint,
        device=torch.device("cpu"),
        raw_weights=False,
    )

    assert receipt["weight_route"] == "all-parameter-ema"
    assert receipt["finite_model_state"] is True
    assert receipt["tensor_boundary"] is True
    assert loaded.training is False
    assert not any(parameter.requires_grad for parameter in loaded.parameters())


def test_operation_bucket_is_host_side_and_deterministic() -> None:
    assert operation_bucket("问：请翻译这句话。") == "translate"
    assert operation_bucket("问：概括这段文章。") == "summarize"
    assert operation_bucket("问：判断这个说法。") == "classify"
    assert operation_bucket("问：写一首短诗。") == "generate"
    assert operation_bucket("问：解释这个现象。") == "explain"
    assert operation_bucket("问：你好。") == "other"

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from ilm.visual_lm.visual_answer_trajectory_data import VisualAnswerTrajectoryRecord
from scripts.build_visual_answer_trajectory_targets_v39 import (
    EMBEDDING_WORK_ARCHITECTURE,
    LENGTH_WORK_ARCHITECTURE,
    VisualAnswerTrajectoryDocumentLayout,
    _reset_work_directory,
    acquire_work_lock,
    bounded_document_batch_end,
    center_normalize_memmap_region,
    embed_documents_resumable,
    measure_lengths_resumable,
    validate_local_embed_endpoint,
)
from scripts.build_visual_semantic_distillation_targets_v37 import synthetic_embeddings


def records() -> tuple[VisualAnswerTrajectoryRecord, ...]:
    return (
        VisualAnswerTrajectoryRecord(
            identifier="r0",
            prompt="问：第一题",
            answer="第一答。继续。",
            segments=("第一答。", "继续。"),
            language="zh",
            source="unit",
            rights="test",
        ),
        VisualAnswerTrajectoryRecord(
            identifier="r1",
            prompt="问：第二题",
            answer="第二答。",
            segments=("第二答。",),
            language="zh",
            source="unit",
            rights="test",
        ),
    )


def embedding_identity(layout: VisualAnswerTrajectoryDocumentLayout) -> dict[str, object]:
    return {
        "architecture": EMBEDDING_WORK_ARCHITECTURE,
        "split": "train",
        "layout_sha256": layout.sha256,
        "rows": layout.document_count,
        "dimension": 1024,
        "dtype": "float16",
    }


def test_document_layout_has_stable_contiguous_regions() -> None:
    layout = VisualAnswerTrajectoryDocumentLayout.from_records(records())
    repeated = VisualAnswerTrajectoryDocumentLayout.from_records(records())

    assert layout.identifiers == ("r0", "r1")
    assert layout.segment_offsets == (0, 2, 3)
    assert layout.documents == (
        "问：第一题",
        "问：第二题",
        "第一答。继续。",
        "第二答。",
        "第一答。",
        "继续。",
        "第二答。",
    )
    assert layout.prompt_slice == slice(0, 2)
    assert layout.answer_slice == slice(2, 4)
    assert layout.segment_slice == slice(4, 7)
    assert layout.sha256 == repeated.sha256


def test_character_bounded_batches_always_make_progress() -> None:
    documents = ("短", "中" * 8, "很" * 50)

    assert bounded_document_batch_end(
        documents,
        0,
        maximum_documents=8,
        maximum_characters=5,
    ) == 1
    assert bounded_document_batch_end(
        documents,
        1,
        maximum_documents=8,
        maximum_characters=5,
    ) == 2


def test_embedding_memmap_resumes_after_a_callback_failure(tmp_path: Path) -> None:
    layout = VisualAnswerTrajectoryDocumentLayout.from_records(records())
    data_path = tmp_path / "raw.f16"
    progress_path = tmp_path / "raw.progress.json"
    calls = 0

    def fail_on_second_call(texts):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("interrupted")
        return synthetic_embeddings(texts)

    with pytest.raises(RuntimeError, match="interrupted"):
        embed_documents_resumable(
            layout.documents,
            data_path=data_path,
            progress_path=progress_path,
            identity=embedding_identity(layout),
            embed=fail_on_second_call,
            maximum_documents=2,
            maximum_characters=1_000,
        )

    resumed_texts: list[str] = []

    def resume(texts):
        resumed_texts.extend(texts)
        return synthetic_embeddings(texts)

    state = embed_documents_resumable(
        layout.documents,
        data_path=data_path,
        progress_path=progress_path,
        identity=embedding_identity(layout),
        embed=resume,
        maximum_documents=2,
        maximum_characters=1_000,
    )

    assert resumed_texts == list(layout.documents[2:])
    assert state["completed_rows"] == layout.document_count
    array = np.memmap(data_path, mode="r", dtype=np.float16, shape=(7, 1024))
    assert np.isfinite(array).all()


def test_resumable_identity_mismatch_is_rejected(tmp_path: Path) -> None:
    layout = VisualAnswerTrajectoryDocumentLayout.from_records(records())
    data_path = tmp_path / "raw.f16"
    progress_path = tmp_path / "raw.progress.json"
    embed_documents_resumable(
        layout.documents,
        data_path=data_path,
        progress_path=progress_path,
        identity=embedding_identity(layout),
        embed=synthetic_embeddings,
        maximum_documents=8,
        maximum_characters=10_000,
    )
    changed = embedding_identity(layout) | {"layout_sha256": "0" * 64}

    with pytest.raises(ValueError, match="identity changed"):
        embed_documents_resumable(
            layout.documents,
            data_path=data_path,
            progress_path=progress_path,
            identity=changed,
            embed=synthetic_embeddings,
            maximum_documents=8,
            maximum_characters=10_000,
        )


def test_length_memmap_resumes_and_preserves_flat_order(tmp_path: Path) -> None:
    layout = VisualAnswerTrajectoryDocumentLayout.from_records(records())
    data_path = tmp_path / "lengths.u8"
    progress_path = tmp_path / "lengths.progress.json"
    identity = {
        "architecture": LENGTH_WORK_ARCHITECTURE,
        "split": "train",
        "layout_sha256": layout.sha256,
        "rows": layout.segment_count,
        "dimension": 1,
        "dtype": "uint8",
    }
    starts: list[int] = []

    def measure(texts, start):
        starts.append(start)
        return tuple(range(start + 1, start + 1 + len(texts)))

    state = measure_lengths_resumable(
        layout.segment_documents,
        data_path=data_path,
        progress_path=progress_path,
        identity=identity,
        measure=measure,
        batch_size=2,
    )
    second = measure_lengths_resumable(
        layout.segment_documents,
        data_path=data_path,
        progress_path=progress_path,
        identity=identity,
        measure=lambda _texts, _start: (_ for _ in ()).throw(AssertionError()),
        batch_size=2,
    )

    assert starts == [0, 2]
    assert state == second
    lengths = np.memmap(data_path, mode="r", dtype=np.uint8, shape=(3, 1))
    assert lengths[:, 0].tolist() == [1, 2, 3]


def test_chunked_centering_produces_normalized_regions(tmp_path: Path) -> None:
    documents = ("甲", "乙", "丙", "丁", "戊")
    path = tmp_path / "raw.f16"
    vectors = synthetic_embeddings(documents).half().numpy()
    array = np.memmap(path, mode="w+", dtype=np.float16, shape=vectors.shape)
    array[:] = vectors
    array.flush()
    del array

    targets = center_normalize_memmap_region(
        path,
        start=1,
        stop=5,
        total_rows=5,
        teacher_mean=torch.linspace(-0.01, 0.01, 1024),
        chunk_size=2,
    )

    assert targets.shape == (4, 1024)
    assert targets.dtype == torch.float16
    assert torch.allclose(targets.float().norm(dim=-1), torch.ones(4), atol=5e-4)


def test_endpoint_boundary_remains_loopback_only() -> None:
    assert (
        validate_local_embed_endpoint("http://127.0.0.1:11434/api/embed")
        == "http://127.0.0.1:11434/api/embed"
    )
    with pytest.raises(ValueError):
        validate_local_embed_endpoint("http://localhost:11434/api/embed")


def test_work_lock_rejects_a_concurrent_builder(tmp_path: Path) -> None:
    work = tmp_path / "train.work"
    first = acquire_work_lock(work)
    try:
        with pytest.raises(RuntimeError, match="already active"):
            acquire_work_lock(work)
    finally:
        first.close()

    second = acquire_work_lock(work)
    second.close()


def test_reset_guard_allows_only_named_work_directories(tmp_path: Path) -> None:
    work = tmp_path / "train.work"
    work.mkdir()
    (work / "partial").write_text("generated", encoding="utf-8")

    _reset_work_directory(work)

    assert not work.exists()
    with pytest.raises(ValueError, match="unsafe"):
        _reset_work_directory(tmp_path)

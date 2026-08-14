#!/usr/bin/env python3
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import shutil
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from multiprocessing import get_context
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from ilm.visual_lm.visual_answer_trajectory_data import (
    V39_DEVELOPMENT_FONT,
    V39_TARGET_ARCHITECTURE,
    VisualAnswerTrajectoryRecord,
    canonical_v39_text_length,
    load_v39_instruction_records,
    select_v39_instruction_records,
)
from ilm.visual_lm.visual_answer_trajectory_training import (
    VisualAnswerTrajectoryTargetBank,
)
from ilm.visual_lm.visual_semantic_distillation import file_sha256
from ilm.visual_lm.visual_semantic_distillation_data import (
    V37_SEMANTIC_DIM,
    V37_TARGET_ARCHITECTURE,
    VisualSemanticDistillationRenderConfig,
)
from ilm.visual_lm.visual_semantic_distillation_training import (
    VisualSemanticDistillationTargetBank,
    centered_effective_rank,
)
from scripts.build_visual_semantic_distillation_targets_v37 import (
    BGE_MANIFEST_SHA256,
    BGE_MODEL,
    BGE_MODEL_SHA256,
    atomic_torch_save,
    atomic_write_json,
    request_bge_embeddings,
    retrieval_metrics,
    synthetic_embeddings,
    unload_bge,
    validate_local_embed_endpoint,
    verify_bge_artifact,
)


EXPERIMENT = V39_TARGET_ARCHITECTURE
RESEARCH_DOCUMENT = "references/visual_answer_trajectory_v39_research.md"
EXPECTED_RESEARCH_SHA256 = (
    "c5ff3500905f3d8ad3d16b7bdb5584a86748cbab9ca0de39a860646f7bce9ef4"
)
DEFAULT_INSTRUCTION_MANIFEST = "data/raw/alpaca_zh.json"
EXPECTED_INSTRUCTION_SHA256 = (
    "6fcb98c6d79691d1f9a88ef513335da9124e7fdeef5103343f5d9f9a6e8f4903"
)
DEFAULT_V37_TRAIN_BANK = "artifacts/visual_semantic_distillation_v37_targets/train.pt"
EXPECTED_V37_TRAIN_BANK_SHA256 = (
    "3cd73f0818d65fd45c7700470cd010e292f359eed5aa3e62859bdf50d301711d"
)
DEFAULT_ENDPOINT = "http://127.0.0.1:11434/api/embed"
DEFAULT_TEACHER_MANIFEST = (
    "../LocalLLM/.local/models/ollama/manifests/"
    "registry.ollama.ai/library/bge-m3/latest"
)
DEFAULT_TEACHER_MODEL_LAYER = (
    "../LocalLLM/.local/models/ollama/blobs/"
    "sha256-daec91ffb5dd0c27411bd71f29932917c49cf529a641d0168496c3a501e3062c"
)
EXPECTED_FULL_GEOMETRY = {
    "train": {"records": 44_637, "segments": 277_815, "documents": 367_089},
    "development": {"records": 1_544, "segments": 9_747, "documents": 12_835},
}
SOURCE_FILES = (
    "ilm/visual_lm/visual_answer_trajectory_data.py",
    "ilm/visual_lm/visual_answer_trajectory_training.py",
    "scripts/build_visual_answer_trajectory_targets_v39.py",
    "scripts/build_visual_semantic_distillation_targets_v37.py",
)
EMBEDDING_WORK_ARCHITECTURE = "visual-answer-trajectory-embedding-work-v39"
LENGTH_WORK_ARCHITECTURE = "visual-answer-trajectory-length-work-v39"


@dataclass(frozen=True)
class VisualAnswerTrajectoryDocumentLayout:
    identifiers: tuple[str, ...]
    documents: tuple[str, ...]
    segment_offsets: tuple[int, ...]
    sha256: str

    def __post_init__(self) -> None:
        records = len(self.identifiers)
        if records < 1 or len(set(self.identifiers)) != records:
            raise ValueError("V39 document layout identifiers must be unique")
        if len(self.segment_offsets) != records + 1:
            raise ValueError("V39 document segment offsets do not align")
        if self.segment_offsets[0] != 0:
            raise ValueError("V39 document segment offsets must start at zero")
        counts = [
            stop - start
            for start, stop in zip(self.segment_offsets, self.segment_offsets[1:])
        ]
        if any(not 1 <= count <= 16 for count in counts):
            raise ValueError("V39 document records require one to sixteen spans")
        if len(self.documents) != 2 * records + self.segment_count:
            raise ValueError("V39 document regions do not cover the corpus")
        if len(self.sha256) != 64:
            raise ValueError("V39 document layout digest is invalid")

    @property
    def record_count(self) -> int:
        return len(self.identifiers)

    @property
    def segment_count(self) -> int:
        return self.segment_offsets[-1]

    @property
    def document_count(self) -> int:
        return len(self.documents)

    @property
    def prompt_slice(self) -> slice:
        return slice(0, self.record_count)

    @property
    def answer_slice(self) -> slice:
        return slice(self.record_count, 2 * self.record_count)

    @property
    def segment_slice(self) -> slice:
        return slice(2 * self.record_count, self.document_count)

    @property
    def segment_documents(self) -> tuple[str, ...]:
        return self.documents[self.segment_slice]

    @classmethod
    def from_records(
        cls,
        records: Sequence[VisualAnswerTrajectoryRecord],
    ) -> VisualAnswerTrajectoryDocumentLayout:
        if not records:
            raise ValueError("V39 cannot lay out an empty record sequence")
        identifiers = tuple(record.identifier for record in records)
        prompts = tuple(record.prompt for record in records)
        answers = tuple(record.answer for record in records)
        segments = tuple(segment for record in records for segment in record.segments)
        offsets = [0]
        for record in records:
            offsets.append(offsets[-1] + len(record.segments))
        digest = hashlib.sha256()
        _update_framed_digest(digest, EXPERIMENT)
        for record in records:
            _update_framed_digest(digest, record.identifier)
            _update_framed_digest(digest, record.prompt)
            _update_framed_digest(digest, record.answer)
            _update_framed_digest(digest, str(len(record.segments)))
            for segment in record.segments:
                _update_framed_digest(digest, segment)
        return cls(
            identifiers=identifiers,
            documents=prompts + answers + segments,
            segment_offsets=tuple(offsets),
            sha256=digest.hexdigest(),
        )


def _update_framed_digest(digest: Any, value: str) -> None:
    encoded = value.encode("utf-8")
    digest.update(len(encoded).to_bytes(8, "big"))
    digest.update(encoded)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build resumable, hash-pinned V39 answer-trajectory targets."
    )
    parser.add_argument("--instruction-manifest", default=DEFAULT_INSTRUCTION_MANIFEST)
    parser.add_argument("--split", choices=("train", "development"), default="train")
    parser.add_argument("--out")
    parser.add_argument("--work-dir")
    parser.add_argument("--v37-train-bank", default=DEFAULT_V37_TRAIN_BANK)
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    parser.add_argument("--model", default=BGE_MODEL)
    parser.add_argument("--teacher-manifest", default=DEFAULT_TEACHER_MANIFEST)
    parser.add_argument("--teacher-model-layer", default=DEFAULT_TEACHER_MODEL_LAYER)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--maximum-batch-characters", type=int, default=8_192)
    parser.add_argument("--length-batch-size", type=int, default=256)
    parser.add_argument("--length-workers", type=int, default=4)
    parser.add_argument("--assembly-chunk-size", type=int, default=8_192)
    parser.add_argument("--diagnostic-samples", type=int, default=512)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--maximum-records", type=int, default=0)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--reset-work", action="store_true")
    return parser.parse_args()


def default_output(split: str) -> Path:
    return Path("artifacts/visual_answer_trajectory_v39_targets") / f"{split}.pt"


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"V39 progress file is not an object: {path}")
    return value


def _validate_progress_identity(
    state: Mapping[str, Any],
    identity: Mapping[str, Any],
) -> None:
    changed = [key for key, value in identity.items() if state.get(key) != value]
    if changed:
        raise ValueError(f"V39 resumable work identity changed: {changed}")


def _open_resumable_memmap(
    *,
    data_path: Path,
    progress_path: Path,
    rows: int,
    dimension: int,
    dtype: np.dtype[Any],
    identity: Mapping[str, Any],
    completed_key: str,
    request_key: str,
) -> tuple[np.memmap, dict[str, Any]]:
    if rows < 1 or dimension < 1:
        raise ValueError("V39 resumable work shape is invalid")
    data_path.parent.mkdir(parents=True, exist_ok=True)
    expected_bytes = rows * dimension * np.dtype(dtype).itemsize
    if progress_path.exists():
        if not data_path.is_file():
            raise FileNotFoundError("V39 progress exists without its data file")
        state = _read_json(progress_path)
        _validate_progress_identity(state, identity)
        if data_path.stat().st_size != expected_bytes:
            raise ValueError("V39 resumable data file has another byte count")
        completed = int(state.get(completed_key, -1))
        requests = int(state.get(request_key, -1))
        if not 0 <= completed <= rows or requests < 0:
            raise ValueError("V39 resumable progress counters are invalid")
        return np.memmap(data_path, mode="r+", dtype=dtype, shape=(rows, dimension)), state
    if data_path.exists():
        raise FileExistsError("V39 resumable data exists without an identity receipt")
    array = np.memmap(data_path, mode="w+", dtype=dtype, shape=(rows, dimension))
    array.flush()
    state = dict(identity) | {completed_key: 0, request_key: 0}
    atomic_write_json(state, progress_path)
    return array, state


def bounded_document_batch_end(
    documents: Sequence[str],
    start: int,
    *,
    maximum_documents: int,
    maximum_characters: int,
) -> int:
    if not 0 <= start < len(documents):
        raise ValueError("V39 document batch start is invalid")
    if min(maximum_documents, maximum_characters) < 1:
        raise ValueError("V39 document batch limits must be positive")
    stop = start
    characters = 0
    while stop < len(documents) and stop - start < maximum_documents:
        next_characters = len(documents[stop])
        if stop > start and characters + next_characters > maximum_characters:
            break
        characters += next_characters
        stop += 1
    return max(start + 1, stop)


def embed_documents_resumable(
    documents: Sequence[str],
    *,
    data_path: Path,
    progress_path: Path,
    identity: Mapping[str, Any],
    embed: Callable[[Sequence[str]], torch.Tensor],
    maximum_documents: int,
    maximum_characters: int,
    progress: Callable[[Mapping[str, Any]], None] | None = None,
) -> dict[str, Any]:
    array, state = _open_resumable_memmap(
        data_path=data_path,
        progress_path=progress_path,
        rows=len(documents),
        dimension=V37_SEMANTIC_DIM,
        dtype=np.dtype("float16"),
        identity=identity,
        completed_key="completed_rows",
        request_key="embedding_requests",
    )
    completed = int(state["completed_rows"])
    requests = int(state["embedding_requests"])
    while completed < len(documents):
        stop = bounded_document_batch_end(
            documents,
            completed,
            maximum_documents=maximum_documents,
            maximum_characters=maximum_characters,
        )
        vectors = embed(documents[completed:stop]).detach().float().cpu()
        if vectors.shape != (stop - completed, V37_SEMANTIC_DIM):
            raise RuntimeError("V39 embedding callback returned another shape")
        if not bool(torch.isfinite(vectors).all()):
            raise FloatingPointError("V39 embedding callback returned non-finite data")
        norms = vectors.norm(dim=-1)
        if not bool((norms > 1e-6).all()):
            raise RuntimeError("V39 embedding callback returned a zero vector")
        vectors = F.normalize(vectors, dim=-1)
        array[completed:stop] = vectors.numpy().astype(np.float16, copy=False)
        array.flush()
        completed = stop
        requests += 1
        state = dict(state) | {
            "completed_rows": completed,
            "embedding_requests": requests,
            "complete": completed == len(documents),
        }
        atomic_write_json(state, progress_path)
        if progress is not None:
            progress(state)
    del array
    return state


def measure_lengths_resumable(
    segments: Sequence[str],
    *,
    data_path: Path,
    progress_path: Path,
    identity: Mapping[str, Any],
    measure: Callable[[Sequence[str], int], Sequence[float]],
    batch_size: int,
    progress: Callable[[Mapping[str, Any]], None] | None = None,
) -> dict[str, Any]:
    array, state = _open_resumable_memmap(
        data_path=data_path,
        progress_path=progress_path,
        rows=len(segments),
        dimension=1,
        dtype=np.dtype("uint8"),
        identity=identity,
        completed_key="completed_rows",
        request_key="measurement_batches",
    )
    completed = int(state["completed_rows"])
    batches = int(state["measurement_batches"])
    while completed < len(segments):
        stop = min(len(segments), completed + batch_size)
        values = tuple(float(value) for value in measure(segments[completed:stop], completed))
        if len(values) != stop - completed:
            raise RuntimeError("V39 visual-length callback returned another shape")
        if any(not math.isfinite(value) or not 1 <= value <= 64 for value in values):
            raise ValueError("V39 visual-length callback returned an invalid length")
        rounded = np.asarray([round(value) for value in values], dtype=np.uint8)
        if any(abs(float(value) - int(result)) > 1e-6 for value, result in zip(values, rounded)):
            raise ValueError("V39 canonical visual lengths must be integral")
        array[completed:stop, 0] = rounded
        array.flush()
        completed = stop
        batches += 1
        state = dict(state) | {
            "completed_rows": completed,
            "measurement_batches": batches,
            "complete": completed == len(segments),
        }
        atomic_write_json(state, progress_path)
        if progress is not None:
            progress(state)
    del array
    return state


def center_normalize_memmap_region(
    path: Path,
    *,
    start: int,
    stop: int,
    total_rows: int,
    teacher_mean: torch.Tensor,
    chunk_size: int,
) -> torch.Tensor:
    if not 0 <= start < stop <= total_rows or chunk_size < 1:
        raise ValueError("V39 target assembly region is invalid")
    if teacher_mean.shape != (V37_SEMANTIC_DIM,):
        raise ValueError("V39 teacher mean has another shape")
    raw = np.memmap(
        path,
        mode="r",
        dtype=np.float16,
        shape=(total_rows, V37_SEMANTIC_DIM),
    )
    output = torch.empty(stop - start, V37_SEMANTIC_DIM, dtype=torch.float16)
    mean = teacher_mean.float().cpu()
    for absolute_start in range(start, stop, chunk_size):
        absolute_stop = min(stop, absolute_start + chunk_size)
        values = torch.from_numpy(
            np.array(raw[absolute_start:absolute_stop], dtype=np.float32, copy=True)
        )
        centered = values - mean
        if not bool((centered.norm(dim=-1) > 1e-6).all()):
            raise RuntimeError("V39 centering produced a zero target")
        output[
            absolute_start - start : absolute_stop - start
        ] = F.normalize(centered, dim=-1).half()
    del raw
    return output


def load_v37_teacher_mean(path: str | Path, *, verify_hash: bool) -> torch.Tensor:
    source = Path(path)
    if verify_hash and file_sha256(source) != EXPECTED_V37_TRAIN_BANK_SHA256:
        raise ValueError("V39 V37 train target bank hash changed")
    state = torch.load(source, map_location="cpu", weights_only=False)
    if not isinstance(state, Mapping) or state.get("architecture") != V37_TARGET_ARCHITECTURE:
        raise ValueError("V39 teacher mean source is not a V37 target bank")
    bank = VisualSemanticDistillationTargetBank.from_state_dict(state)
    if bank.receipt.get("split") != "train":
        raise ValueError("V39 teacher mean must come from the V37 train split")
    teacher = bank.receipt.get("teacher", {})
    if verify_hash and (
        teacher.get("manifest_sha256") != BGE_MANIFEST_SHA256
        or teacher.get("model_layer_sha256") != BGE_MODEL_SHA256
    ):
        raise ValueError("V39 V37 teacher identity changed")
    return bank.teacher_mean.float().cpu()


def _sample_indices(count: int, maximum: int) -> torch.Tensor:
    if min(count, maximum) < 1:
        raise ValueError("V39 diagnostic sample geometry is invalid")
    if count <= maximum:
        return torch.arange(count)
    return torch.linspace(0, count - 1, steps=maximum).round().long().unique()


def target_diagnostics(
    prompt: torch.Tensor,
    answer: torch.Tensor,
    segments: torch.Tensor,
    *,
    maximum_samples: int,
) -> dict[str, Any]:
    global_indices = _sample_indices(len(prompt), maximum_samples)
    segment_indices = _sample_indices(len(segments), maximum_samples)
    global_metrics = retrieval_metrics(
        prompt[global_indices].float(),
        answer[global_indices].float(),
    )
    segment_sample = F.normalize(segments[segment_indices].float(), dim=-1)
    cyclic = F.cosine_similarity(segment_sample, segment_sample.roll(1, dims=0), dim=-1)
    return {
        "global_sample": len(global_indices),
        "segment_sample": len(segment_indices),
        "prompt_answer": global_metrics,
        "segment_effective_rank": centered_effective_rank(segment_sample),
        "segment_cyclic_cosine": float(cyclic.mean()),
        "maximum_prompt_norm_error": float(
            (prompt[global_indices].float().norm(dim=-1) - 1).abs().max()
        ),
        "maximum_answer_norm_error": float(
            (answer[global_indices].float().norm(dim=-1) - 1).abs().max()
        ),
        "maximum_segment_norm_error": float(
            (segment_sample.norm(dim=-1) - 1).abs().max()
        ),
    }


def _exact_length_worker(
    payload: tuple[str, int, VisualSemanticDistillationRenderConfig, str],
) -> float:
    text, variant, render_config, font_path = payload
    return canonical_v39_text_length(
        text,
        render_config=render_config,
        font_path=font_path,
        variant=variant,
    )


def _progress_printer(label: str, total: int, every: int) -> Callable[[Mapping[str, Any]], None]:
    def report(state: Mapping[str, Any]) -> None:
        completed = int(state["completed_rows"])
        counter = int(
            state.get("embedding_requests", state.get("measurement_batches", 0))
        )
        if completed == total or counter % every == 0:
            print(
                f"{label}: {completed}/{total} ({100 * completed / total:.2f}%)",
                flush=True,
            )

    return report


def _reset_work_directory(path: Path) -> None:
    resolved = path.resolve()
    unsafe = {
        Path(resolved.anchor),
        Path("/tmp").resolve(),
        Path.home().resolve(),
        Path.cwd().resolve(),
    }
    if resolved in unsafe or not resolved.name.endswith(".work"):
        raise ValueError("V39 refuses to reset an unsafe work directory")
    if path.exists():
        shutil.rmtree(path)


def acquire_work_lock(work: Path):
    lock_path = work.with_name(f"{work.name}.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    handle = lock_path.open("a+", encoding="utf-8")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        handle.close()
        raise RuntimeError(f"V39 target work is already active: {work}") from error
    return handle


def main() -> None:
    started = time.perf_counter()
    args = parse_args()
    integer_settings = (
        args.batch_size,
        args.maximum_batch_characters,
        args.length_batch_size,
        args.length_workers,
        args.assembly_chunk_size,
        args.diagnostic_samples,
        args.progress_every,
    )
    if min(integer_settings) < 1 or args.timeout <= 0 or args.maximum_records < 0:
        raise ValueError("V39 target-builder arguments are invalid")
    if file_sha256(RESEARCH_DOCUMENT) != EXPECTED_RESEARCH_SHA256:
        raise RuntimeError("V39 research specification changed")
    instruction_sha256 = file_sha256(args.instruction_manifest)
    if not args.smoke and instruction_sha256 != EXPECTED_INSTRUCTION_SHA256:
        raise RuntimeError("V39 instruction data differs from the fixed corpus")
    if args.smoke and not 1 <= args.maximum_records <= 64:
        raise ValueError("V39 smoke builds require 1..64 maximum records")

    output = Path(args.out) if args.out else default_output(args.split)
    work = Path(args.work_dir) if args.work_dir else output.with_suffix(".work")
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"V39 target bank already exists: {output}")
    work_lock = acquire_work_lock(work)
    if args.reset_work:
        _reset_work_directory(work)

    render_config = VisualSemanticDistillationRenderConfig(augment=False)
    all_records = load_v39_instruction_records(args.instruction_manifest)
    records, rejected = select_v39_instruction_records(
        all_records,
        split=args.split,
        render_config=render_config,
    )
    font_fit_selected = len(records)
    if args.maximum_records:
        records = records[: args.maximum_records]
    layout = VisualAnswerTrajectoryDocumentLayout.from_records(records)
    if not args.smoke and not args.maximum_records:
        observed = {
            "records": layout.record_count,
            "segments": layout.segment_count,
            "documents": layout.document_count,
        }
        if observed != EXPECTED_FULL_GEOMETRY[args.split]:
            raise RuntimeError(f"V39 full corpus geometry changed: {observed}")

    work.mkdir(parents=True, exist_ok=True)
    lengths_path = work / "segment_lengths.u8.memmap"
    lengths_progress_path = work / "segment_lengths.progress.json"
    length_identity = {
        "architecture": LENGTH_WORK_ARCHITECTURE,
        "split": args.split,
        "layout_sha256": layout.sha256,
        "rows": layout.segment_count,
        "dimension": 1,
        "dtype": "uint8",
        "render_config": asdict(render_config),
        "font_path": str(Path(V39_DEVELOPMENT_FONT).resolve()),
        "method": "exact-clean-raster-active-patches",
    }

    executor: ProcessPoolExecutor | None = None
    if args.length_workers > 1:
        executor = ProcessPoolExecutor(
            max_workers=args.length_workers,
            mp_context=get_context("fork"),
        )

    def measure_batch(texts: Sequence[str], start: int) -> Sequence[float]:
        payloads = (
            (text, start + offset, render_config, V39_DEVELOPMENT_FONT)
            for offset, text in enumerate(texts)
        )
        if executor is None:
            return tuple(_exact_length_worker(payload) for payload in payloads)
        return tuple(executor.map(_exact_length_worker, payloads, chunksize=16))

    try:
        length_progress = measure_lengths_resumable(
            layout.segment_documents,
            data_path=lengths_path,
            progress_path=lengths_progress_path,
            identity=length_identity,
            measure=measure_batch,
            batch_size=args.length_batch_size,
            progress=_progress_printer(
                "visual lengths",
                layout.segment_count,
                args.progress_every,
            ),
        )
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)

    if args.smoke:
        teacher_mean = torch.zeros(V37_SEMANTIC_DIM)
        teacher_receipt: dict[str, Any] = {
            "route": "deterministic-synthetic-smoke-only",
            "evidence_eligible": False,
            "student_runtime_dependency": False,
        }
        embed = synthetic_embeddings
        teacher_called = False
    else:
        validate_local_embed_endpoint(args.endpoint)
        teacher_mean = load_v37_teacher_mean(args.v37_train_bank, verify_hash=True)
        teacher_receipt = verify_bge_artifact(
            endpoint=args.endpoint,
            model=args.model,
            manifest_path=args.teacher_manifest,
            model_layer_path=args.teacher_model_layer,
            timeout=args.timeout,
        ) | {"evidence_eligible": not bool(args.maximum_records)}
        teacher_called = False

        def embed(texts: Sequence[str]) -> torch.Tensor:
            nonlocal teacher_called
            teacher_called = True
            return request_bge_embeddings(
                texts,
                endpoint=args.endpoint,
                model=args.model,
                timeout=args.timeout,
            )

    embedding_path = work / "raw_embeddings.f16.memmap"
    embedding_progress_path = work / "raw_embeddings.progress.json"
    embedding_identity = {
        "architecture": EMBEDDING_WORK_ARCHITECTURE,
        "split": args.split,
        "layout_sha256": layout.sha256,
        "rows": layout.document_count,
        "dimension": V37_SEMANTIC_DIM,
        "dtype": "float16",
        "teacher_model": "synthetic" if args.smoke else args.model,
        "teacher_manifest_sha256": None if args.smoke else BGE_MANIFEST_SHA256,
        "teacher_model_layer_sha256": None if args.smoke else BGE_MODEL_SHA256,
        "instruction_sha256": instruction_sha256,
    }
    embedding_progress: dict[str, Any]
    try:
        embedding_progress = embed_documents_resumable(
            layout.documents,
            data_path=embedding_path,
            progress_path=embedding_progress_path,
            identity=embedding_identity,
            embed=embed,
            maximum_documents=args.batch_size,
            maximum_characters=args.maximum_batch_characters,
            progress=_progress_printer(
                "BGE documents",
                layout.document_count,
                args.progress_every,
            ),
        )
    finally:
        if not args.smoke and teacher_called:
            unload_bge(
                endpoint=args.endpoint,
                model=args.model,
                timeout=args.timeout,
            )

    prompt_targets = center_normalize_memmap_region(
        embedding_path,
        start=layout.prompt_slice.start,
        stop=layout.prompt_slice.stop,
        total_rows=layout.document_count,
        teacher_mean=teacher_mean,
        chunk_size=args.assembly_chunk_size,
    )
    answer_targets = center_normalize_memmap_region(
        embedding_path,
        start=layout.answer_slice.start,
        stop=layout.answer_slice.stop,
        total_rows=layout.document_count,
        teacher_mean=teacher_mean,
        chunk_size=args.assembly_chunk_size,
    )
    segment_targets = center_normalize_memmap_region(
        embedding_path,
        start=layout.segment_slice.start,
        stop=layout.segment_slice.stop,
        total_rows=layout.document_count,
        teacher_mean=teacher_mean,
        chunk_size=args.assembly_chunk_size,
    )
    length_memmap = np.memmap(
        lengths_path,
        mode="r",
        dtype=np.uint8,
        shape=(layout.segment_count, 1),
    )
    segment_lengths = torch.from_numpy(np.array(length_memmap[:, 0], copy=True)).half()
    del length_memmap
    diagnostics = target_diagnostics(
        prompt_targets,
        answer_targets,
        segment_targets,
        maximum_samples=args.diagnostic_samples,
    )

    receipt: dict[str, Any] = {
        "experiment": EXPERIMENT,
        "label": (
            "smoke"
            if args.smoke
            else "evidence-input"
            if not args.maximum_records
            else "exploratory-subset"
        ),
        "split": args.split,
        "research": {
            "path": RESEARCH_DOCUMENT,
            "sha256": EXPECTED_RESEARCH_SHA256,
        },
        "source_sha256": {path: file_sha256(path) for path in SOURCE_FILES},
        "data": {
            "manifest": str(Path(args.instruction_manifest).resolve()),
            "sha256": instruction_sha256,
            "loaded_records": len(all_records),
            "font_fit_selected_before_limit": font_fit_selected,
            "selected_records": layout.record_count,
            "selected_segments": layout.segment_count,
            "selected_documents": layout.document_count,
            "rejected_records": len(rejected),
            "rejected_identifiers": list(rejected),
            "rights": "CC BY-NC 4.0; research use only",
        },
        "layout_sha256": layout.sha256,
        "document_regions": {
            "prompts": [layout.prompt_slice.start, layout.prompt_slice.stop],
            "answers": [layout.answer_slice.start, layout.answer_slice.stop],
            "segments": [layout.segment_slice.start, layout.segment_slice.stop],
        },
        "render_config": asdict(render_config),
        "teacher": teacher_receipt,
        "teacher_mean_source": (
            "zero-smoke-only"
            if args.smoke
            else str(Path(args.v37_train_bank).resolve())
        ),
        "teacher_mean_source_sha256": (
            None if args.smoke else EXPECTED_V37_TRAIN_BANK_SHA256
        ),
        "target_transform": "l2_normalize(raw_l2_normalized_bge - fixed_v37_train_joint_mean)",
        "embedding_work": embedding_progress
        | {
            "path": str(embedding_path.resolve()),
            "sha256": file_sha256(embedding_path),
        },
        "length_work": length_progress
        | {
            "path": str(lengths_path.resolve()),
            "sha256": file_sha256(lengths_path),
        },
        "diagnostics": diagnostics,
        "source_text_strings_stored": False,
        "record_identifiers_stored": True,
        "token_ids_stored": False,
        "unicode_ids_stored": False,
        "student_runtime_teacher_dependency": False,
        "elapsed_seconds": time.perf_counter() - started,
    }
    bank = VisualAnswerTrajectoryTargetBank(
        identifiers=layout.identifiers,
        prompt_targets=prompt_targets,
        answer_targets=answer_targets,
        segment_targets=segment_targets,
        segment_offsets=torch.tensor(layout.segment_offsets, dtype=torch.long),
        segment_lengths=segment_lengths,
        teacher_mean=teacher_mean.float(),
        receipt=receipt,
    )
    atomic_torch_save(bank.state_dict(), output)
    summary = receipt | {
        "target_bank": str(output.resolve()),
        "target_bank_sha256": file_sha256(output),
        "target_records": layout.record_count,
        "target_segments": layout.segment_count,
        "target_dimension": V37_SEMANTIC_DIM,
        "finite": True,
    }
    atomic_write_json(summary, output.with_suffix(".receipt.json"))
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    work_lock.close()


if __name__ == "__main__":
    main()

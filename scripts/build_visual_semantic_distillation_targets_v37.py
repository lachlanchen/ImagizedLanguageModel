#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import time
import urllib.error
import urllib.request
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence
from urllib.parse import urlsplit

import torch
import torch.nn.functional as F

from ilm.visual_lm.visual_semantic_distillation import file_sha256
from ilm.visual_lm.visual_semantic_distillation_data import (
    V37_SEMANTIC_DIM,
    V37_TARGET_ARCHITECTURE,
    VisualSemanticDistillationRenderConfig,
    canonical_answer_length_v37,
    load_v37_instruction_records,
    select_v37_instruction_records,
)
from ilm.visual_lm.visual_semantic_distillation_training import (
    VisualSemanticDistillationTargetBank,
    centered_effective_rank,
)
from ilm.visual_lm.visual_semantic_raster_data import VisualRasterRecord


EXPERIMENT = V37_TARGET_ARCHITECTURE
PROTOCOL_DOCUMENT = "references/visual_semantic_distillation_v37_protocol.md"
EXPECTED_PROTOCOL_SHA256 = (
    "e3cca1c8eedb387f80a88cf17a93466f59532ea666d6dcbfe57e5d7d5e91f6d7"
)
DEFAULT_INSTRUCTION_MANIFEST = "data/raw/alpaca_zh.json"
EXPECTED_INSTRUCTION_SHA256 = (
    "6fcb98c6d79691d1f9a88ef513335da9124e7fdeef5103343f5d9f9a6e8f4903"
)
BGE_MODEL = "bge-m3:latest"
BGE_ENDPOINT = "http://127.0.0.1:11434/api/embed"
BGE_MANIFEST_SHA256 = "7907646426070047a77226ac3e684fbbe8410524f7b4a74d02837e43f2146bab"
BGE_MODEL_SHA256 = "daec91ffb5dd0c27411bd71f29932917c49cf529a641d0168496c3a501e3062c"
BGE_MODEL_BYTES = 1_157_671_200
MINIMUM_OLLAMA_VERSION = (0, 32, 6)
SEED = 20_263_700
SOURCE_FILES = (
    "ilm/visual_lm/visual_semantic_distillation_data.py",
    "ilm/visual_lm/visual_semantic_distillation_training.py",
    "scripts/build_visual_semantic_distillation_targets_v37.py",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build hash-pinned offline BGE semantic targets for V37."
    )
    parser.add_argument("--instruction-manifest", default=DEFAULT_INSTRUCTION_MANIFEST)
    parser.add_argument("--split", choices=("train", "development"), default="train")
    parser.add_argument(
        "--train-bank",
        default="artifacts/visual_semantic_distillation_v37_targets/train.pt",
    )
    parser.add_argument(
        "--out",
        default="artifacts/visual_semantic_distillation_v37_targets/train.pt",
    )
    parser.add_argument("--endpoint", default=BGE_ENDPOINT)
    parser.add_argument("--model", default=BGE_MODEL)
    parser.add_argument(
        "--teacher-manifest",
        default=(
            "../LocalLLM/.local/models/ollama/manifests/"
            "registry.ollama.ai/library/bge-m3/latest"
        ),
    )
    parser.add_argument(
        "--teacher-model-layer",
        default=(
            "../LocalLLM/.local/models/ollama/blobs/"
            "sha256-daec91ffb5dd0c27411bd71f29932917c49cf529a641d0168496c3a501e3062c"
        ),
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--maximum-records", type=int, default=0)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def validate_local_embed_endpoint(endpoint: str) -> str:
    parsed = urlsplit(endpoint)
    if (
        parsed.scheme != "http"
        or parsed.hostname != "127.0.0.1"
        or parsed.port != 11434
        or parsed.path != "/api/embed"
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("V37 BGE endpoint must be exactly local Ollama /api/embed")
    return endpoint


def _request_json(
    endpoint: str,
    *,
    method: str = "GET",
    payload: Mapping[str, Any] | None = None,
    timeout: float,
) -> dict[str, Any]:
    data = None
    headers: dict[str, str] = {}
    if payload is not None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(
        endpoint,
        data=data,
        headers=headers,
        method=method,
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as error:
        raise RuntimeError(f"V37 local teacher request failed: {error}") from error
    if not isinstance(body, dict):
        raise RuntimeError("V37 local teacher returned a non-object response")
    return body


def _base_url(endpoint: str) -> str:
    parsed = urlsplit(validate_local_embed_endpoint(endpoint))
    return f"{parsed.scheme}://{parsed.hostname}:{parsed.port}"


def _version_tuple(value: str) -> tuple[int, int, int]:
    parts = value.split(".")
    if len(parts) < 3 or not all(part.isdigit() for part in parts[:3]):
        raise ValueError(f"unrecognized Ollama version: {value!r}")
    return tuple(int(part) for part in parts[:3])  # type: ignore[return-value]


def verify_bge_artifact(
    *,
    endpoint: str,
    model: str,
    manifest_path: str | Path,
    model_layer_path: str | Path,
    timeout: float,
) -> dict[str, Any]:
    validate_local_embed_endpoint(endpoint)
    if model != BGE_MODEL:
        raise ValueError(f"V37 production targets require {BGE_MODEL!r}")
    manifest = Path(manifest_path).expanduser().resolve()
    model_layer = Path(model_layer_path).expanduser().resolve()
    if not manifest.is_file() or not model_layer.is_file():
        raise FileNotFoundError("V37 pinned BGE manifest or model layer is missing")
    if file_sha256(manifest) != BGE_MANIFEST_SHA256:
        raise ValueError("V37 BGE manifest hash changed")
    if file_sha256(model_layer) != BGE_MODEL_SHA256:
        raise ValueError("V37 BGE model-layer hash changed")
    if model_layer.stat().st_size != BGE_MODEL_BYTES:
        raise ValueError("V37 BGE model-layer byte count changed")
    manifest_value = json.loads(manifest.read_text(encoding="utf-8"))
    model_layers = [
        layer
        for layer in manifest_value.get("layers", [])
        if layer.get("mediaType") == "application/vnd.ollama.image.model"
    ]
    expected_digest = f"sha256:{BGE_MODEL_SHA256}"
    if len(model_layers) != 1 or model_layers[0].get("digest") != expected_digest:
        raise ValueError("V37 BGE manifest no longer selects the pinned model layer")
    if int(model_layers[0].get("size", -1)) != BGE_MODEL_BYTES:
        raise ValueError("V37 BGE manifest layer size changed")

    base = _base_url(endpoint)
    version = str(
        _request_json(f"{base}/api/version", timeout=timeout).get("version", "")
    )
    if _version_tuple(version) < MINIMUM_OLLAMA_VERSION:
        raise RuntimeError("V37 requires Ollama 0.32.6 or newer")
    tags = _request_json(f"{base}/api/tags", timeout=timeout).get("models", [])
    live = [item for item in tags if item.get("name") == model]
    if len(live) != 1 or str(live[0].get("digest")) != BGE_MANIFEST_SHA256:
        raise RuntimeError("V37 live BGE tag does not match the pinned manifest")
    return {
        "model": model,
        "endpoint": endpoint,
        "server_version": version,
        "manifest": str(manifest),
        "manifest_sha256": BGE_MANIFEST_SHA256,
        "model_layer": str(model_layer),
        "model_layer_sha256": BGE_MODEL_SHA256,
        "model_layer_bytes": BGE_MODEL_BYTES,
        "official_model_card_license": "MIT",
        "role": "offline detached semantic target construction only",
        "student_runtime_dependency": False,
    }


def request_bge_embeddings(
    texts: Sequence[str],
    *,
    endpoint: str,
    model: str,
    timeout: float,
    keep_alive: str | int = "10m",
) -> torch.Tensor:
    if not texts:
        raise ValueError("V37 cannot embed an empty text batch")
    body: dict[str, Any] | None = None
    for attempt in range(3):
        try:
            body = _request_json(
                validate_local_embed_endpoint(endpoint),
                method="POST",
                payload={
                    "model": model,
                    "input": list(texts),
                    "truncate": False,
                    "keep_alive": keep_alive,
                },
                timeout=timeout,
            )
            break
        except RuntimeError:
            if attempt == 2:
                raise
            time.sleep(2**attempt)
    if body is None:
        raise RuntimeError("V37 local teacher returned no response")
    embeddings = torch.tensor(body.get("embeddings"), dtype=torch.float32)
    if embeddings.shape != (len(texts), V37_SEMANTIC_DIM):
        raise RuntimeError("V37 BGE response has the wrong shape")
    if not bool(torch.isfinite(embeddings).all()):
        raise FloatingPointError("V37 BGE response contains non-finite values")
    norms = embeddings.norm(dim=-1)
    if not bool((norms > 1e-6).all()):
        raise RuntimeError("V37 BGE response contains a zero vector")
    return F.normalize(embeddings, dim=-1)


def unload_bge(*, endpoint: str, model: str, timeout: float) -> None:
    request_bge_embeddings(
        ("卸载",),
        endpoint=endpoint,
        model=model,
        timeout=timeout,
        keep_alive=0,
    )


def synthetic_embeddings(texts: Sequence[str]) -> torch.Tensor:
    rows: list[torch.Tensor] = []
    for text in texts:
        values = bytearray()
        counter = 0
        while len(values) < 4 * V37_SEMANTIC_DIM:
            values.extend(hashlib.sha256(f"{counter}:{text}".encode("utf-8")).digest())
            counter += 1
        integer = torch.tensor(
            list(values[: 4 * V37_SEMANTIC_DIM]), dtype=torch.float32
        )
        vector = integer.reshape(V37_SEMANTIC_DIM, 4).mean(dim=1) - 127.5
        rows.append(F.normalize(vector, dim=-1))
    return torch.stack(rows)


def embed_records(
    records: Sequence[VisualRasterRecord],
    *,
    batch_size: int,
    embed: Callable[[Sequence[str]], torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, int]:
    if batch_size < 1 or not records:
        raise ValueError("V37 target embedding sizes are invalid")
    documents = [text for record in records for text in (record.prompt, record.answer)]
    chunks: list[torch.Tensor] = []
    requests = 0
    for start in range(0, len(documents), batch_size):
        chunk = embed(documents[start : start + batch_size]).float()
        if chunk.shape != (
            min(batch_size, len(documents) - start),
            V37_SEMANTIC_DIM,
        ):
            raise RuntimeError("V37 embedding callback returned the wrong shape")
        chunks.append(F.normalize(chunk, dim=-1))
        requests += 1
    all_embeddings = torch.cat(chunks)
    return all_embeddings[0::2], all_embeddings[1::2], requests


def transform_targets(
    prompt_raw: torch.Tensor,
    answer_raw: torch.Tensor,
    *,
    teacher_mean: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if prompt_raw.shape != answer_raw.shape or prompt_raw.ndim != 2:
        raise ValueError("V37 raw target modalities must match [N,D]")
    if prompt_raw.shape[1] != V37_SEMANTIC_DIM:
        raise ValueError("V37 raw target width changed")
    if teacher_mean is None:
        teacher_mean = torch.cat((prompt_raw, answer_raw)).mean(dim=0)
    if teacher_mean.shape != (V37_SEMANTIC_DIM,):
        raise ValueError("V37 teacher mean has the wrong shape")
    prompt = F.normalize(prompt_raw.float() - teacher_mean.float(), dim=-1)
    answer = F.normalize(answer_raw.float() - teacher_mean.float(), dim=-1)
    return prompt, answer, teacher_mean.float()


def retrieval_metrics(
    prompt: torch.Tensor,
    answer: torch.Tensor,
) -> dict[str, float | int]:
    if prompt.shape != answer.shape or prompt.ndim != 2:
        raise ValueError("V37 retrieval matrices must align")
    prompt = F.normalize(prompt.float(), dim=-1)
    answer = F.normalize(answer.float(), dim=-1)
    similarities = prompt @ answer.T
    indices = torch.arange(len(prompt))
    ranking = similarities.argsort(dim=1, descending=True)
    positions = (ranking == indices[:, None]).nonzero(as_tuple=False)[:, 1]
    correct = similarities[indices, indices]
    cyclic = similarities[indices, torch.roll(indices, shifts=1)]
    return {
        "samples": len(prompt),
        "top1": float((positions < 1).float().mean()),
        "top5": float((positions < min(5, len(prompt))).float().mean()),
        "mrr": float((1 / (positions + 1).float()).mean()),
        "correct_cosine": float(correct.mean()),
        "cyclic_margin": float((correct - cyclic).mean()),
        "correct_beats_cyclic": float((correct > cyclic).float().mean()),
        "answer_effective_rank": centered_effective_rank(answer),
    }


def validate_development_target_sanity(metrics: Mapping[str, float | int]) -> None:
    conditions = {
        "top1": float(metrics["top1"]) >= 0.80,
        "top5": float(metrics["top5"]) >= 0.90,
        "mrr": float(metrics["mrr"]) >= 0.85,
        "cyclic_margin": float(metrics["cyclic_margin"]) >= 0.50,
        "answer_effective_rank": float(metrics["answer_effective_rank"]) >= 70,
    }
    failed = [name for name, passed in conditions.items() if not passed]
    if failed:
        raise RuntimeError(f"V37 development target sanity failed: {failed}")


def atomic_torch_save(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent,
        prefix=path.name,
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
    try:
        torch.save(dict(payload), temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_write_json(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=path.name,
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def load_train_bank(path: str | Path) -> VisualSemanticDistillationTargetBank:
    state = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(state, Mapping):
        raise TypeError("V37 train bank must contain a state mapping")
    bank = VisualSemanticDistillationTargetBank.from_state_dict(state)
    if bank.receipt.get("split") != "train":
        raise ValueError("V37 development targets require a train target bank")
    if bank.receipt.get("protocol", {}).get("sha256") != EXPECTED_PROTOCOL_SHA256:
        raise ValueError("V37 train target bank has another protocol")
    return bank


def main() -> None:
    args = parse_args()
    if min(args.batch_size, args.timeout) <= 0 or args.maximum_records < 0:
        raise ValueError("V37 target-builder arguments are invalid")
    if file_sha256(PROTOCOL_DOCUMENT) != EXPECTED_PROTOCOL_SHA256:
        raise RuntimeError("V37 protocol changed after preregistration")
    instruction_sha256 = file_sha256(args.instruction_manifest)
    if not args.smoke and instruction_sha256 != EXPECTED_INSTRUCTION_SHA256:
        raise RuntimeError("V37 instruction data differs from preregistration")
    output = Path(args.out)
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"V37 target bank already exists: {output}")

    render_config = VisualSemanticDistillationRenderConfig(augment=False)
    all_records = load_v37_instruction_records(args.instruction_manifest)
    records, rejected = select_v37_instruction_records(
        all_records,
        split=args.split,
        render_config=render_config,
    )
    if args.maximum_records:
        records = records[: args.maximum_records]
    if not args.smoke:
        expected_count = 5_822 if args.split == "train" else 196
        if len(records) != expected_count:
            raise RuntimeError(
                f"V37 {args.split} record count changed: {len(records)} != {expected_count}"
            )

    teacher_receipt: dict[str, Any]
    if args.smoke:
        teacher_receipt = {
            "route": "deterministic-synthetic-smoke-only",
            "student_runtime_dependency": False,
            "evidence_eligible": False,
        }
        embed = synthetic_embeddings
    else:
        teacher_receipt = verify_bge_artifact(
            endpoint=args.endpoint,
            model=args.model,
            manifest_path=args.teacher_manifest,
            model_layer_path=args.teacher_model_layer,
            timeout=args.timeout,
        ) | {"evidence_eligible": True}

        def embed(texts: Sequence[str]) -> torch.Tensor:
            return request_bge_embeddings(
                texts,
                endpoint=args.endpoint,
                model=args.model,
                timeout=args.timeout,
            )

    started = time.perf_counter()
    try:
        prompt_raw, answer_raw, request_count = embed_records(
            records,
            batch_size=args.batch_size,
            embed=embed,
        )
    finally:
        if not args.smoke:
            unload_bge(endpoint=args.endpoint, model=args.model, timeout=args.timeout)

    train_bank = None
    if args.split == "development":
        train_bank = load_train_bank(args.train_bank)
        teacher_mean = train_bank.teacher_mean.float()
        if not args.smoke:
            train_teacher = train_bank.receipt.get("teacher", {})
            if train_teacher.get("manifest_sha256") != BGE_MANIFEST_SHA256:
                raise ValueError("V37 train and development teacher manifests differ")
            if train_teacher.get("model_layer_sha256") != BGE_MODEL_SHA256:
                raise ValueError("V37 train and development teacher layers differ")
    else:
        teacher_mean = None
    prompt_targets, answer_targets, teacher_mean = transform_targets(
        prompt_raw,
        answer_raw,
        teacher_mean=teacher_mean,
    )
    lengths = torch.tensor(
        [
            canonical_answer_length_v37(
                record,
                split=args.split,
                config=render_config,
            )
            for record in records
        ],
        dtype=torch.float32,
    )
    target_metrics = retrieval_metrics(prompt_targets, answer_targets)
    if args.split == "development" and not args.smoke:
        validate_development_target_sanity(target_metrics)

    receipt: dict[str, Any] = {
        "experiment": EXPERIMENT,
        "label": "smoke" if args.smoke else "evidence-input",
        "split": args.split,
        "protocol": {
            "path": PROTOCOL_DOCUMENT,
            "sha256": EXPECTED_PROTOCOL_SHA256,
        },
        "source_sha256": {path: file_sha256(path) for path in SOURCE_FILES},
        "data": {
            "manifest": str(Path(args.instruction_manifest).resolve()),
            "sha256": instruction_sha256,
            "raw_eligible_records": len(all_records),
            "selected_records": len(records),
            "rejected_records": len(rejected),
            "rejected_identifiers": list(rejected),
        },
        "render_config": asdict(render_config),
        "teacher": teacher_receipt,
        "target_transform": "l2_normalize(raw_l2_normalized_bge - train_joint_mean)",
        "teacher_mean_source": (
            "current-complete-train-bank"
            if args.split == "train"
            else str(Path(args.train_bank).resolve())
        ),
        "teacher_mean_norm": float(teacher_mean.norm()),
        "embedding_requests": request_count,
        "embedding_documents": 2 * len(records),
        "target_metrics": target_metrics,
        "strings_stored": False,
        "token_ids_stored": False,
        "student_runtime_dependency": False,
        "elapsed_seconds": time.perf_counter() - started,
    }
    bank = VisualSemanticDistillationTargetBank(
        identifiers=tuple(record.identifier for record in records),
        prompt_targets=prompt_targets.half().cpu(),
        answer_targets=answer_targets.half().cpu(),
        lengths=lengths.half().cpu(),
        teacher_mean=teacher_mean.float().cpu(),
        receipt=receipt,
    )
    atomic_torch_save(bank.state_dict(), output)
    summary = receipt | {
        "target_bank": str(output.resolve()),
        "target_bank_sha256": file_sha256(output),
        "target_count": len(bank.identifiers),
        "target_dimension": bank.prompt_targets.shape[1],
        "finite": True,
    }
    atomic_write_json(summary, output.with_suffix(".receipt.json"))
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

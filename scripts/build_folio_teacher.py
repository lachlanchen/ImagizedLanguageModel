#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Sequence

import torch

from ilm.visual_lm.instruction_data import VisualInstructionRecord, load_alpaca_records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cache continuous semantic fields for offline visual-retina distillation."
    )
    parser.add_argument("--zh-data", default="data/raw/alpaca_zh.json")
    parser.add_argument("--en-data", default="data/raw/alpaca_en.json")
    parser.add_argument("--disable-zh", action="store_true")
    parser.add_argument("--disable-en", action="store_true")
    parser.add_argument("--max-records-per-language", type=int, default=10_000)
    parser.add_argument("--max-prompt-chars", type=int, default=180)
    parser.add_argument("--max-response-chars", type=int, default=320)
    parser.add_argument("--model", default="bge-m3:latest")
    parser.add_argument("--endpoint", default="http://127.0.0.1:11434/api/embed")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--save-every-batches", type=int, default=10)
    parser.add_argument("--out", default="data/teacher/folio_bge_m3.pt")
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_records(args: argparse.Namespace) -> tuple[list[VisualInstructionRecord], list[dict[str, Any]]]:
    records: list[VisualInstructionRecord] = []
    sources: list[dict[str, Any]] = []
    requested = []
    if not args.disable_zh:
        requested.append((Path(args.zh_data), "zh", "GPT-4-LLM alpaca_gpt4_data_zh", "CC-BY-NC-4.0"))
    if not args.disable_en:
        requested.append((Path(args.en_data), "en", "Stanford Alpaca", "CC-BY-NC-4.0"))
    for path, language, source, license_name in requested:
        if not path.exists():
            raise FileNotFoundError(f"missing {path}; run scripts/download_alpaca.py")
        loaded = load_alpaca_records(
            path,
            language=language,
            source=source,
            max_prompt_chars=args.max_prompt_chars,
            max_response_chars=args.max_response_chars,
            limit=args.max_records_per_language,
        )
        records.extend(loaded)
        sources.append(
            {
                "path": str(path),
                "sha256": file_sha256(path),
                "language": language,
                "source": source,
                "license": license_name,
                "records": len(loaded),
            }
        )
    if not records:
        raise ValueError("no instruction records selected")
    return records, sources


def documents_from_records(records: Sequence[VisualInstructionRecord]) -> list[dict[str, Any]]:
    documents: list[dict[str, Any]] = []
    for record in records:
        for kind, text in (("prompt", record.prompt), ("response", record.response)):
            documents.append(
                {
                    "record_identifier": record.identifier,
                    "kind": kind,
                    "language": record.language,
                    "source": record.source,
                    "text": text,
                }
            )
    return documents


def embed_batch(
    texts: Sequence[str],
    *,
    endpoint: str,
    model: str,
    timeout: float,
) -> torch.Tensor:
    payload = json.dumps({"model": model, "input": list(texts)}, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        endpoint,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError) as error:
        raise RuntimeError(f"semantic teacher request failed: {error}") from error
    embeddings = torch.tensor(body["embeddings"], dtype=torch.float32)
    if embeddings.ndim != 2 or embeddings.shape[0] != len(texts):
        raise RuntimeError("semantic teacher returned a malformed embedding batch")
    return torch.nn.functional.normalize(embeddings, dim=-1)


def atomic_save(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def cache_payload(
    *,
    documents: list[dict[str, Any]],
    embeddings: torch.Tensor,
    sources: list[dict[str, Any]],
    args: argparse.Namespace,
    complete: bool,
    elapsed_seconds: float,
) -> dict[str, Any]:
    return {
        "architecture": "folio-semantic-teacher-v1",
        "teacher_model": args.model,
        "teacher_endpoint": args.endpoint,
        "teacher_role": "offline continuous-field supervision only",
        "student_runtime_dependency": False,
        "documents": documents[: embeddings.shape[0]],
        "embeddings": embeddings.half().cpu(),
        "embedding_mean": embeddings.float().mean(dim=0).half().cpu(),
        "student_target_transform": "center_then_l2_normalize_v1",
        "sources": sources,
        "complete": complete,
        "elapsed_seconds": elapsed_seconds,
    }


def main() -> None:
    args = parse_args()
    output = Path(args.out)
    records, sources = load_records(args)
    documents = documents_from_records(records)
    chunks: list[torch.Tensor] = []
    offset = 0
    if output.exists():
        existing = torch.load(output, map_location="cpu", weights_only=False)
        existing_documents = existing.get("documents", [])
        if (
            existing.get("architecture") == "folio-semantic-teacher-v1"
            and existing.get("teacher_model") == args.model
            and existing_documents == documents[: len(existing_documents)]
        ):
            prior = existing["embeddings"].float()
            chunks.append(prior)
            offset = prior.shape[0]
            if existing.get("complete") and offset == len(documents):
                print(json.dumps({"status": "already_complete", "documents": offset, "out": str(output)}))
                return

    started = time.perf_counter()
    batch_counter = 0
    for begin in range(offset, len(documents), args.batch_size):
        end = min(len(documents), begin + args.batch_size)
        embedded = embed_batch(
            [str(item["text"]) for item in documents[begin:end]],
            endpoint=args.endpoint,
            model=args.model,
            timeout=args.timeout,
        )
        chunks.append(embedded)
        batch_counter += 1
        all_embeddings = torch.cat(chunks, dim=0)
        print(
            json.dumps(
                {
                    "stage": "embed",
                    "documents": int(all_embeddings.shape[0]),
                    "total": len(documents),
                    "dimensions": int(all_embeddings.shape[1]),
                    "elapsed_seconds": time.perf_counter() - started,
                }
            ),
            flush=True,
        )
        if batch_counter % max(1, args.save_every_batches) == 0:
            atomic_save(
                cache_payload(
                    documents=documents,
                    embeddings=all_embeddings,
                    sources=sources,
                    args=args,
                    complete=False,
                    elapsed_seconds=time.perf_counter() - started,
                ),
                output,
            )

    all_embeddings = torch.cat(chunks, dim=0)
    atomic_save(
        cache_payload(
            documents=documents,
            embeddings=all_embeddings,
            sources=sources,
            args=args,
            complete=True,
            elapsed_seconds=time.perf_counter() - started,
        ),
        output,
    )
    print(
        json.dumps(
            {
                "stage": "complete",
                "records": len(records),
                "documents": len(documents),
                "dimensions": int(all_embeddings.shape[1]),
                "out": str(output),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()

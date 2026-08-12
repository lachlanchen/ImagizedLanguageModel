#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.nn.functional as F

from ilm.visual_lm.folio_data import load_teacher_cache, stable_fraction


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a validated held-out paraphrase suite offline.")
    parser.add_argument("--teacher-cache", required=True)
    parser.add_argument("--out", default="data/teacher/folio_paraphrases.jsonl")
    parser.add_argument("--validation-fraction", type=float, default=0.05)
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--group-size", type=int, default=8)
    parser.add_argument("--chat-endpoint", default="http://127.0.0.1:8008/v1/chat/completions")
    parser.add_argument("--chat-model", default="qwen3:8b-q4_K_M")
    parser.add_argument("--api-key", default="local-dev-key")
    parser.add_argument("--embed-endpoint", default="http://127.0.0.1:11434/api/embed")
    parser.add_argument("--embed-model", default="bge-m3:latest")
    parser.add_argument("--minimum-semantic-cosine", type=float, default=0.65)
    parser.add_argument("--timeout", type=float, default=180.0)
    return parser.parse_args()


def request_json(
    endpoint: str,
    payload: dict[str, Any],
    *,
    timeout: float,
    authorization: str | None = None,
) -> dict[str, Any]:
    headers = {"Content-Type": "application/json"}
    if authorization is not None:
        headers["Authorization"] = f"Bearer {authorization}"
    request = urllib.request.Request(
        endpoint,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError) as error:
        raise RuntimeError(f"local teacher request failed: {error}") from error


def extract_object(content: str) -> dict[str, Any]:
    content = content.strip()
    if content.startswith("```"):
        content = content.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    try:
        value = json.loads(content)
    except json.JSONDecodeError:
        start, end = content.find("{"), content.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("paraphrase teacher returned no JSON object")
        value = json.loads(content[start : end + 1])
    if not isinstance(value, dict):
        raise ValueError("paraphrase teacher JSON must be an object")
    return value


def extract_items(payload: dict[str, Any], expected_ids: set[str]) -> list[Any]:
    """Accept common local-model JSON key variants without accepting prose."""

    for key in ("items", "key_items", "paraphrases"):
        value = payload.get(key)
        if isinstance(value, list):
            return value
    list_values = [value for value in payload.values() if isinstance(value, list)]
    if len(list_values) == 1:
        return list_values[0]
    keyed_items: list[dict[str, str]] = []
    for identifier, value in payload.items():
        if identifier not in expected_ids:
            continue
        if isinstance(value, str):
            paraphrase = value.strip()
        elif isinstance(value, dict) and isinstance(value.get("paraphrase"), str):
            paraphrase = value["paraphrase"].strip()
        else:
            continue
        if paraphrase:
            keyed_items.append({"id": identifier, "paraphrase": paraphrase})
    if keyed_items:
        return keyed_items
    keys = ", ".join(sorted(str(key) for key in payload))
    raise ValueError(f"paraphrase teacher omitted an item array; keys={keys}")


def paraphrase_group(items: Sequence[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, str]]:
    source = [{"id": item["identifier"], "language": item["language"], "prompt": item["text"]} for item in items]
    system = (
        "Rewrite each user prompt once without answering it. Preserve every factual constraint and requested task. "
        "Keep the original language. Make wording materially different, concise, and natural. "
        "Return only one JSON object with key items; each item has exactly id and paraphrase. /no_think"
    )
    payload = {
        "model": args.chat_model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(source, ensure_ascii=False)},
        ],
        "temperature": 0.2,
        "max_tokens": 2400,
        "stream": False,
        "think": False,
        "response_format": {"type": "json_object"},
    }
    body = request_json(
        args.chat_endpoint,
        payload,
        timeout=args.timeout,
        authorization=args.api_key,
    )
    parsed = extract_object(body["choices"][0]["message"]["content"])
    returned = extract_items(parsed, {item["identifier"] for item in items})
    by_id: dict[str, str] = {}
    for value in returned:
        if isinstance(value, dict) and isinstance(value.get("id"), str) and isinstance(value.get("paraphrase"), str):
            paraphrase = value["paraphrase"].strip()
            if paraphrase:
                by_id.setdefault(value["id"], paraphrase)
    output = []
    for item in items:
        paraphrase = by_id.get(item["identifier"], "")
        if paraphrase and paraphrase != item["text"]:
            output.append({"id": item["identifier"], "paraphrase": paraphrase})
    return output


def semantic_cosines(pairs: Sequence[tuple[str, str]], args: argparse.Namespace) -> list[float]:
    flattened = [text for pair in pairs for text in pair]
    body = request_json(
        args.embed_endpoint,
        {"model": args.embed_model, "input": flattened},
        timeout=args.timeout,
    )
    fields = F.normalize(torch.tensor(body["embeddings"], dtype=torch.float32), dim=-1)
    return [float(torch.dot(fields[index], fields[index + 1])) for index in range(0, len(fields), 2)]


def atomic_write(path: Path, records: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    os.replace(temporary, path)


def main() -> None:
    args = parse_args()
    cache = load_teacher_cache(args.teacher_cache)
    candidates = []
    for document in cache["documents"]:
        if document["kind"] != "prompt":
            continue
        if stable_fraction(str(document["record_identifier"])) >= args.validation_fraction:
            continue
        candidates.append(
            {
                "identifier": str(document["record_identifier"]),
                "language": str(document["language"]),
                "text": str(document["text"]),
            }
        )
        if len(candidates) >= args.limit:
            break
    output_path = Path(args.out)
    accepted: list[dict[str, Any]] = []
    completed = set()
    if output_path.exists():
        for line in output_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                record = json.loads(line)
                accepted.append(record)
                completed.add(str(record["identifier"]))
    candidates = [candidate for candidate in candidates if candidate["identifier"] not in completed]
    started = time.perf_counter()
    source_by_id = {candidate["identifier"]: candidate for candidate in candidates}
    for offset in range(0, len(candidates), args.group_size):
        group = candidates[offset : offset + args.group_size]
        try:
            generated = paraphrase_group(group, args)
        except (KeyError, TypeError, ValueError, RuntimeError) as error:
            print(json.dumps({"stage": "skip_group", "offset": offset, "error": str(error)}), flush=True)
            continue
        pairs = [(source_by_id[item["id"]]["text"], item["paraphrase"]) for item in generated]
        similarities = semantic_cosines(pairs, args) if pairs else []
        for generated_item, similarity in zip(generated, similarities):
            if similarity < args.minimum_semantic_cosine:
                continue
            source = source_by_id[generated_item["id"]]
            accepted.append(
                {
                    "identifier": generated_item["id"],
                    "language": source["language"],
                    "paraphrase": generated_item["paraphrase"],
                    "semantic_cosine": similarity,
                    "teacher_model": args.chat_model,
                    "validator_model": args.embed_model,
                }
            )
        atomic_write(output_path, accepted)
        print(
            json.dumps(
                {
                    "stage": "paraphrase",
                    "processed": min(len(candidates), offset + len(group)),
                    "accepted": len(accepted),
                    "elapsed_seconds": time.perf_counter() - started,
                }
            ),
            flush=True,
        )
    print(json.dumps({"stage": "complete", "accepted": len(accepted), "out": str(output_path)}), flush=True)


if __name__ == "__main__":
    main()

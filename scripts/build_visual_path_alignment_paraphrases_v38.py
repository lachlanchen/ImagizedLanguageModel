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
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
from urllib.parse import urlsplit

from ilm.visual_lm.visual_semantic_distillation import file_sha256
from ilm.visual_lm.visual_semantic_distillation_data import (
    V37_TRAIN_FONTS,
    VisualSemanticDistillationRenderConfig,
    load_v37_instruction_records,
    select_v37_instruction_records,
    visual_text_fits_v37,
)
from ilm.visual_lm.visual_semantic_raster_data import (
    VisualRasterRecord,
    normalize_visible_text,
)
from scripts.build_visual_semantic_distillation_targets_v37 import (
    BGE_MANIFEST_SHA256,
    BGE_MODEL,
    BGE_MODEL_BYTES,
    BGE_MODEL_SHA256,
    request_bge_embeddings,
    unload_bge,
    verify_bge_artifact,
)


EXPERIMENT = "visual-path-alignment-paraphrases-v38"
SEED = 20_263_800
DEFAULT_INSTRUCTION_MANIFEST = "data/raw/alpaca_zh.json"
DEFAULT_HOLDOUT_MANIFEST = "data/teacher/folio_paraphrases_zh_holdout.jsonl"
DEFAULT_OUT = "data/teacher/visual_path_alignment_paraphrases_v38.jsonl"
DEFAULT_CANDIDATES = (
    "artifacts/visual_path_alignment_v38_paraphrases/candidates.jsonl"
)
EXPECTED_INSTRUCTION_SHA256 = (
    "6fcb98c6d79691d1f9a88ef513335da9124e7fdeef5103343f5d9f9a6e8f4903"
)
EXPECTED_HOLDOUT_SHA256 = (
    "132a8a257d31be3cf607d0313bf042dafaee7f2df08440bf7e012a502ed6c02f"
)

QWEN_MODEL = "qwen3:4b-q8_0"
QWEN_ENDPOINT = "http://127.0.0.1:11434/api/chat"
QWEN_MANIFEST_SHA256 = (
    "6461746fd6b5a2327ba63d5cd1359af119852d82aa8c981efe948d1868a4dc20"
)
QWEN_MODEL_SHA256 = (
    "fb684cd1056921c526f12a9efbad10c4627e151ecc1e28314fae1c2cce0c2c15"
)
QWEN_MODEL_BYTES = 4_368_878_272
QWEN_LICENSE_SHA256 = (
    "d18a5cc71b84bc4af394a31116bd3932b42241de70c77d2b76d69a314ec8aa12"
)

BGE_ENDPOINT = "http://127.0.0.1:11434/api/embed"
V38_EXTRA_TRAIN_FONTS = (
    "/usr/share/fonts/truetype/arphic-gkai00mp/gkai00mp.ttf",
    "/usr/share/fonts/truetype/arphic/ukai.ttc",
    "/usr/share/fonts/truetype/arphic/uming.ttc",
)
V38_TRAIN_FONTS = tuple(dict.fromkeys((*V37_TRAIN_FONTS, *V38_EXTRA_TRAIN_FONTS)))

SYSTEM_PROMPT = (
    "Rewrite the supplied Chinese task as one concise Chinese instruction with "
    "exactly the same intent and all conditions preserved. Do not answer the "
    "task. Return only the rewritten instruction, without labels or explanation."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the hash-pinned offline V38 training paraphrases."
    )
    parser.add_argument("--instruction-manifest", default=DEFAULT_INSTRUCTION_MANIFEST)
    parser.add_argument("--holdout-manifest", default=DEFAULT_HOLDOUT_MANIFEST)
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--candidates", default=DEFAULT_CANDIDATES)
    parser.add_argument("--target-count", type=int, default=1_024)
    parser.add_argument("--candidate-count", type=int, default=1_600)
    parser.add_argument("--minimum-cosine", type=float, default=0.82)
    parser.add_argument("--qwen-endpoint", default=QWEN_ENDPOINT)
    parser.add_argument("--qwen-model", default=QWEN_MODEL)
    parser.add_argument(
        "--qwen-manifest",
        default=(
            "../LocalLLM/.local/models/ollama/manifests/"
            "registry.ollama.ai/library/qwen3/4b-q8_0"
        ),
    )
    parser.add_argument(
        "--qwen-model-layer",
        default=(
            "../LocalLLM/.local/models/ollama/blobs/"
            "sha256-fb684cd1056921c526f12a9efbad10c4627e151ecc1e28314fae1c2cce0c2c15"
        ),
    )
    parser.add_argument(
        "--qwen-license-layer",
        default=(
            "../LocalLLM/.local/models/ollama/blobs/"
            "sha256-d18a5cc71b84bc4af394a31116bd3932b42241de70c77d2b76d69a314ec8aa12"
        ),
    )
    parser.add_argument("--bge-endpoint", default=BGE_ENDPOINT)
    parser.add_argument("--bge-model", default=BGE_MODEL)
    parser.add_argument(
        "--bge-manifest",
        default=(
            "../LocalLLM/.local/models/ollama/manifests/"
            "registry.ollama.ai/library/bge-m3/latest"
        ),
    )
    parser.add_argument(
        "--bge-model-layer",
        default=(
            "../LocalLLM/.local/models/ollama/blobs/"
            "sha256-daec91ffb5dd0c27411bd71f29932917c49cf529a641d0168496c3a501e3062c"
        ),
    )
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--embedding-batch-size", type=int, default=96)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _request_json(
    endpoint: str,
    *,
    payload: Mapping[str, Any] | None = None,
    timeout: float,
) -> dict[str, Any]:
    data = None
    headers: dict[str, str] = {}
    if payload is not None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(endpoint, data=data, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as error:
        raise RuntimeError(f"local Ollama request failed: {error}") from error
    if not isinstance(body, dict):
        raise RuntimeError("local Ollama response is not a JSON object")
    if body.get("error"):
        raise RuntimeError(str(body["error"]))
    return body


def _validate_local_endpoint(endpoint: str, *, path: str) -> str:
    parsed = urlsplit(endpoint)
    if (
        parsed.scheme != "http"
        or parsed.hostname != "127.0.0.1"
        or parsed.port != 11434
        or parsed.path != path
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError(f"V38 endpoint must be local Ollama {path}")
    return endpoint


def verify_qwen_artifact(
    *,
    endpoint: str,
    model: str,
    manifest_path: str | Path,
    model_layer_path: str | Path,
    license_layer_path: str | Path,
    timeout: float,
) -> dict[str, Any]:
    _validate_local_endpoint(endpoint, path="/api/chat")
    if model != QWEN_MODEL:
        raise ValueError(f"V38 paraphrases require {QWEN_MODEL!r}")
    manifest = Path(manifest_path).expanduser().resolve()
    model_layer = Path(model_layer_path).expanduser().resolve()
    license_layer = Path(license_layer_path).expanduser().resolve()
    for path in (manifest, model_layer, license_layer):
        if not path.is_file():
            raise FileNotFoundError(path)
    if file_sha256(manifest) != QWEN_MANIFEST_SHA256:
        raise ValueError("V38 Qwen manifest hash changed")
    if (
        file_sha256(model_layer) != QWEN_MODEL_SHA256
        or model_layer.stat().st_size != QWEN_MODEL_BYTES
    ):
        raise ValueError("V38 Qwen model layer changed")
    if file_sha256(license_layer) != QWEN_LICENSE_SHA256:
        raise ValueError("V38 Qwen license layer changed")
    manifest_value = json.loads(manifest.read_text(encoding="utf-8"))
    model_layers = [
        layer
        for layer in manifest_value.get("layers", [])
        if layer.get("mediaType") == "application/vnd.ollama.image.model"
    ]
    if len(model_layers) != 1 or model_layers[0].get("digest") != (
        f"sha256:{QWEN_MODEL_SHA256}"
    ):
        raise ValueError("V38 Qwen manifest model selection changed")
    base = "http://127.0.0.1:11434"
    version = str(_request_json(f"{base}/api/version", timeout=timeout).get("version"))
    tags = _request_json(f"{base}/api/tags", timeout=timeout).get("models", [])
    live = [item for item in tags if item.get("name") == model]
    if len(live) != 1 or live[0].get("digest") != QWEN_MANIFEST_SHA256:
        raise RuntimeError("V38 live Qwen tag differs from the pinned manifest")
    return {
        "model": model,
        "endpoint": endpoint,
        "server_version": version,
        "manifest": str(manifest),
        "manifest_sha256": QWEN_MANIFEST_SHA256,
        "model_layer": str(model_layer),
        "model_layer_sha256": QWEN_MODEL_SHA256,
        "model_layer_bytes": QWEN_MODEL_BYTES,
        "license_layer": str(license_layer),
        "license_layer_sha256": QWEN_LICENSE_SHA256,
        "license": "Apache-2.0",
        "role": "offline training-paraphrase preparation only",
        "student_runtime_dependency": False,
    }


def _holdout_source_identifiers(path: str | Path) -> set[str]:
    identifiers: set[str] = set()
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            item = json.loads(line)
            try:
                index = int(str(item.get("identifier", "")).rsplit(":", 1)[1])
            except (IndexError, ValueError):
                continue
            identifiers.add(f"alpaca-zh:{index}")
    if not identifiers:
        raise ValueError("V38 found no fixed paraphrase holdout identifiers")
    return identifiers


def deterministic_candidates(
    records: Sequence[VisualRasterRecord],
    *,
    excluded: set[str],
    seed: int,
) -> tuple[VisualRasterRecord, ...]:
    eligible = [record for record in records if record.identifier not in excluded]
    return tuple(
        sorted(
            eligible,
            key=lambda record: (
                hashlib.sha256(
                    f"{seed}:{record.identifier}".encode("utf-8")
                ).digest(),
                record.identifier,
            ),
        )
    )


def clean_paraphrase(value: str) -> str:
    text = str(value).strip().strip("`").strip()
    for prefix in (
        "Paraphrase:",
        "Rewrite:",
        "\u6539\u5199\uff1a",
        "\u91cd\u5199\uff1a",
        "\u6539\u5199\uff1a",
    ):
        if text.startswith(prefix):
            text = text[len(prefix) :].strip()
    text = text.strip('"\'\u201c\u201d\u300c\u300d ')
    text = normalize_visible_text(text)
    if text and not text.startswith("\u95ee\uff1a"):
        text = "\u95ee\uff1a" + text
    return text


def request_paraphrase(
    record: VisualRasterRecord,
    *,
    endpoint: str,
    model: str,
    seed: int,
    timeout: float,
) -> tuple[str, dict[str, int]]:
    body: dict[str, Any] | None = None
    for attempt in range(3):
        try:
            body = _request_json(
                _validate_local_endpoint(endpoint, path="/api/chat"),
                payload={
                    "model": model,
                    "stream": False,
                    "think": False,
                    "keep_alive": "10m",
                    "options": {
                        "temperature": 0.2,
                        "seed": int(seed),
                        "num_predict": 128,
                    },
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": record.prompt},
                    ],
                },
                timeout=timeout,
            )
            break
        except RuntimeError:
            if attempt == 2:
                raise
            time.sleep(2**attempt)
    if body is None:
        raise RuntimeError("V38 Qwen returned no response")
    message = body.get("message", {})
    paraphrase = clean_paraphrase(
        message.get("content", "") if isinstance(message, Mapping) else ""
    )
    usage = {
        "prompt_eval_count": int(body.get("prompt_eval_count", 0)),
        "eval_count": int(body.get("eval_count", 0)),
        "total_duration_ns": int(body.get("total_duration", 0)),
    }
    return paraphrase, usage


def unload_qwen(*, endpoint: str, model: str, timeout: float) -> None:
    base = endpoint.rsplit("/", 2)[0]
    _request_json(
        f"{base}/api/generate",
        payload={"model": model, "keep_alive": 0},
        timeout=timeout,
    )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                item = json.loads(line)
                if isinstance(item, dict):
                    rows.append(item)
    return rows


def _append_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        if os.path.exists(temporary):
            os.unlink(temporary)
        raise


def _chunks(values: Sequence[Any], size: int) -> Iterable[Sequence[Any]]:
    if size < 1:
        raise ValueError("V38 chunk size must be positive")
    for start in range(0, len(values), size):
        yield values[start : start + size]


def paraphrase_fits(text: str, config: VisualSemanticDistillationRenderConfig) -> bool:
    if not text or len(text) > 160:
        return False
    return all(
        Path(font).is_file()
        and visual_text_fits_v37(
            text,
            config=config,
            font_path=font,
            font_size=config.maximum_font_size,
            origin=config.maximum_origin,
        )
        for font in V38_TRAIN_FONTS
    )


def validate_candidates(
    rows: Sequence[Mapping[str, Any]],
    records: Mapping[str, VisualRasterRecord],
    *,
    endpoint: str,
    model: str,
    timeout: float,
    batch_size: int,
    minimum_cosine: float,
    render_config: VisualSemanticDistillationRenderConfig,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    accepted: list[dict[str, Any]] = []
    reasons: dict[str, int] = {}
    for chunk in _chunks(list(rows), batch_size):
        valid_rows: list[Mapping[str, Any]] = []
        texts: list[str] = []
        for row in chunk:
            identifier = str(row.get("identifier", ""))
            paraphrase = normalize_visible_text(str(row.get("paraphrase", "")))
            record = records.get(identifier)
            reason = None
            if record is None:
                reason = "missing-source"
            elif paraphrase == record.prompt:
                reason = "exact-copy"
            elif paraphrase == record.answer:
                reason = "answer-copy"
            elif not paraphrase_fits(paraphrase, render_config):
                reason = "does-not-fit"
            elif not 0.45 <= len(paraphrase) / max(1, len(record.prompt)) <= 1.80:
                reason = "length-ratio"
            if reason is not None:
                reasons[reason] = reasons.get(reason, 0) + 1
                continue
            valid_rows.append(row)
            texts.extend((record.prompt, paraphrase, record.answer))
        if not valid_rows:
            continue
        embeddings = request_bge_embeddings(
            texts,
            endpoint=endpoint,
            model=model,
            timeout=timeout,
        ).reshape(len(valid_rows), 3, -1)
        for row, vectors in zip(valid_rows, embeddings):
            original_cosine = float(vectors[0] @ vectors[1])
            paraphrase_answer_cosine = float(vectors[1] @ vectors[2])
            original_answer_cosine = float(vectors[0] @ vectors[2])
            if original_cosine < minimum_cosine:
                reasons["semantic-cosine"] = reasons.get("semantic-cosine", 0) + 1
                continue
            if paraphrase_answer_cosine > max(0.90, original_answer_cosine + 0.20):
                reasons["answer-like"] = reasons.get("answer-like", 0) + 1
                continue
            accepted.append(
                dict(row)
                | {
                    "semantic_cosine": original_cosine,
                    "paraphrase_answer_cosine": paraphrase_answer_cosine,
                    "original_answer_cosine": original_answer_cosine,
                    "validator_model": model,
                }
            )
    return accepted, reasons


def main() -> None:
    args = parse_args()
    if args.smoke:
        args.target_count = min(args.target_count, 2)
        args.candidate_count = min(args.candidate_count, 4)
    if not 1 <= args.target_count <= args.candidate_count:
        raise ValueError("V38 target count must fit inside candidate count")
    if not 0.5 <= args.minimum_cosine < 1.0:
        raise ValueError("V38 semantic threshold is invalid")
    if min(args.timeout, args.embedding_batch_size) <= 0:
        raise ValueError("V38 timeout and embedding batch must be positive")

    instruction_sha = file_sha256(args.instruction_manifest)
    holdout_sha = file_sha256(args.holdout_manifest)
    if not args.smoke and (
        instruction_sha != EXPECTED_INSTRUCTION_SHA256
        or holdout_sha != EXPECTED_HOLDOUT_SHA256
    ):
        raise RuntimeError("V38 source data differs from the fixed inputs")

    qwen_receipt = verify_qwen_artifact(
        endpoint=args.qwen_endpoint,
        model=args.qwen_model,
        manifest_path=args.qwen_manifest,
        model_layer_path=args.qwen_model_layer,
        license_layer_path=args.qwen_license_layer,
        timeout=args.timeout,
    )
    bge_receipt = verify_bge_artifact(
        endpoint=args.bge_endpoint,
        model=args.bge_model,
        manifest_path=args.bge_manifest,
        model_layer_path=args.bge_model_layer,
        timeout=args.timeout,
    )
    all_records = load_v37_instruction_records(args.instruction_manifest)
    render_config = VisualSemanticDistillationRenderConfig(augment=True)
    train_records, _ = select_v37_instruction_records(
        all_records,
        split="train",
        render_config=render_config,
    )
    excluded = _holdout_source_identifiers(args.holdout_manifest)
    candidates = deterministic_candidates(train_records, excluded=excluded, seed=args.seed)
    if len(candidates) < args.candidate_count:
        raise RuntimeError("V38 has too few eligible non-holdout training records")
    candidates = candidates[: args.candidate_count]
    by_identifier = {record.identifier: record for record in candidates}

    candidate_path = Path(args.candidates)
    out_path = Path(args.out)
    receipt_path = out_path.with_suffix(".receipt.json")
    if args.overwrite:
        for path in (candidate_path, out_path, receipt_path):
            path.unlink(missing_ok=True)
    elif out_path.exists():
        raise FileExistsError(f"V38 final paraphrase manifest already exists: {out_path}")

    generated = _read_jsonl(candidate_path)
    generated_ids = {str(row.get("identifier")) for row in generated}
    usage = {"prompt_eval_count": 0, "eval_count": 0, "total_duration_ns": 0}
    generation_errors: list[dict[str, str]] = []
    started = time.monotonic()
    try:
        for position, record in enumerate(candidates):
            if record.identifier in generated_ids:
                continue
            try:
                paraphrase, item_usage = request_paraphrase(
                    record,
                    endpoint=args.qwen_endpoint,
                    model=args.qwen_model,
                    seed=args.seed + position * 1_000_003,
                    timeout=args.timeout,
                )
                row = {
                    "identifier": record.identifier,
                    "paraphrase": paraphrase,
                    "teacher_model": args.qwen_model,
                    "candidate_position": position,
                    "source_prompt_sha256": hashlib.sha256(
                        record.prompt.encode("utf-8")
                    ).hexdigest(),
                    "generation_seed": args.seed + position * 1_000_003,
                }
                _append_jsonl(candidate_path, row)
                generated.append(row)
                generated_ids.add(record.identifier)
                for key, value in item_usage.items():
                    usage[key] += value
            except Exception as error:  # Per-row errors are resumable and nonfatal.
                generation_errors.append(
                    {"identifier": record.identifier, "error": str(error)}
                )
            if (position + 1) % 25 == 0 or position + 1 == len(candidates):
                print(
                    json.dumps(
                        {
                            "generated": len(generated),
                            "attempted": position + 1,
                            "errors": len(generation_errors),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
    finally:
        unload_qwen(
            endpoint=args.qwen_endpoint,
            model=args.qwen_model,
            timeout=args.timeout,
        )

    accepted, rejection_reasons = validate_candidates(
        generated,
        by_identifier,
        endpoint=args.bge_endpoint,
        model=args.bge_model,
        timeout=args.timeout,
        batch_size=args.embedding_batch_size,
        minimum_cosine=args.minimum_cosine,
        render_config=render_config,
    )
    unload_bge(
        endpoint=args.bge_endpoint,
        model=args.bge_model,
        timeout=args.timeout,
    )
    if len(accepted) < args.target_count:
        raise RuntimeError(
            f"V38 accepted {len(accepted)} paraphrases, fewer than {args.target_count}; "
            "rerun with a larger --candidate-count"
        )
    selected = accepted[: args.target_count]
    lines = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in selected
    )
    _atomic_text(out_path, lines)
    receipt = {
        "experiment": EXPERIMENT,
        "created_unix": time.time(),
        "elapsed_seconds": time.monotonic() - started,
        "seed": args.seed,
        "target_count": args.target_count,
        "candidate_count": args.candidate_count,
        "generated_count": len(generated),
        "accepted_before_truncation": len(accepted),
        "minimum_semantic_cosine": args.minimum_cosine,
        "semantic_cosine": {
            "minimum": min(float(row["semantic_cosine"]) for row in selected),
            "mean": sum(float(row["semantic_cosine"]) for row in selected)
            / len(selected),
            "maximum": max(float(row["semantic_cosine"]) for row in selected),
        },
        "rejection_reasons": rejection_reasons,
        "generation_errors": generation_errors,
        "generation_usage": usage,
        "instruction_manifest": str(Path(args.instruction_manifest).resolve()),
        "instruction_sha256": instruction_sha,
        "holdout_manifest": str(Path(args.holdout_manifest).resolve()),
        "holdout_sha256": holdout_sha,
        "holdout_source_identifiers_excluded": len(excluded),
        "candidate_manifest": str(candidate_path.resolve()),
        "candidate_manifest_sha256": file_sha256(candidate_path),
        "output": str(out_path.resolve()),
        "output_sha256": file_sha256(out_path),
        "training_fonts": list(V38_TRAIN_FONTS),
        "training_font_sha256": {
            path: file_sha256(path) for path in V38_TRAIN_FONTS
        },
        "qwen": qwen_receipt,
        "bge": bge_receipt
        | {
            "manifest_sha256": BGE_MANIFEST_SHA256,
            "model_layer_sha256": BGE_MODEL_SHA256,
            "model_layer_bytes": BGE_MODEL_BYTES,
            "role": "offline paraphrase validation only",
        },
        "student_runtime_dependency": False,
    }
    _atomic_text(receipt_path, json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

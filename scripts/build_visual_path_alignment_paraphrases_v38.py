#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
import time
import urllib.error
import urllib.request
from collections import Counter
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
DEFAULT_JUDGMENTS = (
    "artifacts/visual_path_alignment_v38_paraphrases/judgments.jsonl"
)
DEFAULT_ADJUDICATIONS = (
    "artifacts/visual_path_alignment_v38_paraphrases/adjudications.jsonl"
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

JUDGE_MODEL = "qwen3:8b-q8_0"
JUDGE_MANIFEST_SHA256 = (
    "e56358ca25dd14db6853a9f68a92d717aaa6f0a94250a72d1a0f3d86a9f30130"
)
JUDGE_MODEL_SHA256 = (
    "d87f4a5a2f1a6051d9fac010c12f76f3ba2137b137d413ba8f4d3a3d06b3a25b"
)
JUDGE_MODEL_BYTES = 8_851_075_872
JUDGE_PROTOCOL_VERSION = 2

ADJUDICATOR_MODEL = "qwen3:30b-a3b-instruct-2507-q4_K_M"
ADJUDICATOR_MANIFEST_SHA256 = (
    "19e422b0231392335cfc49cfd172de7034bb1aeabb08aa307cce745c60b272fe"
)
ADJUDICATOR_MODEL_SHA256 = (
    "78b329e716e7e9775973d392cd132b1f1ff1c8287a992887caeb6fd6c56ba9cc"
)
ADJUDICATOR_MODEL_BYTES = 18_556_685_856
ADJUDICATOR_PROTOCOL_VERSION = 1

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
JUDGE_SYSTEM_PROMPT = (
    "You audit Chinese instruction-paraphrase training data. Decompose the "
    "requested operation before judging wording similarity. A candidate passes "
    "only when it is itself a request or question, asks for exactly the same "
    "operation, preserves every input and condition, and does not perform any "
    "part of the task. A literal leading '问：' is metadata added by the data "
    "pipeline; ignore it when deciding whether the remaining sentence is a "
    "request. A declarative answer after '问：' is still an answer. Filled "
    "blanks, completed analogies, translated text, calculations, summaries, "
    "rewritten sentences, and other task results must be marked as performing "
    "the task. Return the required JSON object only."
)

JUDGE_EXAMPLES: tuple[tuple[str, str, Mapping[str, Any]], ...] = (
    (
        "\u89e3\u91ca\u6c34\u5faa\u73af\u7684\u4e09\u4e2a\u4e3b\u8981\u9636\u6bb5\u3002",
        "\u95ee\uff1a\u8bf7\u8bf4\u660e\u6c34\u5faa\u73af\u5305\u542b\u54ea\u4e09\u4e2a\u4e3b\u8981\u9636\u6bb5\u3002",
        {
            "original_operation": "\u89e3\u91ca\u5e76\u5217\u51fa\u4e09\u4e2a\u9636\u6bb5",
            "candidate_operation": "\u8bf7\u6c42\u8bf4\u660e\u5e76\u5217\u51fa\u4e09\u4e2a\u9636\u6bb5",
            "candidate_is_instruction": True,
            "same_requested_operation": True,
            "preserves_all_inputs_and_conditions": True,
            "performs_or_answers_task": False,
            "reason": "\u5019\u9009\u53e5\u4ecd\u5728\u8bf7\u6c42\u540c\u4e00\u89e3\u91ca\uff0c\u672a\u7ed9\u51fa\u9636\u6bb5\u3002",
        },
    ),
    (
        "\u9009\u62e9\u6b63\u786e\u7684\u8bcd\u586b\u7a7a\uff1a\u5929\u6c14\u53d8\u5f97____\u3002",
        "\u95ee\uff1a\u5929\u6c14\u53d8\u5f97\u5bd2\u51b7\u3002",
        {
            "original_operation": "\u9009\u8bcd\u586b\u7a7a",
            "candidate_operation": "\u9648\u8ff0\u5df2\u586b\u597d\u7684\u53e5\u5b50",
            "candidate_is_instruction": False,
            "same_requested_operation": False,
            "preserves_all_inputs_and_conditions": False,
            "performs_or_answers_task": True,
            "reason": "\u5019\u9009\u53e5\u586b\u4e86\u7a7a\uff0c\u662f\u7b54\u6848\u800c\u975e\u6539\u5199\u540e\u7684\u6307\u4ee4\u3002",
        },
    ),
    (
        "\u75c5\u4eba\u54b3\u55fd\u3002\u8bf7\u66f4\u7cbe\u786e\u5730\u7f16\u8f91\u8fd9\u53e5\u8bdd\u3002",
        "\u95ee\uff1a\u75c5\u4eba\u51fa\u73b0\u54b3\u55fd\u75c7\u72b6\u3002",
        {
            "original_operation": "\u7cbe\u786e\u6539\u5199\u7ed9\u5b9a\u53e5\u5b50",
            "candidate_operation": "\u7ed9\u51fa\u5df2\u6539\u5199\u7684\u53e5\u5b50",
            "candidate_is_instruction": False,
            "same_requested_operation": False,
            "preserves_all_inputs_and_conditions": True,
            "performs_or_answers_task": True,
            "reason": "\u5019\u9009\u53e5\u5df2\u6267\u884c\u6539\u5199\u4efb\u52a1\u3002",
        },
    ),
    (
        "\u5b8c\u6210\u4ee5\u4e0b\u7c7b\u6bd4\uff1a'\u5feb\u5f97\u50cf____\u4e00\u6837\u3002'",
        "\u95ee\uff1a\u5feb\u5f97\u50cf\u95ea\u7535\u4e00\u6837\u3002",
        {
            "original_operation": "\u8865\u5168\u7c7b\u6bd4",
            "candidate_operation": "\u7ed9\u51fa\u5df2\u8865\u5168\u7684\u7c7b\u6bd4",
            "candidate_is_instruction": False,
            "same_requested_operation": False,
            "preserves_all_inputs_and_conditions": False,
            "performs_or_answers_task": True,
            "reason": "\u5019\u9009\u53e5\u5b8c\u6210\u4e86\u7c7b\u6bd4\uff0c\u6ca1\u6709\u8bf7\u6c42\u5b8c\u6210\u5b83\u3002",
        },
    ),
)

HIGH_RISK_OPERATION_FAMILIES: tuple[
    tuple[str, tuple[str, ...], tuple[str, ...]], ...
] = (
    (
        "fill-or-complete",
        (
            "\u586b\u7a7a",
            "\u586b\u5199\u7a7a",
            "\u8865\u5168",
            "\u5b8c\u6210\u4ee5\u4e0b\u7c7b\u6bd4",
            "\u5b8c\u6210\u8fd9\u4e2a\u7c7b\u6bd4",
            "\u5b8c\u6210\u4ee5\u4e0b\u53e5\u5b50",
            "\u8865\u5145\u5b8c\u6574",
        ),
        (
            "\u586b\u7a7a",
            "\u586b\u5199",
            "\u8865\u5168",
            "\u8865\u4e0a",
            "\u7a7a\u767d",
            "\u6a2a\u7ebf",
            "\u7f3a\u5931",
            "\u5b8c\u6210\u7c7b\u6bd4",
            "\u5b8c\u6210\u53e5\u5b50",
        ),
    ),
    (
        "rewrite-or-edit",
        (
            "\u6539\u5199",
            "\u91cd\u5199",
            "\u7f16\u8f91\u8fd9\u53e5",
            "\u7f16\u8f91\u4ee5\u4e0b",
            "\u4fee\u6539\u8fd9\u53e5",
            "\u4fee\u6539\u4ee5\u4e0b",
            "\u6da6\u8272",
            "\u7ea0\u6b63",
            "\u4fee\u6b63",
        ),
        (
            "\u6539\u5199",
            "\u91cd\u5199",
            "\u7f16\u8f91",
            "\u4fee\u6539",
            "\u6da6\u8272",
            "\u7ea0\u6b63",
            "\u4fee\u6b63",
            "\u91cd\u65b0\u8868\u8fbe",
            "\u91cd\u65b0\u8868\u8ff0",
        ),
    ),
    (
        "translate",
        ("\u7ffb\u8bd1", "\u8bd1\u6210", "\u8bd1\u4e3a", "\u82f1\u8bd1", "\u4e2d\u8bd1"),
        ("\u7ffb\u8bd1", "\u8bd1\u6210", "\u8bd1\u4e3a", "\u8bd1\u6587", "\u82f1\u8bd1", "\u4e2d\u8bd1"),
    ),
    (
        "calculate",
        ("\u8ba1\u7b97", "\u6c42\u89e3", "\u6c42\u51fa", "\u7b97\u51fa"),
        ("\u8ba1\u7b97", "\u6c42\u89e3", "\u6c42\u51fa", "\u7b97\u51fa", "\u6c42\u5f97"),
    ),
)

ADJUDICATOR_RELATIONS = (
    "equal",
    "candidate_broader",
    "candidate_narrower",
    "different",
    "not_applicable",
)
ADJUDICATOR_SYSTEM_PROMPT = (
    "Conservatively audit Chinese instruction paraphrases. Precision is more "
    "important than recall. Ignore a leading '问：' metadata prefix. Extract "
    "all atomic requirements. For every relation field choose an explicit "
    "relation: equal, candidate_broader, candidate_narrower, different, or "
    "not_applicable. Do not call a broader or narrower category equal. "
    "Quantity/unit includes all numbers, counts, units, and relations. Named "
    "inputs include supplied names, quoted text, and items. Output requirements "
    "include format, language, tone, ordering, length, and exclusions. Candidate "
    "form is request only if it still asks the task. State whether task work was "
    "performed. Return the required JSON object only."
)
ADJUDICATOR_EXAMPLES: tuple[tuple[str, str, Mapping[str, Any]], ...] = (
    (
        "\u5217\u51fa\u4e24\u79cd\u6d77\u6d0b\u54fa\u4e73\u52a8\u7269\u3002",
        "\u95ee\uff1a\u5217\u51fa\u4e24\u79cd\u6d77\u6d0b\u52a8\u7269\u3002",
        {
            "candidate_form": "request",
            "operation_relation": "equal",
            "quantity_unit_relation": "equal",
            "category_scope_relation": "candidate_broader",
            "named_input_relation": "not_applicable",
            "output_requirement_relation": "equal",
            "task_execution": "not_performed",
            "original_requirements": [
                "\u5217\u51fa",
                "\u4e24\u79cd",
                "\u6d77\u6d0b\u54fa\u4e73\u52a8\u7269",
            ],
            "candidate_requirements": [
                "\u5217\u51fa",
                "\u4e24\u79cd",
                "\u6d77\u6d0b\u52a8\u7269",
            ],
            "reason": "\u5019\u9009\u7c7b\u522b\u66f4\u5bbd\u3002",
        },
    ),
    (
        "\u5199\u4e00\u9996\u4e09\u884c\u8bd7\uff0c\u6bcf\u884c\u56db\u4e2a\u5b57\u3002",
        "\u95ee\uff1a\u5199\u4e00\u9996\u4e09\u8a00\u4e09\u884c\u8bd7\u3002",
        {
            "candidate_form": "request",
            "operation_relation": "equal",
            "quantity_unit_relation": "different",
            "category_scope_relation": "not_applicable",
            "named_input_relation": "not_applicable",
            "output_requirement_relation": "different",
            "task_execution": "not_performed",
            "original_requirements": [
                "\u5199\u8bd7",
                "\u4e09\u884c",
                "\u6bcf\u884c\u56db\u5b57",
            ],
            "candidate_requirements": [
                "\u5199\u8bd7",
                "\u4e09\u884c",
                "\u6bcf\u884c\u4e09\u5b57",
            ],
            "reason": "\u6570\u91cf\u4e0e\u8f93\u51fa\u5f62\u5f0f\u4e0d\u540c\u3002",
        },
    ),
    (
        "\u8bf4\u660e\u6c34\u5faa\u73af\u7684\u4e09\u4e2a\u9636\u6bb5\u3002",
        "\u95ee\uff1a\u6c34\u5faa\u73af\u5305\u542b\u54ea\u4e09\u4e2a\u9636\u6bb5\uff1f",
        {
            "candidate_form": "request",
            "operation_relation": "equal",
            "quantity_unit_relation": "equal",
            "category_scope_relation": "equal",
            "named_input_relation": "equal",
            "output_requirement_relation": "equal",
            "task_execution": "not_performed",
            "original_requirements": [
                "\u8bf4\u660e",
                "\u6c34\u5faa\u73af",
                "\u4e09\u4e2a\u9636\u6bb5",
            ],
            "candidate_requirements": [
                "\u8be2\u95ee",
                "\u6c34\u5faa\u73af",
                "\u4e09\u4e2a\u9636\u6bb5",
            ],
            "reason": "\u8981\u6c42\u7b49\u4ef7\u3002",
        },
    ),
    (
        "\u9009\u62e9\u8bcd\u8bed\u586b\u7a7a\uff1a\u5929\u6c14\u53d8\u5f97____\u3002",
        "\u95ee\uff1a\u5929\u6c14\u53d8\u5f97\u5bd2\u51b7\u3002",
        {
            "candidate_form": "task_result",
            "operation_relation": "different",
            "quantity_unit_relation": "not_applicable",
            "category_scope_relation": "not_applicable",
            "named_input_relation": "equal",
            "output_requirement_relation": "different",
            "task_execution": "fully_performed",
            "original_requirements": [
                "\u9009\u8bcd\u586b\u7a7a",
                "\u7ed9\u5b9a\u53e5\u5b50",
            ],
            "candidate_requirements": ["\u5b8c\u6210\u53e5"],
            "reason": "\u5019\u9009\u5df2\u6267\u884c\u4efb\u52a1\u3002",
        },
    ),
)

CHINESE_NUMERAL_VALUES = {
    "\u96f6": 0,
    "\u3007": 0,
    "\u4e00": 1,
    "\u4e8c": 2,
    "\u4e24": 2,
    "\u4e09": 3,
    "\u56db": 4,
    "\u4e94": 5,
    "\u516d": 6,
    "\u4e03": 7,
    "\u516b": 8,
    "\u4e5d": 9,
}
CHINESE_NUMERAL_UNITS = {"\u5341": 10, "\u767e": 100, "\u5343": 1_000, "\u4e07": 10_000}
CHINESE_QUANTITY_UNITS = (
    "\u4e2a|\u79cd|\u7c7b|\u53ea|\u6761|\u672c|\u7bc7|\u9996|\u884c|\u5217|\u53e5|\u6bb5|\u5b57|\u8bcd|\u8a00|"
    "\u97f3\u8282|\u9879|\u6b21|\u5e74|\u6708|\u65e5|\u5929|\u5c0f\u65f6|\u5206\u949f|\u79d2|\u516c\u91cc|\u5343\u7c73|"
    "\u7c73|\u5398\u7c73|\u6beb\u7c73|\u7f8e\u5143|\u5143|\u4eba|\u56fd|\u9636\u6bb5|\u4f8b|\u70b9|\u7ae0|\u9875|\u90e8\u5206|"
    "\u95ee\u9898|\u6b65\u9aa4|\u65b9\u5f0f|\u65b9\u6cd5|\u7b54\u6848|\u539f\u56e0|\u7279\u70b9|\u8981\u7d20|\u65b9\u9762|\u89d2\u8272|"
    "\u4e8b\u4ef6|\u5355\u8bcd|\u77ed\u8bed|\u53e5\u5b50|%"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the hash-pinned offline V38 training paraphrases."
    )
    parser.add_argument("--instruction-manifest", default=DEFAULT_INSTRUCTION_MANIFEST)
    parser.add_argument("--holdout-manifest", default=DEFAULT_HOLDOUT_MANIFEST)
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--candidates", default=DEFAULT_CANDIDATES)
    parser.add_argument("--judgments", default=DEFAULT_JUDGMENTS)
    parser.add_argument("--adjudications", default=DEFAULT_ADJUDICATIONS)
    parser.add_argument("--target-count", type=int, default=1_024)
    parser.add_argument("--candidate-count", type=int, default=2_000)
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
    parser.add_argument("--judge-endpoint", default=QWEN_ENDPOINT)
    parser.add_argument("--judge-model", default=JUDGE_MODEL)
    parser.add_argument(
        "--judge-manifest",
        default=(
            "../LocalLLM/.local/models/ollama/manifests/"
            "registry.ollama.ai/library/qwen3/8b-q8_0"
        ),
    )
    parser.add_argument(
        "--judge-model-layer",
        default=(
            "../LocalLLM/.local/models/ollama/blobs/"
            "sha256-d87f4a5a2f1a6051d9fac010c12f76f3ba2137b137d413ba8f4d3a3d06b3a25b"
        ),
    )
    parser.add_argument(
        "--judge-license-layer",
        default=(
            "../LocalLLM/.local/models/ollama/blobs/"
            "sha256-d18a5cc71b84bc4af394a31116bd3932b42241de70c77d2b76d69a314ec8aa12"
        ),
    )
    parser.add_argument("--adjudicator-endpoint", default=QWEN_ENDPOINT)
    parser.add_argument("--adjudicator-model", default=ADJUDICATOR_MODEL)
    parser.add_argument(
        "--adjudicator-manifest",
        default=(
            "../LocalLLM/.local/models/ollama/manifests/"
            "registry.ollama.ai/library/qwen3/30b-a3b-instruct-2507-q4_K_M"
        ),
    )
    parser.add_argument(
        "--adjudicator-model-layer",
        default=(
            "../LocalLLM/.local/models/ollama/blobs/"
            "sha256-78b329e716e7e9775973d392cd132b1f1ff1c8287a992887caeb6fd6c56ba9cc"
        ),
    )
    parser.add_argument(
        "--adjudicator-license-layer",
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
    parser.add_argument("--rebuild-final", action="store_true")
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


def _verify_chat_model_artifact(
    *,
    endpoint: str,
    model: str,
    manifest_path: str | Path,
    model_layer_path: str | Path,
    license_layer_path: str | Path,
    timeout: float,
    expected_model: str,
    expected_manifest_sha256: str,
    expected_model_sha256: str,
    expected_model_bytes: int,
    role: str,
) -> dict[str, Any]:
    _validate_local_endpoint(endpoint, path="/api/chat")
    if model != expected_model:
        raise ValueError(f"V38 chat stage requires {expected_model!r}")
    manifest = Path(manifest_path).expanduser().resolve()
    model_layer = Path(model_layer_path).expanduser().resolve()
    license_layer = Path(license_layer_path).expanduser().resolve()
    for path in (manifest, model_layer, license_layer):
        if not path.is_file():
            raise FileNotFoundError(path)
    if file_sha256(manifest) != expected_manifest_sha256:
        raise ValueError("V38 chat-model manifest hash changed")
    if (
        file_sha256(model_layer) != expected_model_sha256
        or model_layer.stat().st_size != expected_model_bytes
    ):
        raise ValueError("V38 chat-model layer changed")
    if file_sha256(license_layer) != QWEN_LICENSE_SHA256:
        raise ValueError("V38 Qwen license layer changed")
    manifest_value = json.loads(manifest.read_text(encoding="utf-8"))
    model_layers = [
        layer
        for layer in manifest_value.get("layers", [])
        if layer.get("mediaType") == "application/vnd.ollama.image.model"
    ]
    if len(model_layers) != 1 or model_layers[0].get("digest") != (
        f"sha256:{expected_model_sha256}"
    ):
        raise ValueError("V38 chat-model manifest selection changed")
    base = "http://127.0.0.1:11434"
    version = str(_request_json(f"{base}/api/version", timeout=timeout).get("version"))
    tags = _request_json(f"{base}/api/tags", timeout=timeout).get("models", [])
    live = [item for item in tags if item.get("name") == model]
    if len(live) != 1 or live[0].get("digest") != expected_manifest_sha256:
        raise RuntimeError("V38 live chat-model tag differs from the pinned manifest")
    return {
        "model": model,
        "endpoint": endpoint,
        "server_version": version,
        "manifest": str(manifest),
        "manifest_sha256": expected_manifest_sha256,
        "model_layer": str(model_layer),
        "model_layer_sha256": expected_model_sha256,
        "model_layer_bytes": expected_model_bytes,
        "license_layer": str(license_layer),
        "license_layer_sha256": QWEN_LICENSE_SHA256,
        "license": "Apache-2.0",
        "role": role,
        "student_runtime_dependency": False,
    }


def verify_qwen_artifact(
    *,
    endpoint: str,
    model: str,
    manifest_path: str | Path,
    model_layer_path: str | Path,
    license_layer_path: str | Path,
    timeout: float,
) -> dict[str, Any]:
    return _verify_chat_model_artifact(
        endpoint=endpoint,
        model=model,
        manifest_path=manifest_path,
        model_layer_path=model_layer_path,
        license_layer_path=license_layer_path,
        timeout=timeout,
        expected_model=QWEN_MODEL,
        expected_manifest_sha256=QWEN_MANIFEST_SHA256,
        expected_model_sha256=QWEN_MODEL_SHA256,
        expected_model_bytes=QWEN_MODEL_BYTES,
        role="offline training-paraphrase preparation only",
    )


def verify_judge_artifact(
    *,
    endpoint: str,
    model: str,
    manifest_path: str | Path,
    model_layer_path: str | Path,
    license_layer_path: str | Path,
    timeout: float,
) -> dict[str, Any]:
    return _verify_chat_model_artifact(
        endpoint=endpoint,
        model=model,
        manifest_path=manifest_path,
        model_layer_path=model_layer_path,
        license_layer_path=license_layer_path,
        timeout=timeout,
        expected_model=JUDGE_MODEL,
        expected_manifest_sha256=JUDGE_MANIFEST_SHA256,
        expected_model_sha256=JUDGE_MODEL_SHA256,
        expected_model_bytes=JUDGE_MODEL_BYTES,
        role="offline instruction-versus-answer validation only",
    )


def verify_adjudicator_artifact(
    *,
    endpoint: str,
    model: str,
    manifest_path: str | Path,
    model_layer_path: str | Path,
    license_layer_path: str | Path,
    timeout: float,
) -> dict[str, Any]:
    return _verify_chat_model_artifact(
        endpoint=endpoint,
        model=model,
        manifest_path=manifest_path,
        model_layer_path=model_layer_path,
        license_layer_path=license_layer_path,
        timeout=timeout,
        expected_model=ADJUDICATOR_MODEL,
        expected_manifest_sha256=ADJUDICATOR_MANIFEST_SHA256,
        expected_model_sha256=ADJUDICATOR_MODEL_SHA256,
        expected_model_bytes=ADJUDICATOR_MODEL_BYTES,
        role="offline final paraphrase adjudication only",
    )


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


def _judgment_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "original_operation": {"type": "string"},
            "candidate_operation": {"type": "string"},
            "candidate_is_instruction": {"type": "boolean"},
            "same_requested_operation": {"type": "boolean"},
            "preserves_all_inputs_and_conditions": {"type": "boolean"},
            "performs_or_answers_task": {"type": "boolean"},
            "reason": {"type": "string"},
        },
        "required": [
            "original_operation",
            "candidate_operation",
            "candidate_is_instruction",
            "same_requested_operation",
            "preserves_all_inputs_and_conditions",
            "performs_or_answers_task",
            "reason",
        ],
        "additionalProperties": False,
    }


def _judge_pair(original: str, candidate: str) -> str:
    return f"ORIGINAL TASK:\n{original}\n\nCANDIDATE:\n{candidate}"


def _judge_messages(original: str, candidate: str) -> list[dict[str, str]]:
    messages = [{"role": "system", "content": JUDGE_SYSTEM_PROMPT}]
    for example_original, example_candidate, verdict in JUDGE_EXAMPLES:
        messages.extend(
            (
                {
                    "role": "user",
                    "content": _judge_pair(example_original, example_candidate),
                },
                {
                    "role": "assistant",
                    "content": json.dumps(verdict, ensure_ascii=False, sort_keys=True),
                },
            )
        )
    messages.append({"role": "user", "content": _judge_pair(original, candidate)})
    return messages


def judge_protocol_sha256() -> str:
    protocol = {
        "version": JUDGE_PROTOCOL_VERSION,
        "system": JUDGE_SYSTEM_PROMPT,
        "examples": JUDGE_EXAMPLES,
        "schema": _judgment_schema(),
        "high_risk_operation_families": HIGH_RISK_OPERATION_FAMILIES,
    }
    encoded = json.dumps(
        protocol, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def deterministic_operation_gate(original: str, candidate: str) -> tuple[bool, str]:
    candidate_body = normalize_visible_text(candidate)
    if candidate_body.startswith("\u95ee\uff1a"):
        candidate_body = candidate_body[2:].strip()
    for family, source_markers, candidate_markers in HIGH_RISK_OPERATION_FAMILIES:
        if any(marker in original for marker in source_markers) and not any(
            marker in candidate_body for marker in candidate_markers
        ):
            return False, f"missing-{family}-operation"
    return True, ""


def judgment_passes(verdict: Mapping[str, Any]) -> bool:
    return bool(
        verdict.get("candidate_is_instruction", False)
        and verdict.get("same_requested_operation", False)
        and verdict.get("preserves_all_inputs_and_conditions", False)
        and not verdict.get("performs_or_answers_task", True)
    )


def judgment_failure_code(verdict: Mapping[str, Any]) -> str:
    if bool(verdict.get("performs_or_answers_task", True)):
        return "task-performed-or-answered"
    if not bool(verdict.get("candidate_is_instruction", False)):
        return "candidate-not-instruction"
    if not bool(verdict.get("same_requested_operation", False)):
        return "requested-operation-changed"
    if not bool(verdict.get("preserves_all_inputs_and_conditions", False)):
        return "input-or-condition-dropped"
    return "unspecified-judge-failure"


def request_judgment(
    record: VisualRasterRecord,
    paraphrase: str,
    *,
    endpoint: str,
    model: str,
    seed: int,
    timeout: float,
) -> tuple[dict[str, Any], dict[str, int]]:
    schema = _judgment_schema()
    body = _request_json(
        _validate_local_endpoint(endpoint, path="/api/chat"),
        payload={
            "model": model,
            "stream": False,
            "think": False,
            "keep_alive": "10m",
            "format": schema,
            "options": {
                "temperature": 0.0,
                "seed": int(seed),
                "num_predict": 192,
            },
            "messages": _judge_messages(record.prompt, paraphrase),
        },
        timeout=timeout,
    )
    message = body.get("message", {})
    content = message.get("content", "") if isinstance(message, Mapping) else ""
    try:
        verdict = json.loads(str(content))
    except json.JSONDecodeError as error:
        raise RuntimeError("V38 judge returned invalid JSON") from error
    bool_fields = (
        "candidate_is_instruction",
        "same_requested_operation",
        "preserves_all_inputs_and_conditions",
        "performs_or_answers_task",
    )
    if not isinstance(verdict, Mapping) or any(
        not isinstance(verdict.get(name), bool) for name in bool_fields
    ):
        raise RuntimeError("V38 judge returned an invalid verdict")
    normalized_verdict = {
        "original_operation": normalize_visible_text(
            str(verdict.get("original_operation", ""))
        )[:160],
        "candidate_operation": normalize_visible_text(
            str(verdict.get("candidate_operation", ""))
        )[:160],
        **{name: bool(verdict[name]) for name in bool_fields},
        "reason": normalize_visible_text(str(verdict.get("reason", "")))[:240],
    }
    usage = {
        "prompt_eval_count": int(body.get("prompt_eval_count", 0)),
        "eval_count": int(body.get("eval_count", 0)),
        "total_duration_ns": int(body.get("total_duration", 0)),
    }
    return normalized_verdict, usage


def _chinese_numeral_to_int(value: str) -> int:
    total = 0
    section = 0
    number = 0
    for character in value:
        if character in CHINESE_NUMERAL_VALUES:
            number = CHINESE_NUMERAL_VALUES[character]
            continue
        unit = CHINESE_NUMERAL_UNITS.get(character)
        if unit is None:
            raise ValueError(f"unsupported Chinese numeral: {value!r}")
        if unit == 10_000:
            section += number
            total += (section or 1) * unit
            section = 0
            number = 0
        else:
            section += (number or 1) * unit
            number = 0
    return total + section + number


def normalized_numeric_constraints(text: str) -> Counter[str]:
    constraints: Counter[str] = Counter()
    for match in re.finditer(r"\d+(?:\.\d+)?", text):
        raw = match.group(0)
        if "." in raw:
            normalized = raw.rstrip("0").rstrip(".")
        else:
            normalized = raw.lstrip("0") or "0"
        constraints[normalized] += 1
    chinese_number = "\u96f6\u3007\u4e00\u4e8c\u4e24\u4e09\u56db\u4e94\u516d\u4e03\u516b\u4e5d\u5341\u767e\u5343\u4e07"
    patterns = (
        rf"([{chinese_number}]+)(?=(?:{CHINESE_QUANTITY_UNITS}))",
        rf"\u767e\u5206\u4e4b([{chinese_number}]+)",
    )
    for pattern in patterns:
        for match in re.finditer(pattern, text):
            constraints[str(_chinese_numeral_to_int(match.group(1)))] += 1
    return constraints


def deterministic_numeric_gate(original: str, candidate: str) -> tuple[bool, str]:
    original_numbers = normalized_numeric_constraints(original)
    candidate_numbers = normalized_numeric_constraints(candidate)
    if original_numbers != candidate_numbers:
        return False, "numeric-constraint-changed"
    return True, ""


def _adjudicator_schema() -> dict[str, Any]:
    relation = {"type": "string", "enum": list(ADJUDICATOR_RELATIONS)}
    return {
        "type": "object",
        "properties": {
            "candidate_form": {
                "type": "string",
                "enum": ["request", "task_result", "declarative_nonrequest", "unclear"],
            },
            "operation_relation": relation,
            "quantity_unit_relation": relation,
            "category_scope_relation": relation,
            "named_input_relation": relation,
            "output_requirement_relation": relation,
            "task_execution": {
                "type": "string",
                "enum": [
                    "not_performed",
                    "partly_performed",
                    "fully_performed",
                    "unclear",
                ],
            },
            "original_requirements": {"type": "array", "items": {"type": "string"}},
            "candidate_requirements": {"type": "array", "items": {"type": "string"}},
            "reason": {"type": "string"},
        },
        "required": [
            "candidate_form",
            "operation_relation",
            "quantity_unit_relation",
            "category_scope_relation",
            "named_input_relation",
            "output_requirement_relation",
            "task_execution",
            "original_requirements",
            "candidate_requirements",
            "reason",
        ],
        "additionalProperties": False,
    }


def _adjudicator_messages(original: str, candidate: str) -> list[dict[str, str]]:
    messages = [{"role": "system", "content": ADJUDICATOR_SYSTEM_PROMPT}]
    for example_original, example_candidate, verdict in ADJUDICATOR_EXAMPLES:
        messages.extend(
            (
                {
                    "role": "user",
                    "content": _judge_pair(example_original, example_candidate),
                },
                {
                    "role": "assistant",
                    "content": json.dumps(verdict, ensure_ascii=False, sort_keys=True),
                },
            )
        )
    messages.append({"role": "user", "content": _judge_pair(original, candidate)})
    return messages


def adjudicator_protocol_sha256() -> str:
    protocol = {
        "version": ADJUDICATOR_PROTOCOL_VERSION,
        "system": ADJUDICATOR_SYSTEM_PROMPT,
        "examples": ADJUDICATOR_EXAMPLES,
        "schema": _adjudicator_schema(),
        "numeric_constraint_units": CHINESE_QUANTITY_UNITS,
    }
    encoded = json.dumps(
        protocol, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def adjudication_passes(verdict: Mapping[str, Any]) -> bool:
    acceptable = {"equal", "not_applicable"}
    return bool(
        verdict.get("candidate_form") == "request"
        and verdict.get("task_execution") == "not_performed"
        and all(
            verdict.get(name) in acceptable
            for name in (
                "operation_relation",
                "quantity_unit_relation",
                "category_scope_relation",
                "named_input_relation",
                "output_requirement_relation",
            )
        )
    )


def adjudication_failure_code(verdict: Mapping[str, Any]) -> str:
    if verdict.get("task_execution") != "not_performed":
        return f"task-{verdict.get('task_execution', 'invalid')}"
    if verdict.get("candidate_form") != "request":
        return f"candidate-{verdict.get('candidate_form', 'invalid')}"
    for label, field in (
        ("operation", "operation_relation"),
        ("quantity-unit", "quantity_unit_relation"),
        ("category-scope", "category_scope_relation"),
        ("named-input", "named_input_relation"),
        ("output-requirement", "output_requirement_relation"),
    ):
        relation = str(verdict.get(field, "invalid"))
        if relation not in {"equal", "not_applicable"}:
            return f"{label}-{relation}"
    return "unspecified-adjudication-failure"


def request_adjudication(
    record: VisualRasterRecord,
    paraphrase: str,
    *,
    endpoint: str,
    model: str,
    seed: int,
    timeout: float,
) -> tuple[dict[str, Any], dict[str, int]]:
    body = _request_json(
        _validate_local_endpoint(endpoint, path="/api/chat"),
        payload={
            "model": model,
            "stream": False,
            "think": False,
            "keep_alive": "10m",
            "format": _adjudicator_schema(),
            "options": {
                "temperature": 0.0,
                "seed": int(seed),
                "num_predict": 320,
            },
            "messages": _adjudicator_messages(record.prompt, paraphrase),
        },
        timeout=timeout,
    )
    message = body.get("message", {})
    content = message.get("content", "") if isinstance(message, Mapping) else ""
    try:
        verdict = json.loads(str(content))
    except json.JSONDecodeError as error:
        raise RuntimeError("V38 adjudicator returned invalid JSON") from error
    relation_fields = (
        "operation_relation",
        "quantity_unit_relation",
        "category_scope_relation",
        "named_input_relation",
        "output_requirement_relation",
    )
    if (
        not isinstance(verdict, Mapping)
        or verdict.get("candidate_form")
        not in {"request", "task_result", "declarative_nonrequest", "unclear"}
        or verdict.get("task_execution")
        not in {"not_performed", "partly_performed", "fully_performed", "unclear"}
        or any(verdict.get(name) not in ADJUDICATOR_RELATIONS for name in relation_fields)
    ):
        raise RuntimeError("V38 adjudicator returned an invalid verdict")
    normalized_verdict = {
        "candidate_form": str(verdict["candidate_form"]),
        **{name: str(verdict[name]) for name in relation_fields},
        "task_execution": str(verdict["task_execution"]),
        "original_requirements": [
            normalize_visible_text(str(item))[:160]
            for item in list(verdict.get("original_requirements", []))[:24]
        ],
        "candidate_requirements": [
            normalize_visible_text(str(item))[:160]
            for item in list(verdict.get("candidate_requirements", []))[:24]
        ],
        "reason": normalize_visible_text(str(verdict.get("reason", "")))[:240],
    }
    usage = {
        "prompt_eval_count": int(body.get("prompt_eval_count", 0)),
        "eval_count": int(body.get("eval_count", 0)),
        "total_duration_ns": int(body.get("total_duration", 0)),
    }
    return normalized_verdict, usage


def judge_candidates(
    rows: Sequence[Mapping[str, Any]],
    records: Mapping[str, VisualRasterRecord],
    *,
    journal_path: Path,
    endpoint: str,
    model: str,
    seed: int,
    timeout: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    existing = _read_jsonl(journal_path)
    protocol_sha = judge_protocol_sha256()
    by_key = {
        (
            str(row.get("identifier", "")),
            str(row.get("source_prompt_sha256", "")),
            str(row.get("paraphrase_sha256", "")),
        ): row
        for row in existing
        if row.get("judge_protocol_version") == JUDGE_PROTOCOL_VERSION
        and row.get("judge_protocol_sha256") == protocol_sha
        and isinstance(row.get("pass"), bool)
    }
    usage = {"prompt_eval_count": 0, "eval_count": 0, "total_duration_ns": 0}
    errors: list[dict[str, str]] = []
    passed: list[dict[str, Any]] = []
    failed_reasons: dict[str, int] = {}
    for position, row in enumerate(rows):
        identifier = str(row["identifier"])
        paraphrase = str(row["paraphrase"])
        source_prompt_sha = hashlib.sha256(
            records[identifier].prompt.encode("utf-8")
        ).hexdigest()
        paraphrase_sha = hashlib.sha256(paraphrase.encode("utf-8")).hexdigest()
        key = (identifier, source_prompt_sha, paraphrase_sha)
        judgment = by_key.get(key)
        if judgment is None:
            try:
                gate_passed, gate_reason = deterministic_operation_gate(
                    records[identifier].prompt, paraphrase
                )
                base_judgment = {
                    "identifier": identifier,
                    "source_prompt_sha256": source_prompt_sha,
                    "paraphrase_sha256": paraphrase_sha,
                    "judge_protocol_version": JUDGE_PROTOCOL_VERSION,
                    "judge_protocol_sha256": protocol_sha,
                    "seed": seed + position * 1_000_033,
                }
                if gate_passed:
                    verdict, item_usage = request_judgment(
                        records[identifier],
                        paraphrase,
                        endpoint=endpoint,
                        model=model,
                        seed=seed + position * 1_000_033,
                        timeout=timeout,
                    )
                    accepted = judgment_passes(verdict)
                    judgment = base_judgment | verdict | {
                        "pass": accepted,
                        "failure_code": (
                            "pass" if accepted else judgment_failure_code(verdict)
                        ),
                        "judge_model": model,
                        "decision_path": "structured-model-judge",
                    }
                    for name, value in item_usage.items():
                        usage[name] += value
                else:
                    judgment = base_judgment | {
                        "pass": False,
                        "failure_code": gate_reason,
                        "reason": (
                            "candidate omitted the source's explicit high-risk "
                            "operation marker"
                        ),
                        "judge_model": "deterministic-operation-gate",
                        "decision_path": "deterministic-operation-gate",
                    }
                _append_jsonl(journal_path, judgment)
                by_key[key] = judgment
            except Exception as error:  # Failed judgments cannot enter training.
                errors.append({"identifier": identifier, "error": str(error)})
                continue
        if bool(judgment.get("pass", False)):
            passed.append(dict(row) | {"instruction_judge": "pass"})
        else:
            key_reason = normalize_visible_text(
                str(judgment.get("failure_code", "unspecified"))
            )
            failed_reasons[key_reason] = failed_reasons.get(key_reason, 0) + 1
        if (position + 1) % 25 == 0 or position + 1 == len(rows):
            print(
                json.dumps(
                    {
                        "judged": position + 1,
                        "judge_passed": len(passed),
                        "judge_errors": len(errors),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    return passed, {
        "journal_rows": len(_read_jsonl(journal_path)),
        "passed": len(passed),
        "failed": len(rows) - len(passed) - len(errors),
        "failed_reasons": failed_reasons,
        "errors": errors,
        "usage": usage,
        "judge_protocol_version": JUDGE_PROTOCOL_VERSION,
        "judge_protocol_sha256": protocol_sha,
    }


def adjudicate_candidates(
    rows: Sequence[Mapping[str, Any]],
    records: Mapping[str, VisualRasterRecord],
    *,
    journal_path: Path,
    endpoint: str,
    model: str,
    seed: int,
    timeout: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    existing = _read_jsonl(journal_path)
    protocol_sha = adjudicator_protocol_sha256()
    by_key = {
        (
            str(row.get("identifier", "")),
            str(row.get("source_prompt_sha256", "")),
            str(row.get("paraphrase_sha256", "")),
        ): row
        for row in existing
        if row.get("adjudicator_protocol_version") == ADJUDICATOR_PROTOCOL_VERSION
        and row.get("adjudicator_protocol_sha256") == protocol_sha
        and isinstance(row.get("pass"), bool)
    }
    usage = {"prompt_eval_count": 0, "eval_count": 0, "total_duration_ns": 0}
    errors: list[dict[str, str]] = []
    passed: list[dict[str, Any]] = []
    failed_reasons: dict[str, int] = {}
    for position, row in enumerate(rows):
        identifier = str(row["identifier"])
        paraphrase = str(row["paraphrase"])
        record = records[identifier]
        source_prompt_sha = hashlib.sha256(record.prompt.encode("utf-8")).hexdigest()
        paraphrase_sha = hashlib.sha256(paraphrase.encode("utf-8")).hexdigest()
        key = (identifier, source_prompt_sha, paraphrase_sha)
        adjudication = by_key.get(key)
        if adjudication is None:
            item_seed = seed + position * 1_000_037
            base_adjudication = {
                "identifier": identifier,
                "source_prompt_sha256": source_prompt_sha,
                "paraphrase_sha256": paraphrase_sha,
                "adjudicator_protocol_version": ADJUDICATOR_PROTOCOL_VERSION,
                "adjudicator_protocol_sha256": protocol_sha,
                "seed": item_seed,
            }
            numeric_passed, numeric_reason = deterministic_numeric_gate(
                record.prompt, paraphrase
            )
            if not numeric_passed:
                adjudication = base_adjudication | {
                    "pass": False,
                    "failure_code": numeric_reason,
                    "reason": "normalized numeric constraints differ",
                    "adjudicator_model": "deterministic-numeric-gate",
                    "decision_path": "deterministic-numeric-gate",
                }
                _append_jsonl(journal_path, adjudication)
                by_key[key] = adjudication
            else:
                try:
                    verdict, item_usage = request_adjudication(
                        record,
                        paraphrase,
                        endpoint=endpoint,
                        model=model,
                        seed=item_seed,
                        timeout=timeout,
                    )
                    accepted = adjudication_passes(verdict)
                    adjudication = base_adjudication | verdict | {
                        "pass": accepted,
                        "failure_code": (
                            "pass" if accepted else adjudication_failure_code(verdict)
                        ),
                        "adjudicator_model": model,
                        "decision_path": "relation-enum-model-adjudicator",
                    }
                    _append_jsonl(journal_path, adjudication)
                    by_key[key] = adjudication
                    for name, value in item_usage.items():
                        usage[name] += value
                except Exception as error:  # Failed adjudications fail closed.
                    errors.append({"identifier": identifier, "error": str(error)})
                    continue
        if bool(adjudication.get("pass", False)):
            passed.append(dict(row) | {"constraint_adjudicator": "pass"})
        else:
            failure_code = normalize_visible_text(
                str(adjudication.get("failure_code", "unspecified"))
            )
            failed_reasons[failure_code] = failed_reasons.get(failure_code, 0) + 1
        if (position + 1) % 25 == 0 or position + 1 == len(rows):
            print(
                json.dumps(
                    {
                        "adjudicated": position + 1,
                        "adjudicator_passed": len(passed),
                        "adjudicator_errors": len(errors),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    return passed, {
        "journal_rows": len(_read_jsonl(journal_path)),
        "passed": len(passed),
        "failed": len(rows) - len(passed) - len(errors),
        "failed_reasons": failed_reasons,
        "errors": errors,
        "usage": usage,
        "adjudicator_protocol_version": ADJUDICATOR_PROTOCOL_VERSION,
        "adjudicator_protocol_sha256": protocol_sha,
    }


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
    judge_receipt = verify_judge_artifact(
        endpoint=args.judge_endpoint,
        model=args.judge_model,
        manifest_path=args.judge_manifest,
        model_layer_path=args.judge_model_layer,
        license_layer_path=args.judge_license_layer,
        timeout=args.timeout,
    )
    adjudicator_receipt = verify_adjudicator_artifact(
        endpoint=args.adjudicator_endpoint,
        model=args.adjudicator_model,
        manifest_path=args.adjudicator_manifest,
        model_layer_path=args.adjudicator_model_layer,
        license_layer_path=args.adjudicator_license_layer,
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
    judgment_path = Path(args.judgments)
    adjudication_path = Path(args.adjudications)
    out_path = Path(args.out)
    receipt_path = out_path.with_suffix(".receipt.json")
    previous_receipt: dict[str, Any] = {}
    if receipt_path.exists():
        previous_receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if args.overwrite:
        for path in (
            candidate_path,
            judgment_path,
            adjudication_path,
            out_path,
            receipt_path,
        ):
            path.unlink(missing_ok=True)
        previous_receipt = {}
    elif args.rebuild_final:
        out_path.unlink(missing_ok=True)
        receipt_path.unlink(missing_ok=True)
    elif out_path.exists() or receipt_path.exists():
        raise FileExistsError(f"V38 final paraphrase manifest already exists: {out_path}")

    generated = _read_jsonl(candidate_path)
    generated_ids = {str(row.get("identifier")) for row in generated}
    previous_usage = previous_receipt.get("generation_usage", {})
    usage = {
        name: int(previous_usage.get(name, 0))
        for name in ("prompt_eval_count", "eval_count", "total_duration_ns")
    }
    generation_errors: list[dict[str, str]] = list(
        previous_receipt.get("generation_errors", [])
    )
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
    try:
        judged, judgment_summary = judge_candidates(
            accepted,
            by_identifier,
            journal_path=judgment_path,
            endpoint=args.judge_endpoint,
            model=args.judge_model,
            seed=args.seed + 38_000_000,
            timeout=args.timeout,
        )
    finally:
        unload_qwen(
            endpoint=args.judge_endpoint,
            model=args.judge_model,
            timeout=args.timeout,
        )
    try:
        adjudicated, adjudication_summary = adjudicate_candidates(
            judged,
            by_identifier,
            journal_path=adjudication_path,
            endpoint=args.adjudicator_endpoint,
            model=args.adjudicator_model,
            seed=args.seed + 76_000_000,
            timeout=args.timeout,
        )
    finally:
        unload_qwen(
            endpoint=args.adjudicator_endpoint,
            model=args.adjudicator_model,
            timeout=args.timeout,
        )

    unique: list[dict[str, Any]] = []
    seen_paraphrases: set[str] = set()
    duplicate_paraphrases = 0
    for row in adjudicated:
        paraphrase = str(row["paraphrase"])
        if paraphrase in seen_paraphrases:
            duplicate_paraphrases += 1
            continue
        seen_paraphrases.add(paraphrase)
        unique.append(row)
    if duplicate_paraphrases:
        rejection_reasons["duplicate-paraphrase"] = duplicate_paraphrases

    if len(unique) < args.target_count:
        raise RuntimeError(
            f"V38 accepted {len(unique)} adjudicated unique paraphrases, fewer "
            f"than {args.target_count}; "
            "rerun with a larger --candidate-count"
        )
    selected = unique[: args.target_count]
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
        "accepted_by_numeric_filters": len(accepted),
        "accepted_by_instruction_judge": len(judged),
        "accepted_by_constraint_adjudicator": len(adjudicated),
        "accepted_unique_before_truncation": len(unique),
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
        "judgment_manifest": str(judgment_path.resolve()),
        "judgment_manifest_sha256": file_sha256(judgment_path),
        "judgment": judgment_summary,
        "judge_protocol_version": JUDGE_PROTOCOL_VERSION,
        "judge_protocol_sha256": judge_protocol_sha256(),
        "adjudication_manifest": str(adjudication_path.resolve()),
        "adjudication_manifest_sha256": file_sha256(adjudication_path),
        "adjudication": adjudication_summary,
        "adjudicator_protocol_version": ADJUDICATOR_PROTOCOL_VERSION,
        "adjudicator_protocol_sha256": adjudicator_protocol_sha256(),
        "output": str(out_path.resolve()),
        "output_sha256": file_sha256(out_path),
        "training_fonts": list(V38_TRAIN_FONTS),
        "training_font_sha256": {
            path: file_sha256(path) for path in V38_TRAIN_FONTS
        },
        "qwen": qwen_receipt,
        "instruction_judge": judge_receipt,
        "constraint_adjudicator": adjudicator_receipt,
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

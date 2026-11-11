#!/usr/bin/env python3
"""Measure a local Qwen3 text reference on the registered visual tasks."""

import argparse
from collections import defaultdict
from contextlib import contextmanager
import json
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import time

import numpy as np
import requests

from ilm.image_edit.baseline import ROOT, save_json, sha
from ilm.visual_lm.compact_visual_span_weights import CONTEXT_SHA, MANIFEST_SHA
from ilm.visual_lm.visual_sequence_corpus import VisualSequenceCorpus
from scripts.infer_compact_visual_span import read_image
from scripts.prepare_compact_visual_span_evaluation import normalized, source_records
from scripts.prepare_visual_sequence_corpus import FONT, Rasterizer
from scripts.review_compact_visual_span import registered_inputs
from scripts.score_pixel_language_first_blocks import host_resources


MODEL = "qwen3:1.7b"
OPTIONS = {"num_ctx": 2048, "num_predict": 128, "temperature": 0, "seed": 202611153,
           "num_gpu": 0, "num_thread": 2}
TASKS = {
    "qa": "請根據給定資料摘錄最短而完整的答案。只輸出原文答案。",
    "book": "請延續以下段落的內容與文體，續寫約一百字。只輸出接續文字。",
}


def load(path):
    return json.loads(path.read_text())


def terminate(_signum, _frame):
    raise SystemExit("CPU comparison termination requested")


def references(evaluation, preparation):
    path = evaluation / "references.json"
    if sha(path) != preparation["references_sha256"]:
        raise ValueError("Registered evaluation references changed")
    return {row["id"]: row for row in load(path)["reference_rows"]}


def prepare(data, evaluation, output):
    registration, inputs = registered_inputs(evaluation)
    corpus = VisualSequenceCorpus(data, split="development",
                                  expected_report_sha256=registration["visual_corpus_report_sha256"])
    source = {row.identifier: row for groups in source_records(corpus).values()
              for records in groups.values() for row in records}
    if sha(FONT) != load(data / "report.json")["font_sha256"]:
        raise ValueError("The original input renderer changed")
    renderer = Rasterizer()
    refs = references(evaluation, registration)
    cases = []
    for case in inputs.values():
        if case["condition"] != "normal":
            continue
        ref = refs[case["id"].removesuffix("_normal")]
        original = source[ref["source_id"]]
        prompt = original.prompt if ref["kind"] == "qa" else original.answer[:ref["prefix_glyphs"]]
        path = evaluation / "inputs" / case["image"]
        if sha(path) != case["image_sha256"] or len(prompt) != case["glyphs"]:
            raise ValueError("Registered visual prompt changed")
        packed = b"".join(renderer.tile(char) for char in prompt)
        pixels = np.unpackbits(np.frombuffer(packed, dtype=np.uint8), bitorder="big").reshape(-1, 1, 32, 32)
        if not np.array_equal(pixels, read_image(path, case["glyphs"]).numpy()):
            raise ValueError("Text prompt does not render to the registered input")
        cases.append({"id": case["id"], "kind": ref["kind"], "prompt": prompt, "image_sha256": case["image_sha256"]})
    if len(cases) != registration["normal_inputs"]:
        raise ValueError("Text prompts must cover the whole normal cohort")
    output.mkdir(parents=True, exist_ok=False)
    path = output / "prompts.json"
    save_json(path, {"cases": cases, "input_manifest_sha256": registration["input_manifest_sha256"],
                     "pixel_identical_roundtrips": len(cases)})
    print(json.dumps({"input_count": len(cases), "pixel_identical_roundtrips": len(cases), "prompts_sha256": sha(path)}))


def pinned_model(local):
    ollama = local / ".local/ollama/bin/ollama"
    models = local / ".local/models/ollama"
    manifest = models / "manifests/registry.ollama.ai/library/qwen3/1.7b"
    if sha(manifest) != MANIFEST_SHA:
        raise ValueError("The named Qwen3 model manifest changed")
    metadata = load(manifest)
    for layer in [metadata["config"], *metadata["layers"]]:
        kind, digest = layer["mediaType"], layer["digest"].removeprefix("sha256:")
        if kind.endswith(".model") and digest != CONTEXT_SHA:
            raise ValueError("Text reference must use the compact body's source archive")
        blob = models / "blobs" / f"sha256-{digest}"
        if blob.stat().st_size != layer["size"] or sha(blob) != digest:
            raise ValueError(f"A local model layer changed: {kind}")
    return ollama, models


@contextmanager
def cpu_server(ollama, models, output):
    host = host_resources()
    if host["available_gib"] < 24 or host["swap_fraction"] > .75:
        raise RuntimeError("Insufficient shared workstation memory for the CPU comparison")
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    address = f"127.0.0.1:{port}"
    base = f"http://{address}"
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", HIP_VISIBLE_DEVICES="", ROCR_VISIBLE_DEVICES="",
               OLLAMA_VULKAN="0", OLLAMA_NO_CLOUD="1", OLLAMA_NOPRUNE="1", OLLAMA_HOST=address,
               OLLAMA_MODELS=str(models), OLLAMA_NUM_PARALLEL="1", OLLAMA_MAX_LOADED_MODELS="1",
               OMP_NUM_THREADS="2")
    cores = ",".join(map(str, sorted(os.sched_getaffinity(0))[:2]))
    with (output / "server.log").open("a") as log, requests.Session() as session:
        session.trust_env = False
        server = subprocess.Popen(["nice", "-n", "10", "taskset", "-c", cores, str(ollama), "serve"],
                                  env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            deadline = time.monotonic() + 20
            while server.poll() is None and time.monotonic() < deadline:
                try:
                    response = session.get(base + "/api/version", timeout=1)
                    response.raise_for_status()
                    break
                except requests.ConnectionError:
                    time.sleep(.2)
            else:
                raise RuntimeError("The owned CPU Ollama server did not start")
            yield session, base, response.json()["version"]
        finally:
            if server.poll() is None:
                os.killpg(server.pid, signal.SIGTERM)
                try:
                    server.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(server.pid, signal.SIGKILL)
                    server.wait()


def generate(prompts, local, output):
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("Hide CUDA for the bounded CPU quality comparison")
    model, models = pinned_model(local)
    identity = {"model": MODEL, "manifest_sha256": MANIFEST_SHA, "archive_sha256": CONTEXT_SHA,
                "ollama_binary_sha256": sha(model), "prompts_sha256": sha(prompts),
                "tasks": TASKS, "options": OPTIONS, "think": False, "device": "cpu",
                "source_sha256": sha(Path(__file__))}
    cases = load(prompts)["cases"]
    if not cases or len({row["id"] for row in cases}) != len(cases):
        raise ValueError("Expected a nonempty set of distinct text prompts")
    output.mkdir(parents=True, exist_ok=True)
    request = output / "request.json"
    if request.exists() and load(request) != identity:
        raise ValueError("Resume requires the original baseline configuration")
    save_json(request, identity)
    records = []
    with cpu_server(model, models, output) as (session, base, version):
        for case in cases:
            key = case["id"]
            if not re.fullmatch(r"(?:qa_\d+_[ab]|book_\d+)_normal", key):
                raise ValueError("Invalid registered baseline case identifier")
            path = output / (key + ".json")
            if not path.is_file():
                start = time.perf_counter()
                response = session.post(base + "/api/chat", json={
                    "model": MODEL, "stream": False, "think": False, "keep_alive": "5m", "options": OPTIONS,
                    "messages": [{"role": "system", "content": TASKS[case["kind"]]},
                                 {"role": "user", "content": case["prompt"]}],
                }, timeout=(5, 180))
                response.raise_for_status()
                elapsed = time.perf_counter() - start
                raw = response.json()
                resident = session.get(base + "/api/ps", timeout=5)
                resident.raise_for_status()
                loaded, = resident.json()["models"]
                if (loaded["size_vram"] != 0 or loaded["digest"].removeprefix("sha256:") != MANIFEST_SHA
                        or raw["model"] != MODEL or not raw["done"] or raw["message"].get("thinking")):
                    raise ValueError("Unexpected model, unfinished response or non-CPU execution")
                save_json(path, {"id": key, "response": raw, "elapsed_seconds": elapsed,
                                 "resident_model_bytes": loaded["size"], "vram_bytes": loaded["size_vram"],
                                 "quantization": loaded["details"]["quantization_level"]})
                print(json.dumps({"id": key, "reason": raw["done_reason"], "seconds": elapsed}), flush=True)
            records.append({"id": key, "sha256": sha(path)})
    save_json(output / "report.json", {"request_sha256": sha(request), "ollama_version": version,
                                       "raw_responses": records, "references_opened_during_generation": False})


def score(evaluation, output):
    registration, inputs = registered_inputs(evaluation)
    report = load(output / "report.json")
    refs = references(evaluation, registration)
    request = output / "request.json"
    prompts = output.parent / "prompts.json"
    if sha(request) != report["request_sha256"] or sha(prompts) != load(request)["prompts_sha256"]:
        raise ValueError("Text baseline request changed")
    expected = {key: row["image_sha256"] for key, row in inputs.items() if row["condition"] == "normal"}
    prepared = load(prompts)
    if (prepared["input_manifest_sha256"] != registration["input_manifest_sha256"]
            or {row["id"]: row["image_sha256"] for row in prepared["cases"]} != expected):
        raise ValueError("Baseline prompts must be matched to the registered images")
    records = report["raw_responses"]
    if len(records) != len(expected) or {row["id"] for row in records} != expected.keys():
        raise ValueError("Text baseline results must cover the normal cohort")
    pairs = defaultdict(list)
    results = []
    for row in records:
        path = output / (row["id"] + ".json")
        if sha(path) != row["sha256"]:
            raise ValueError("Saved Qwen output changed")
        actual = load(path)
        if actual["id"] != row["id"] or actual["vram_bytes"] != 0:
            raise ValueError("Baseline response identity or device differs")
        ref = refs[row["id"].removesuffix("_normal")]
        raw = actual["response"]
        answer = raw["message"]["content"]
        stopped = raw["done_reason"] == "stop"
        exact = stopped and len(answer.strip()) <= 128 and normalized(answer) == normalized(ref["answer"])
        if ref["kind"] == "qa":
            pairs[ref["paired_group"]].append(exact)
        results.append({"id": row["id"], "kind": ref["kind"], "exact": exact, "stopped": stopped,
                        "nonempty": bool(answer.strip()), "answer_codepoints": len(answer.strip()),
                        "elapsed_seconds": actual["elapsed_seconds"], "prompt_tokens": raw.get("prompt_eval_count"),
                        "output_tokens": raw.get("eval_count"), "quantization": actual["quantization"],
                        "resident_model_bytes": actual["resident_model_bytes"], "vram_bytes": actual["vram_bytes"]})
    summary = {kind: {"cases": len(rows), "exact": sum(row["exact"] for row in rows),
                      "stopped": sum(row["stopped"] for row in rows), "nonempty": sum(row["nonempty"] for row in rows)}
               for kind in TASKS if (rows := [row for row in results if row["kind"] == kind])}
    if len(pairs) != registration["paired_qa_groups"] or any(len(pair) != 2 for pair in pairs.values()):
        raise ValueError("A text baseline question pair is incomplete")
    scores = {"baseline": MODEL, "summary": summary, "paired_qa_both_exact": sum(all(pair) for pair in pairs.values()),
              "raw_report_sha256": sha(output / "report.json"), "references_sha256": registration["references_sha256"],
              "input_manifest_sha256": registration["input_manifest_sha256"], "results": results,
              "book_coherence_rated": False, "single_gpu_efficiency_measured": False}
    save_json(output / "scores.json", scores)
    print(json.dumps({"summary": summary, "paired_qa_both_exact": scores["paired_qa_both_exact"]}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", type=Path, default=ROOT / "artifacts/compact_visual_span_evaluation_v1")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/compact_visual_span_qwen_reference_v1")
    stages = parser.add_subparsers(dest="stage", required=True)
    prep = stages.add_parser("prepare")
    prep.add_argument("--data", type=Path, default=ROOT / "data/visual_sequence_corpus_v1")
    infer = stages.add_parser("generate")
    infer.add_argument("--local", type=Path, default=ROOT.parent / "LocalLLM")
    stages.add_parser("score")
    args = parser.parse_args()
    if args.stage == "prepare":
        prepare(args.data, args.evaluation, args.output)
    elif args.stage == "generate":
        signal.signal(signal.SIGTERM, terminate)
        generate(args.output / "prompts.json", args.local.resolve(), args.output / "generation")
    else:
        score(args.evaluation, args.output / "generation")

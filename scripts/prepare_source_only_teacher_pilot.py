"""Draft modern Chinese for the training-only source audit using a local Ollama model."""

import argparse
import fcntl
import os
from pathlib import Path
import socket
import subprocess
import time
import requests
from scripts.evaluate_anchor_gap_writer import EVIDENCE, ROOT, load, sha, write_new

PROMPT = (
    "请将以下整段古文译成通顺、准确的现代汉语。仅依据给出的字句，保留人名、地名、数字、"
    "否定和说话者，不补写上下文，没有明示的人名不猜测。只写译文，不加解释。\n\n"
)
MODEL = "qwen3:8b-q4_K_M"


def collect(local):
    sample = EVIDENCE / "training_sample.json"
    out = ROOT / "artifacts/source_only_teacher_pilot_v1"
    out.mkdir(exist_ok=True)
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    addr = f"127.0.0.1:{port}"
    base = f"http://{addr}/api/"
    folder = local / ".local"
    env = os.environ | {
        "CUDA_VISIBLE_DEVICES": "0",
        "OLLAMA_HOST": addr,
        "OLLAMA_MODELS": str(folder / "models/ollama"),
        "OLLAMA_NUM_PARALLEL": "1",
        "OLLAMA_MAX_LOADED_MODELS": "1",
    }
    start = time.monotonic()
    with (out / "server.log").open("a") as log:
        proc = subprocess.Popen(
            [str(folder / "ollama/bin/ollama"), "serve"],
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        try:
            for _ in range(60):
                if proc.poll() is not None:
                    raise RuntimeError("Owned Ollama server exited; inspect server.log")
                try:
                    tags = requests.get(base + "tags", timeout=1).json()["models"]
                    break
                except (requests.ConnectionError, requests.Timeout):
                    time.sleep(0.5)
            else:
                raise RuntimeError("Owned local server failed to start")
            digest = next(t["digest"] for t in tags if t["name"] == MODEL)
            drafts = []
            for r in load(sample):
                query = dict(
                    model=MODEL,
                    prompt=PROMPT + r["source"],
                    think=False,
                    stream=False,
                    options=dict(
                        temperature=0, seed=44193, num_ctx=2048, num_predict=300
                    ),
                )
                d = requests.post(base + "generate", json=query, timeout=120)
                d.raise_for_status()
                v = d.json()
                assert v.get("done") and v["response"].strip()
                drafts.append(
                    dict(
                        row=r["row"],
                        draft=v["response"].strip(),
                        done_reason=v["done_reason"],
                        eval_count=v["eval_count"],
                        eval_duration_ns=v["eval_duration"],
                    )
                )
            write_new(
                EVIDENCE / "teacher_drafts.json",
                dict(
                    model=MODEL,
                    digest=digest,
                    prompt=PROMPT,
                    training_sample_sha=sha(sample),
                    seconds=time.monotonic() - start,
                    drafts=drafts,
                    reference_was_supplied=False,
                    student_updates=0,
                ),
            )
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-llm", type=Path, default=ROOT.parent / "LocalLLM")
    args = parser.parse_args()
    with (ROOT / "artifacts/ilm_gpu0.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        collect(args.local_llm.resolve())

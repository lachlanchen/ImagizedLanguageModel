"""Own a temporary loopback Ollama worker backed by the installed local weights."""

from contextlib import contextmanager
import os
import signal
import socket
import subprocess
import time

import requests
from scripts.align_visual_translation import ROOT, sha, load

SYSTEM = "你是文言文译者。将用户给出的一句文言文忠实地译成通顺的现代汉语，保留人物、事件、否定和引语。只写一份完整译文，不加标题、解释或原文。"


@contextmanager
def local_teacher(model, directory):
    local = ROOT.parent / "LocalLLM" / ".local"
    models = local / "models/ollama"
    name, tag = model.split(":", 1)
    manifest = models / "manifests/registry.ollama.ai/library" / name / tag
    digest = sha(manifest)
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    url = f"http://127.0.0.1:{port}"
    env = dict(
        os.environ,
        CUDA_VISIBLE_DEVICES="0",
        OLLAMA_HOST=f"127.0.0.1:{port}",
        OLLAMA_MODELS=str(models),
        OLLAMA_NUM_PARALLEL="1",
        OLLAMA_MAX_LOADED_MODELS="1",
    )
    with (directory / "server.log").open("a") as log, requests.Session() as session:
        session.trust_env = False
        proc = subprocess.Popen(
            [str(local / "ollama/bin/ollama"), "serve"],
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            for _ in range(100):
                if proc.poll() is not None:
                    raise RuntimeError("Local inference worker exited")
                try:
                    tags = session.get(url + "/api/tags", timeout=2).json()["models"]
                    break
                except requests.RequestException:
                    time.sleep(0.3)
            else:
                raise TimeoutError("Local inference startup timeout")
            if not any(t["name"] == model and t["digest"] == digest for t in tags):
                raise ValueError("Local model tag does not match its stored manifest")

            def ask(system, user, **settings):
                response = session.post(
                    url + "/api/chat",
                    json=dict(
                        model=model,
                        stream=False,
                        keep_alive="2m",
                        messages=[dict(role="system", content=system), user],
                        **settings,
                    ),
                    timeout=150,
                )
                response.raise_for_status()
                raw = response.json()
                if not raw.get("done") or raw["model"] != model:
                    raise ValueError("Incomplete local response")
                return dict(
                    text=raw["message"]["content"].strip(),
                    stop=raw["done_reason"],
                    tokens=raw["eval_count"],
                    duration_ns=raw["total_duration"],
                    prompt_tokens=raw["prompt_eval_count"],
                    generation_ns=raw["eval_duration"],
                )

            yield ask, digest, sum(layer["size"] for layer in load(manifest)["layers"])
        finally:
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGTERM)
                try:
                    proc.wait(timeout=12)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait()

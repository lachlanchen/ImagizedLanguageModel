"""Compare autonomous answers to the same printed passages on one local 4090."""

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from time import monotonic, sleep
import base64, os, signal, socket, subprocess
import numpy as np
import requests
from scripts.prepare_competing_facts import PUB, WORK, ROOT, read, dump, sha
from scripts.evaluate_competing_facts import dataset
from scripts.evaluate_visual_discourse import packed
from scripts.evaluate_visual_cursor import normalize
from scripts.train_paired_evidence import gpu_owner
from scripts.measure_writer_energy import snapshot, consumed
from ilm.visual_lm.portable_writer import save_grid

STORE = ROOT.parent / "LocalLLM/.local/models/ollama"
BINARY = ROOT.parent / "LocalLLM/.local/ollama/bin/ollama"
MODELS = {
    "qwen3:8b-q4_K_M": "a3de86cd1c132c822487ededd47a324c50491393e6565cd14bafa40d0b8e686f",
    "qwen3-vl:8b-instruct-q4_K_M": "1329cd5ab37e5ef6faef37cd8ff8660ffc96ed5b47b34fddf8eb0912e6c82466",
}
INSTRUCTION = "Answer using only the passage. Use the language of the question. Give the shortest natural answer that completely answers the question. Copy its wording when appropriate. For a yes/no question answer yes or no. If the passage cannot answer the question, answer unknown. Output only the answer."


@contextmanager
def server():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    url = f"http://127.0.0.1:{port}"
    env = dict(
        os.environ,
        CUDA_VISIBLE_DEVICES="0",
        OLLAMA_HOST=f"127.0.0.1:{port}",
        OLLAMA_MODELS=str(STORE),
        OLLAMA_NUM_PARALLEL="4",
        OLLAMA_MAX_LOADED_MODELS="1",
        OLLAMA_NO_CLOUD="1",
        OLLAMA_CONTEXT_LENGTH="4096",
    )
    with (WORK / "qwen-server.log").open("ab") as log:
        p = subprocess.Popen(
            [str(BINARY), "serve"],
            stdout=log,
            stderr=subprocess.STDOUT,
            env=env,
            start_new_session=True,
        )
        try:
            for _ in range(100):
                if p.poll() is not None:
                    raise RuntimeError("Local Ollama exited at startup")
                try:
                    if requests.get(url + "/api/version", timeout=1).ok:
                        break
                except requests.RequestException:
                    pass
                sleep(0.2)
            else:
                raise TimeoutError("Ollama startup")
            yield url
        finally:
            if p.poll() is None:
                os.killpg(p.pid, signal.SIGTERM)
                try:
                    p.wait(20)
                except subprocess.TimeoutExpired:
                    os.killpg(p.pid, signal.SIGKILL)
                    p.wait()


def png(tiles, ids):
    # Encode the original ink tiles with the same PNG writer as the optical CLI.
    import tempfile

    with tempfile.TemporaryDirectory(dir=WORK) as directory:
        path = Path(directory) / "page.png"
        save_grid(tiles[np.asarray(ids, dtype=int)].tobytes(), path)
        return base64.b64encode(path.read_bytes()).decode()


def run(url, model, mode, rows, tiles, tag):
    out = PUB / (tag + "-outputs.json.gz")
    rc = PUB / (tag + "-scores.json")
    if rc.is_file():
        if sha(out) != read(rc)["sha256"]:
            raise ValueError("Changed comparison outputs")
        return
    expected = MODELS[model]
    if sha(STORE / "blobs" / ("sha256-" + expected)) != expected:
        raise ValueError("Model hash mismatch")
    meta = requests.post(url + "/api/show", json={"model": model}, timeout=30).json()
    if meta["details"]["quantization_level"] != "Q4_K_M":
        raise ValueError("Wrong quantization")

    def ask(r):
        if mode == "image":
            user = dict(
                role="user",
                content="The first image is the passage. The second image is the question.",
                images=[png(tiles, r["s"]), png(tiles, r["q"])],
            )
        else:
            user = dict(
                role="user",
                content="Passage:\n" + r["source"] + "\n\nQuestion:\n" + r["question"],
            )
        opts = dict(
            model=model,
            stream=False,
            keep_alive="10m",
            options=dict(temperature=0, seed=1157, num_predict=96, num_ctx=4096),
            messages=[dict(role="system", content=INSTRUCTION), user],
        )
        if model.startswith("qwen3:"):
            opts["think"] = False
        res = requests.post(url + "/api/chat", json=opts, timeout=180)
        if not res.ok:
            raise RuntimeError(str(res.status_code) + " " + res.text[:400])
        obj = res.json()
        answer = obj["message"]["content"].strip()
        ended = obj.get("done_reason") == "stop"
        return dict(
            key=r["key"],
            doc=r["doc"],
            kind=r["kind"],
            answer=answer,
            stopped=ended,
            correct=bool(ended and normalize(answer) in set(map(normalize, r["refs"]))),
            generated_tokens=obj.get("eval_count"),
            input_tokens=obj.get("prompt_eval_count"),
        )

    meter = snapshot()
    start = monotonic()
    detail = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        for i, r in enumerate(pool.map(ask, rows)):
            detail.append(r)
            if (i + 1) % 32 == 0:
                print(tag, i + 1, len(rows), round(monotonic() - start, 1), flush=True)
    seconds = monotonic() - start
    energy = consumed(meter, snapshot())
    loaded = requests.get(url + "/api/ps", timeout=5).json()["models"]
    assert any(
        r["name"].lower() == model.lower() and r["size_vram"] > 0 for r in loaded
    ), loaded
    split = {}
    for kind in ("all", "squad", "coqa", "cmrc", "drcd"):
        group = [r for r in detail if kind == "all" or r["kind"] == kind]
        pages = {}
        for r in group:
            pages.setdefault(r["doc"], []).append(r["correct"])
        assert all(len(v) == 2 for v in pages.values())
        split[kind] = dict(
            n=len(group),
            correct=sum(r["correct"] for r in group),
            stopped=sum(r["stopped"] for r in group),
            both=sum(all(v) for v in pages.values()),
            pages=len(pages),
        )
    packed(out, detail)
    dump(
        rc,
        dict(
            model=model,
            weights=expected,
            modality=mode,
            quantization="Q4_K_M",
            ollama=requests.get(url + "/api/version").json(),
            seconds=seconds,
            joules=energy["joules"],
            size_vram=loaded[0]["size_vram"],
            sha256=sha(out),
            metrics=split,
        ),
    )
    print(tag, split, flush=True)


def main():
    rows, enc = dataset()
    all_rows = [r for r in rows if r["test"] == "human"]
    short = []
    for k in ("squad", "cmrc", "coqa", "drcd"):
        short.extend([r for r in all_rows if r["kind"] == k][:32])
    tiles = np.array([np.frombuffer(v, np.uint8) for v in enc.tiles])
    with gpu_owner(), server() as url:
        text, vl = MODELS
        run(url, text, "text", all_rows, tiles, "qwen8-text")
        run(url, vl, "image", short, tiles, "qwen8vl-image")
        run(url, vl, "text", short, tiles, "qwen8vl-text")


if __name__ == "__main__":
    from scripts.verse_gpu_runtime import configure_cuda_cli

    configure_cuda_cli()
    main()

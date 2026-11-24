"""Compare the visual writer and a local quantized vision model on identical question images."""

from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
from time import monotonic
import argparse
from pathlib import Path
import requests
import torch

from scripts.benchmark_competing_qwen import MODELS, STORE, png, server
from scripts.benchmark_open_exam import VLM, ask
from scripts.evaluate_evidence_views import groups, official_index
from scripts.evaluate_visual_discourse import unpack
from scripts.measure_writer_energy import snapshot, consumed
from scripts.prepare_evidence_views import Alphabet, MODEL, PUB
from scripts.prepare_open_exam import dump, sha
from scripts.train_paired_evidence import gpu_owner
from scripts.write_open_exam import load, infer
from scripts.write_visual_discourse import Writer


def measure(root, report):
    judge, _, _, dev, _ = official_index()
    sets = groups(judge, dev)
    selected = []
    for group in ("coqa", "drop"):
        docs = sorted(
            {r["doc"] for r in sets[group]},
            key=lambda k: sha256(("view-benchmark " + k).encode()).digest(),
        )[:8]
        selected.extend(dict(r, group=group) for r in sets[group] if r["doc"] in docs)
    assert len(selected) == 32
    a = Alphabet()
    seq = [(a.encode(r["question"]), a.encode(r["page"])) for r in selected]
    tiles = a.array()
    if sha(STORE / "blobs" / ("sha256-" + MODELS[VLM])) != MODELS[VLM]:
        raise ValueError("Unexpected comparison weights")
    outcomes = {}
    costs = {}
    with gpu_owner(), torch.inference_mode():
        writer = Writer(device="cuda")
        net = load("highlight", "cuda", root)
        infer(writer, net, tiles[seq[0][0]], tiles[seq[0][1]], geometry=(320, 4096))
        torch.cuda.synchronize()
        start = monotonic()
        meter = snapshot()
        bypixels = {bytes(b): c for b, c in zip(tiles, a.chars)}
        local = []
        for q, p in seq:
            out, stop = infer(writer, net, tiles[q], tiles[p], geometry=(320, 4096))
            local.append(
                (
                    "".join(bypixels[bytes(b)] for b in out),
                    stop,
                    sha256(out.tobytes()).hexdigest(),
                )
            )
        torch.cuda.synchronize()
        costs["highlight"] = dict(
            seconds=monotonic() - start, joules=consumed(meter, snapshot())["joules"]
        )
        original = {
            (r["doc"], r["turn"], r["group"]): r
            for r in unpack(report / "highlight-outcomes.json.gz")
        }
        serial_differences = 0
        for r, (_, stop, digest) in zip(selected, local, strict=True):
            rec = original[r["doc"], r["turn"], r["group"]]
            if stop != rec["stop"] or digest != rec["sha256"]:
                serial_differences += 1
        del net, writer
        torch.cuda.empty_cache()
        with server() as url:
            ask(url, VLM, dict(role="user", content="Reply OK."))
            messages = [
                dict(
                    role="user",
                    content="The first image is the passage. The second image ends with the current question. Earlier dialogue, when present, precedes a vertical bar. Answer the current question from the passage. Give only the shortest complete answer.",
                    images=[png(tiles, p), png(tiles, q)],
                )
                for q, p in seq
            ]
            start = monotonic()
            meter = snapshot()
            with ThreadPoolExecutor(max_workers=4) as pool:
                replies = list(pool.map(lambda msg: ask(url, VLM, msg), messages))
            costs["qwen"] = dict(
                seconds=monotonic() - start,
                joules=consumed(meter, snapshot())["joules"],
            )
            assert any(
                r["name"].casefold() == VLM.casefold() and r["size_vram"] > 0
                for r in requests.get(url + "/api/ps", timeout=15).json()["models"]
            )
    for name, vals in (("highlight", [(t, e) for t, e, h in local]), ("qwen", replies)):
        results = []
        for r, (text, stop) in zip(selected, vals, strict=True):
            results.append(
                dict(
                    doc=r["doc"],
                    turn=r["turn"],
                    group=r["group"],
                    exact=stop
                    and judge.normalize_answer(text)
                    in list(map(judge.normalize_answer, r["refs"])),
                    stopped=stop,
                    text=text,
                )
            )
        outcomes[name] = results
    dump(
        report / "qwen.json",
        dict(
            local_model=MODELS[VLM],
            serial_batch_pixel_or_stop_differences=serial_differences,
            cost=costs,
            answers=outcomes,
        ),
    )


if __name__ == "__main__":
    from scripts.verse_gpu_runtime import configure_cuda_cli

    configure_cuda_cli()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-dir", type=Path, default=MODEL)
    p.add_argument("--report", type=Path, default=PUB)
    a = p.parse_args()
    measure(a.model_dir, a.report)

"""Measure timed PNG inference and transfer to the fixed historical dictionary entries."""

from hashlib import sha256
from tempfile import TemporaryDirectory
from pathlib import Path
from time import monotonic

import numpy as np
import torch
from ilm.visual_lm.book_prefix import observations
from ilm.visual_lm.portable_writer import read_image
from scripts.evaluate_recurrent_history import historical_pages
from scripts.evaluate_visual_discourse import draw, unpack
from scripts.prepare_open_exam import MODEL, PUB, WORK, read, dump
from scripts.measure_writer_energy import snapshot, consumed
from scripts.train_paired_evidence import gpu_owner
from scripts.train_open_exam import corpus
from scripts.write_open_exam import infer, load
from scripts.write_recurrent_evidence import generate
from scripts.write_visual_discourse import Writer, encode_pages


def inference_cost(writer):
    ids = read(PUB / "qwen.json")["ids"]
    rows, tiles, _ = corpus("held")
    indexed = {r["id"]: r for r in rows}
    exact_run = {
        r["id"]: r
        for r in unpack(WORK / "matched-responses.json.gz")
        if r["condition"] == "intact"
    }
    net = load("matched", "cuda")
    with TemporaryDirectory(dir=WORK) as tmp:
        paths = []
        for i, identifier in enumerate(ids):
            row = indexed[identifier]
            d = Path(tmp) / str(i)
            d.mkdir()
            for name in ("qi", "pi"):
                draw(d, name, tiles[row[name]])
            paths.append((d, len(row["qi"]), len(row["pi"])))

        def write(d, nq, npg):
            pix, end = infer(
                writer, net, read_image(d / "qi.png", nq), read_image(d / "pi.png", npg)
            )
            draw(d, "answer", pix)
            return sha256(pix.tobytes()).hexdigest(), end

        write(*paths[0])
        torch.cuda.synchronize()
        meter, start = snapshot(), monotonic()
        results = [write(*args) for args in paths]
        torch.cuda.synchronize()
        seconds = monotonic() - start
        energy = consumed(meter, snapshot())
    for identifier, (digest, end) in zip(ids, results, strict=True):
        r = exact_run[identifier]
        if (digest, end) != (r["output_sha256"], r["stop"]):
            raise ValueError(f"Physical replay differs for {identifier}")
    dump(
        PUB / "optical-cost.json",
        dict(
            n=len(ids),
            seconds=seconds,
            joules=energy["joules"],
            verified_against_full_run=len(results),
            scope="warm, serial PNG read, neural answer and PNG write; no model or source-page preparation",
        ),
    )


def history(writer):
    rows, pages, targets = historical_pages()
    tiles, sequences = encode_pages(writer.palette, pages)
    features = torch.from_numpy(observations({"tiles": tiles}, "ink")).cuda()
    report = {}
    for arm in ("parent", "exchanged", "matched"):
        net = load(arm, "cuda")
        out = []
        for i in range(0, len(rows), 16):
            q = sequences[2 * i : 2 * (i + 16) : 2]
            p = sequences[2 * i + 1 : 2 * (i + 16) : 2]
            out.extend(
                generate(
                    writer.model,
                    net,
                    features,
                    len(writer.palette),
                    q,
                    p,
                    [a + b for a, b in zip(p, q, strict=True)],
                    64,
                )
            )
        items = []
        for i, (r, (ids, stop)) in enumerate(zip(rows, out, strict=True)):
            pixels = tiles[np.asarray(ids, int)]
            entry = dict(
                lang=r["language"],
                stage=r["stage"],
                label=r["label"],
                correct=stop and np.array_equal(pixels, targets[i]),
                stopped=stop,
                cells=len(pixels),
                sha256=sha256(pixels.tobytes()).hexdigest(),
            )
            if r["example"] and arm == "matched":
                d = MODEL / "examples/yan" / entry["lang"] / entry["stage"]
                d.mkdir(parents=True, exist_ok=True)
                for k, a in (
                    ("question", pages[2 * i]),
                    ("passage", pages[2 * i + 1]),
                    ("reference", targets[i]),
                    ("answer", pixels),
                ):
                    draw(d, k, a)
                dump(
                    d / "receipt.json",
                    dict(
                        entry,
                        question_cells=len(pages[2 * i]),
                        page_cells=len(pages[2 * i + 1]),
                    ),
                )
            items.append(entry)
        report[arm] = {
            lang: sum(r["correct"] for r in items if r["lang"] == lang)
            for lang in ("en", "zh")
        }
        dump(PUB / (arm + "-history.json"), items)
    dump(PUB / "history-summary.json", report)


if __name__ == "__main__":
    from scripts.verse_gpu_runtime import configure_cuda_cli

    configure_cuda_cli()
    with gpu_owner(), torch.inference_mode():
        writer = Writer(device="cuda")
        inference_cost(writer)
        history(writer)

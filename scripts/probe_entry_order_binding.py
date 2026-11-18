"""Freeze the writer and vary only the order or presence of the two pictured records."""
from time import monotonic

import json

import numpy as np
import torch

from scripts.audit_general_entry_writer import EVIDENCE
from scripts.evaluate_general_entry_writer import english_request, pictured
from scripts.evaluate_same_passage_writer import generate, image_archive
from scripts.prepare_pictured_entry_answers import sha, write_json
from scripts.train_general_entry_writer import (AUTHORED, DEVICE, Entries, PairCorpus, Rasterizer, WORK, load_writer, small,
                                                memory_bank, physical_tables)


@torch.inference_mode()
def main():
    start = monotonic()
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = True
    bank = memory_bank()
    corpus = Entries(bank, "development")
    # Match the original evaluation's feature matrix dimensions and GPU precision.
    PairCorpus(bank, "development")
    small(bank, "development")
    raster = Rasterizer()
    for m in corpus.meta:
        pictured(bank, raster, english_request(m))
    for line in AUTHORED.read_text().splitlines():
        pictured(bank, raster, json.loads(line)["text"])
    marker = bank.lookup[raster.tile("【")]
    parts = []
    for c in corpus.cards:
        starts = np.flatnonzero(c == marker)
        assert len(starts) == 2 and starts[0] == 0
        parts.append(np.split(c, [starts[1]]))
    features, mem, _ = physical_tables(bank, DEVICE)
    net = load_writer(WORK / "paired", DEVICE)
    ink = net.ink(features)
    keys = net.write_key(ink[mem])
    answers = [b"".join(bank.tiles[i] for i in y) for y in corpus.y]
    modes = {}
    cache = WORK / "order_probe"
    cache.mkdir()
    for mode in ("original", "swap", "requested_only", "distractor_only"):
        sources = []
        for j, meta in enumerate(corpus.meta):
            a, b = parts[meta["card"]]
            assert a[1] != b[1] and corpus.y[j][0] in (a[1], b[1])
            wanted, other = (a, b) if a[1] == corpus.y[j][0] else (b, a)
            if mode == "original":
                s = corpus.cards[meta["card"]]
            elif mode == "swap":
                s = np.r_[b, a]
            else:
                s = wanted if mode == "requested_only" else other
            sources.append(s)
        images, stopped = generate(net, ink, mem, keys, sources, corpus.q, bank)
        exact = np.array([e and x == y for x, e, y in zip(images, stopped, answers, strict=True)])
        form = [e and x[-256:-128] == y[-256:-128] for x, e, y in zip(images, stopped, answers, strict=True)]
        path = cache / (mode + ".npz")
        image_archive(path, images, stopped)
        modes[mode] = dict(exact=int(exact.sum()), form=sum(form), stopped=sum(stopped), file_sha256=sha(path))
        if mode == "original":
            with np.load(WORK / "paired/simple.npz") as old, np.load(path) as fresh:
                assert all(np.array_equal(old[k], fresh[k]) for k in old.files)
            baseline = images
            initial = exact
        else:
            modes[mode].update(identical_streams=sum(x == y for x,y in zip(baseline, images, strict=True)),
                               gained=int((~initial & exact).sum()), lost=int((initial & ~exact).sum()))
    write_json(EVIDENCE / "order_probe.json", dict(modes=modes, cases=len(corpus.q), seconds=monotonic()-start,
               weights=sha(WORK / "paired/final.safetensors"), program_sha256=sha(__file__)))
    print(modes)


if __name__ == "__main__":
    main()

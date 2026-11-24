"""Measure which human-fact conditioning errors survive every local fact replacement."""

from collections import Counter, defaultdict
from hashlib import sha256
import numpy as np
import torch
from scripts.evaluate_fact_channel import observed
from scripts.prepare_fact_ranking_confirmation import PUB, WORK
from scripts.prepare_fact_channel import MODEL, Ink, read, dump
from scripts.write_fact_channel import VisualAnswer
from scripts.train_fact_channel import gpu_owner
from scripts.verse_gpu_runtime import configure_cuda_cli


def offsets(seq, word):
    if not word:
        raise ValueError("Empty annotated fact")
    return [
        i for i in range(len(seq) - len(word) + 1) if seq[i : i + len(word)] == word
    ]


def recoverable(emitted, reference, offset, length):
    prefix, suffix = reference[:offset], reference[offset + length :]
    return (
        len(emitted) >= len(prefix) + len(suffix)
        and emitted[: len(prefix)] == prefix
        and emitted[len(emitted) - len(suffix) :] == suffix
    )


def main():
    rows = read(WORK / "confirmation.json")
    with gpu_owner(), torch.inference_mode():
        net = VisualAnswer("human", "cuda")
        ink = Ink(np.load(MODEL / "palette.npy"))
        records = observed(net, rows, ink, gold=True)
    totals = defaultdict(Counter)
    examples = []
    for r, rec in zip(rows, records, strict=True):
        group = totals[r["task"]]
        fact = [ink.tiles[k] for k in ink.encode(r["facts"]["answer"])]
        refs = [[ink.tiles[k] for k in ink.encode(a)] for a in r["answers"]]
        raw = bytes.fromhex(rec["ink_hex"])
        assert sha256(raw).hexdigest() == rec["sha256"]
        emitted = [raw[i : i + 128] for i in range(0, len(raw), 128)]
        exact = bool(rec["stop"] and emitted in refs)
        group.update(n=1, exact=int(exact))
        # Keep all accepted references, including alternate Chinese wordings.
        framed = [(ref, ix[0]) for ref in refs if len(ix := offsets(ref, fact)) == 1]
        if not framed:
            continue
        repeats = len(offsets(emitted, fact))
        group.update(framed=1)
        if exact:
            group.update(framed_correct=1)
            continue
        real = any(recoverable(emitted, y, o, len(fact)) for y, o in framed)
        group.update(
            framed_failed=1,
            stopped_failures=int(rec["stop"]),
            recoverable=int(real),
            recoverable_stopped=int(real and rec["stop"]),
            fact_missing=int(repeats == 0),
            fact_once=int(repeats == 1),
            fact_repeated=int(repeats >= 2),
        )
        examples.append(
            dict(
                task=r["task"],
                q=r["q"],
                reference=r["answers"],
                fact=r["facts"]["answer"],
                output=rec["text"],
                stopped=rec["stop"],
                repeats=repeats,
                recoverable=real,
            )
        )
    known = read(PUB / "formulation.json")
    if any(g["exact"] != known[t]["bitmap_exact"] for t, g in totals.items()):
        raise ValueError("Supplied-fact physical replay differs")
    dump(PUB / "frame.json", dict(totals=totals, examples=examples))
    print(dict(totals), flush=True)


if __name__ == "__main__":
    configure_cuda_cli()
    main()

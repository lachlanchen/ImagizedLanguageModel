"""Build writing actions using only training bitmap identities and their order."""
from pathlib import Path
import os

import numpy as np
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer

from scripts.reserve_independent_optical_actions import BASE, PROTOCOL, PUB, ROOT, RUN, dump, read, sha
from scripts.sample_visual_library import library

OFFSET = 0xE000
MAX_CELLS = 8
ARMS = ("cell", "strip")


def prepare():
    RUN.mkdir(exist_ok=True)
    PUB.mkdir(parents=True, exist_ok=True)
    receipt = PUB / "preparation.json"
    if receipt.exists():
        assert all(sha(RUN / name) == digest for name, digest in read(receipt)["files"].items())
        return
    corpus = ROOT / "artifacts/bilingual_ink_v1"
    source = read(ROOT / "references/evidence/bilingual_ink_v1/corpus.json")["files"]
    for name in ("images.npy", "lengths.npy"):
        assert sha(corpus / name) == source[name]
    books = library("train")
    opus = np.load(corpus / "images.npy", mmap_mode="r")
    n = len(opus) + len(books)
    cells = np.zeros((n, 2, 96), dtype=np.uint16)
    cells[:len(opus)] = opus
    for i, row in enumerate(books, len(opus)):
        for t, line in enumerate(row["cells"]):
            cells[i, t, :len(line)] = line
    lengths = np.count_nonzero(cells, axis=2)
    assert np.array_equal(lengths[:len(opus)], np.load(corpus / "lengths.npy"))
    assert np.all(lengths > 0)
    np.save(RUN / "cells.npy", cells)
    inventory = read(BASE / "inventories.json")
    tables = {arm: [] for arm in ARMS}
    stats = {}
    for t, inv in enumerate(inventory):
        assert inv[0] == 1
        atoms = inv[1:]
        inverse = np.zeros(max(inv) + 1, dtype=np.uint16)
        inverse[atoms] = np.arange(1, len(inv))
        strings = ["".join(chr(OFFSET + int(j)) for j in line[:size])
                   for line, size in zip(cells[:, t], lengths[:, t], strict=True)]
        encoder = Tokenizer(BPE())
        encoder.train_from_iterator(strings, trainer=BpeTrainer(
            vocab_size=len(atoms) + 2048, min_frequency=2, max_token_length=MAX_CELLS,
            initial_alphabet=[chr(OFFSET + j) for j in atoms], show_progress=False))
        vocab = encoder.get_vocab()
        strip = [()] + [tuple(ord(c) - OFFSET for c in s) for s in sorted(vocab, key=vocab.get)]
        assert max(map(len, strip)) <= MAX_CELLS
        assert {a[0] for a in strip if len(a) == 1} == set(atoms)
        tables["strip"].append(strip)
        tables["cell"].append([()] + [(k,) for k in atoms])
        targets = np.zeros((n, 96), dtype=np.uint16)
        for i in range(0, n, 1024):
            encoded = encoder.encode_batch(strings[i:i+1024])
            for k, item in enumerate(encoded, i):
                ids = [j + 1 for j in item.ids]
                flat = [v for j in ids for v in strip[j]]
                if not np.array_equal(flat, cells[k, t, :lengths[k,t]]):
                    raise ValueError(f"Physical target round trip failed on {k}/{t}")
                targets[k, :len(ids)] = ids
        np.save(RUN / f"strip_{t}.npy", targets)
        cell = inverse[cells[:, t]]
        assert np.array_equal(cell != 0, cells[:, t] != 0)
        np.save(RUN / f"cell_{t}.npy", cell)
        stats[t] = dict(glyphs=int(lengths[:, t].sum()), actions=int(np.count_nonzero(targets)),
                        types=len(strip)-1, max_cells=max(map(len, strip)))
    dump(RUN / "tables.json", tables)
    files = ["cells.npy", "tables.json"] + [f"{arm}_{t}.npy" for arm in ARMS for t in (0, 1)]
    dump(receipt, dict(pairs=n, books=len(books), opus=len(opus), stats=stats,
                       opus_files=source, library=sha(ROOT/"artifacts/visual_library_v1/train.jsonl"),
                       palette=sha(BASE/"palette.npy"), files={k: sha(RUN/k) for k in files},
                       program=sha(Path(__file__)), protocol=sha(PROTOCOL)))
    print(stats, flush=True)


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    prepare()

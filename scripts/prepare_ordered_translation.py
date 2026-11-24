"""Prepare leakage-grouped Mandarin/English translation pairs and their physical glyphs."""

from collections import Counter, defaultdict
from hashlib import sha256
from pathlib import Path
import re
import unicodedata
from zipfile import ZipFile
from urllib.request import urlopen

import numpy as np
from opencc import OpenCC
from scripts.prepare_region_writer import Ink, ROOT, sha, read, dump

WORK = ROOT / "artifacts/ordered_translation_v1"
PUB = ROOT / "references/evidence/ordered_translation_v1"
MODEL = ROOT / "models/ordered_translation"
RAW = ROOT / "data/raw/tatoeba_cmn/cmn-eng.zip"
URL = "https://www.manythings.org/anki/cmn-eng.zip"
DIGEST = "d606361aa7312dcdfb15ef4fc51cb1e7ed8bc2dc608a7b192626cd36e8403106"
SCRIPT = OpenCC("t2s")


def key(s):
    return "".join(
        c
        for c in SCRIPT.convert(unicodedata.normalize("NFKC", s)).casefold()
        if c.isalnum()
    )


def hashed(s):
    return sha256(s.encode()).hexdigest()


def prepare():
    WORK.mkdir(parents=True, exist_ok=True)
    PUB.mkdir(parents=True, exist_ok=True)
    MODEL.mkdir(parents=True, exist_ok=True)
    if not RAW.exists():
        RAW.parent.mkdir(parents=True, exist_ok=True)
        with urlopen(URL, timeout=60) as r:
            content = r.read()
        if sha256(content).hexdigest() != DIGEST:
            raise ValueError("Archive revision differs")
        RAW.write_bytes(content)
    if sha(RAW) != DIGEST:
        raise ValueError("Archive revision differs")
    links = {}

    def root(x):
        links.setdefault(x, x)
        while x != links[x]:
            links[x] = links[links[x]]
            x = links[x]
        return x

    data = []
    with ZipFile(RAW) as z:
        for line in z.read("cmn.txt").decode("utf-8-sig").splitlines():
            en, zh, credit = line.split("\t")
            a, b = "en/" + key(en), "zh/" + key(zh)
            if len(a) == 3 or len(b) == 3:
                continue
            x, y = root(a), root(b)
            links[max(x, y)] = min(x, y)
            data.append((en, zh, credit, a))
    ink = Ink()
    # Identify non-renderable samples before partitioning the output inventory.
    for c in sorted(set("".join(en + zh for en, zh, _, _ in data))):
        ink.encode(c)
    bad = set(ink.bad)
    ink = Ink()
    groups = defaultdict(list)
    excluded = Counter()
    for en, zh, credit, a in data:
        if max(len(en), len(zh)) > 96:
            excluded["length"] += 1
            continue
        if bad.intersection(en + zh):
            excluded["missing_font"] += 1
            continue
        groups[root(a)].append((en, zh, credit))
    order = sorted(groups, key=lambda g: hashed("ordered-translation-01/" + g))
    splits = {"train": order[512:], "dev": order[256:512], "test": order[:256]}
    serial = {}
    inventories = [set(), set()]
    for split, gs in splits.items():
        rows = []
        for g in gs:
            for en, zh, credit in groups[g]:
                pair = [ink.encode(s) for s in (en, zh)]
                if split == "train":
                    for vocab, ids in zip(inventories, pair):
                        vocab.update(ids)
                rows.append(
                    dict(family=hashed(g), text=[en, zh], cells=pair, credit=credit)
                )
        serial[split] = rows
        dump(WORK / (split + ".json"), rows)
        if split == "train":
            np.save(MODEL / "palette.npy", ink.array())
            dump(
                MODEL / "readback.json", dict((str(i), c) for c, i in ink.chars.items())
            )
    np.save(WORK / "inputs.npy", ink.array())
    dump(MODEL / "inventories.json", [[1, *sorted(v)] for v in inventories])
    rc = dict(
        archive=DIGEST,
        excluded=excluded,
        raw_rows=len(data),
        groups={k: len(v) for k, v in splits.items()},
        pairs={k: len(v) for k, v in serial.items()},
        shapes=[len(v) for v in inventories],
        files={
            p.name: sha(p)
            for p in [
                *(WORK / (k + ".json") for k in splits),
                WORK / "inputs.npy",
                MODEL / "palette.npy",
                MODEL / "readback.json",
                MODEL / "inventories.json",
            ]
        },
        unsupported_held_targets=[
            sum(
                any(i not in inventories[t] for i in r["cells"][t])
                for r in serial["test"]
            )
            for t in (0, 1)
        ],
    )
    dump(PUB / "corpus.json", rc)
    print(rc)


if __name__ == "__main__":
    prepare()

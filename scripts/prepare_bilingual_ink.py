"""Select optical parallel supervision from the official OPUS-100 English–Chinese splits."""

from collections import Counter
from functools import lru_cache
from hashlib import sha256
from pathlib import Path
import os
import numpy as np
import pyarrow.parquet as pq
import requests
from scripts.prepare_proposition_translation import (
    ROOT,
    BASE,
    OLD,
    WORK as FIRST,
    Ink,
    read,
    dump,
    key,
    sha,
)

WORK = ROOT / "artifacts/bilingual_ink_v1"
PUB = ROOT / "references/evidence/bilingual_ink_v1"
MODEL = ROOT / "models/bilingual_ink"
RAW = ROOT / "data/raw/opus100-en-zh"
REV = "805090dc28bf78897da9641cdf08b61287580df9"
HASH = {
    "train": "e48b522a92493526c39bb9417a5ece77cf406b07c59797bc652072b296914dd9",
    "validation": "ce0428a37c741a3c55ae707ca7b12fc0410b015212ad604f9b305f810188434c",
    "test": "571b811b20859d801561f62a21cc0e8a2fb034dd54bc960bb4b879dc6345fd19",
}


def source(split):
    p = RAW / (split + ".parquet")
    if not p.exists():
        RAW.mkdir(parents=True, exist_ok=True)
        addr = f"https://huggingface.co/datasets/Helsinki-NLP/opus-100/resolve/{REV}/en-zh/{split}-00000-of-00001.parquet"
        with requests.get(addr, stream=True, timeout=90) as response:
            response.raise_for_status()
            temp = p.with_suffix(".partial")
            with temp.open("wb") as output:
                for block in response.iter_content(1 << 20):
                    output.write(block)
            if sha(temp) != HASH[split]:
                raise ValueError(split)
            os.replace(temp, p)
    if sha(p) != HASH[split]:
        raise ValueError(split)
    for batch in pq.ParquetFile(p).iter_batches(batch_size=4096):
        for r in batch.column(0).to_pylist():
            yield [r["en"].strip(), r["zh"].strip()]


def prepare():
    for p in (WORK, PUB, MODEL):
        p.mkdir(parents=True, exist_ok=True)
    ink = Ink(np.load(BASE / "palette.npy"))
    invs = [set(v) for v in read(BASE / "inventories.json")]
    shape = lru_cache(None)(lambda c: ink.encode(c)[0])

    def encode(sides):
        if any(not s or len(s) > 96 for s in sides):
            return None, "length"
        if not any("a" <= c.lower() <= "z" for c in sides[0]) or not any(
            "\u3400" <= c <= "\u9fff" for c in sides[1]
        ):
            return None, "script"
        cells = [list(map(shape, s)) for s in sides]
        if any(c in ink.bad for s in sides for c in s) or any(
            not set(x) <= v for x, v in zip(cells, invs)
        ):
            return None, "outside_palette"
        return cells, None

    used = [set(), set()]

    def reserve(sides):
        for t, s in enumerate(sides):
            used[t].add(key(s))

    # Exclude all prior supervised/held natural domains from the fresh OPUS set.
    for split in ("train", "dev", "test", "news-train", "news-held"):
        for r in read(OLD / (split + ".json")):
            reserve(r["text"])
    first = read(FIRST / "held.json")
    for p in first["pairs"]:
        for z in p["forms"]:
            reserve(z)
    base = read(BASE / "readback.json")
    for p in read(FIRST / "contrasts.json"):
        for fork in p["cells"]:
            reserve(["".join(base[str(j)] for j in ids) for ids in fork])
    stats = {}
    held = []
    for split in ("test", "validation"):
        count = Counter()
        for s in source(split):
            norm = list(map(key, s))
            duplicate = any(n in known for n, known in zip(norm, used))
            reserve(s)
            x, reason = encode(s)
            if duplicate:
                count["overlap"] += 1
            elif reason:
                count[reason] += 1
            else:
                count["eligible"] += 1
                if split == "test" and len(held) < 128:
                    held.append(dict(text=s, cells=x))
        stats[split] = count
    cap = 240000
    n = 0
    arr = np.zeros((cap, 2, 96), dtype=np.uint16)
    lengths = np.zeros((cap, 2), dtype=np.uint8)
    count = Counter()
    for s in source("train"):
        count["screened"] += 1
        x, reason = encode(s)
        if reason:
            count[reason] += 1
            continue
        norm = list(map(key, s))
        if any(not k or k in known for k, known in zip(norm, used)):
            count["overlap"] += 1
            continue
        for t, (ids, k) in enumerate(zip(x, norm)):
            used[t].add(k)
            arr[n, t, : len(ids)] = ids
            lengths[n, t] = len(ids)
        n += 1
        if n == cap:
            break
    if n < cap or len(held) != 128:
        raise ValueError((n, len(held)))
    np.save(WORK / "images.npy", arr)
    np.save(WORK / "lengths.npy", lengths)
    dump(WORK / "held.json", held)
    stats["train"] = dict(count, accepted=n)
    dump(
        PUB / "corpus.json",
        dict(
            revision=REV,
            sources=HASH,
            screening=stats,
            held=128,
            files={
                k: sha(WORK / k) for k in ("images.npy", "lengths.npy", "held.json")
            },
        ),
    )
    print(stats)


if __name__ == "__main__":
    prepare()

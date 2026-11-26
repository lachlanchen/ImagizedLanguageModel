"""Prepare aligned book units, TED talks and independent FLORES articles as visual cells."""

from collections import Counter, defaultdict
import csv
from functools import lru_cache
from hashlib import sha256
from io import TextIOWrapper
from pathlib import Path
import tarfile
from urllib.request import urlopen
from zipfile import ZipFile
import xml.etree.ElementTree as ET

import numpy as np
from scripts.prepare_paired_ink import (
    BASE,
    FIRST,
    OLD,
    ROOT,
    Ink,
    key,
    read,
    dump,
    sha,
    opus_source,
)
from scripts.prepare_translation_paraphrases import WORK as PARAPHRASES

WORK = ROOT / "artifacts/book_breadth_v1"
PUB = ROOT / "references/evidence/book_breadth_v1"
MODEL = ROOT / "models/book_breadth"
SOURCES = {
    "ted2020/en-zh_cn.txt.zip": (
        "https://object.pouta.csc.fi/OPUS-TED2020/v1/moses/en-zh_cn.txt.zip",
        "f1a4e205458d67318d1eee31121c5350b3748caa2bca654a6b52a4a69e8d6202",
    ),
    "flores101/flores101_dataset.tar.gz": (
        "https://dl.fbaipublicfiles.com/flores101/dataset/flores101_dataset.tar.gz",
        "49fa80207b09fcc0eca8253ed13303b3a0ae0f16081af862601c73ac76f2cba6",
    ),
}
BOOKS = {
    "train": "david-copperfield little-women pride-and-prejudice three-musketeers brothers-karamazov anna-karenina dream-red-chamber call-to-arms-lu-xun".split(),
    "dev": "wuthering-heights oliver-twist daodejing".split(),
    "test": "jane-eyre huckleberry-finn tale-of-two-cities shanhaijing".split(),
}
DOMAINS = ("book", "talk", "flores")


def rank(s):
    return sha256(("book-breadth-01/" + s).encode()).hexdigest()


def archive(name):
    url, expected = SOURCES[name]
    p = ROOT / "data/raw" / name
    if not p.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".download")
        with urlopen(url, timeout=90) as r, tmp.open("wb") as w:
            while block := r.read(1 << 20):
                w.write(block)
        if sha(tmp) != expected:
            raise ValueError(name)
        tmp.replace(p)
    if sha(p) != expected:
        raise ValueError(name)
    return p


def paired_windows(units):
    prev = None
    for en, zh in units:
        if en and zh:
            yield en, zh
            if prev:
                yield prev[0] + " " + en, prev[1] + zh
        prev = (en, zh) if en and zh else None


def books(split, hashes):
    for work in BOOKS[split]:
        base = ROOT.parent / "ZhJpBook/books" / work / "work"
        inputs = sorted(base.glob("*/interlinear/chunks/*.json"))
        if not inputs:
            raise FileNotFoundError(work)
        digest = sha256()
        for p in inputs:
            digest.update(str(p.relative_to(base)).encode() + bytes.fromhex(sha(p)))
            for para in read(p)["paragraphs"]:
                units = [
                    (
                        str(
                            u.get("source_en")
                            or "".join(s["t"] for s in u.get("en", []))
                        ).strip(),
                        "".join(s["t"] for s in u.get("zh", [])).strip(),
                    )
                    for u in para["units"]
                ]
                for pair in paired_windows(units):
                    yield split, "book", work, pair
        hashes[work] = dict(chunks=len(inputs), sha256=digest.hexdigest())


def talks():
    with ZipFile(archive("ted2020/en-zh_cn.txt.zip")) as z:
        prefix = "TED2020.en-zh_cn."
        with z.open(prefix + "en") as en, z.open(prefix + "zh_cn") as zh, z.open(
            prefix + "xml"
        ) as xml:
            for _, node in ET.iterparse(xml, events=("end",)):
                if node.tag != "linkGrp":
                    continue
                doc = node.attrib["fromDoc"]
                bucket = int(rank(doc)[:8], 16) % 20
                split = "test" if bucket == 0 else "dev" if bucket == 1 else "train"
                units = [
                    (next(en).decode().strip(), next(zh).decode().strip()) for _ in node
                ]
                if next(en).strip() or next(zh).strip():
                    raise ValueError("Talk alignment boundary")
                for pair in paired_windows(units):
                    yield split, "talk", doc, pair
                node.clear()
            if en.read() or zh.read():
                raise ValueError("Unconsumed talk rows")


def flores():
    claimed = {}
    with tarfile.open(archive("flores101/flores101_dataset.tar.gz")) as tar:
        for split, tag in (("dev", "dev"), ("test", "devtest")):

            def lines(name):
                return TextIOWrapper(
                    tar.extractfile("flores101_dataset/" + name), encoding="utf8"
                )

            meta = csv.DictReader(lines("metedata_" + tag + ".tsv"), delimiter="\t")
            for en, zh, m in zip(
                lines(f"{tag}/eng.{tag}"),
                lines(f"{tag}/zho_simpl.{tag}"),
                meta,
                strict=True,
            ):
                doc = m["URL"]
                if claimed.setdefault(doc, split) == split:
                    yield split, "flores", doc, (en.strip(), zh.strip())


def prepare():
    WORK.mkdir(parents=True, exist_ok=True)
    PUB.mkdir(parents=True, exist_ok=True)
    inventory = [set(x) for x in read(BASE / "inventories.json")]
    ink = Ink(np.load(BASE / "palette.npy"))
    glyph = lru_cache(None)(lambda c: ink.encode(c)[0])
    normal = lru_cache(65536)(key)

    def encode(pair):
        en, zh = pair
        if (
            not 4 <= len(en) <= 96
            or not 2 <= len(zh) <= 96
            or not any("a" <= c.lower() <= "z" for c in en)
            or not any("\u3400" <= c <= "\u9fff" for c in zh)
        ):
            return None
        cells = [list(map(glyph, s)) for s in pair]
        if any(
            not set(v) <= inventory[t] or ink.bad.intersection(pair[t])
            for t, v in enumerate(cells)
        ):
            return None
        return cells

    manifests = {}
    candidates = defaultdict(list)

    def gather(rows):
        for split, dom, doc, pair in rows:
            if split != "train" and encode(pair):
                candidates[split, dom, doc].append(pair)

    for split in ("dev", "test"):
        gather(books(split, manifests))
    gather(talks())
    gather(flores())
    # Check only the prospective held strings, using a superset of earlier exposure.
    wanted = [
        {normal(z[t]) for arr in candidates.values() for z in arr} for t in (0, 1)
    ]
    excluded = [set(), set()]

    def screen(pair):
        for t, s in enumerate(pair):
            if len(s) <= 96 and (k := normal(s)) in wanted[t]:
                excluded[t].add(k)

    for tag in ("train", "news-train", "test", "dev", "news-held"):
        for row in read(OLD / (tag + ".json")):
            screen(row["text"])
    for pair in opus_source("train"):
        screen(pair)
    lookup = read(BASE / "readback.json")

    def screen_ink(pair):
        screen(["".join(lookup[str(i)] for i in side) for side in pair])

    for row in read(FIRST / "contrasts.json"):
        for pair in row["cells"]:
            screen_ink(pair)
    for row in read(PARAPHRASES / "examples.json"):
        for t in (0, 1):
            for side in [row["target"][t], *row["variants"][t]]:
                screen_ink([side, []] if t == 0 else [[], side])
    from scripts.prepare_translation_paraphrases import published

    for t, lang in enumerate(("en", "zh")):
        for row in published(lang, "train").values():
            for s in (row["sentence1"], row["sentence2"]):
                screen([s, ""] if t == 0 else ["", s])

    held = {}
    for split in ("dev", "test"):
        held[split] = []
        for dom in DOMAINS:
            remaining = {}
            for (sp, d, doc), arr in candidates.items():
                if (sp, d) == (split, dom):
                    remaining[doc] = sorted(
                        set(arr), key=lambda z: rank(doc + z[0] + z[1]), reverse=True
                    )
            rows = []
            for _ in range(24):
                gain = 0
                for doc in sorted(remaining, key=rank):
                    while remaining[doc] and len(rows) < 24:
                        pair = remaining[doc].pop()
                        keys = [normal(s) for s in pair]
                        if any(not k or k in excluded[t] for t, k in enumerate(keys)):
                            continue
                        for t, k in enumerate(keys):
                            excluded[t].add(k)
                        rows.append(
                            dict(domain=dom, doc=doc, text=pair, cells=encode(pair))
                        )
                        gain += 1
                        break
                if len(rows) == 24 or not gain:
                    break
            if len(rows) != 24:
                raise ValueError(f"{split} {dom} has only {len(rows)}")
            held[split].extend(rows)
        dump(WORK / (split + ".json"), held[split])
    # Store only packed bitmap addresses in the private training arrays.
    seen = [
        {normal(r["text"][t]) for rows in held.values() for r in rows} for t in (0, 1)
    ]
    storage = []
    labels = []
    docs = defaultdict(set)
    for src in (books("train", manifests), talks()):
        for split, domain, doc, pair in src:
            if split != "train" or not (cells := encode(pair)):
                continue
            keys = [normal(s) for s in pair]
            if any(not k or k in seen[t] for t, k in enumerate(keys)):
                continue
            for t, k in enumerate(keys):
                seen[t].add(k)
            a = np.zeros((2, 96), np.uint16)
            for t, v in enumerate(cells):
                a[t, : len(v)] = v
            storage.append(a)
            labels.append(DOMAINS.index(domain))
            docs[domain].add(doc)
    np.save(WORK / "cells.npy", np.stack(storage))
    np.save(WORK / "domains.npy", np.asarray(labels, np.uint8))
    # This separate page library explicitly marks its translations as machine generated.
    private = Counter()
    for page in sorted((ROOT.parent / "zhjpbook/books").glob("*/pages/*.json")):
        p = read(page)
        private["pages"] += 1
        pair = [
            p.get("text_" + l + "_readable") or p.get("text_" + l) or ""
            for l in ("en", "zh")
        ]
        if all(pair):
            private["bilingual"] += 1
        if encode(pair):
            private["within_both_inventories_and_cell_limit"] += 1
    rc = dict(
        training_pairs=Counter(DOMAINS[i] for i in labels),
        training_documents={k: len(v) for k, v in docs.items()},
        assessment_documents={
            s: {d: len({r["doc"] for r in rows if r["domain"] == d}) for d in DOMAINS}
            for s, rows in held.items()
        },
        local_book_sources=manifests,
        private_page_inventory=private,
        archives={k: v[1] for k, v in SOURCES.items()},
        files={
            n: sha(WORK / n)
            for n in ("cells.npy", "domains.npy", "dev.json", "test.json")
        },
    )
    dump(PUB / "corpus.json", rc)
    print(
        {
            k: rc[k]
            for k in (
                "training_pairs",
                "training_documents",
                "assessment_documents",
                "private_page_inventory",
            )
        },
        flush=True,
    )


if __name__ == "__main__":
    prepare()

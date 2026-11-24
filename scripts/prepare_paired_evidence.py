"""Prepare common-passage visual question pairs and training-only human evidence spans."""

from collections import Counter
import hashlib
import re
import numpy as np
import pyarrow.parquet as pq
from scripts.prepare_visual_cursor import (
    ROOT,
    FILES,
    WORK as CURSOR,
    hash_text,
    read,
    dump,
    sha,
)
from scripts.prepare_visual_sequence_corpus import Rasterizer

WORK = ROOT / "artifacts/paired_evidence_v1"
PUB = ROOT / "references/evidence/paired_evidence_v1"
MODEL = ROOT / "models/paired_evidence"
SOURCES = {
    **FILES,
    "squad": [
        (
            "visual_semantic_teacher/squad_train.json",
            "3527663986b8295af4f7fcdff1ba1ff3f72d07d61a20f487cb238a6ef92fd955",
        ),
        (
            "squad_v1.1/dev-v1.1.json",
            "95aa6a52d5d6a735563366753ca50492a658031da74f301ac5238b03966972c9",
        ),
    ],
}
PASSAGE = 600
QUERY = 160


def clean(s):
    return " ".join(s.split())


def evidence(raw, start, end):
    if start < 0 or end <= start or end > len(raw):
        return (-1, -1)
    marks = list(re.finditer(r"\S+", raw))
    offset = 0
    left = right = None
    for m in marks:
        a, b = max(start, m.start()), min(end, m.end())
        if a < b:
            if left is None:
                left = offset + a - m.start()
            right = offset + b - m.start()
        offset += len(m.group()) + 1
    return (left, right) if left is not None else (-1, -1)


def source(k, phase):
    filename, digest = SOURCES[k][phase]
    f = ROOT / "data/raw" / filename
    if sha(f) != digest:
        raise ValueError(filename)
    if k == "coqa":
        for d in pq.read_table(f).to_pylist():
            raw = d["story"]
            a = d["answers"]
            last = ""
            rows = []
            for q, ans, l, r in zip(
                d["questions"],
                a["input_text"],
                a["answer_start"],
                a["answer_end"],
                strict=True,
            ):
                rows.append((clean(q), last, [clean(ans)], evidence(raw, l, r)))
                last = clean(q + " " + ans)
            yield clean(raw), rows
    else:
        for article in read(f)["data"]:
            for d in article["paragraphs"]:
                raw = d["context"]
                rows = []
                for qa in d["qas"]:
                    a = qa["answers"]
                    refs = list(
                        dict.fromkeys(clean(r["text"]) for r in a if r["text"].strip())
                    )
                    if not refs:
                        continue
                    l = a[0]["answer_start"]
                    v = a[0]["text"]
                    span = (
                        evidence(raw, l, l + len(v))
                        if raw[l : l + len(v)] == v
                        else (-1, -1)
                    )
                    rows.append((clean(qa["question"]), "", refs, span))
                yield clean(raw), rows


def fragments(p):
    text = "".join(p.casefold().split())
    # Passage-family screen uses a shared 96-character substring at any alignment.
    if len(text) < 96:
        return {hash_text(text)}
    return {
        hashlib.blake2b(text[i : i + 96].encode(), digest_size=12).digest()
        for i in range(0, len(text) - 95, 64)
    }


class Encoder:
    def __init__(self):
        with np.load(CURSOR / "pages.npz") as f:
            self.tiles = list(map(bytes, f["tiles"]))
            self.chars = list(map(str, f["chars"]))
        self.shapes = {v: i for i, v in enumerate(self.tiles) if i >= 2}
        self.cache = {}
        self.raster = Rasterizer()

    def ids(self, s):
        values = []
        for c in s:
            if c not in self.cache:
                g = self.raster.tile(c)
                if g not in self.shapes:
                    self.shapes[g] = len(self.tiles)
                    self.tiles.append(g)
                    self.chars.append(c)
                self.cache[c] = self.shapes[g]
            values.append(self.cache[c])
        return values


def prep():
    WORK.mkdir(exist_ok=True, parents=True)
    PUB.mkdir(exist_ok=True, parents=True)
    enc = Encoder()
    old = {r["doc"] for r in read(CURSOR / "held.json")}
    fit = {}
    known = set()
    known_docs = set()
    dev = set()
    drop = Counter()
    for k in SOURCES:
        fit[k] = list(source(k, 0))
        for p, _ in fit[k]:
            known_docs.add(hash_text(p))
            known.update(fragments(p))
    held = []
    hold_frag = set()
    for k in SOURCES:
        docs = list(source(k, 1))
        dev.update(hash_text(p) for p, _ in docs)
        docs.sort(key=lambda d: hash_text("paired-evidence-1 " + d[0]))
        count = 0
        for p, rows in docs:
            doc = hash_text(p)
            if doc in old or doc in known_docs:
                continue
            chars = "".join(p.casefold().split())
            probes = {
                hashlib.blake2b(chars[i : i + 96].encode(), digest_size=12).digest()
                for i in range(max(0, len(chars) - 95))
            } or {hash_text(chars)}
            if probes & (known | hold_frag):
                continue
            selected = []
            try:
                s = enc.ids(p[:PASSAGE])
            except ValueError:
                continue
            for q, h, refs, _ in rows:
                cur = q[:QUERY]
                visible = (cur + (" Earlier: " + h if h else ""))[:QUERY]
                if (
                    not cur
                    or not refs[0]
                    or any(
                        t["question"] == cur or set(t["refs"]) & set(refs)
                        for t in selected
                    )
                ):
                    continue
                try:
                    x = enc.ids(visible)
                except ValueError:
                    continue
                selected.append(
                    dict(kind=k, doc=doc, question=cur, query=visible, q=x, refs=refs)
                )
                if len(selected) == 2:
                    break
            if len(selected) == 2:
                held.append(
                    dict(
                        doc=doc,
                        kind=k,
                        source=p[:PASSAGE],
                        s=s,
                        rows=selected,
                        clipped=len(p) > PASSAGE,
                    )
                )
                hold_frag.update(fragments(p))
                count += 1
                if count == 64:
                    break
        if count != 64:
            raise ValueError((k, count))
    passages = []
    queries = []
    answers = []
    labels = []
    groups = {k: [] for k in SOURCES}
    seen = set()
    for k, docs in fit.items():
        for p, rows in docs:
            doc = hash_text(p)
            if doc in dev or doc in seen:
                continue
            seen.add(doc)
            try:
                s = enc.ids(p[:PASSAGE])
            except ValueError:
                drop[k + "_page"] += 1
                continue
            if not s:
                continue
            group = []
            seenqa = set()
            for q, h, refs, (l, r) in rows:
                if not q or not refs[0] or q in seenqa:
                    continue
                seenqa.add(q)
                query = (q[:QUERY] + (" Earlier: " + h if h else ""))[:QUERY]
                try:
                    x = enc.ids(query)
                    y = enc.ids(refs[0]) + [1]
                except ValueError:
                    drop[k + "_row"] += 1
                    continue
                if len(x) == 0:
                    continue
                if l < 0 or r > len(s):
                    l = r = -1
                group.append(len(queries))
                queries.append(x)
                answers.append(y)
                labels.append((len(passages), l, r))
            if group:
                passages.append(s)
            if len(group) >= 2:
                groups[k].append(group)
    arrays = {}
    for field, seqs in (("p", passages), ("q", queries), ("y", answers)):
        arrays[field] = np.concatenate([np.asarray(v, np.uint16) for v in seqs])
        arrays[field + "o"] = np.r_[0, np.cumsum([len(v) for v in seqs])]
    arrays.update(
        labels=np.array(labels, np.int32),
        tiles=np.frombuffer(b"".join(enc.tiles), np.uint8).reshape(-1, 128),
        chars=np.array(enc.chars),
    )
    np.savez(WORK / "corpus.npz", **arrays)
    dump(WORK / "groups.json", groups)
    dump(WORK / "held.json", held)
    dump(
        PUB / "corpus.json",
        dict(
            train_pages=len(passages),
            qa_rows=len(queries),
            paired_training_pages={k: len(g) for k, g in groups.items()},
            held={k: [r["doc"] for r in held if r["kind"] == k] for k in SOURCES},
            human_evidence_visible=sum(l >= 0 for _, l, _ in labels),
            dropped=drop,
            sources=SOURCES,
            sha256={
                f: sha(WORK / f) for f in ("corpus.npz", "groups.json", "held.json")
            },
        ),
    )
    print(len(passages), len(queries), len(held), flush=True)


if __name__ == "__main__":
    prep()

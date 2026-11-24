"""Build bilingual image QA with human evidence spans and intact dialogue context."""

import re
from collections import Counter
from hashlib import sha256

import numpy as np
import pyarrow.parquet as pq

from scripts.evaluate_competing_facts import CorpusFace
from scripts.prepare_multiple_ink_runs import prepared_files
from scripts.prepare_open_exam import ROOT, read, sha
from scripts.score_dialogue_ink_references import official_index
from scripts.train_open_exam import corpus
from scripts.write_visual_discourse import DEFAULT

WORK = ROOT / "artifacts/evidence_views_v1"
PUB = ROOT / "references/evidence/evidence_views_v1"
MODEL = ROOT / "models/evidence_views"
TRAIN = ROOT / "data/raw/coqa/train-official.json"
DROP = ROOT / "data/raw/drop/validation.parquet"
SOURCES = {
    TRAIN: "b0fdb2bc1bd38dd3ca2ce5fa2ac3e02c6288ac914f241ac409a655ffb6619fa6",
    DROP: "7f9a3bdbb1b5909abfa25cbab693f89f47568c98e6e03473500d604f044c8f68",
}
URLS = {
    TRAIN: "https://downloads.cs.stanford.edu/nlp/data/coqa/coqa-train-v1.0.json",
    DROP: "https://huggingface.co/datasets/ucinlp/drop/resolve/95cda593fae71b60b5b19f82de3fcf3298c1239c/data/validation-00000-of-00001.parquet",
}


def text_key(s):
    return re.sub(r"\s+", "", s).casefold()


def windows(s):
    return (s[i : i + 72] for i in range(len(s) - 71)) if len(s) >= 72 else (s,)


def query(q, previous):
    """Keep every current-question glyph and the most recent visible previous exchange."""
    if not previous:
        return q
    sep = " | "
    history = previous[-max(0, 320 - len(sep) - len(q)) :]
    return history + sep + q


class Alphabet:
    def __init__(self, tiles=None):
        base = np.load(DEFAULT / "glyphs.npy")
        self.tiles = list(map(bytes, base if tiles is None else tiles))
        assert self.tiles[: len(base)] == list(map(bytes, base))
        self.n = len(base)
        self.chars = list(read(DEFAULT / "glyphs.json")["transcript"]) + ["�"] * (
            len(self.tiles) - len(base)
        )
        self.lookup = {b: i for i, b in enumerate(self.tiles) if i >= 2}
        self.face = CorpusFace()
        self.cache = {}
        self.missing = set()

    def encode(self, s):
        for c in dict.fromkeys(s):
            if c not in self.cache:
                try:
                    b = self.face.tile(c)
                except ValueError:
                    self.missing.add(c)
                    b = self.face.tile("�")
                if b not in self.lookup:
                    self.lookup[b] = len(self.tiles)
                    self.tiles.append(b)
                    self.chars.append(c)
                self.cache[c] = self.lookup[b]
        return [self.cache[c] for c in s]

    def array(self):
        return np.frombuffer(b"".join(self.tiles), np.uint8).reshape(-1, 128)


def build(stage):
    for p, h in SOURCES.items():
        if not p.exists():
            import requests

            rsp = requests.get(URLS[p], timeout=90)
            rsp.raise_for_status()
            if sha256(rsp.content).hexdigest() != h:
                raise ValueError(f"Unexpected download for {p.name}")
            p.parent.mkdir(parents=True, exist_ok=True)
            tmp = p.with_suffix(".part")
            tmp.write_bytes(rsp.content)
            tmp.replace(p)
        if sha(p) != h:
            raise ValueError(f"Changed source {p.name}")
    judge, _, _, dev, _ = official_index()
    held = {text_key(r["story"]) for r in read(dev)["data"]}
    held.update(
        text_key(r["passage"])
        for r in pq.read_table(DROP, columns=["passage"]).to_pylist()
    )
    forbidden = {w for d in held for w in windows(d)}
    chinese, tiles, n = corpus("train")
    alphabet = Alphabet(tiles)
    assert n == alphabet.n
    docs = []
    qs = []
    ys = []
    meta = []
    stats = Counter()
    seen = set()

    def add(q, y, doc, lo, hi, chinese):
        qs.append(q)
        ys.append(y)
        meta.append((doc, lo, hi, int(chinese)))

    for r in read(TRAIN)["data"]:
        story = r["story"]
        ident = text_key(story)
        stats["official_turns"] += len(r["questions"])
        reserved = (
            int(sha256(" ".join(story.split()).encode()).hexdigest(), 16) % 17 == 0
        )
        if (
            reserved
            or ident in seen
            or not 1 <= len(story) <= 4096
            or any(w in forbidden for w in windows(ident))
        ):
            stats["excluded_stories"] += 1
            continue
        seen.add(ident)
        doc = len(docs)
        page = alphabet.encode(story)
        docs.append(page)
        previous = ""
        for question, answer in zip(r["questions"], r["answers"], strict=True):
            assert question["turn_id"] == answer["turn_id"]
            q = question["input_text"]
            a = answer["input_text"]
            lo = answer["span_start"]
            hi = answer["span_end"]
            full_q = query(q, previous) if len(q) <= 160 else q
            previous = q + " " + a
            if not (
                0 <= lo < hi <= len(page)
                and hi - lo <= 600
                and story[lo:hi] == answer["span_text"]
                and 0 < len(q) <= 160
                and 0 < len(a) < 64
                and judge.normalize_answer(a)
            ):
                stats["excluded_geometry_or_evidence"] += 1
                continue
            qi = alphabet.encode(full_q)
            y = alphabet.encode(a) + [1]
            evidence = page[lo:hi]
            novel = {i for i in y if i >= n}
            if alphabet.missing.intersection(a) or any(
                not novel.issubset(set(p) | set((p + qi)[-144:]))
                for p in (page, evidence)
            ):
                stats["unspellable_in_a_view"] += 1
                continue
            if len(qi) > 320:
                raise ValueError("Lost current question")
            add(qi, y, doc, lo, hi, False)
            stats["english_turns"] += 1
            phrase = judge.normalize_answer(a)
            if (
                len(phrase.split()) >= 2
                and " " + phrase + " "
                not in " " + judge.normalize_answer(story + " " + full_q) + " "
            ):
                stats["absent_multiword"] += 1
    indices = {}
    for r in chinese:
        if r["doc"] not in indices:
            indices[r["doc"]] = len(docs)
            docs.append(r["pi"])
        doc = indices[r["doc"]]
        assert docs[doc] == r["pi"]
        add(r["qi"], r["y"], doc, 0, len(docs[doc]), True)
    stats["chinese_questions"] = len(chinese)
    arrays = dict(meta=np.asarray(meta, np.int32), tiles=alphabet.array())
    for key, rows in (("p", docs), ("q", qs), ("y", ys)):
        arrays[key] = np.concatenate([np.asarray(row, np.int32) for row in rows])
        arrays[key + "ptr"] = np.cumsum([0] + list(map(len, rows)), dtype=np.int64)
    np.savez(stage / "train.npz", **arrays)
    return dict(
        stats=stats,
        english_stories=len(seen),
        canonical=n,
        sources={str(k.relative_to(ROOT)): v for k, v in SOURCES.items()},
        c3=sha(ROOT / "references/evidence/open_exam_v1/corpus.json"),
        held=sha(dev),
        files={"train.npz": sha(stage / "train.npz")},
    )


if __name__ == "__main__":
    prepared_files(
        WORK, PUB / "corpus.json", {"train.npz": ("files", "train.npz")}, build
    )

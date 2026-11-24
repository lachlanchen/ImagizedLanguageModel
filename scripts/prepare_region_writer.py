"""Complete human answers with visible candidate regions for a bilingual visual writer."""

import csv
from collections import defaultdict
from hashlib import sha256
import re
import numpy as np

from scripts.prepare_evidence_views import ROOT, sha
from scripts.prepare_open_exam import read, dump
from scripts.prepare_grounded_phrasings import phrase_key
from scripts.prepare_visual_statements import collect, detok, squad
from scripts.prepare_chinese_card_reply import DOMAINS, RAW, extract
from scripts.prepare_semantic_page import RAW as E2E, FILES
from scripts.evaluate_competing_facts import CorpusFace

WORK = ROOT / "artifacts/region_writer_v1"
PUB = ROOT / "references/evidence/region_writer_v1"
MODEL = ROOT / "models/region_writer"
LIMIT = 256


def digest(s):
    return sha256(s.encode()).hexdigest()


def regions(page):
    parts = re.findall(r".+?(?:[.!?。！？](?=\s|$)|$)", page.strip(), flags=re.S)
    return [
        s.strip()
        for part in parts
        for i in range(0, len(part), 256)
        if (s := part[i : i + 256]).strip()
    ]


def example(task, family, q, blocks, valid, answers, facts):
    return dict(
        task=task,
        family=digest(family),
        q=q,
        blocks=blocks,
        valid=valid,
        answers=answers,
        facts=facts,
    )


def restaurant(split):
    filename = E2E / FILES[split]
    assert (
        sha(filename)
        == read(ROOT / "references/evidence/semantic_page_v1/corpus.json")["files"][
            filename.name
        ]
    )
    groups = defaultdict(dict)
    with filename.open(newline="") as f:
        for row in csv.DictReader(f):
            fields = tuple(
                sorted(
                    (k.strip(), v)
                    for k, v in re.findall(r"([^,\[]+)\[([^\]]*)\]", row["mr"])
                )
            )
            target = row["ref"].strip()
            if "name" in dict(fields) and 0 < len(target) < LIMIT:
                groups[fields].setdefault(phrase_key(target), target)
    keys = sorted(groups)
    for i, k in enumerate(keys):
        named = dict(k)["name"]
        j = (i + 1 + int(digest(str(k))[:8], 16) % (len(keys) - 1)) % len(keys)
        while dict(keys[j])["name"] == named:
            j = (j + 1) % len(keys)
        fmt = lambda fields: " | ".join(f"{n} = {v}" for n, v in fields)
        blocks = [fmt(k), fmt(keys[j])]
        gold = int(digest(repr(k))[8:10], 16) % 2
        if gold:
            blocks.reverse()
        yield example(
            "restaurant",
            fmt(k),
            f"Describe {named}.",
            blocks,
            [gold],
            list(groups[k].values()),
            dict(k),
        )


def chinese(split):
    s = {"dev": "val", "train": "train"}[split]
    receipt = read(ROOT / "references/evidence/chinese_reply_v1/corpus.json")
    for f in [
        RAW / (s + ".json.zip"),
        *(RAW / (n + "_db.json") for n in DOMAINS.values()),
    ]:
        path = (
            "data/crosswoz/database/" if f.suffix == ".json" else "data/crosswoz/"
        ) + f.name
        assert sha(f) == receipt["files"][path]
    db = {d: dict(read(RAW / (n + "_db.json"))) for d, n in DOMAINS.items()}
    rows, _ = extract(s, db)
    for r in rows:
        fields = list(r["card"])
        blocks = [r["name"] + " " + k + " " + r["card"][k] for k in fields]
        yield example(
            "catalogue",
            r["domain"] + r["name"],
            r["q"],
            blocks,
            [fields.index(r["key"])],
            [r["answer"]],
            {r["key"]: r["card"][r["key"]]},
        )


def statements(split, pages):
    for r in collect(split):
        if r["example_uid"] not in pages:
            continue
        page = pages[r["example_uid"]]
        q, f, y = [detok(r[k]) for k in ("question", "answer", "turker_answer")]
        blocks = regions(page)
        valid = [i for i, b in enumerate(blocks) if f and f.casefold() in b.casefold()]
        if valid:
            yield example("statements", page, q, blocks, valid, [y], {"answer": f})


def inventory():
    pages = {k: p for split in (0, 1) for k, p, _, _ in squad(split)}

    def eligible(split):
        for source in (restaurant(split), chinese(split), statements(split, pages)):
            for row in source:
                if not 0 < len(row["q"]) <= 192 or not 1 <= len(row["blocks"]) <= 24:
                    continue
                if (
                    max(map(len, row["blocks"])) > 256
                    or sum(map(len, row["blocks"])) > 1536
                ):
                    continue
                row["answers"] = [
                    s for s in row["answers"] if 0 < len(s) < LIMIT and phrase_key(s)
                ]
                if row["answers"]:
                    yield row

    dev = list(eligible("dev"))
    train = list(eligible("train"))
    # Split by family first. The Chinese dev dialogues reuse many training entities.
    # Reserve a deterministic subset of dev entities in their entirety for measurement.
    chosen = []
    for task in ("restaurant", "catalogue", "statements"):
        pool = [r for r in dev if r["task"] == task]
        pool.sort(key=lambda r: digest("region-1" + r["family"] + r["q"]))
        seen = set()
        for r in pool:
            if r["family"] not in seen:
                chosen.append(r)
                seen.add(r["family"])
                if len(seen) == 96:
                    break
    held_sources = {r["family"] for r in chosen}
    held_text = {phrase_key(t) for r in chosen for t in r["answers"]}
    held_qs = {phrase_key(r["q"]) for r in chosen if r["task"] == "statements"}
    result = []
    for r in train:
        if r["family"] in held_sources or (
            r["task"] == "statements" and phrase_key(r["q"]) in held_qs
        ):
            continue
        if r["task"] == "restaurant" and any(
            digest(b) in held_sources for b in r["blocks"]
        ):
            continue
        r["answers"] = [s for s in r["answers"] if phrase_key(s) not in held_text]
        if r["answers"]:
            result.append(r)
    return result, chosen


class Ink:
    def __init__(self, tiles=None):
        self.tiles = list(map(bytes, tiles)) if tiles is not None else [bytes(128)] * 2
        self.byink = {b: i for i, b in enumerate(self.tiles) if i >= 2}
        self.face = CorpusFace()
        self.chars = {}
        self.bad = set()

    def encode(self, s):
        for c in dict.fromkeys(s):
            if c in self.chars:
                continue
            try:
                b = self.face.tile(c)
            except ValueError:
                self.bad.add(c)
                b = self.face.tile("�")
            if b not in self.byink:
                self.byink[b] = len(self.tiles)
                self.tiles.append(b)
            self.chars[c] = self.byink[b]
        return [self.chars[c] for c in s]

    def array(self):
        return np.frombuffer(b"".join(self.tiles), np.uint8).reshape(-1, 128)


def main():
    WORK.mkdir(exist_ok=True, parents=True)
    PUB.mkdir(exist_ok=True, parents=True)
    train, dev = inventory()
    ink = Ink()
    for r in train:
        for s in r["answers"]:
            ink.encode(s)
    vocab = len(ink.tiles)
    np.save(WORK / "palette.npy", ink.array())
    strings = {}
    seq = []

    def index(s):
        if s not in strings:
            strings[s] = len(seq)
            seq.append(ink.encode(s))
        return strings[s]

    sizes = {}
    for split, rows in (("train", train), ("dev", dev)):
        encoded = []
        for r in rows:
            q = index(r["q"])
            ps = list(map(index, r["blocks"]))
            ys = [index(s) for s in r["answers"] if not ink.bad.intersection(s)]
            if ys:
                encoded.append([q, ps, r["valid"], ys, r["task"], r["family"]])
        dump(WORK / (split + ".json"), encoded)
        sizes[split] = {
            t: dict(
                questions=sum(z[4] == t for z in encoded),
                answers=sum(len(z[3]) for z in encoded if z[4] == t),
            )
            for t in ("restaurant", "catalogue", "statements")
        }
    dump(WORK / "dev-source.json", dev)
    np.save(WORK / "tiles.npy", ink.array())
    np.savez(
        WORK / "strings.npz",
        ids=np.concatenate([np.asarray(x, np.int32) for x in seq]),
        ptr=np.cumsum([0] + list(map(len, seq))),
    )
    dump(
        PUB / "corpus.json",
        dict(
            counts=sizes,
            output_bitmaps=vocab,
            missing_codepoints=sorted(map(ord, ink.bad)),
            max_decisions=LIMIT,
            files={
                f: sha(WORK / f)
                for f in (
                    "train.json",
                    "dev.json",
                    "dev-source.json",
                    "palette.npy",
                    "tiles.npy",
                    "strings.npz",
                )
            },
        ),
    )
    print(sizes, flush=True)


if __name__ == "__main__":
    main()

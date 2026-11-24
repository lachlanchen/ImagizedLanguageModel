"""Prepare disjoint optical fact-reader and answer-writer corpora and reserve new paired pages."""

from collections import Counter, defaultdict
import numpy as np
from scripts.prepare_region_writer import (
    ROOT,
    MODEL as BASE,
    Ink,
    inventory,
    regions,
    digest,
    phrase_key,
    example,
    statements,
    squad,
    ensure_raw_sources,
    read,
    dump,
    sha,
)
from scripts.prepare_phrase_confirmation import WORK as FIRST, exam_rows
from scripts.prepare_marginal_confirmation import WORK as SECOND
from scripts.prepare_multiple_ink_runs import prepared_files
from scripts.prepare_visual_cursor import FILES

WORK = ROOT / "artifacts/fact_channel_v1"
PUB = ROOT / "references/evidence/fact_channel_v1"
MODEL = ROOT / "models/fact_channel"
NAMES = (
    "reader.json",
    "writer.json",
    "confirmation.json",
    "strings.npz",
    "tiles.npy",
    "palette.npy",
    "readback.json",
)


def eligible(r):
    b = r["blocks"]
    return (
        0 < len(r["q"]) <= 192
        and 0 < len(b) <= 24
        and max(map(len, b)) <= 256
        and sum(map(len, b)) <= 1536
        and all(0 < len(s) < 256 for s in r["answers"])
    )


def chinese_training(corpus, split=0):
    f, h = FILES[corpus][split]
    path = ROOT / "data/raw" / f
    if sha(path) != h:
        raise ValueError(f)
    for a in read(path)["data"]:
        for p in a["paragraphs"]:
            page = p["context"].strip()
            bs = regions(page)
            for qa in p["qas"]:
                answers = list(
                    dict.fromkeys(
                        x["text"].strip() for x in qa["answers"] if x["text"].strip()
                    )
                )
                if not answers:
                    continue
                valid = [i for i, b in enumerate(bs) if any(s in b for s in answers)]
                if valid:
                    yield example(
                        "exam",
                        page,
                        qa["question"].strip(),
                        bs,
                        valid,
                        answers,
                        {"answer": answers[0]},
                    )


def reservation(writer, history):
    excluded = writer + history
    families = {r["family"] for r in excluded}
    queries = {phrase_key(r["q"]) for r in excluded}
    targets = {phrase_key(t) for r in writer + history for t in r["answers"]}
    pages = {k: p for split in (0, 1) for k, p, _, _ in squad(split)}
    out = []
    for source in (statements("dev", pages), exam_rows()):
        groups = defaultdict(list)
        for r in source:
            if (
                eligible(r)
                and r["family"] not in families
                and phrase_key(r["q"]) not in queries
                and all(phrase_key(t) not in targets for t in r["answers"])
            ):
                groups[r["family"]].append(r)
        n = 0
        for f in sorted(groups, key=lambda k: digest("channel-page" + k)):
            pool = sorted(groups[f], key=lambda r: digest("channel-query" + r["q"]))
            a = pool[0]
            b = next(
                (
                    b
                    for b in pool[1:]
                    if phrase_key(a["q"]) != phrase_key(b["q"])
                    and a["blocks"] == b["blocks"]
                    and phrase_key(a["facts"]["answer"])
                    != phrase_key(b["facts"]["answer"])
                ),
                None,
            )
            if b is not None:
                out.extend((a, b))
                n += 1
            if n == 96:
                break
        if n == 0:
            raise ValueError("No fresh paired passage")
    return out


def build(stage):
    ensure_raw_sources()
    writer, old = inventory()
    history = (
        old + read(FIRST / "confirmation.json") + read(SECOND / "confirmation.json")
    )
    held = {r["family"] for r in history}
    old_queries = {phrase_key(r["q"]) for r in history}
    old_answers = {phrase_key(a) for r in history for a in r["answers"]}
    reader = []
    for corpus in ("cmrc", "drcd"):
        for r in chinese_training(corpus):
            if (
                not eligible(r)
                or r["family"] in held
                or phrase_key(r["q"]) in old_queries
            ):
                continue
            if int(r["family"][:8], 16) % 3:
                reader.append(r)
            elif all(phrase_key(a) not in old_answers for a in r["answers"]):
                writer.append(r)
    fresh = reservation(writer, history)
    occupied = {r["family"] for r in writer + history + fresh}
    qs = {phrase_key(r["q"]) for r in writer + history + fresh}
    for _, page, q, answers in squad(0):
        if digest(page) in occupied or phrase_key(q) in qs:
            continue
        bs = regions(page)
        valid = [i for i, b in enumerate(bs) if any(a in b for a in answers)]
        if valid:
            r = example("squad", page, q, bs, valid, answers, {"answer": answers[0]})
            if eligible(r):
                reader.append(r)
    reader = [
        r
        for r in reader
        if r["family"] not in occupied and phrase_key(r["q"]) not in qs
    ]
    dump(stage / "confirmation.json", fresh)
    ink = Ink(np.load(BASE / "palette.npy"))
    for r in writer:
        for a in r["answers"]:
            ink.encode(a)
    n = len(ink.tiles)
    np.save(stage / "palette.npy", ink.array())
    chars = list(read(BASE / "readback.json")) + ["�"] * (
        n - len(read(BASE / "readback.json"))
    )
    for c, i in ink.chars.items():
        if chars[i] == "�":
            chars[i] = c
    dump(stage / "readback.json", chars)
    index = {}
    ids = []

    def number(s):
        if s not in index:
            index[s] = len(ids)
            ids.append(np.asarray(ink.encode(s), dtype=np.int32))
        return index[s]

    sizes = {}
    for name, rows in (("writer", writer), ("reader", reader)):
        result = []
        for r in rows:
            ys = [number(a) for a in r["answers"] if not ink.bad.intersection(a)]
            if not ys:
                continue
            q = number(r["q"])
            bs = [number(b) for b in r["blocks"]]
            if name == "writer":
                fact = (
                    r["blocks"][r["valid"][0]]
                    if r["task"] == "restaurant"
                    else next(iter(r["facts"].values()))
                )
                result.append([q, bs, number(fact), ys, r["task"], r["family"]])
            else:
                locations = []
                offset = 0
                for b in r["blocks"]:
                    for a in set(r["answers"]):
                        for i in positions(b, a):
                            locations.append((offset + i, len(a)))
                    offset += len(b) + 1
                if locations:
                    result.append(
                        [q, bs, sorted(set(locations)), r["task"], r["family"]]
                    )
        dump(stage / (name + ".json"), result)
        sizes[name] = dict(Counter(row[-2] for row in result))
        if name == "writer":
            sizes["wordings"] = sum(len(r[3]) for r in result)
    np.save(stage / "tiles.npy", ink.array())
    np.savez(
        stage / "strings.npz",
        ids=np.concatenate(ids),
        ptr=np.cumsum([0] + list(map(len, ids))),
    )
    return dict(
        counts=sizes,
        confirmation=dict(Counter(r["task"] for r in fresh)),
        output_bitmaps=n,
        reader_writer_family_overlap=0,
        reader_writer_question_overlap=0,
        files={f: sha(stage / f) for f in NAMES},
    )


def positions(text, needle):
    if needle:
        i = text.find(needle)
        while i >= 0:
            yield i
            i = text.find(needle, i + 1)


def main():
    prepared_files(WORK, PUB / "corpus.json", {f: ("files", f) for f in NAMES}, build)
    print(read(PUB / "corpus.json")["counts"], flush=True)


if __name__ == "__main__":
    main()

"""Reserve new passages and prepare complete human and instruction answers."""

from collections import Counter, defaultdict
from pathlib import Path
import json
import numpy as np

from scripts.prepare_fact_ranking_confirmation import (
    FIRST,
    SECOND,
    PREVIOUS,
    PREVIOUS_PUB,
    ORIGINAL,
    WORK as FOURTH,
    Ink,
    identity,
    read,
    dump,
    sha,
    ROOT,
    eligible,
    exam_rows,
    statements_from_qamr,
    digest,
    phrase_key,
    prepared_files,
)
from scripts.prepare_region_writer import regions, example
from scripts.prepare_recurrent_evidence import raw, DOLLY, WORK as PROSE, ARCHIVE
from scripts.measure_expanded_ink_inventory import family
from scripts.measure_visual_answer_support import instructions, SOURCES

WORK = ROOT / "artifacts/sentence_context_v1"
PUB = ROOT / "references/evidence/sentence_context_v1"
MODEL = ROOT / "models/sentence_context"
BANK = ROOT / "artifacts/visual_answer_support_v1/training-ink.npy"
FILES = ("confirmation.json", "long.json", "additional.json")


def long_rows(language, train):
    if language == "en":
        with (DOLLY / ("train.jsonl" if train else "sealed_test.jsonl")).open() as f:
            for line in f:
                r = json.loads(line)
                q, p = r["instruction"], r["context"]
                is_held = int.from_bytes(family(q)[:8], "big") % 5 == 0
                if is_held == train:
                    continue
                yield r["group"], q, p, [r["answer"]]
    else:
        for k, q, p, refs, _ in raw("zh", "train" if train else "development"):
            yield str(k), q, p, refs


class PhysicalIndex:
    def __init__(self, ink):
        self.ink = ink
        self.families, self.q, self.p, self.y = set(), set(), set(), set()

    def keys(self, row):
        encode = lambda s: identity(self.ink.encode(s))
        return (
            row["family"],
            encode(row["q"]),
            tuple(map(encode, row["blocks"])),
            tuple(map(encode, row["answers"])),
        )

    def add(self, f, q, p, ys):
        self.families.add(f)
        self.q.add(q)
        self.p.add(p)
        self.y.update(ys)

    def overlaps(self, keys):
        f, q, p, ys = keys
        return (
            f in self.families
            or q in self.q
            or p in self.p
            or bool(self.y.intersection(ys))
        )


def prior_index():
    spec = read(PREVIOUS_PUB / "corpus.json")
    for name, expected in spec["files"].items():
        if sha(PREVIOUS / name) != expected:
            raise ValueError(name)
    idx = PhysicalIndex(Ink(np.load(PREVIOUS / "tiles.npy")))
    with np.load(PREVIOUS / "strings.npz") as d:
        ids, ptr = d["ids"], d["ptr"]
    sig = [identity(ids[a:b]) for a, b in zip(ptr[:-1], ptr[1:])]
    for split in ("reader", "writer"):
        for r in read(PREVIOUS / (split + ".json")):
            idx.add(
                r[-1],
                sig[r[0]],
                tuple(sig[k] for k in r[1]),
                [sig[k] for k in r[3]] if split == "writer" else [],
            )
    history = [
        ORIGINAL / "dev-source.json",
        *[p / "confirmation.json" for p in (FIRST, SECOND, PREVIOUS, FOURTH)],
    ]
    rc = read(ROOT / "references/evidence/fact_span_ranking_v1/reservation.json")
    if [sha(p) for p in history] != [*rc["previous"], rc["sha256"]]:
        raise ValueError("Different prior evaluations")
    for path in history:
        for row in read(path):
            idx.add(*idx.keys(row))
    for r in read(PROSE / "held.json"):
        row = example(
            "prose",
            r["lang"] + str(r["id"]),
            r["question"],
            regions(r["source"]),
            [],
            r["refs"],
            {},
        )
        idx.add(*idx.keys(row))
    return idx


def paired(idx):
    result = []
    for stream in (statements_from_qamr(), exam_rows()):
        groups = defaultdict(list)
        for row in stream:
            if eligible(row) and not idx.overlaps(idx.keys(row)):
                groups[row["family"]].append(row)
        seen = set()
        n = 0
        for f in sorted(groups, key=lambda s: digest("context-passage" + s)):
            pool = sorted(groups[f], key=lambda r: digest("context-question" + r["q"]))
            pair = next(
                (
                    (a, b)
                    for i, a in enumerate(pool)
                    for b in pool[i + 1 :]
                    if a["blocks"] == b["blocks"]
                    and phrase_key(a["q"]) != phrase_key(b["q"])
                    and not {phrase_key(a["q"]), phrase_key(b["q"])} & seen
                    and phrase_key(a["facts"]["answer"])
                    != phrase_key(b["facts"]["answer"])
                ),
                None,
            )
            if pair:
                result.extend(pair)
                seen.update(phrase_key(r["q"]) for r in pair)
                n += 1
            if n == 96:
                break
        if not n:
            raise ValueError("No unused paired passages")
    return result


def prose(task, k, q, p, refs):
    q, p = q.strip(), p.strip()
    refs = list(dict.fromkeys(s.strip() for s in refs if s and s.strip()))
    if not q or len(q) > 192 or not refs or len(p) > 1536:
        return None
    blocks = regions(p) if p else [q]
    if not blocks or len(blocks) > 24:
        return None
    return example(task, task + str(k), q, blocks, [], refs, {})


def build(stage):
    bank_rc = read(
        ROOT / "references/evidence/visual_answer_support_v1/training-inventory.json"
    )
    if (
        sha(BANK) != bank_rc["inventory_sha256"]
        or sha(ARCHIVE)
        != "643aa0d555b684bf3da68dc92516711223cecc899ed53f9e1ef1113dd44af571"
    ):
        raise ValueError("Input checksum")
    idx = prior_index()
    original = paired(idx)
    held = PhysicalIndex(idx.ink)
    for row in original:
        held.add(*held.keys(row))
    longer = []
    counts = Counter()
    for lang in ("en", "zh"):
        task = "dolly" if lang == "en" else "dureader"
        pool = sorted(
            long_rows(lang, False), key=lambda r: digest("context-long" + lang + r[0])
        )
        for category in ("short", "long"):
            n = 0
            for k, q, p, refs in pool:
                row = prose(task, k, q, p, refs)
                if (
                    not row
                    or len(row["answers"][0]) < 8
                    or (len(row["answers"][0]) >= 256) != (category == "long")
                ):
                    continue
                if any(
                    phrase_key(a) in phrase_key(q) or phrase_key(a) in phrase_key(p)
                    for a in row["answers"]
                ):
                    continue
                keys = idx.keys(row)
                if idx.overlaps(keys) or held.overlaps(keys):
                    continue
                held.add(*keys)
                longer.append(row)
                n += 1
                if n == 32:
                    break
            counts[task + "_" + category] = n
    extras = []
    stats = defaultdict(Counter)
    for name, source, limit in [
        ("dolly", long_rows("en", True), 8192),
        ("dureader", long_rows("zh", True), 8192),
        *[(t, teacher(t), 4096) for t in ("alpaca_en", "alpaca_zh")],
    ]:
        distinct = set()
        for k, q, p, refs in source:
            stats[name]["examined"] += 1
            row = prose(name, k, q, p, refs)
            if not row or not all(len(a) < 1024 for a in row["answers"]):
                stats[name]["outside_geometry"] += 1
                continue
            keys = held.keys(row)
            if held.overlaps(keys) or idx.overlaps(keys) or keys[1] in distinct:
                stats[name]["overlap"] += 1
                continue
            distinct.add(keys[1])
            extras.append(row)
            stats[name]["admitted"] += 1
            if len(distinct) == limit:
                break
    for name, data in zip(FILES, (original, longer, extras), strict=True):
        dump(stage / name, data)
    return dict(
        files={f: sha(stage / f) for f in FILES},
        paired=Counter(r["task"] for r in original),
        long=counts,
        extra=stats,
        palette=sha(BANK),
        base_corpus=sha(PREVIOUS_PUB / "corpus.json"),
        protocol=sha(ROOT / "references/visual_sentence_context_protocol.md"),
        code=sha(Path(__file__)),
        dolly={n: sha(DOLLY / n) for n in ("train.jsonl", "sealed_test.jsonl")},
    )


def teacher(name):
    path = ROOT / "data/raw" / (name + ".json")
    if sha(path) != SOURCES[path.name]:
        raise ValueError(name)
    for i, (q, p, y) in enumerate(instructions(path)):
        if int.from_bytes(family(q)[:8], "big") % 5:
            yield str(i), q, p, [y]


if __name__ == "__main__":
    prepared_files(WORK, PUB / "corpus.json", {f: ("files", f) for f in FILES}, build)
    print(read(PUB / "corpus.json"))

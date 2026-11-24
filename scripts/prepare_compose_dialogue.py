"""Prepare source-disjoint conversational and multi-span reading data."""

from collections import Counter
from functools import lru_cache
import gzip
import hashlib
import json
import zipfile

import pyarrow.parquet as pq
from scripts.dialogue_ink_data import read_data
from scripts.dialogue_teacher_stories import fold, key
from scripts.evaluate_optical_passage_qa import normalized
from scripts.prepare_dialogue_ink import ROOT, SOURCES, dump, printable, sha
from scripts.score_dialogue_ink_references import official_index

WORK = ROOT / "artifacts/compose_dialogue_v1"
PUB = ROOT / "references/evidence/compose_dialogue_v1"
MODEL = ROOT / "models/compose_dialogue"
ARCHIVE = ROOT / "data/raw/multispanqa/release.zip"
ARCHIVE_SHA = "f192915e34e4d9b03bd917e9e20ff4dd204da8f1fdbdb175432485c7028905bb"


def ident(s):
    return hashlib.sha256(s.encode()).hexdigest()


@lru_cache(maxsize=16384)
def surfaces(text):
    norm = " " + normalized(text) + " "
    possible = {
        n
        for w in text.split()
        for v in (w, w[:-1], w[1:], w[1:-1])
        if (n := normalized(v))
    }
    return norm, possible


def kind(answer, fields):
    goal = normalized(answer)
    words = goal.split()
    if len(words) < 2:
        return "ordinary"
    vals = [surfaces(s) for s in fields]
    if any(" " + goal + " " in n for n, _ in vals):
        return "ordinary"
    return (
        "assembly"
        if all(any(w in opts for _, opts in vals) for w in words)
        else "spell"
    )


def spans(sample):
    parts = []
    for w, label in zip(sample["context"], sample["label"], strict=True):
        if label == "B":
            parts.append([w])
        elif label == "I":
            parts[-1].append(w)
        elif label != "O":
            raise ValueError(label)
    if len(parts) != sample["num_span"]:
        raise ValueError("Source span count differs")
    return [fold(" ".join(part)) for part in parts]


def prepare():
    if sha(ARCHIVE) != ARCHIVE_SHA:
        raise ValueError("Different MultiSpanQA archive")
    raw = ROOT / "data/raw/coqa/train.parquet"
    if sha(raw) != SOURCES["train"]:
        raise ValueError("Different CoQA source")
    registry = read_data()
    orig = [r for r in registry["rows"] if r[0] == "train"]
    alphabet = set("".join(r[5] for r in orig))
    pools = {"assembly": [], "spell": []}
    for i, r in enumerate(orig):
        group = kind(r[5], (registry["pages"][r[1]], r[3], r[4]))
        if group in pools:
            pools[group].append(i)
    with zipfile.ZipFile(ARCHIVE) as z:
        ms = {
            s: json.loads(z.read("MultiSpanQA_data/" + s + ".json"))["data"]
            for s in ("train", "valid")
        }
    old_keys = {key(registry["pages"][k]) for k in {r[1] for r in registry["rows"]}}
    _, _, _, official, _ = official_index()
    dev_keys = {key(x["story"]) for x in json.loads(official.read_text())["data"]}
    ms_keys = {s: {key(" ".join(r["context"])) for r in src} for s, src in ms.items()}
    dialogue_keys = old_keys | dev_keys
    excluded = dialogue_keys | ms_keys["train"] | ms_keys["valid"]
    candidates = {}
    for item in pq.read_table(raw).to_pylist():
        page = fold(item["story"])
        canonical = key(page)
        if (
            canonical not in excluded
            and printable(page)
            and 321 <= len(page.split()) <= 448
        ):
            candidates.setdefault(canonical, (page, item))
    fresh = {"pages": {}, "rows": []}
    for canonical in sorted(candidates, key=lambda k: ident("compose-v1/" + k))[:200]:
        page, item = candidates[canonical]
        pid = ident(page)
        history = ""
        for j, (q, a) in enumerate(
            zip(item["questions"], item["answers"]["input_text"], strict=True)
        ):
            q, a = fold(q), fold(a)
            prev = " ".join(history.split()[-48:])
            history = q + " " + a
            if a and printable(q) and printable(prev) and 1 <= len(q.split()) <= 32:
                fresh["pages"][pid] = page
                fresh["rows"].append(
                    ["fresh", pid, j, q, prev, a, [-1, -1], item["source"]]
                )
    ext = {}
    counts = {}
    for split, items in ms.items():
        d = {"pages": {}, "rows": []}
        ct = Counter()
        seen = set()
        for r in items:
            page = fold(" ".join(r["context"]))
            q = fold(" ".join(r["question"]))
            pid = ident(page)
            if key(page) in dialogue_keys or (
                split == "train" and key(page) in ms_keys["valid"]
            ):
                ct["overlap"] += 1
                continue
            if (
                not printable(page)
                or not printable(q)
                or not 1 <= len(page.split()) <= 448
                or not 1 <= len(q.split()) <= 32
            ):
                ct["input_geometry"] += 1
                continue
            a = " and ".join(spans(r))
            if split == "train" and (
                len(a.split()) > 16
                or max(map(len, a.split())) > 24
                or not set(a) <= alphabet
            ):
                ct["answer_geometry"] += 1
                continue
            if (pid, q, a) in seen:
                ct["duplicate"] += 1
                continue
            seen.add((pid, q, a))
            d["pages"][pid] = page
            d["rows"].append(
                [
                    "train" if split == "train" else "fresh",
                    pid,
                    r["id"],
                    q,
                    "",
                    a,
                    [-1, -1],
                    "MultiSpanQA",
                ]
            )
        ext[split] = d
        counts[split] = dict(
            ct, eligible=len(d["rows"]), pages=len(d["pages"]), raw=len(items)
        )
    WORK.mkdir(exist_ok=True, parents=True)
    PUB.mkdir(exist_ok=True, parents=True)
    result = dict(pools=pools, extra=ext["train"], lists=ext["valid"], fresh=fresh)
    file = WORK / "prepared.json.gz"
    file.write_bytes(
        gzip.compress(json.dumps(result, separators=(",", ":")).encode(), mtime=0)
    )
    manifest = dict(
        archive_sha256=ARCHIVE_SHA,
        coqa_train=SOURCES["train"],
        prepared_sha256=sha(file),
        fitting=len(orig),
        pools={k: len(v) for k, v in pools.items()},
        multispan=counts,
        fresh_candidates=len(candidates),
        fresh_stories=len(fresh["pages"]),
        fresh_turns=len(fresh["rows"]),
        fresh_kinds=dict(
            Counter(
                kind(r[5], (fresh["pages"][r[1]], r[3], r[4])) for r in fresh["rows"]
            )
        ),
    )
    dump(PUB / "split.json", manifest)
    print(json.dumps(manifest), flush=True)


def prepared():
    file = WORK / "prepared.json.gz"
    if sha(file) != json.loads((PUB / "split.json").read_text())["prepared_sha256"]:
        raise ValueError("Prepared corpus changed")
    return json.loads(gzip.decompress(file.read_bytes()))


if __name__ == "__main__":
    prepare()

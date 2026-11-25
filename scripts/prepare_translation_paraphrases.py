"""Prepare matched semantic events with varied visual sources and the same pictured target."""

from collections import Counter
from functools import lru_cache
from hashlib import sha256
from pathlib import Path
import re
import requests
import numpy as np
import pyarrow.parquet as pq
from scripts.prepare_proposition_translation import (
    ROOT,
    BASE,
    OLD,
    WORK as FIRST,
    NAMES,
    VERBS,
    NUMS,
    OBJECTS,
    ACTIONS,
    Ink,
    capitalize,
    hashed,
    read,
    dump,
    sha,
    key,
)
from scripts.prepare_bilingual_ink import WORK as LARGE, source as opus_source

WORK = ROOT / "artifacts/translation_paraphrases_v1"
PUB = ROOT / "references/evidence/translation_paraphrases_v1"
MODEL = ROOT / "models/translation_paraphrases"
REV = "4cd8187c404bda33cb1f62b49b001115862acf37"
DIGEST = {
    "en-train": "0de6e8a4ba4bd88ee79c883349d112ec1ca85cda78726b59c215fab5849e49f3",
    "zh-train": "87aa77e8c763193c40b7ddf38c0cad5b35550f5186e5922bd2a6c631bc73ce0e",
    "en-test": "fdb1e43f1068cf0cec955327cfdedc2d3065f15d3cd673c9d0002e6e87cc7915",
    "zh-test": "cdae3237c12a49165aacf489d3ab60b832b7672e7775abf937a1495f3ab75c80",
    "en-validation": "e91c4c89a3d74deaf82bf2e87e626cf8d69d85f3c8e736b1ab3b6c3503ff26da",
    "zh-validation": "9c830a0a82e411c3587537380f88abd53d7dba08624a1f3ec1da43bf8265fbef",
}


def published(language, split):
    tag = language + "-" + split
    raw = ROOT / "data/raw/paws-x-en-zh" / (tag + ".parquet")
    if not raw.exists():
        raw.parent.mkdir(parents=True, exist_ok=True)
        r = requests.get(
            f"https://huggingface.co/datasets/google-research-datasets/paws-x/resolve/{REV}/{language}/{split}-00000-of-00001.parquet",
            timeout=60,
        )
        r.raise_for_status()
        if sha256(r.content).hexdigest() != DIGEST[tag]:
            raise ValueError(tag)
        raw.write_bytes(r.content)
    if sha(raw) != DIGEST[tag]:
        raise ValueError(tag)
    return {r["id"]: r for r in pq.read_table(raw).to_pylist()}


def alternatives():
    # The historical training split grouped unordered names; retain that partition.
    pastparts = [
        "helped",
        "visited",
        "thanked",
        "called",
        "invited",
        "followed",
        "seen",
        "found",
        "hugged",
        "praised",
    ]
    zhverbs = [
        "帮助",
        "拜访",
        "感谢",
        "致电",
        "邀请",
        "跟随",
        "看见",
        "找到",
        "拥抱",
        "表扬",
    ]
    for i, (a, x) in enumerate(NAMES):
        for j, (b, y) in enumerate(NAMES):
            if (
                i == j
                or int(hashed(f"proposition-names/{min(i,j)}/{max(i,j)}")[:4], 16) % 5
                == 0
            ):
                continue
            for (past, inf, zp, zb), pp, v in zip(
                VERBS, pastparts, zhverbs, strict=True
            ):
                for neg in (False, True):
                    canonical = (
                        capitalize(
                            f"{a} did not {inf} {b}." if neg else f"{a} {past} {b}."
                        ),
                        x + ("没有" + zb if neg else zp) + y + "。",
                    )
                    # Each source asserts the original event. No time, purpose or extra object is introduced.
                    if neg:
                        en = [
                            f"{b} was not {pp} by {a}.",
                            f"{a} didn't {inf} {b}.",
                            f"It is not true that {a} {past} {b}.",
                        ]
                        zh = [f"{x}并没有{v}{y}。", f"{x}没有{v}{y}。"]
                    else:
                        en = [
                            f"{b} was {pp} by {a}.",
                            f"It was {a} who {past} {b}.",
                            f"The one who {past} {b} was {a}.",
                        ]
                        zh = [f"{v}{y}的人是{x}。", f"是{x}{v}了{y}。"]
                    yield "event", canonical, ([capitalize(s) for s in en], zh)
    participles = ["bought", "found", "sold", "lost", "needed", "counted"]
    for i, (a, x) in enumerate(NAMES):
        for j, (obj, zo) in enumerate(OBJECTS):
            if int(hashed(f"proposition-quant/{i}/{j}")[:4], 16) % 5 == 0:
                continue
            for (verb, zv), pp, stem in zip(
                ACTIONS,
                participles,
                ("买", "找到", "卖", "丢", "需要", "数过"),
                strict=True,
            ):
                for word, num in NUMS:
                    canonical = (
                        capitalize(f"{a} {verb} {word} {obj}."),
                        f"{x}{zv}{num}{zo}。",
                    )
                    en = [
                        capitalize(f"{word} {obj} were {pp} by {a}."),
                        capitalize(f"It was {a} who {verb} {word} {obj}."),
                    ]
                    zh = [
                        f"是{x}{zv}{num}{zo}。",
                        f"{x}{stem}的{zo[1:]}有{num}{zo[0]}。",
                    ]
                    yield "quantity", canonical, (en, zh)


def prepare():
    for d in (WORK, PUB, MODEL):
        d.mkdir(parents=True, exist_ok=True)
    ink = Ink(np.load(BASE / "palette.npy"))
    inv = [set(a) for a in read(BASE / "inventories.json")]
    encode = lru_cache(maxsize=200000)(ink.encode)

    def allowed(side, t):
        return (
            0 < len(side) <= 96
            and set(encode(side)) <= inv[t]
            and not any(c in ink.bad for c in side)
        )

    paired = {}
    disagreements = Counter()
    inconsistent = []
    for split in ("test", "validation", "train"):
        en = published("en", split)
        zh = published("zh", split)
        if en.keys() != zh.keys():
            raise ValueError("Unaligned release")
        pairs = []
        for i, x in en.items():
            y = zh[i]
            forms = [[r[k] for r in (x, y)] for k in ("sentence1", "sentence2")]
            if x["label"] != y["label"]:
                disagreements[split] += 1
                inconsistent.extend(forms)
                continue
            pairs.append(dict(id=i, label=x["label"], forms=forms))
        paired[split] = pairs
    # Check official test candidates against preceding training without storing the training texts.
    candidates = sorted(
        [
            r
            for r in paired["test"]
            if not r["label"]
            and all(allowed(s, t) for z in r["forms"] for t, s in enumerate(z))
        ],
        key=lambda r: hashed("paws-ink-hold/" + str(r["id"])),
    )
    wanted = [{key(z[t]) for r in candidates for z in r["forms"]} for t in (0, 1)]
    seen = [set(), set()]

    def inspect(z):
        for t, s in enumerate(z):
            if len(s) <= 96 and (k := key(s)) in wanted[t]:
                seen[t].add(k)

    for k in ("train", "news-train"):
        for r in read(OLD / (k + ".json")):
            inspect(r["text"])
    # A superset of the 240,000 training rows is safe for excluding evaluation overlap.
    for z in opus_source("train"):
        inspect(z)
    lookup = read(BASE / "readback.json")
    for r in read(FIRST / "contrasts.json"):
        for pair in r["cells"]:
            inspect(["".join(lookup[str(c)] for c in side) for side in pair])
    held = []
    for r in candidates:
        if all(key(z[t]) not in seen[t] for z in r["forms"] for t in (0, 1)):
            held.append(r)
            for z in r["forms"]:
                for t, s in enumerate(z):
                    seen[t].add(key(s))
        if len(held) == 16:
            break
    if len(held) != 16:
        raise ValueError("Insufficient human cases")
    excluded = [set(), set()]

    def exclude(z):
        for t, s in enumerate(z):
            excluded[t].add(key(s))

    for z in inconsistent:
        exclude(z)
    for split in ("test", "validation"):
        for r in paired[split]:
            for z in r["forms"]:
                exclude(z)
    for r in read(FIRST / "held.json")["pairs"]:
        for z in r["forms"]:
            exclude(z)
    for r in read(FIRST / "held.json")["natural"] + read(LARGE / "held.json"):
        exclude(r["text"])
    examples = []
    counts = Counter()

    def add(kind, canonical, alts):
        if not all(
            allowed(s, t) and key(s) not in excluded[t] for t, s in enumerate(canonical)
        ):
            return
        forms = [
            [encode(s) for s in words if allowed(s, t) and key(s) not in excluded[t]]
            for t, words in enumerate(alts)
        ]
        if not all(forms):
            return
        examples.append(dict(target=[encode(s) for s in canonical], variants=forms))
        counts[kind] += 1

    for k, z, v in alternatives():
        add(k, z, v)
    for r in paired["train"]:
        a, b = r["forms"]
        for z, other in ((a, b), (b, a)):
            # Only labelled-equivalent siblings can supply an alternative source.
            v = other if r["label"] else z
            add("paws-positive" if r["label"] else "paws-negative", z, [[s] for s in v])
    dump(WORK / "examples.json", examples)
    dump(WORK / "held.json", held)
    dump(
        PUB / "corpus.json",
        dict(
            revision=REV,
            archives=DIGEST,
            examples=counts,
            label_disagreements=disagreements,
            held=[r["id"] for r in held],
            files={k: sha(WORK / (k + ".json")) for k in ("examples", "held")},
        ),
    )
    print(counts)


if __name__ == "__main__":
    prepare()

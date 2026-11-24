"""Reserve image-to-gloss examples for identical headwords used in different senses."""

from collections import Counter, defaultdict
from pathlib import Path
import hashlib
import json
import os
import re

import numpy as np

from scripts.train_typeface_writer import Face, FACES

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / "artifacts/visual_word_senses_v1"
PUBLIC = ROOT / "references/evidence/visual_word_senses_20261124"
if experiment_dir := os.environ.get("ILM_WORD_SENSE_RUN_DIR"):
    WORK = Path(experiment_dir).expanduser().resolve()
    PUBLIC = WORK / "evidence"
SOURCE = Path(
    os.environ.get(
        "ILM_KYOTO_CONLLU",
        ROOT / "data/raw/ud_classical_kyoto_r216/lzh_kyoto-ud-train.conllu",
    )
)
SOURCE_HASH = "dde87435a4590bde8346cc6b792ff0c6dcd301f876815143187bd9b2c839e02b"
BOOKS = ("KR1h0004", "KR4a0001")
RADIUS = 8
LIMIT = 24
SEED = 241199
ALIASES = {
    "speak": ["talk"],
    "speech": ["utterance"],
    "affair": ["matter"],
    "serve": ["serve-under"],
    "general": ["commander"],
    "clothing": ["clothes"],
    "submit": ["yield"],
    "road": ["path"],
    "scholar": ["learned-person"],
    "child": ["offspring"],
    "sun": ["the-sun"],
    "buy": ["purchase"],
    "eat": ["consume"],
    "go": ["proceed"],
    "receive": ["accept"],
    "wish": ["want"],
    "inherit": ["succeed-to"],
    "death": ["demise"],
    "ancient": ["old"],
}


def save(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def key(data):
    return hashlib.sha256(data.encode()).hexdigest()


def reserve():
    if sha(SOURCE) != SOURCE_HASH:
        raise ValueError("Expected the pinned Kyoto 2.16 training file")
    WORK.mkdir(parents=True, exist_ok=True)
    PUBLIC.mkdir(exist_ok=True)
    raw = SOURCE.read_text()
    faces = [Face(f) for f in FACES[:2]]
    rows = []
    windows = defaultdict(set)
    split_seen = set()
    for block in raw.strip().split("\n\n"):
        m = re.search(r"^# sent_id = (.+)$", block, re.M)
        if not m:
            continue
        sid = m[1]
        work = sid.split("_")[0]
        split = work if work in BOOKS else "train"
        words = [
            line.split("\t")
            for line in block.splitlines()
            if line.split("\t", 1)[0].isdigit()
        ]
        page = "".join(w[1] for w in words)
        pos = 0
        for w in words:
            form = w[1]
            start = pos
            pos += len(form)
            g = re.search(r"(?:^|\|)Gloss=([^|]+)", w[-1])
            if not (
                g
                and re.fullmatch(r"[a-z]+(?:-[a-z]+)*", g[1])
                and len(g[1]) <= LIMIT
                and len(form) == 1
                and "\u3400" <= form <= "\u9fff"
            ):
                continue
            left = page[max(0, start - RADIUS) : start]
            right = page[pos : pos + RADIUS]
            context = left.rjust(RADIUS) + form + right.ljust(RADIUS)
            if any(
                not (
                    c == " "
                    or "\u3400" <= c <= "\u9fff"
                    and all(ord(c) in f.coverage for f in faces)
                )
                for c in context
            ):
                continue
            if not (left or right):
                continue
            rec = (work, sid, start, left, form, right, g[1])
            dup = (split, context, g[1])
            if dup in split_seen:
                continue
            split_seen.add(dup)
            rows.append(rec)
            windows[context].add((split, g[1]))
    clean = []
    for r in rows:
        ctx = r[3].rjust(RADIUS) + r[4] + r[5].ljust(RADIUS)
        if len(windows[ctx]) == 1:
            clean.append(r)
    letters = sorted({c for r in clean for c in r[3] + r[4] + r[5] + r[6]})
    chars = [" "] + letters
    pixels = np.array([[np.frombuffer(f(c), np.uint8) for c in chars] for f in faces])
    assert not pixels[:, 0].any()
    # Different printed Unicode variants may be visually identical. Check and
    # exclude any ambiguous or cross-split rendered window as well.
    index = {c: i for i, c in enumerate(chars)}
    phys = np.unique(pixels[0], axis=0, return_inverse=True)[1]
    views = defaultdict(set)
    material = []
    for r in clean:
        ids = [index[c] for c in r[3].rjust(RADIUS) + r[4] + r[5].ljust(RADIUS)]
        v = phys[ids].astype("<u2").tobytes()
        s = r[0] if r[0] in BOOKS else "train"
        views[v].add((s, r[-1]))
        material.append((s, v, ids, r))
    grouped = defaultdict(list)
    seen = set()
    for s, v, ids, r in material:
        if len(views[v]) != 1 or v in seen:
            continue
        seen.add(v)
        grouped[s].append((ids, r))
    np.save(WORK / "pixels.npy", pixels)
    save(WORK / "cells.json", chars)
    train_senses = Counter((r[4], r[6]) for _, r in grouped["train"])
    cp = {}
    stats = {}
    for split, items in grouped.items():
        ys = np.zeros((len(items), LIMIT + 1), np.int32)
        for i, (_, r) in enumerate(items):
            for j, c in enumerate(r[6]):
                ys[i, j] = index[c]
        path = WORK / (split + ".npz")
        np.savez_compressed(path, x=np.array([x for x, _ in items], np.int32), y=ys)
        meta = [r for _, r in items]
        save(WORK / (split + ".json"), meta)
        stats[split] = dict(
            rows=len(items),
            forms=len({r[4] for r in meta}),
            glosses=len({r[6] for r in meta}),
            inputs=sha(path),
            meta=sha(WORK / (split + ".json")),
        )
        if split == "train":
            continue
        by = defaultdict(lambda: defaultdict(list))
        for i, r in enumerate(meta):
            if train_senses[r[4], r[6]] >= 3:
                by[r[4]][r[6]].append(i)
        pairs = []
        for form, options in sorted(by.items()):
            for g, ids in options.items():
                ids.sort(key=lambda i: key(str(meta[i])))
            while len(active := [g for g, ids in options.items() if ids]) >= 2:
                a, b = sorted(active, key=lambda g: (-len(options[g]), g))[:2]
                pairs.append([options[a].pop(), options[b].pop()])
        cp[split] = pairs
        stats[split]["pairs"] = len(pairs)
    save(PUBLIC / "pairs.json", cp)
    save(PUBLIC / "aliases.json", ALIASES)
    save(
        PUBLIC / "split.json",
        dict(
            input=sha(SOURCE),
            data=stats,
            pixels=sha(WORK / "pixels.npy"),
            cells=sha(WORK / "cells.json"),
            pairs=sha(PUBLIC / "pairs.json"),
            aliases=sha(PUBLIC / "aliases.json"),
            fonts=[dict(index=f[1], sha=sha(f[0])) for f in FACES[:2]],
        ),
    )
    print(stats)


if __name__ == "__main__":
    reserve()

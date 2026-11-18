"""Freeze paired sense contrasts from chapters outside the prior translation rosters."""

from collections import Counter, defaultdict
from hashlib import sha256
from pathlib import Path
import tarfile

import numpy as np

from scripts.prepare_visual_event_roles import (
    BASE,
    OLD,
    ROOT,
    order,
    load,
    sha,
    write_new,
)
from scripts.prepare_visual_edit_translation import pairs
from scripts.prepare_visual_sequence_corpus import Rasterizer
from scripts.survey_pictured_predicate_contexts import PREDICATES, clauses
from scripts.train_causal_glyph_translator import read_rows, tiles

OUT = ROOT / "artifacts/predicate_likelihood_v1"
EVENT = ROOT / "artifacts/visual_event_roles_v1"
SURVEY = (
    ROOT / "references/evidence/pictured_predicate_training_candidates_20261119.json"
)
# Whole-gloss reading of the frozen training inventory. The two distance examples
# labelled 离开 would teach the competing sense under the quantitative reading.
REJECT = {
    4: "九/久 transcription",
    8: "distance ambiguity",
    11: "destination supplied by context",
    13: "incomplete source",
    16: "unexpressed subject and place",
    25: "unexpressed destination",
    29: "lost origin",
    52: "broken proper name",
    56: "uncertain alignment",
    60: "added participants",
    62: "damaged place names",
    64: "distance ambiguity",
    73: "unsupported details",
    99: "tree supplied only in gloss",
    102: "corrupt phrase",
    104: "added together",
    105: "added together",
    108: "damaged place name",
    111: "context supplied",
    117: "added together",
    125: "incomplete gloss",
    126: "added time subordination",
    133: "damaged place name",
    136: "added together",
}


def window(text, predicate):
    i = text.index(predicate) + 1
    return ("^" + text + "$")[i - 1 : i + 2]


def candidates(x, y):
    for p, glosses in PREDICATES.items():
        if x.count(p) != 1 or any(g in x for g in glosses):
            continue
        matched = [g for g in glosses if g in y]
        if len(matched) != 1 or y.count(matched[0]) != 1:
            continue
        a, b = clauses(x), clauses(y)
        if len(a) != len(b):
            continue
        for src, tgt in zip(a, b):
            if (
                p in src
                and matched[0] in tgt
                and 3 <= len(src) <= 22
                and 2 <= len(tgt) <= 40
            ):
                yield dict(source=src, target=tgt, predicate=p, gloss=matched[0])


def prepare():
    OUT.mkdir()
    train = [
        dict(r, index=i)
        for i, r in enumerate(load(SURVEY)["candidates"])
        if i not in REJECT
    ]
    write_new(OUT / "train.json", train)
    renderer = Rasterizer()

    def digest(s):
        return sha256(b"".join(map(renderer.tile, s))).digest()

    used = set()

    def add(s):
        used.add(digest(s))

    windows = {window(r["source"], r["predicate"]) for r in train}
    chapters = set()
    bank = tiles()
    for split in ("train", "heldout"):
        for row in read_rows(split):
            chapters.add(row["group"])
            for col in ("x", "y", "z"):
                used.add(sha256(bank[row[col]].tobytes()).digest())
    for d, roster, src in OLD:
        folder = ROOT / "artifacts" / d
        chapters.update(r["chapter"] for r in load(folder / (roster + ".json")))
        for name in (src, "reference"):
            with np.load(folder / (name + ".npz")) as f:
                raw, offsets = f["packed"], f["offsets"]
                for a, b in zip(offsets, offsets[1:]):
                    used.add(sha256(raw[a:b]).digest())
    chapters.update(
        r["chapter"]
        for r in load(
            ROOT
            / "references/evidence/block_translation_20261118/calibration_mapping.json"
        )
    )
    for name in ("train", "roles", "broad"):
        for r in load(EVENT / (name + ".json")):
            chapters.add(r["chapter"])
            for k in ("source", "target"):
                add(r[k])
                if name != "broad":
                    add(r[k].translate(str.maketrans({r["a"]: r["b"], r["b"]: r["a"]})))
    for r in load(SURVEY)["candidates"]:
        for k in ("source", "target"):
            add(r[k])
            add(r[k] + "。")
    archive = BASE / "archive.tar.gz"
    assert sha(archive) == load(EVENT / "preparation.json")["archive"]
    by = defaultdict(list)
    reasons = Counter()
    with tarfile.open(archive, "r|gz") as tar:
        for item in tar:
            path = Path(item.name)
            if (
                len(path.parts) < 4
                or path.parts[1] != "双语数据"
                or path.name != "bitext.txt"
            ):
                continue
            ch = "/".join(path.parts[2:-1])
            if ch in chapters:
                continue
            for i, (x, y) in enumerate(pairs(tar.extractfile(item).read())):
                for r in candidates(x, y):
                    if window(r["source"], r["predicate"]) in windows:
                        reasons["familiar_predicate_window"] += 1
                        continue
                    try:
                        keys = {
                            digest(r[k] + end)
                            for k in ("source", "target")
                            for end in ("", "。")
                        }
                    except ValueError:
                        reasons["unrenderable"] += 1
                        continue
                    if keys & used:
                        reasons["old_bitmap"] += 1
                        continue
                    by[r["gloss"]].append(
                        (
                            order(f"sense/v1/{ch}/{i}/{r['source']}"),
                            dict(r, chapter=ch, ordinal=i),
                            keys,
                        )
                    )
    selected = []
    test_ch = set()
    test_ink = set()
    for p, (g, h) in PREDICATES.items():
        count = 0
        for _, a, ak in sorted(by[g], key=lambda item: item[0]):
            if a["chapter"] in test_ch or ak & test_ink:
                continue
            for _, b, bk in sorted(by[h], key=lambda item: item[0]):
                if (
                    b["chapter"] in test_ch | {a["chapter"]}
                    or bk & (test_ink | ak)
                    or len(a["source"]) != len(b["source"])
                ):
                    continue
                selected.extend([a, b])
                test_ch.update((a["chapter"], b["chapter"]))
                test_ink.update(ak | bk)
                count += 1
                break
            if count == 12:
                break
    assert all(
        len(selected[i]["source"]) == len(selected[i + 1]["source"])
        and selected[i]["gloss"] != selected[i + 1]["gloss"]
        for i in range(0, len(selected), 2)
    )
    write_new(OUT / "proposed.json", selected)
    write_new(
        OUT / "preparation.json",
        dict(
            archive=sha(archive),
            training_sha=sha(OUT / "train.json"),
            proposed_sha=sha(OUT / "proposed.json"),
            rejected_train=REJECT,
            prior_chapters=len(chapters),
            excluded=dict(reasons),
            eligible_by_gloss={g: len(items) for g, items in by.items()},
            training_counts=dict(Counter(r["gloss"] for r in train)),
            proposed_counts=dict(Counter(r["gloss"] for r in selected)),
        ),
    )
    for i, r in enumerate(selected):
        print(i, r["source"], r["target"], flush=True)


if __name__ == "__main__":
    prepare()

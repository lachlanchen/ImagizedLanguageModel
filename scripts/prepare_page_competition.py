"""Reserve fresh pages and align all visible annotated alternatives on training pages."""

from collections import Counter
from itertools import chain
import json
import numpy as np
from scripts.audit_visible_answer_sets import placement
from scripts.prepare_multiple_ink_runs import (
    ROOT,
    WORK as OLD,
    PUB as OLD_PUB,
    english,
    chinese,
    normalize,
    fragments,
    EN,
    ZH,
    CorpusFace,
    prepared_files,
    read,
    dump,
    sha,
)
from scripts.prepare_complete_fact_sets import WORK as SETS
from scripts.prepare_fact_binding import WORK as BIND
from scripts.reserve_survival_pages import reserve

WORK = ROOT / "artifacts/page_competition_v1"
PUB = ROOT / "references/evidence/page_competition_v1"
MODEL = ROOT / "models/page_competition"


def previous():
    return (
        read(OLD / "held.json")
        + read(OLD / "confirm.json")
        + read(SETS / "confirm.json")
        + read(BIND / "valid.json")
    )


def build(stage):
    hashes = {"en": sha(EN), "zh": sha(ZH)}
    if hashes != read(OLD_PUB / "corpus.json")["raw"]:
        raise ValueError("Changed original corpora")
    held = previous() + read(WORK / "confirm.json")
    qs = {normalize(r["q"]) for r in held}
    excerpts = set().union(*(fragments(r["p"], 24) for r in held))
    face = CorpusFace()
    with np.load(OLD / "training.npz") as f:
        tiles = list(map(bytes, f["tiles"]))
    index = {b: i for i, b in enumerate(tiles) if i >= 2}

    def encode(s):
        ids = []
        for c in s:
            b = face.tile(c)
            if b not in index:
                index[b] = len(tiles)
                tiles.append(b)
            ids.append(index[b])
        return ids

    records = []
    seen = set()
    counts = Counter()
    for r in chain(english("train"), chinese("train")):
        lang = r["lang"]
        q = normalize(r["q"])
        ident = (q, normalize(r["p"]))
        if not q or not ident[1] or q in qs or ident in seen:
            continue
        cache = {}
        refs = []
        for ref in r["refs"]:
            # Preserve one literal spelling for each member under the scoring normalization.
            words = list({normalize(w): w for w in reversed(ref)}.values())
            if placement(r["p"], words, cache) == "placed":
                refs.append(words)
        if not refs or fragments(r["p"], 1) & excerpts:
            continue
        try:
            qi, pi = encode(r["q"]), encode(r["p"])
        except ValueError:
            counts[lang + "_unrenderable"] += 1
            continue
        seen.add(ident)
        options = [[[[i, i + len(w)] for i in cache[w]] for w in ref] for ref in refs]
        records.append(
            dict(
                id=r["id"],
                lang=lang,
                qi=qi,
                pi=pi,
                p=r["p"],
                refs=r["refs"],
                options=options,
            )
        )
        counts[lang] += 1
        counts[lang + "_multiple_lists"] += int(len(refs) > 1)
    with (stage / "training.json").open("w", encoding="utf-8") as out:
        json.dump(records, out, ensure_ascii=False, separators=(",", ":"))
    np.save(
        stage / "training-tiles.npy",
        np.frombuffer(b"".join(tiles), np.uint8).reshape(-1, 128),
    )
    return dict(
        raw=hashes,
        counts=dict(counts),
        held=sha(WORK / "confirm.json"),
        files={n: sha(stage / n) for n in ("training.json", "training-tiles.npy")},
    )


def main():
    prepared_files(
        WORK,
        PUB / "fresh.json",
        {"confirm.json": ("sha256",), "confirm-tiles.npy": ("tiles",)},
        lambda s: reserve(
            OLD,
            s,
            previous(),
            (("en", True, 48), ("zh", True, 112), ("zh", False, 32)),
            "page-competition-v1 ",
            450,
        ),
    )
    prepared_files(
        WORK,
        PUB / "corpus.json",
        {n: ("files", n) for n in ("training.json", "training-tiles.npy")},
        build,
    )


if __name__ == "__main__":
    main()

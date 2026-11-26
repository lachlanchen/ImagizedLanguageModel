"""Reserve source-disjoint book passages for evaluation before comparison of the two paired fits."""

import numpy as np
from scripts import prepare_visual_training_breadth as breadth
from scripts.prepare_opus_paired_supervision import ROOT, PUB, WORK, base
from scripts.prepare_visual_training_breadth import (
    prior,
    past,
    previous,
    last,
    pool,
    key,
    word,
    candidates,
)

RULE = ROOT / "references/paired_branch_training_protocol.md"
BOOKS = (
    "meditations-marcus-aurelius",
    "metamorphosis-kafka",
    "walden",
    "great-gatsby",
    "the-trial-kafka",
    "letter-from-an-unknown-woman",
    "the-overcoat-gogol",
    "samurai-world-warrior",
    "why-nations-fail",
    "simple-path-to-wealth",
    "secrets-question-based-selling",
    "resurrection",
    "trading-in-the-zone",
    "the-count-of-monte-cristo",
    "a-storm-of-swords",
    "return-of-the-king",
)


def main():
    if (PUB / "books.json").exists():
        raise SystemExit("Existing book reserve")
    first = past.previous
    old = first.previous() + base.read(first.WORK / "held.json")
    earlier = (
        first.FIELD_RUN,
        first.TARGET_RUN,
        first.RUN,
        past.span.RUN,
        past.pictured.RUN,
        past.drafts.RUN,
        past.actions.RUN,
        past.preceding.RUN,
        past.RUN,
        base.FIRST,
        base.WORK,
        prior.WORK,
        previous.parent.WORK,
        previous.WORK,
        last.last.WORK,
        last.WORK,
        breadth.WORK,
    )
    for p in (
        first.PARENT / "unseen-books.json",
        first.PRIOR_RUN / "fresh.json",
        *(d / "panel.json" for d in earlier),
    ):
        old.extend(base.read(p))
    used = set(base.read(first.RECEIPT)["works"]) | set(
        base.read(ROOT / "references/evidence/book_breadth_v1/corpus.json")[
            "local_book_sources"
        ]
    )
    for split in ("train", "dev", "test"):
        rows = first.library(split)
        used.update(r["work"] for r in rows)
        old.extend(rows)
    used.update(r["doc"] for r in old if "doc" in r)
    pairs = prior.training_pairs()
    events = np.load(previous.WORK / "pairs.npy", mmap_mode="r")[
        base.read(previous.PUB / "selection.json")["real"] :
    ]
    forbidden = [
        {base.compact(row["cells"][t]) for row in old}
        | {line.tobytes() for line in pairs[:, t]}
        | {line.tobytes() for line in events[:, t]}
        for t in (0, 1)
    ]
    normalized = [{key(row["text"][t]) for row in old} for t in (0, 1)]
    surfaces = [{word(row["cells"][t]) for row in old} for t in (0, 1)]
    families = base.read(WORK / "families.json")
    assert base.sha(WORK / "families.json") == base.read(PUB / "selection.json")["text"]
    for t in (0, 1):
        normalized[t].update(key(pair[t]) for fam in families for pair in fam)
        for a, b in pool(t):
            surfaces[1 - t].add(a)
            surfaces[t].add(b)
    face = base.Ink(np.load(base.BASE / "palette.npy"))
    inv = list(map(set, base.read(base.BASE / "inventories.json")))
    reserved = []
    editions = {}
    for title in BOOKS:
        if any(
            title == u or title.startswith(u + "-") or u.startswith(title + "-")
            for u in used
        ):
            continue
        entries, digest = candidates(title)
        group = []
        for _, en, zh, ch, paragraph, unit in entries:
            text = [en, zh]
            cells = list(map(face.encode, text))
            if any(
                not 1 <= len(v) <= 96
                or not set(v) <= inv[t]
                or face.bad.intersection(text[t])
                or key(text[t]) in normalized[t]
                or base.compact(v) in forbidden[t]
                or word(v) in surfaces[t]
                for t, v in enumerate(cells)
            ):
                continue
            group.append(
                dict(
                    doc=title,
                    text=text,
                    cells=cells,
                    chunk=ch,
                    paragraph=paragraph,
                    unit=unit,
                )
            )
            for t, v in enumerate(cells):
                normalized[t].add(key(text[t]))
                surfaces[t].add(word(v))
            if len(group) == 12:
                break
        if len(group) == 12:
            editions[title] = digest
            reserved.extend(group)
        if len(editions) == 4:
            break
    if len(reserved) != 48:
        raise ValueError(f"Insufficient disjoint passages from {list(editions)}")
    base.dump(WORK / "panel.json", reserved)
    base.dump(
        PUB / "books.json",
        dict(
            editions=editions,
            panel=base.sha(WORK / "panel.json"),
            past_panel=base.sha(breadth.WORK / "panel.json"),
            corpus=base.sha(PUB / "selection.json"),
            protocol=base.sha(RULE),
            cases_per_direction=len(reserved),
        ),
    )
    print("reserved without assessing", list(editions), flush=True)


if __name__ == "__main__":
    main()

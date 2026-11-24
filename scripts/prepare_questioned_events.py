"""Reserve physically disjoint classical lines with multiple annotated events."""

from collections import Counter, defaultdict
import csv
from functools import lru_cache
from pathlib import Path
import os
import re

from scripts.prepare_asked_gloss import ROOT, PRIOR, HELD, SOURCES, read, dump, sha
from scripts.prepare_pronoun_writer import CANONICAL, window
from scripts.reproduce_pictured_action import corpus
from scripts.train_asked_gloss import rows_and_split
from scripts.train_typeface_writer import Face, FACES

PUB = ROOT / "references/evidence/questioned_events_v1"
WORK = Path(os.environ.get("ILM_EVENT_WORK", ROOT / "artifacts/questioned_events_v1"))


def extract(path, part):
    for block in path.read_text().split("\n\n"):
        sid = re.search(r"(?m)^# sent_id = (.+)$", block)
        if not sid:
            continue
        nodes = {}
        pos = 0
        for line in block.splitlines():
            w = line.split("\t")
            if w[0].isdigit():
                m = re.search(r"(?:^|\|)Gloss=([^|]+)", w[-1])
                nodes[w[0]] = (w, pos, m[1] if m else "")
                pos += len(w[1])
        page = "".join(w[1] for w, _, _ in nodes.values())
        if not 3 <= len(page) <= 32 or not all("\u3400" <= c <= "\u9fff" for c in page):
            continue
        arcs = []
        for w, o, noun in nodes.values():
            if w[7] != "obj" or w[6] not in nodes:
                continue
            v, i, verb = nodes[w[6]]
            noun = CANONICAL.get(noun, noun)
            if v[3] != "VERB" or len(v[1]) != 1 or len(w[1]) != 1:
                continue
            if not all(
                re.fullmatch(r"[a-z]+(?:-[a-z]+)*", t) and len(t) <= 24
                for t in (noun, verb)
            ):
                continue
            arcs.append([i, o, verb, [noun]])
        if not arcs or len({k for v, o, _, _ in arcs for k in (v, o)}) != 2 * len(arcs):
            continue
        if len({a[2] for a in arcs}) != len(arcs) or len(
            {a[3][0] for a in arcs}
        ) != len(arcs):
            continue
        book = sid[1].split("_")[0]
        group = "work" if book in HELD else ("train" if part == "train" else "section")
        yield dict(id=sid[1], book=book, group=group, page=page, events=sorted(arcs))


def literature():
    path = ROOT / "data/raw/visual_action_semantics/pu.txt"
    assert (
        sha(path)
        == read(ROOT / "references/evidence/visual_action_meaning_v1/pu_screen.json")[
            "chinese"
        ]
    )
    text = path.read_text()
    with (PUB / "literature.tsv").open() as f:
        for n, r in enumerate(csv.DictReader(f, delimiter="\t")):
            page = r["page"]
            story = r["story"]
            source = text.split("〈" + story + "〉", 1)[1].split("〈", 1)[0]
            assert page in "".join(c for c in source if "\u3400" <= c <= "\u9fff"), page
            events = []
            for k in ("1", "2"):
                v, o = r["verb" + k], r["object" + k]
                assert page.count(v) == page.count(o) == 1
                events.append(
                    [
                        page.index(v),
                        page.index(o),
                        r["action" + k],
                        r["word" + k].split("|"),
                    ]
                )
            yield dict(
                id="pu-" + str(n + 1),
                book=story,
                group="literature",
                page=page,
                events=events,
            )


def prepare():
    WORK.mkdir(parents=True, exist_ok=True)
    face = Face(FACES[0])
    codes = {}

    @lru_cache(None)
    def cell(c):
        return codes.setdefault(face.cell(c), len(codes))

    def key(s):
        return tuple(map(cell, s))

    fit_pages = set()
    fit_windows = set()

    def mark(page):
        fit_pages.add(key(page))
        fit_windows.update(key(window(page, i)) for i in range(len(page)))

    prior = read(PRIOR / "train.json")
    for _, _, _, l, c, r, _ in prior:
        fit_windows.add(key(l.rjust(8) + c + r.ljust(8)))
    fit_windows.update(
        key(r[0]) for r in read(ROOT / "artifacts/pronoun_writer_v1/train.json")
    )
    older, _ = rows_and_split()
    for r in older:
        if r["split"] == "train":
            mark(r["page"])
    for r in read(ROOT / "artifacts/visual_clause_roles_v1/train.json"):
        mark(r["page"])
    groups = defaultdict(dict)
    counts = Counter()
    folder = corpus()
    for part in SOURCES:
        for r in extract(folder / f"lzh_kyoto-ud-{part}.conllu", part):
            if all(ord(c) in face.coverage for c in r["page"]):
                groups[key(r["page"])].setdefault((r["group"], str(r["events"])), r)
    train = []
    test = []
    for choices in groups.values():
        if len(choices) != 1:
            counts["conflict_or_split_duplicate"] += 1
            continue
        r = next(iter(choices.values()))
        if r["group"] == "train":
            train.append(r)
            mark(r["page"])
        elif len(r["events"]) >= 2:
            test.append(r)
    test.extend(literature())
    held = []
    for r in test:
        if key(r["page"]) in fit_pages or any(
            key(window(r["page"], j)) in fit_windows
            for v, o, _, _ in r["events"]
            for j in (v, o)
        ):
            counts["overlap_" + r["group"]] += 1
        else:
            held.append(r)
    dump(WORK / "train.json", train)
    dump(WORK / "held.json", held)
    stats = dict(
        train=len(train),
        multiple=sum(len(r["events"]) > 1 for r in train),
        held=Counter(r["group"] for r in held),
        excluded=counts,
        training_sha256=sha(WORK / "train.json"),
        held_sha256=sha(WORK / "held.json"),
        sources=SOURCES,
    )
    dump(WORK / "split.json", stats)
    print(stats)


if __name__ == "__main__":
    prepare()

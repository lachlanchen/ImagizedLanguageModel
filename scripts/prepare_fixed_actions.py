"""Freeze printed pairs in which exchanging the actions leaves both name images fixed."""

import csv
import hashlib
import re
from collections import Counter
from pathlib import Path

from scripts.prepare_classical_argument_reader import normalized
from scripts.train_optical_role_adapter import (
    PUBLIC as OLD,
    WORK as PARENT,
    ROOT,
    read,
    record,
    sha,
)
from scripts.train_typeface_writer import Face, FACES

PUBLIC = ROOT / "references/evidence/fixed_action_20261124"
WORK = ROOT / "artifacts/fixed_action_v1"
BOOKS = {"romance-three-kingdoms": "three", "water-margin-120": "water"}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def prepare():
    face = Face(FACES[0])
    train = PARENT / "learn.json"
    assert sha(train) == read(OLD / "corpus.json")["learn_sha"]
    training = {
        normalized(p)
        for records in read(train).values()
        for r in records
        for p in r["pages"]
    }
    previous = ROOT / "references/evidence"
    older = [
        normalized(x["page"])
        for name, file in (
            ("role_question_binding", "classical"),
            ("classical_argument", "zuo"),
            ("optical_role_adapter", "guoyu"),
        )
        for x in csv.DictReader(
            (previous / (name + "_20261124") / (file + ".tsv")).open(), delimiter="\t"
        )
    ]
    sources = {}
    chapters = {}
    for b in BOOKS:
        folder = ROOT.parent / "ZhJpBook/books" / b
        book = folder / "work/source-extraction/canon-exact.md"
        sources[b] = sha(book)
        assert sources[b] == read(folder / "book-plan.json")["source_sha256"]["zh"]
        chapters[b] = re.split(r"^## 第[^\n]+回\s*\n", book.read_text(), flags=re.M)[1:]
    rows = []
    fingerprints = []
    episodes = set()
    counts = Counter()
    for v in csv.DictReader((PUBLIC / "passages.tsv").open(), delimiter="\t"):
        b = v["work"]
        a, c = v["left"], v["right"]
        x, y = v["action_left"], v["action_right"]
        sep = v["separator"]
        orig = a + x + sep + c + y
        ch = int(v["chapter"])
        assert orig in chapters[b][ch - 1] and len(x) == len(y) and a != c and x != y
        assert orig.count(a) == orig.count(c) == 1 and not {x, y} & {a, c}
        predicate_pair = tuple(sorted([x, y]))
        assert predicate_pair not in episodes
        episodes.add(predicate_pair)
        assert not any(normalized(x) in p and normalized(y) in p for p in training)
        counts[b] += 1
        scene = f"{BOOKS[b]}-{counts[b]:02}"
        for order in (0, 1):
            names = (a, c) if order == 0 else (c, a)
            for world in (0, 1):
                acts = (x, y) if (order ^ world) == 0 else (y, x)
                page = names[0] + acts[0] + sep + names[1] + acts[1]
                assert not any(normalized(page) in p for p in training | set(older))
                offsets = [0, len(names[0]) + len(x) + 1]
                raw = face(page)
                if world == 0:
                    reference = raw
                for name, j in zip(names, offsets):
                    assert (
                        raw[128 * j : 128 * (j + len(name))]
                        == reference[128 * j : 128 * (j + len(name))]
                        == face(name)
                    )
                for lang in ("zh", "en"):
                    for h, side in enumerate(("left", "right")):
                        question = v[lang + "_" + side]
                        assert x not in question or y not in question
                        assert all(name not in question for name in names)
                        target = (a, c)[h ^ world]
                        rival = (a, c)[1 ^ h ^ world]
                        assert page.count(target) == page.count(rival) == 1
                        assert all(
                            t.isspace() or ord(t) in face.coverage
                            for t in page + question
                        )
                        rows.append(
                            dict(
                                group=BOOKS[b],
                                scene=scene,
                                key=f"{scene}/{order}",
                                chapter=ch,
                                order=order,
                                world=world,
                                h=h,
                                lang=lang,
                                page=page,
                                question=question,
                                answer=target,
                                other=rival,
                                source_start=page.index(target),
                                rival_start=page.index(rival),
                            )
                        )
                        fingerprints.append(
                            [
                                scene,
                                order,
                                world,
                                lang,
                                h,
                                *[
                                    digest(face(t))
                                    for t in (page, question, target, rival)
                                ],
                            ]
                        )
    assert list(counts.values()) == [16, 16] and len(rows) == 512
    record(WORK / "inputs.json", rows)
    record(
        PUBLIC / "inputs.json",
        dict(
            works=sources,
            selection=sha(PUBLIC / "passages.tsv"),
            training=sha(train),
            questions=len(rows),
            passages=dict(counts),
            rendered=digest(repr(fingerprints).encode()),
            inputs=sha(WORK / "inputs.json"),
        ),
    )


if __name__ == "__main__":
    prepare()

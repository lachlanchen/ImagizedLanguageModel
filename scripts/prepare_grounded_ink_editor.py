"""Prepare real labelled clauses and register unseen two-clause writing problems."""

import csv
import re
from pathlib import Path

from scripts.prepare_fixed_actions import digest
from scripts.train_optical_role_adapter import (
    ROOT,
    WORK as FIT,
    PUBLIC as OPTICAL,
    read,
    sha,
    record,
)
from scripts.train_typeface_writer import Face, FACES
from scripts.prepare_classical_argument_reader import normalized

PUBLIC = ROOT / "references/evidence/grounded_ink_editor_20261124"
WORK = ROOT / "artifacts/grounded_ink_editor_v1"
STYLES = ("列出兩個人名", "分別敘述兩件事")
SEPARATORS = "，。；：！？、"
ANCHOR = ROOT / "models/optical_role_reader/adapter.safetensors"


def prepare():
    face = Face(FACES[0])
    held = []
    sources = {}
    for i, r in enumerate(
        csv.DictReader((PUBLIC / "passages.tsv").open(), delimiter="\t")
    ):
        book = r["work"]
        p = ROOT.parent / "ZhJpBook/books" / book
        src = p / (
            "markdown/zh.md"
            if book == "dream-red-chamber"
            else "work/source-extraction/canon-exact.md"
        )
        chapters = re.split(r"^## 第[^\n]+?回[^\n]*\n", src.read_text(), flags=re.M)[1:]
        a, b = r["left"], r["right"]
        x, y = r["action_left"], r["action_right"]
        assert (a + x + r["separator"] + b + y) in chapters[
            int(r["chapter"]) - 1
        ] and len(x) == len(y)
        sources[book] = sha(src)
        for w in (0, 1):
            clause = [a + (y if w else x), b + (x if w else y)]
            page = r["separator"].join(clause)
            assert page.count(a) == page.count(b) == 1
            assert all(
                c.isspace() or ord(c) in face.coverage
                for c in page
                + "".join(r[k] for k in ("zh_left", "zh_right", "en_left", "en_right"))
            )
            for lang in ("zh", "en"):
                held.append(
                    dict(
                        scene=i,
                        work=book,
                        world=w,
                        lang=lang,
                        page=page,
                        questions=[r[lang + "_left"], r[lang + "_right"]],
                        names=[a, b][:: 1 - 2 * w],
                        clauses=clause[:: 1 - 2 * w],
                    )
                )
    path = FIT / "learn.json"
    assert sha(path) == read(OPTICAL / "corpus.json")["learn_sha"]
    data = read(path)
    pool = {}
    for kind, rows in data.items():
        for r in rows:
            # Only original printed passages contribute trainable source fragments.
            page = r["pages"][0]
            for clause in re.split(r"[，。；：！？、,.!?:;\n]+", page):
                clause = clause.strip()
                if not 2 <= len(clause) <= 30 or not all(
                    c.isalnum() and ord(c) in face.coverage for c in clause
                ):
                    continue
                for q in r["qs"]:
                    name = q["answer"]
                    if (
                        not 1 <= len(name) <= 8
                        or len(name) == len(clause)
                        or clause.count(name) != 1
                    ):
                        continue
                    if any(
                        normalized(clause) in normalized(h["page"])
                        or normalized(h["page"]) in normalized(page)
                        for h in held
                    ):
                        continue
                    pool[(clause, name)] = dict(
                        text=clause, name=name, kind=kind, source=q["id"]
                    )
    corpus = sorted(pool.values(), key=lambda r: (r["text"], r["name"]))
    assert (
        len(corpus) > 1500
        and len(held) == 96
        and all(
            read(ROOT / "references/evidence/fixed_action_20261124/result.json")[
                "eligible"
            ][l]
            for l in ("zh",)
        )
    )
    record(WORK / "clauses.json", corpus)
    record(WORK / "held.json", held)
    record(
        PUBLIC / "corpus.json",
        dict(
            training_source=sha(path),
            source_works=sources,
            annotated_clauses=len(corpus),
            unique_texts=len({r["text"] for r in corpus}),
            kinds={k: sum(r["kind"] == k for r in corpus) for k in data},
            reserved=sha(PUBLIC / "passages.tsv"),
            held=sha(WORK / "held.json"),
            pool=sha(WORK / "clauses.json"),
            reader=sha(ANCHOR),
            style_rasters=[digest(face(s)) for s in STYLES],
        ),
    )


if __name__ == "__main__":
    prepare()

"""Reserve independent reading pairs and assemble the actual optical writer curriculum."""

import re
import sqlite3
from itertools import combinations
from zipfile import ZipFile

import numpy as np
from scripts.build_origin_library import OUT
from scripts.prepare_directional_reading import row, digest, prior_index
from scripts.prepare_fact_channel import WORK as PARENT, PUB as PARENT_PUB, Ink
from scripts.prepare_story_answer_fit import (
    ROOT,
    WORK as LAST,
    PUB as LAST_PUB,
    TRAIN,
    read,
    dump,
    sha,
    prepared_files,
)

WORK = ROOT / "artifacts/visual_story_writer_v1"
PUB = ROOT / "references/evidence/visual_story_writer_v1"
MODEL = ROOT / "models/visual_story_writer"
RAW = ROOT / "data/raw/narrative_writer"
FILES = {
    "MCTest.zip": "31007cfc3a760bc9d3f9aca8579cf5a9408ffc04c34a9c241a26cfb8190ac2d9",
    "c3-m-dev.json": "357d0d8d2a29bc845cbe50e048c263629f5e527b70f24c3e0838c387c8d3cb54",
}
STAGES = {
    "oracle": ("甲骨文", "oracle bone"),
    "bronze": ("金文", "bronze"),
    "seal": ("篆書", "seal"),
    "liushutong": ("六書通", "Liushutong"),
}


def groups():
    with ZipFile(RAW / "MCTest.zip") as z:
        for series in ("mc160", "mc500"):
            root = "MCTest/" + series + ".dev."
            inputs = z.read(root + "tsv").decode("utf-8-sig").splitlines()
            refs = z.read(root + "ans").decode().splitlines()
            for line, answers in zip(inputs, refs, strict=True):
                rec = line.split("\t")
                page = " ".join(rec[2].replace("\\newline", " ").split())
                qs = [
                    (
                        rec[3 + 5 * i].split(":", 1)[1].strip(),
                        rec[4 + 5 * i + ord(a) - 65],
                    )
                    for i, a in enumerate(answers.split())
                ]
                yield "mctest", rec[0], page, qs
    for parts, qs, key in read(RAW / "c3-m-dev.json"):
        yield "c3", key, " ".join(parts), [(v["question"], v["answer"]) for v in qs]


def reserve(stage):
    for name, h in FILES.items():
        if sha(RAW / name) != h:
            raise ValueError(name)
    idx = prior_index()
    for p in (
        TRAIN / "training.json",
        LAST / "confirmation.json",
        ROOT / "artifacts/optical_competition_v1/confirmation.json",
    ):
        for r in read(p):
            idx.add(*idx.keys(r))
    rows = []
    counts = {}
    for task in ("mctest", "c3"):
        n = 0
        seen_q = set()
        seen_page = set()
        for _, name, page, qs in sorted(
            (g for g in groups() if g[0] == task),
            key=lambda x: digest("visual-story-reserve" + x[1]),
        ):
            pool = []
            for q, a in qs:
                if re.search(
                    r"following|above|below|which of these|all of these|none of these|下列|以下|下面|上述|选项|以上|都对",
                    q + " " + a,
                    re.I,
                ):
                    continue
                r = row(task, task + "/" + name, page, q, [a], name + "/" + q)
                if not r:
                    continue
                family, question, section, _ = idx.keys(r)
                if (
                    family in idx.families
                    or question in idx.q
                    or section in idx.p
                    or question in seen_q
                    or section in seen_page
                ):
                    continue
                if idx.ink.bad.intersection(q + page + a):
                    continue
                pool.append(r)
            for a, b in combinations(pool, 2):
                if a["q"] == b["q"] or a["answers"] == b["answers"]:
                    continue
                rows.extend((a, b))
                seen_q.update(idx.keys(r)[1] for r in (a, b))
                seen_page.add(idx.keys(a)[2])
                n += 1
                break
            if n == 32:
                break
        if n < 8:
            raise ValueError(f"Only {n} independent {task} pages")
        counts[task] = n
    dump(stage / "confirmation.json", rows)
    return dict(
        files={"confirmation.json": sha(stage / "confirmation.json")},
        pages=counts,
        raw=FILES,
        protocol=sha(ROOT / "references/visual_story_writer_protocol.md"),
    )


def history(ink, split, limit=None):
    db = sqlite3.connect(
        (OUT / "origin.sqlite3").resolve().as_uri() + "?mode=ro", uri=True
    )
    forms = {}
    sql = """SELECT headword,entry_id,stage,label,pixels32,family FROM form JOIN image USING(svg_sha)
    JOIN entry USING(headword,entry_id) WHERE split=? AND has_ink=1 ORDER BY headword,entry_id,stage,label"""
    for c, e, s, l, p, f in db.execute(sql, (split,)):
        forms.setdefault((c, e, f), {}).setdefault(s, []).append((l, p))
    db.close()
    result = []
    used = set()
    for (c, e, f), stages in sorted(
        forms.items(), key=lambda x: digest("history-page/" + str(x[0]))
    ):
        names = [s for s in STAGES if s in stages]
        if len(names) < 2 or f in used:
            continue
        ordered = sorted(names, key=lambda s: digest(e + s))[:2]
        used.add(f)
        images = []
        for s in ordered:
            label, raw = min(stages[s], key=lambda v: digest(e + v[0]))
            if raw not in ink.byink:
                ink.byink[raw] = len(ink.tiles)
                ink.tiles.append(raw)
            images.append(ink.byink[raw])
        if images[0] == images[1]:
            continue
        segments = [ink.encode(c + "。")] + [
            ink.encode(STAGES[s][0] + "（" + STAGES[s][1] + "）") + [i]
            for s, i in zip(ordered, images)
        ]
        for j, s in enumerate(ordered):
            question = (
                f"请写出{c}的{STAGES[s][0]}字形。"
                if j == 0
                else f"Show the {STAGES[s][1]} form of {c}."
            )
            result.append(
                dict(
                    q=ink.encode(question),
                    p=segments,
                    y=[[images[j]]],
                    task="forms",
                    family=f,
                    char=c,
                    stage=s,
                    source=e,
                    question=question,
                    images=images,
                )
            )
        if limit and len(result) >= limit:
            break
    return result


def training():
    reservation = read(LAST_PUB / "reservation.json")

    def verify(path, h):
        if sha(path) != h:
            raise ValueError(path)

    verify(LAST / "indices.json", reservation["files"]["indices.json"])
    verify(TRAIN / "training.json", reservation["story_source"])
    for name in ("tiles.npy", "writer.json", "strings.npz"):
        verify(PARENT / name, read(PARENT_PUB / "corpus.json")["files"][name])
    ink = Ink(np.load(PARENT / "tiles.npy"))
    ix = read(LAST / "indices.json")
    stories = read(TRAIN / "training.json")
    rows = []
    for i in ix["stories"]:
        r = stories[i]
        rows.append(
            dict(
                q=ink.encode(r["q"]),
                p=[ink.encode(b) for b in r["blocks"]],
                y=[ink.encode(a) for a in r["answers"]],
                task="stories",
            )
        )
    replay = read(PARENT / "writer.json")
    with np.load(PARENT / "strings.npz") as z:
        ids, ptr = z["ids"], z["ptr"]

    def seq(i):
        return ids[ptr[i] : ptr[i + 1]].tolist()

    for task, indices in ix["replay"].items():
        for i in indices:
            q, blocks, _, ans, name, _ = replay[i]
            assert name == task
            rows.append(
                dict(
                    q=seq(q), p=list(map(seq, blocks)), y=list(map(seq, ans)), task=task
                )
            )
    rows.extend(history(ink, "train"))
    return ink, rows


if __name__ == "__main__":
    prepared_files(
        WORK,
        PUB / "reservation.json",
        {"confirmation.json": ("files", "confirmation.json")},
        reserve,
    )
    print(read(PUB / "reservation.json"))

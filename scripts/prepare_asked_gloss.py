"""Prepare paired visual questions about the predicate and object of a real clause."""

from collections import Counter, defaultdict
import csv
from functools import lru_cache
import json
from pathlib import Path
import os
import re

import numpy as np

from scripts.prepare_visual_word_senses import SOURCE_HASH, WORK as PRIOR, sha
from scripts.train_typeface_writer import Face, FACES

ROOT = Path(__file__).resolve().parents[1]
WORK = Path(os.environ.get("ILM_ASKED_GLOSS_DIR", ROOT / "artifacts/asked_gloss_v1"))
PUBLIC = ROOT / "references/evidence/asked_gloss_v1"
QUESTIONS = [
    ("句中主要動詞的英文是什麼", "句中直接賓語的英文是什麼"),
    ("請用英文解釋主要動詞", "請用英文解釋直接賓語"),
    ("Give the main verb in English", "Give the direct object in English"),
    ("Translate the main verb", "Translate the direct object"),
    # Never used for fitting.
    ("這段話的主要動詞怎樣譯成英文", "這段話的直接賓語怎樣譯成英文"),
    (
        "How would you translate the main verb",
        "How would you translate the direct object",
    ),
]
SOURCES = {
    "train": SOURCE_HASH,
    "dev": "a67c34fa60e2fc73480cea81cbca18cc7f3e0f730f40fd983f3ca5078e64ad9d",
    "test": "335bfdf23fb93bd61b08ee7a35391321264145930f0ccb91c512f69da792e722",
}
HELD = {"KR1h0004", "KR4a0001", "KR6c0127", "KR6f0082"}


def read(path):
    return json.loads(path.read_text())


def dump(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")


def iter_rows(path, part):
    for block in path.read_text().strip().split("\n\n"):
        sid = re.search(r"^# sent_id = (.+)$", block, re.M)
        if not sid:
            continue
        sid = sid[1]
        book = sid.split("_")[0]
        if (part == "train") == (book in HELD):
            continue
        words = [
            l.split("\t") for l in block.splitlines() if l.split("\t")[0].isdigit()
        ]
        roots = [w for w in words if w[7] == "root" and w[3] == "VERB"]
        if len(roots) != 1:
            continue
        objs = [w for w in words if w[7] == "obj" and w[6] == roots[0][0]]
        if len(objs) != 1:
            continue
        page = "".join(w[1] for w in words)
        if not 3 <= len(page) <= 32 or not all("\u3400" <= c <= "\u9fff" for c in page):
            continue
        focus, gloss = [], []
        for target in (roots[0], objs[0]):
            g = re.search(r"(?:^|\|)Gloss=([a-z]+(?:-[a-z]+)*)(?:\||$)", target[-1])
            if len(target[1]) != 1 or not g or len(g[1]) > 24:
                break
            focus.append(sum(len(w[1]) for w in words[: words.index(target)]))
            gloss.append([g[1]])
        if (
            len(gloss) == 2
            and gloss[0] != gloss[1]
            and page[focus[0]] != page[focus[1]]
        ):
            yield dict(
                book=book,
                split="train" if part == "train" else book,
                id=sid,
                page=page,
                focus=focus,
                gloss=gloss,
            )


def reserve():
    baseline = read(ROOT / "references/evidence/visual_word_senses_20261124/split.json")
    for name, h in [
        ("pixels.npy", baseline["pixels"]),
        ("train.npz", baseline["data"]["train"]["inputs"]),
    ]:
        if sha(PRIOR / name) != h:
            raise ValueError("Rebuild the authenticated visual gloss training split")
    face = Face(FACES[0])
    old_bank = np.load(PRIOR / "pixels.npy")[0]
    physical = {bits.tobytes(): i for i, bits in enumerate(np.unique(old_bank, axis=0))}

    # Unknown bitmaps are allocated new physical IDs, never a shared unknown symbol.
    @lru_cache(None)
    def cell_id(c):
        ink = face.cell(c)
        return physical.setdefault(ink, len(physical))

    ids = np.array([physical[p.tobytes()] for p in old_bank], dtype="<u2")
    with np.load(PRIOR / "train.npz") as data:
        trained = {r.tobytes() for r in ids[data["x"]]}
    raw = []
    source_dir = Path(
        os.environ.get("ILM_KYOTO_DIR", ROOT / "data/raw/ud_classical_kyoto_r216")
    )
    for part, expected in SOURCES.items():
        p = source_dir / f"lzh_kyoto-ud-{part}.conllu"
        if sha(p) != expected:
            raise ValueError("Wrong Kyoto edition: " + str(p))
        raw.extend(iter_rows(p, part))
    external = PUBLIC / "daodejing.tsv"
    for n, r in enumerate(
        csv.DictReader(external.read_text().splitlines(), delimiter="\t")
    ):
        page = r["passage"]
        assert all(page.count(r[k]) == 1 for k in ("predicate", "object"))
        raw.append(
            dict(
                book="Daodejing",
                split="Daodejing",
                id=f"{r['chapter']}.{n+1}",
                page=page,
                focus=[page.index(r[k]) for k in ("predicate", "object")],
                gloss=[r[k + "_glosses"].split("|") for k in ("predicate", "object")],
            )
        )
    groups = defaultdict(list)
    exclusions = Counter()
    for r in raw:
        if any(ord(c) not in face.coverage for c in r["page"]):
            exclusions[r["split"], "font"] += 1
            continue
        groups[tuple(map(cell_id, r["page"]))].append(r)
    kept = []
    for page, rows in groups.items():
        tags = {(r["split"], tuple(r["focus"]), repr(r["gloss"])) for r in rows}
        if len(tags) != 1:
            for r in rows:
                exclusions[r["split"], "conflict_or_overlap"] += 1
            continue
        r = rows[0]
        exclusions[r["split"], "duplicate"] += len(rows) - 1
        pad = (cell_id(" "),) * 8 + page + (cell_id(" "),) * 8
        if r["split"] != "train" and any(
            np.asarray(pad[i : i + 17], dtype="<u2").tobytes() in trained
            for i in r["focus"]
        ):
            exclusions[r["split"], "prior_writer_window"] += 1
            continue
        kept.append(r)
    WORK.mkdir(parents=True, exist_ok=True)
    path = WORK / "rows.json"
    dump(path, kept)
    result = dict(
        rows_sha256=sha(path),
        external_sha256=sha(external),
        sources=SOURCES,
        rows=dict(Counter(r["split"] for r in kept)),
        exclusions={f"{s}/{k}": v for (s, k), v in exclusions.items() if v},
        questions=QUESTIONS,
        held_ids=[r["id"] for r in kept if r["split"] != "train"],
    )
    dump(WORK / "split.json", result)
    print(result["rows"], result["exclusions"])


if __name__ == "__main__":
    reserve()

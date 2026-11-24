"""Test frozen CrossWOZ reading on ten DuReader attraction-admission questions."""

import json
import re
import zipfile
import hashlib
import torch
from scripts.prepare_prose_fields import ROOT, PUBLIC, LIMIT, dump
from scripts.prepare_chinese_card_reply import Rasterizer
from scripts.write_prose_fields import ProseReply, render

# Extracted from the human references before inference. Alternative or conditional
# core amounts receive a separate, deliberately permissive count.
CASES = {
    182400: ("北京园博园", ["20元"], False),
    183192: ("宝泉风景区", ["80元", "110元", "100元"], False),
    183903: ("无锡鼋头渚", ["90元"], False),
    184484: ("世博会博物馆", ["免费"], True),
    184936: ("丹霞山", ["200元", "140元"], False),
    187424: ("北京故宫", ["60元", "40元"], False),
    189500: ("青龙峡", ["40元"], False),
    189973: ("白鹿原影视城", ["80元", "免费"], False),
    190104: ("壶口瀑布", ["100元"], False),
    190402: ("嵩山少林寺", ["100元"], True),
}


def fragments(doc):
    for i, s in enumerate(doc["paragraphs"]):
        if not s.strip():
            continue
        starts = sorted(
            {*range(0, max(1, len(s) - LIMIT + 1), 80), max(0, len(s) - LIMIT)}
        )
        for start in starts:
            yield i, start, s[start : start + LIMIT]


def pick(q, name, docs):
    bigrams = {q[i : i + 2] for i in range(len(q) - 1)}
    candidates = []
    for j, doc in enumerate(docs):
        if not doc["is_selected"]:
            continue
        for k, start, s in fragments(doc):
            rank = (
                bool(re.search(r"(?:\d+(?:\.\d+)?\s*元|免费|免票)", s)),
                sum(c in s for c in bigrams),
                name in s,
                "门票" in s,
            )
            candidates.append((rank, j, k, start, s))
    if candidates:
        return max(candidates, key=lambda x: x[0])
    return None


def run():
    rows = []
    archive = ROOT / "data/raw/dureader2/dureader_raw.zip"
    digest = hashlib.sha256()
    with archive.open("rb") as handle:
        for chunk in iter(lambda: handle.read(2**20), b""):
            digest.update(chunk)
    if (
        digest.hexdigest()
        != "643aa0d555b684bf3da68dc92516711223cecc899ed53f9e1ef1113dd44af571"
    ):
        raise ValueError("Different DuReader archive")
    with zipfile.ZipFile(archive) as z:
        for part in ("search", "zhidao"):
            for line in z.open("raw/devset/" + part + ".dev.json"):
                r = json.loads(line)
                id = r["question_id"]
                q = r["question"]
                if "门票" not in q or id == 188547:
                    continue
                name, prices, simple = CASES[id]
                found = pick(q, name, r["documents"])
                if not found:
                    rows.append(dict(id=id, part=part, missing=True))
                    continue
                _, j, k, start, s = found
                rows.append(
                    dict(
                        id=id,
                        part=part,
                        domain="景点",
                        name=name,
                        q=q,
                        text=s,
                        where=[j, k, start],
                        prices=prices,
                        simple=simple,
                    )
                )
    assert {r["id"] for r in rows} == set(CASES)
    torch.set_num_threads(2)
    raster = Rasterizer()
    for arm in ("base", "entity-ink"):
        model = ProseReply("cpu", arm=None if arm == "base" else arm)
        for r in rows:
            if r.get("missing"):
                continue
            page = render(r["q"], r["domain"], r["name"], r["text"], raster)
            s = model.batch([page])[0]
            ink = s["pixels"]
            value = None
            if ink is not None and s["keys"] == ["门票"]:
                a, b = s["intervals"]["门票"]
                value = r["text"][a:b]
            r[arm] = dict(
                fields=s["keys"],
                value=value,
                compatible_core=value in r["prices"],
                line=None if ink is None else model.writer.transcript(ink),
            )
    for r in rows:
        if r.get("missing"):
            continue
        s = r.pop("text")
        r["excerpt_sha256"] = hashlib.sha256(s.encode()).hexdigest()
        r["core_present"] = any(v in s for v in r["prices"])
    dump(PUBLIC / "dureader-admissions.json", rows)
    print(
        [
            (r["id"], r.get("core_present"), r.get("base"), r.get("entity-ink"))
            for r in rows
        ]
    )


if __name__ == "__main__":
    run()

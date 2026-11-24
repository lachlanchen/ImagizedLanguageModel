"""Measure complete contrast pairs from the public Winograd Schema Collection."""

from collections import Counter
from difflib import SequenceMatcher
import hashlib
import re
from xml.etree import ElementTree

import requests
import torch
from scripts.prepare_story_mentions import ROOT, PUBLIC, CELLS, MAX_SPAN, dump
from scripts.audit_story_cast import png
from scripts.write_masked_mentions import MaskedReader, cast_cells
from scripts.write_story_mentions import render_cells, tiles
from scripts.prepare_visual_sequence_corpus import Rasterizer

URL = "https://cs.nyu.edu/~davise/papers/WinogradSchemas/WSCollection.xml"
DIGEST = "98c845a5b33c109f2c49742e726fac592bbb1fdc3b3706b604de12f9c9f58c91"


def schemas():
    path = ROOT / "data/raw/story_mentions_external/winograd.xml"
    if path.is_file():
        raw = path.read_bytes()
    else:
        response = requests.get(URL, timeout=45)
        response.raise_for_status()
        raw = response.content
    if hashlib.sha256(raw).hexdigest() != DIGEST:
        raise ValueError("Winograd collection digest changed")
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
    return ElementTree.fromstring(raw).findall("schema")


def extract(node):
    def clean(field):
        return " ".join((node.findtext(field) or "").split())

    left, pron, right = [clean("text/" + key) for key in ("txt1", "pron", "txt2")]
    text = " ".join((left, pron, right)).strip()
    start = len(left) + 1
    names = [" ".join((a.text or "").split()) for a in node.findall("answers/answer")]
    label = clean("correctAnswer").strip(" .").upper()
    if (
        len(names) != 2
        or label not in ("A", "B")
        or not pron
        or not left
        or len(text) > CELLS
    ):
        return None
    spans = []
    for name in names:
        forms = [name, re.sub(r"^(the|an?|some)\s+", "", name, flags=re.I)]
        found = []
        for form in forms:
            if form and len(form) <= MAX_SPAN:
                found = list(
                    re.finditer(
                        r"(?<!\w)" + re.escape(form) + r"(?!\w)", left, flags=re.I
                    )
                )
            if found:
                break
        if not found:
            return None
        spans.append(found[-1].span())
    if max(x[0] for x in spans) < min(x[1] for x in spans):
        return None
    return (
        text,
        (start, start + len(pron)),
        spans,
        ("A", "B").index(label),
        tuple(x.casefold() for x in names),
    )


def main():
    torch.set_num_threads(2)
    items = [extract(s) for s in schemas()]
    reader = MaskedReader(trained_cast=True, ordered=True)
    raster = Rasterizer()
    cases = []
    used = set()
    for j, (one, two) in enumerate(zip(items, items[1:])):
        if not one or not two or j in used:
            continue
        if (
            one[3] == two[3]
            or one[4] != two[4]
            or one[0][slice(*one[1])] != two[0][slice(*two[1])]
        ):
            continue
        if SequenceMatcher(None, one[0], two[0], autojunk=False).ratio() < 0.7:
            continue
        if [one[0][a:b] for a, b in one[2]] != [two[0][a:b] for a, b in two[2]]:
            continue
        used.update((j, j + 1))
        rows = []
        for text, q, spans, target, _ in (one, two):
            source = tiles(text, raster)
            panel = cast_cells([source[a:b] for a, b in spans])
            _, out = reader.answer(png(render_cells(source, q)), png(panel))
            word = None if out is None else out.tobytes()
            answer = next(
                (i for i, (a, b) in enumerate(spans) if source[a:b].tobytes() == word),
                None,
            )
            rows.append(
                dict(
                    wanted=target,
                    answer=answer,
                    nearest=max(range(2), key=lambda i: spans[i][1]),
                )
            )
        cases.append(dict(xml_rows=[j + 1, j + 2], rows=rows))
    totals = Counter()
    for case in cases:
        rows = case["rows"]
        for arm in ("answer", "nearest"):
            hits = [r["wanted"] == r[arm] for r in rows]
            totals[arm + "_both"] += all(hits)
            totals[arm + "_individual"] += sum(hits)
        totals["answer_changed"] += rows[0]["answer"] != rows[1]["answer"]
    dump(
        PUBLIC / "winograd-pairs.json",
        dict(
            source=URL,
            sha256=DIGEST,
            xml_rows=len(items),
            individually_eligible=sum(r is not None for r in items),
            pairs=len(cases),
            scores=totals,
            decisions=cases,
        ),
    )
    print(len(items), len(cases), totals)


if __name__ == "__main__":
    main()

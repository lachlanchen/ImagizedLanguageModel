"""Construct paired, pictured questions and score actual replies under two typefaces."""

from collections import Counter
from dataclasses import dataclass
import json
from scripts.interval_teacher_corpus import fingerprints
from scripts.fit_font_invariant_reader import Face
from scripts.span_reservation import reservation
from scripts.train_causal_glyph_writer import in_cells, sha
from scripts.train_typeface_writer import PREVIOUS, ROOT, FACES, HELD

TRIAL_SHA = "f3fae95b57da8e03afb2b57467dd221417060ef4d82db13bf22fc88589f3a6f3"


def extra_sources():
    path = ROOT / "data/raw/cmrc2018/cmrc2018_trial.json"
    assert sha(path) == TRIAL_SHA
    blocked, ids = set(), set()
    for name in ("train", "dev"):
        prior = json.loads(path.with_name(f"cmrc2018_{name}.json").read_text())
        for article in prior["data"]:
            for v in article["paragraphs"]:
                blocked.update(fingerprints(v["context"][:416]))
                ids.update(q["id"] for q in v["qas"])
    guard = reservation()
    stats = Counter()
    rows = []
    for article in json.loads(path.read_text())["data"]:
        for v in article["paragraphs"]:
            p = v["context"][:416]
            stats["source_paragraphs"] += 1
            fingerprint = fingerprints(p)
            if fingerprint & blocked or guard.overlaps(v["context"]):
                stats["overlaps"] += 1
                continue
            pair = []
            for q in v["qas"]:
                question = q["question"]
                refs = list(
                    dict.fromkeys(
                        a["text"]
                        for a in q["answers"]
                        if 0 < len(a["text"]) <= 48 and a["text"] in p
                    )
                )
                if (
                    q["id"] in ids
                    or not refs
                    or not question
                    or len(p) + 2 * len(question) + 52 > 1024
                ):
                    continue
                if pair and (
                    question == pair[0]["q"]
                    or any(a in b or b in a for a in refs for b in pair[0]["refs"])
                ):
                    continue
                pair.append(
                    dict(
                        p=p,
                        q=question,
                        refs=refs,
                        id=q["id"],
                        board=f"trial/{stats['source_paragraphs']}",
                    )
                )
                if len(pair) == 2:
                    rows.extend(pair)
                    blocked.update(fingerprint)
                    ids.update(a["id"] for a in pair)
                    break
            if len(pair) < 2:
                stats["insufficient_questions"] += 1
    return rows, dict(stats)


@dataclass
class Styled:
    board: str
    p: bytes
    q: bytes
    canonical: tuple
    source: tuple
    question_id: str

    def grade(self, y, stop):
        if not stop:
            return False, False, False
        same = y in self.source
        regular = y in self.canonical
        if same or regular:
            return same, regular, True
        cells = in_cells(y)
        flexible = any(
            len(y) == len(a)
            and all(
                out == u or out == v
                for out, u, v in zip(cells, in_cells(a), in_cells(b), strict=True)
            )
            for a, b in zip(self.source, self.canonical, strict=True)
        )
        return same, regular, flexible


def pictures(rows, font, canonical):
    cases = []
    for a, b in zip(rows[::2], rows[1::2], strict=True):
        if a["board"] != b["board"]:
            raise ValueError("Question pairs out of order")
        if any(
            not c.isspace()
            and (ord(c) not in font.coverage or ord(c) not in canonical.coverage)
            for v in (a, b)
            for s in (v["p"], v["q"], *v["refs"])
            for c in s
        ):
            continue
        for v in (a, b):
            cases.append(
                Styled(
                    v["board"],
                    font(v["p"]),
                    font(v["q"]),
                    tuple(map(canonical, v["refs"])),
                    tuple(map(font, v["refs"])),
                    v["id"],
                )
            )
    return cases


def sets():
    path = PREVIOUS / "pairs_dev.jsonl"
    assert (
        sha(path)
        == json.loads((PREVIOUS / "corpus.json").read_text())["fresh"]["sha256"]
    )
    fresh = list(map(json.loads, path.read_text().splitlines()))
    extra, excluded = extra_sources()
    canonical = Face(FACES[0])
    cohorts = {}
    for name, desc in [
        ("regular", FACES[0]),
        ("bold", FACES[1]),
        ("kai", FACES[2]),
        ("hana", HELD),
    ]:
        font = Face(desc)
        cohorts[name] = pictures(fresh, font, canonical)
        if name in ("regular", "hana"):
            cases = pictures(extra, font, canonical)
            if cases:
                cohorts["trial_" + name] = cases
    cohorts["hana_erased"] = [
        Styled(c.board, c.p, bytes(128), c.canonical, c.source, c.question_id)
        for c in cohorts["hana"]
    ]
    return cohorts, dict(
        extra_pairs=len(extra) // 2,
        source_counts=excluded,
        trial_sha=TRIAL_SHA,
        held_font_sha=sha(HELD[0]),
    )

"""Test extra weight on the first physical action of each bilingual book reply."""

import argparse
from collections import Counter
import hashlib
import json

from scripts.interval_teacher_corpus import load_rows
from scripts.pointing_glyph_corpus import digest
from scripts.train_bilingual_novel_writer import WORK as BOOK, PARENT, SEED, novel_rows
from scripts.train_causal_glyph_writer import record
from scripts.train_mixed_visual_answers import fit_rows
from scripts.train_span_choice import gpu_owner
from scripts.train_typeface_writer import ROOT

WORK = ROOT / "artifacts/first_glyph_book_v1"
PUBLIC = ROOT / "references/evidence/first_glyph_book_20261124"


def positions(rows, arm):
    if arm not in ("first", "later"):
        raise ValueError(arm)
    result = []
    for row in rows:
        if row.task not in ("novel_chinese", "novel_second"):
            result.append(None)
            continue
        n, remainder = divmod(len(row.y), 128)
        if not n or remainder:
            raise ValueError("Expected a nonempty glyph stream")
        h = hashlib.sha256(row.key.encode()).digest()
        result.append(0 if arm == "first" else 1 + int.from_bytes(h[:8], "big") % n)
    return result


def train(arms):
    book, binding = novel_rows("bilingual")
    rows = load_rows() + book
    old = json.loads((BOOK / "bilingual/corpus.json").read_text())
    assert (
        digest(rows) == old["image_digest"]
        and len(rows) == old["rows"]
        and SEED == old["seed"]
    )
    assert binding == json.loads((BOOK / "bilingual/assignments.json").read_text())
    PUBLIC.mkdir(parents=True, exist_ok=True)
    WORK.mkdir(parents=True, exist_ok=True)
    for arm in arms:
        path = WORK / arm
        path.mkdir()
        focus = positions(rows, arm)
        counts = Counter(i for i in focus if i is not None)
        stops = sum(
            i is not None and i == len(r.y) // 128
            for i, r in zip(focus, rows, strict=True)
        )
        assert sum(counts.values()) == binding["presentations"]
        record(
            path / "focus.json",
            dict(positions=counts, stop=stops, presentations=sum(counts.values())),
        )
        fit_rows(path, PARENT / "mapped/page", old["parent"], rows, SEED, focus=focus)
        report = json.loads((path / "page_fit.json").read_text())
        report.update(
            focus=json.loads((path / "focus.json").read_text()),
            image_digest=old["image_digest"],
        )
        record(PUBLIC / (arm + "_fit.json"), report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("arms", nargs="+", choices=("first", "later"))
    with gpu_owner():
        train(parser.parse_args().arms)

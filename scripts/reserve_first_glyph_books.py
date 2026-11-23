"""Authenticate and render the four separately sourced book reading evaluations."""

import argparse
from functools import lru_cache
import json
from pathlib import Path
import re
import subprocess

from bs4 import BeautifulSoup
from scripts.evaluate_bilingual_novel_writer import overlapping_images
from scripts.train_bilingual_novel_writer import FACES, Face, novel_rows, opposite
from scripts.interval_teacher_corpus import load_rows
from scripts.train_causal_glyph_writer import record, sha
from scripts.train_first_glyph_book import PUBLIC
from scripts.typeface_writer_cases import Styled

PAGES = PUBLIC / "reserved_books.json"
RESERVATION = PUBLIC / "reservation.json"


@lru_cache(maxsize=16)
def paragraphs(path):
    raw = path.read_text()
    if path.suffix == ".html":
        parts = [p.get_text() for p in BeautifulSoup(raw, "html.parser").find_all("p")]
    else:
        parts = re.split(r"\n\s*\n", raw)
    joined = "\n".join("".join(p.split()) for p in parts)
    return (
        subprocess.check_output(
            ["uconv", "-x", "Traditional-Simplified"], input=joined.encode()
        )
        .decode()
        .split("\n")
    )


def passage(source):
    short = source[:210]
    ends = [m.end() for m in re.finditer("[。！？；]", short)]
    if len(source) > 210 and ends and ends[-1] > 68:
        short = short[: ends[-1]]
    return short.replace("\u200b", "")


def cases(pages, keys, face):
    result = {k: [] for k in ("fresh_zh", "fresh_en", "fresh_erased")}
    for key in keys:
        row = pages[key]
        page = face(row["page"])
        for i, q in enumerate(row["questions"]):
            refs = tuple(map(face, q["refs"]))
            for group, items in result.items():
                lang = group.split("_")[1]
                prompt = bytes(128) if lang == "erased" else face(q[lang])
                items.append(Styled(key, page, prompt, refs, refs, f"{key}/{i}"))
        assert not overlapping_images(*result["fresh_en"][-2:])
    return result


def reserved(pages_path=PAGES, reservation_path=RESERVATION):
    receipt = json.loads(reservation_path.read_text())
    assert sha(pages_path) == receipt["pages_sha256"]
    return cases(
        json.loads(pages_path.read_text()), receipt["eligible"], Face(FACES[0])
    )


def verify(library, pages_path=PAGES, reservation_path=RESERVATION, earlier_pages=None):
    pages = json.loads(pages_path.read_text())
    face = Face(FACES[0])
    excluded = {}
    eligible = []
    for k, v in pages.items():
        path = library / v["source_file"]
        assert sha(path) == v["source_sha256"]
        assert passage(paragraphs(path)[v["paragraph"]]) == v["page"], k
        assert opposite(*v["questions"]), k
        strings = [v["page"]]
        for q in v["questions"]:
            assert all(r and r in v["page"] for r in q["refs"]), (k, q)
            strings.extend([q["en"], q["zh"], *q["refs"]])
        missing = sorted(
            {
                c
                for s in strings
                for c in s
                if not c.isspace()
                and (ord(c) not in face.coverage or not any(face.cell(c)))
            }
        )
        if missing:
            excluded[k] = missing
        else:
            eligible.append(k)
    groups = cases(pages, eligible, face)
    # Sampling training starts at stride 16 guarantees detection of any
    # contiguous aligned match of 47 or more glyphs using width-32 windows.
    width = 32 * 128
    fresh = {
        r.p[i : i + width]
        for r in groups["fresh_zh"][::2]
        for i in range(0, len(r.p) - width + 1, 128)
    }
    rows = load_rows() + novel_rows("bilingual")[0]
    train = {r.p for r in rows}
    for page in train:
        if any(
            page[i : i + width] in fresh
            for i in range(0, len(page) - width + 1, 16 * 128)
        ):
            raise ValueError("Overlapping book and fit passages")
    evidence = dict(
        pages_sha256=sha(pages_path),
        eligible=eligible,
        missing_glyphs=excluded,
        proposed=len(pages),
        trained_rows=len(rows),
        unique_fit_pages=len(train),
        queried_windows=len(fresh),
        overlap_at_least_47_cells=False,
    )
    if earlier_pages is not None:
        earlier = json.loads(earlier_pages.read_text())
        assert {r["title"] for r in earlier.values()}.isdisjoint(
            r["title"] for r in pages.values()
        )
        for old in earlier.values():
            bitmap = face(old["page"])
            if any(
                bitmap[i : i + width] in fresh
                for i in range(0, len(bitmap) - width + 1, 128)
            ):
                raise ValueError("Literary reservations share a 32-cell window")
        evidence["previous_literary_sha256"] = sha(earlier_pages)
    if (
        reservation_path.exists()
        and json.loads(reservation_path.read_text()) != evidence
    ):
        raise ValueError("Reservation changed")
    record(reservation_path, evidence)
    print(evidence, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, default=Path("../ZhJpBook/sources"))
    verify(parser.parse_args().library)

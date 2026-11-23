"""Verify reserved bilingual questions against the sources and the fitting corpus."""
import json
from pathlib import Path
from scripts.prepare_bilingual_book_intervals import PUBLIC
from scripts.reserve_first_glyph_books import PAGES as FIRST, cases, verify
from scripts.reserve_preference_books import PAGES as SECOND
from scripts.train_typeface_writer import Face, FACES
from scripts.train_causal_glyph_writer import sha

PAGES = PUBLIC / "reserved_books.json"
CHECK = PUBLIC / "reservation.json"


def packets():
    receipt = json.loads(CHECK.read_text())
    assert sha(PAGES) == receipt["pages_sha256"]
    pages = json.loads(PAGES.read_text())
    face = Face(FACES[0])
    return {
        register: cases(pages, [k for k in receipt["eligible"] if pages[k]["register"] == register], face)
        for register in ("classical", "modern")
    }


if __name__ == "__main__":
    verify(Path.cwd(), PAGES, CHECK, earlier_pages=SECOND)
    verify(Path.cwd(), PAGES, PUBLIC / "first_reservation_check.json", earlier_pages=FIRST)

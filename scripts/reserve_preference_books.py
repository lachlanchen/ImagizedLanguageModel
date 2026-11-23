"""Check the six-work evaluation against the actual source files and earlier pages."""
from pathlib import Path

from scripts.mine_stopped_book_replies import PUBLIC
from scripts.reserve_first_glyph_books import PAGES as PREVIOUS, reserved, verify

PAGES = PUBLIC / "reserved_books.json"
CHECK = PUBLIC / "reservation.json"


def literary_cases():
    return reserved(PAGES, CHECK)


if __name__ == "__main__":
    verify(Path("../ZhJpBook/sources"), PAGES, CHECK, earlier_pages=PREVIOUS)

"""Export attested SVGs with their entry metadata or render an indexed book page."""

import argparse
import json
import sqlite3
from hashlib import sha256
from pathlib import Path

import fitz
from scripts.build_origin_library import OUT, ROOT, sha


def lookup(db, char, dest):
    entries = db.execute(
        "SELECT entry_id,fields,split FROM entry WHERE headword=? ORDER BY entry_id",
        (char,),
    ).fetchall()
    if not entries:
        raise ValueError(f"No local etymology entry for {char}")
    result = []
    for entry, fields, split in entries:
        rows = []
        for stage, label, h, svg, usable in db.execute(
            """SELECT stage,label,svg_sha,svg,has_ink FROM form JOIN image USING(svg_sha)
              WHERE headword=? AND entry_id=? ORDER BY stage,label""",
            (char, entry),
        ):
            if sha256(svg).hexdigest() != h:
                raise ValueError(label)
            name = f"{entry}_{stage}_{label}.svg"
            (dest / name).write_bytes(svg)
            rows.append(
                dict(
                    stage=stage,
                    label=label,
                    svg=name,
                    sha256=h,
                    visible_at_32=bool(usable),
                )
            )
        result.append(
            dict(
                entry=entry,
                headword=char,
                split=split,
                source_fields=json.loads(fields),
                forms=rows,
            )
        )
    (dest / "entries.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    )


def page(db, book, page_no, dest, books):
    row = db.execute(
        "SELECT filename,sha,pages FROM book WHERE id=?", (book,)
    ).fetchone()
    if not row or not 1 <= page_no <= row[2]:
        raise ValueError("Unknown book or page")
    file = books / row[0]
    if sha(file) != row[1]:
        raise ValueError("Book checksum differs from the index")
    with fitz.open(file) as doc:
        if doc.is_reflowable:
            doc.layout(width=612, height=792, fontsize=12)
        if doc.page_count != row[2]:
            raise ValueError("Page layout changed")
        pg = doc[page_no - 1]
        pg.get_pixmap(matrix=fitz.Matrix(1.5, 1.5), alpha=False).save(dest / "page.png")
        (dest / "native-text.txt").write_text(pg.get_text())
    (dest / "source.json").write_text(
        json.dumps(dict(book=book, page=page_no, source_sha256=row[1]), indent=2) + "\n"
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--char")
    group.add_argument("--book")
    p.add_argument("--page", type=int, default=1)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--db", type=Path, default=OUT / "origin.sqlite3")
    p.add_argument(
        "--books", type=Path, default=ROOT / "references/source_books/library"
    )
    a = p.parse_args()
    db = sqlite3.connect(a.db.resolve().as_uri() + "?mode=ro", uri=True)
    a.output.mkdir(parents=True, exist_ok=False)
    if a.char:
        lookup(db, a.char, a.output)
    else:
        page(db, a.book, a.page, a.output, a.books)
    db.close()

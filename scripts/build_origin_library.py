"""Build the local Hanziyuan and book-page corpus from verified source assets."""

import argparse
import json
import sqlite3
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory

import fitz
import numpy as np
import torch

from ilm.image_edit.baseline import ROOT, sha, save_json

OUT = ROOT / "data/chinese_origin_library_v1"
PUBLIC = ROOT / "references/evidence/origin_library_v1"
SPLITS = ("train", "development", "sealed")
SCHEMA = """
PRAGMA foreign_keys=ON;
CREATE TABLE query(headword TEXT PRIMARY KEY, cache_name TEXT, cache_sha TEXT, unavailable TEXT);
CREATE TABLE entry(headword TEXT REFERENCES query, entry_id TEXT, family TEXT, split TEXT, fields TEXT,
 PRIMARY KEY(headword,entry_id));
CREATE TABLE image(svg_sha TEXT PRIMARY KEY, svg BLOB NOT NULL, pixels32 BLOB NOT NULL CHECK(length(pixels32)=128));
CREATE TABLE form(headword TEXT,entry_id TEXT,stage TEXT,label TEXT,svg_sha TEXT REFERENCES image,has_ink INTEGER,
 PRIMARY KEY(headword,entry_id,stage,label), FOREIGN KEY(headword,entry_id) REFERENCES entry);
CREATE INDEX lookup_stage ON form(stage,headword);
CREATE TABLE book(id TEXT PRIMARY KEY,title TEXT,filename TEXT,sha TEXT,language TEXT,pages INTEGER);
CREATE TABLE page(book_id TEXT REFERENCES book, page_no INTEGER, PRIMARY KEY(book_id,page_no));
"""


def read(p):
    return json.loads(p.read_text())


def checked(path, expected):
    if sha(path) != expected:
        raise ValueError(f"Changed source: {path}")
    return path


def build(db, legacy, books):
    src = ROOT / "artifacts/historic_entry_metadata_v1"
    base = ROOT / "artifacts/historical_glyph_splits_v1"
    extra = ROOT / "artifacts/historical_glyph_addendum_v1"
    reports = [read(p / "report.json") for p in (base, extra)]
    entries = read(checked(src / "entries.json", reports[0]["source_entries_sha256"]))
    cache = ROOT / "artifacts/cache/v34_historic_glyph_rasters_32.pt"
    files = [
        checked(cache, reports[0]["raster_cache"]["cache_sha256"]),
        checked(extra / "later_rasters.pt", reports[1]["later_rasters_sha256"]),
    ]
    rasters = [
        np.packbits(
            torch.load(f, weights_only=True, map_location="cpu")["pixels"]
            .numpy()
            .reshape(-1, 1024),
            axis=-1,
        )
        for f in files
    ]
    families, pixels = {}, {}
    for i, folder in enumerate((base, extra)):
        for split in SPLITS:
            file = checked(
                folder / f"{split}.json", reports[i]["partitions"][split]["file_sha256"]
            )
            for r in read(file):
                k = (r["char"], r["entry_id"])
                meta = (r["family_sha256"], split)
                if families.setdefault(k, meta) != meta:
                    raise ValueError(k)
                for g in r["glyphs"]:
                    b = rasters[i][g["raster_index"]].tobytes()
                    if sha256(b).hexdigest() != g["raster_sha256"]:
                        raise ValueError(g["label"])
                    pixels[g["sha256"]] = b
    blank_labels = (
        {r["label"] for r in reports[1]["previously_blank_source_glyphs"]}
        if isinstance(reports[1]["previously_blank_source_glyphs"], list)
        else set(reports[1]["previously_blank_source_glyphs"])
    )
    missing = {r["char"]: r["reason"] for r in read(src / "unavailable.json")}
    db.executescript(SCHEMA)
    for q in read(src / "cached_sources.json"):
        checked(legacy / "data/historic/cache" / q["file"], q["sha256"])
        db.execute(
            "INSERT INTO query VALUES(?,?,?,?)",
            (q["char"], q["file"], q["sha256"], missing.get(q["char"])),
        )
    seen = set()
    for r in entries:
        k = (r["char"], r["entry_id"])
        db.execute(
            "INSERT INTO entry VALUES(?,?,?,?,?)",
            (*k, *families[k], json.dumps(r["fields"], ensure_ascii=False)),
        )
        for g in r["glyphs"] + r["source_glyphs_without_matching_asset"]:
            h = g["sha256"]
            if h not in seen:
                path = (
                    legacy / g["source_local_path"]
                    if g.get("source_local_path")
                    else extra / "svg" / (h + ".svg")
                )
                svg = path.read_bytes()
                if sha256(svg).hexdigest() != h:
                    raise ValueError(path)
                b = pixels.get(h)
                if b is None:
                    if g["label"] not in blank_labels:
                        raise ValueError(g)
                    b = b"\xff" * 128
                db.execute("INSERT INTO image VALUES(?,?,?)", (h, svg, b))
                seen.add(h)
            db.execute(
                "INSERT INTO form VALUES(?,?,?,?,?,?)",
                (*k, g["stage"], g["label"], h, int(h in pixels)),
            )
    sources = read(ROOT / "references/source_books/manifest.json")["items"]
    for item in sources:
        file = checked(books / item["filename"], item["sha256"])
        with fitz.open(file) as doc:
            if doc.is_reflowable:
                doc.layout(width=612, height=792, fontsize=12)
            n = doc.page_count
        db.execute(
            "INSERT INTO book VALUES(?,?,?,?,?,?)",
            (item["id"], item["title"], file.name, item["sha256"], item["language"], n),
        )
        db.executemany(
            "INSERT INTO page VALUES(?,?)", ((item["id"], i + 1) for i in range(n))
        )
    db.commit()
    if (
        db.execute("PRAGMA integrity_check").fetchone() != ("ok",)
        or db.execute("PRAGMA foreign_key_check").fetchone()
    ):
        raise ValueError("Corpus integrity failed")
    split_overlap = db.execute(
        "SELECT count(*) FROM (SELECT svg_sha FROM form JOIN entry USING(headword,entry_id) GROUP BY svg_sha HAVING count(distinct split)>1)"
    ).fetchone()[0]
    assert not split_overlap
    return dict(
        counts={
            t: db.execute(f"SELECT count(*) FROM {t}").fetchone()[0]
            for t in ("query", "entry", "image", "form", "book", "page")
        },
        by_stage=dict(db.execute("SELECT stage,count(*) FROM form GROUP BY stage")),
        by_book=dict(db.execute("SELECT id,pages FROM book ORDER BY id")),
        nonblank_forms=db.execute("SELECT sum(has_ink) FROM form").fetchone()[0],
        queried_without_entries=len(missing),
        source_split_svg_overlap=split_overlap,
        book_manifest=sha(ROOT / "references/source_books/manifest.json"),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy", type=Path, default=ROOT.parent / "incoder")
    parser.add_argument(
        "--books", type=Path, default=ROOT / "references/source_books/library"
    )
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    dest = args.out / "origin.sqlite3"
    with TemporaryDirectory(dir=args.out) as tmp:
        p = Path(tmp) / "origin.sqlite3"
        with sqlite3.connect(p) as db:
            report = build(db, args.legacy, args.books)
        db.close()
        report["database_sha256"] = sha(p)
        if dest.exists() and sha(dest) != report["database_sha256"]:
            raise ValueError(
                "Existing collection differs; select another output directory"
            )
        if not dest.exists():
            p.replace(dest)
    save_json(args.out / "manifest.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

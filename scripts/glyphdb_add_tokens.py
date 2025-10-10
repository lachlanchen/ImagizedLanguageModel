#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

try:
    from ilm.db.glyph_db import GlyphDB
except ModuleNotFoundError:
    import sys
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))
    from ilm.db.glyph_db import GlyphDB


def read_tsv(path: str):
    with open(path, "r", encoding="utf-8") as f:
        r = csv.DictReader(f, delimiter="\t")
        for row in r:
            yield row


def main():
    ap = argparse.ArgumentParser(description="Add tokens to glyph DB from TSV (columns: lang, token)")
    ap.add_argument("--db", default="data/glyphdb/glyphs.sqlite3")
    ap.add_argument("--tsv", required=True)
    ap.add_argument("--size", type=int, default=128)
    args = ap.parse_args()

    db = GlyphDB(args.db)
    n = 0
    for row in read_tsv(args.tsv):
        lang = (row.get("lang") or "en").strip()
        token = (row.get("token") or "").strip()
        if not token:
            continue
        path = db.ensure_glyph(lang, token, size=args.size)
        n += 1
    db.close()
    print(json.dumps({"db": args.db, "added": n, "size": args.size}))


if __name__ == "__main__":
    main()


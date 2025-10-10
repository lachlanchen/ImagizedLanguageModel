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


def main():
    ap = argparse.ArgumentParser(description="Ingest an index.tsv (lang token path) into glyph DB; re-render to DB-managed path")
    ap.add_argument("--db", default="data/glyphdb/glyphs.sqlite3")
    ap.add_argument("--index", required=True)
    ap.add_argument("--size", type=int, default=128)
    args = ap.parse_args()

    db = GlyphDB(args.db)
    added = 0
    with open(args.index, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            lang = row.get("lang") or "en"
            token = row.get("token") or ""
            if not token:
                continue
            db.ensure_glyph(lang, token, size=args.size)
            added += 1
    db.close()
    print(json.dumps({"db": args.db, "ingested": added}))


if __name__ == "__main__":
    main()


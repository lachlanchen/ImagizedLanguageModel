#!/usr/bin/env python3
from __future__ import annotations

import argparse
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
    ap = argparse.ArgumentParser(description="Initialize SQLite glyph database")
    ap.add_argument("--db", default="data/glyphdb/glyphs.sqlite3")
    ap.add_argument("--images-root", default=None, help="Directory to store glyph images (default next to DB)")
    args = ap.parse_args()

    Path(args.db).parent.mkdir(parents=True, exist_ok=True)
    db = GlyphDB(args.db, images_root=args.images_root)
    db.close()
    print(json.dumps({"db": args.db, "images_root": db.images_root}))


if __name__ == "__main__":
    main()


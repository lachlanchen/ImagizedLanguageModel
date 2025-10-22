#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sqlite3
from pathlib import Path


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Purge glyphs from SQLite DB (and optionally delete image files)")
    ap.add_argument("--db", required=True, help="Path to glyphs.sqlite3")
    ap.add_argument("--lang", default=None, help="Filter by language code (e.g., en, zh)")
    ap.add_argument("--size", type=int, default=None, help="Filter by glyph size (e.g., 128)")
    ap.add_argument("--delete-files", action="store_true", help="Also delete image files on disk")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    db_path = Path(args.db)
    if not db_path.exists():
        raise SystemExit(f"DB not found: {db_path}")
    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()

    where = []
    params = []
    if args.lang:
        where.append("lang = ?")
        params.append(args.lang)
    if args.size is not None:
        where.append("size = ?")
        params.append(args.size)
    where_clause = (" WHERE " + " AND ".join(where)) if where else ""

    # Fetch records to optionally delete files
    cur.execute(f"SELECT path FROM glyphs{where_clause}", params)
    rows = cur.fetchall()
    file_count = 0
    if args.delete_files:
        for (p,) in rows:
            try:
                if p and os.path.exists(p):
                    os.remove(p)
                    file_count += 1
            except Exception:
                pass

    # Delete DB rows
    cur.execute(f"DELETE FROM glyphs{where_clause}", params)
    conn.commit()
    print(f"Purged {cur.rowcount} DB rows. Deleted {file_count} files.")


if __name__ == "__main__":
    main()


#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Iterable, Optional, Tuple

import requests

from ilm.etymology import db as dbm
from ilm.etymology.hanziyuan import (
    build_char_info,
    guess_source_site,
    parse_page,
    fetch_url,
    save_glyph_assets,
)


def parse_inputs(from_file: Optional[Path], url: Optional[str], char: Optional[str]) -> Iterable[Tuple[Optional[str], str]]:
    if from_file:
        for line in from_file.read_text("utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            # Support: "char<TAB>url" or "url" or "char url"
            if "\t" in line:
                c, u = line.split("\t", 1)
                yield (c.strip() or None, u.strip())
            else:
                parts = line.split()
                if len(parts) == 1:
                    yield (None, parts[0])
                else:
                    yield (parts[0], parts[1])
    elif url:
        yield (char, url)
    else:
        raise SystemExit("Provide --from-file or --url (optionally with --char)")


def ingest_one(
    *,
    url: str,
    char_override: Optional[str],
    db_path: Path,
    out_root: Path,
    cache_dir: Optional[Path],
    delay: float,
    session: Optional[requests.Session] = None,
) -> None:
    print(f"[ingest] url={url}")
    html = fetch_url(url, cache_dir=cache_dir, delay=delay)
    meta, glyphs = parse_page(html, base_url=url)
    meta = build_char_info(char_override, meta)
    if not meta:
        raise RuntimeError("Failed to detect character; pass --char to override")

    print(f"  char={meta.char} codepoint={meta.codepoint} glyphs={len(glyphs)}")

    # Save glyphs to disk
    saved = save_glyph_assets(
        glyphs=glyphs,
        out_root=out_root,
        char=meta.char,
        base_url=url,
        session=session,
        delay=delay,
    )

    # Insert into DB
    conn = dbm.connect(db_path)
    try:
        dbm.ensure_schema(conn)
        char_id = dbm.upsert_char(
            conn,
            meta.char,
            codepoint=meta.codepoint,
            pinyin=meta.pinyin,
            main_meaning=meta.main_meaning,
            importance_freq=meta.importance_freq,
            sources=guess_source_site(url),
        )
        for g, local_path, w, h in saved:
            dbm.add_glyph(
                conn,
                char_id=char_id,
                stage=g.stage or "unknown",
                label=g.label,
                source_site=guess_source_site(url),
                url=url if not (g.src or "").startswith("data:") else None,
                local_path=str(local_path),
                width=w,
                height=h,
            )
    finally:
        conn.close()


def main() -> None:
    ap = argparse.ArgumentParser(description="Ingest etymology pages and glyphs into SQLite and filesystem")
    ap.add_argument("--db", default="data/historic/etymology.sqlite3", help="SQLite DB path")
    ap.add_argument("--out", default="data/historic/glyphs", help="Root output dir for saved glyph images")
    ap.add_argument("--from-file", type=Path, default=None, help="Input file: lines of 'char\turl' or 'url' or 'char url'")
    ap.add_argument("--url", default=None, help="Single URL to ingest")
    ap.add_argument("--char", default=None, help="Character override if not detectable from page")
    ap.add_argument("--cache-dir", type=Path, default=Path("data/historic/cache"), help="Directory for cached HTML")
    ap.add_argument("--delay", type=float, default=0.5, help="Delay between network fetches (seconds)")
    args = ap.parse_args()

    db_path = Path(args.db)
    out_root = Path(args.out)
    cache_dir = Path(args.cache_dir) if args.cache_dir else None
    session = requests.Session()

    errors = 0
    for ch, url in parse_inputs(args.from_file, args.url, args.char):
        try:
            ingest_one(
                url=url,
                char_override=ch,
                db_path=db_path,
                out_root=out_root,
                cache_dir=cache_dir,
                delay=args.delay,
                session=session,
            )
        except Exception as e:
            errors += 1
            print(f"ERROR ingesting {url}: {e}")
            continue

    if errors:
        print(f"Completed with {errors} errors")
        sys.exit(1)
    print("Done.")


if __name__ == "__main__":
    main()


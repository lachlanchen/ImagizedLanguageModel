#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import requests

from ilm.etymology import db as dbm
from ilm.etymology.hanziyuan import (
    build_char_info,
    fetch_hanziyuan_ajax,
    guess_source_site,
    parse_page,
    save_glyph_assets,
)


DEFAULT_DB = Path("data/historic/etymology.sqlite3")
DEFAULT_OUT = Path("data/historic/glyphs")
DEFAULT_CACHE = Path("data/historic/cache")


def ensure_wordfreq() -> None:
    try:
        import wordfreq  # noqa: F401
        return
    except Exception:
        pass
    # Attempt a polite install
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "wordfreq"])


def top_zh_chars(n_chars: int, sample_words: int = 50000) -> List[str]:
    # Derive common characters from wordfreq's top_n_list("zh")
    ensure_wordfreq()
    from collections import Counter
    from wordfreq import top_n_list

    words = top_n_list("zh", sample_words)
    cnt = Counter()
    for w in words:
        for ch in w:
            oc = ord(ch)
            if (0x3400 <= oc <= 0x4DBF) or (0x4E00 <= oc <= 0x9FFF) or (0xF900 <= oc <= 0xFAFF):
                cnt[ch] += 1
    return [ch for ch, _ in cnt.most_common(n_chars)]


def read_chars_file(p: Path, limit: Optional[int]) -> List[str]:
    text = p.read_text("utf-8")
    chars: List[str] = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if len(line) == 1:
            chars.append(line)
        else:
            # Support space/comma-separated lines
            for ch in line.replace(",", " ").split():
                if ch:
                    chars.append(ch)
    # Deduplicate preserving order
    seen = set()
    uniq = []
    for ch in chars:
        if ch not in seen:
            seen.add(ch)
            uniq.append(ch)
    if limit is not None:
        uniq = uniq[:limit]
    return uniq


def already_done(out_root: Path, ch: str) -> bool:
    # Consider done if we have at least one saved glyph file in any stage dir
    d = out_root / ch
    if not d.exists():
        return False
    try:
        for stage_dir in d.iterdir():
            if stage_dir.is_dir():
                for _ in stage_dir.iterdir():
                    return True
    except Exception:
        return False
    return False


@dataclass
class IngestStats:
    attempted: int = 0
    succeeded: int = 0
    skipped: int = 0
    failed: int = 0


def ingest_char(
    ch: str,
    *,
    db_path: Path,
    out_root: Path,
    cache_dir: Optional[Path],
    delay: float,
    session: Optional[requests.Session] = None,
) -> Tuple[bool, Optional[str]]:
    try:
        html, base_url = fetch_hanziyuan_ajax(char=ch, cache_dir=cache_dir, delay=delay, session=session)
        meta, glyphs = parse_page(html, base_url=base_url, filter_related=True)
        meta = build_char_info(ch, meta)
        if not meta:
            return False, "char detection failed"
        saved = save_glyph_assets(glyphs=glyphs, out_root=out_root, char=meta.char, base_url=base_url, session=session, delay=delay)
        conn = dbm.connect(db_path)
        try:
            dbm.ensure_schema(conn)
            char_id = dbm.upsert_char(
                conn,
                meta.char,
                codepoint=meta.codepoint,
                pinyin=meta.pinyin,
                main_meaning=meta.main_meaning,
                importance_freq=None,
                sources=guess_source_site(base_url),
            )
            for g, local_path, w, h in saved:
                dbm.add_glyph(
                    conn,
                    char_id=char_id,
                    stage=g.stage or "unknown",
                    label=g.label,
                    source_site=guess_source_site(base_url),
                    url=base_url if not (g.src or "").startswith("data:") else None,
                    local_path=str(local_path),
                    width=w,
                    height=h,
                )
        finally:
            conn.close()
        return True, None
    except Exception as e:
        return False, str(e)


def main() -> None:
    ap = argparse.ArgumentParser(description="Bulk-ingest hanziyuan glyphs for common Chinese characters (polite)")
    ap.add_argument("--db", default=str(DEFAULT_DB), help="SQLite DB path")
    ap.add_argument("--out", default=str(DEFAULT_OUT), help="Output root for glyph images")
    ap.add_argument("--cache-dir", default=str(DEFAULT_CACHE), help="HTML cache dir")
    ap.add_argument("--limit", type=int, default=1000, help="Number of characters to ingest")
    ap.add_argument("--delay", type=float, default=0.5, help="Base delay between requests (seconds)")
    ap.add_argument("--chars-file", type=Path, default=None, help="Optional file of characters (one per line or space-separated)")
    ap.add_argument("--resume", action="store_true", help="Skip characters that already have saved glyph files")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    db_path = Path(args.db)
    out_root = Path(args.out)
    cache_dir = Path(args.cache_dir) if args.cache_dir else None
    session = requests.Session()

    if args.chars_file and Path(args.chars_file).exists():
        chars = read_chars_file(Path(args.chars_file), args.limit)
    else:
        print("[bulk] deriving top Chinese characters via wordfreq…")
        chars = top_zh_chars(args.limit, sample_words=max(20000, args.limit * 50))

    stats = IngestStats()
    start = time.time()
    for i, ch in enumerate(chars, 1):
        if args.resume and already_done(out_root, ch):
            stats.skipped += 1
            if i % 25 == 0:
                print(f"[bulk] {i}/{len(chars)} skipped={stats.skipped} ok={stats.succeeded} failed={stats.failed}")
            continue
        stats.attempted += 1
        ok, err = ingest_char(ch, db_path=db_path, out_root=out_root, cache_dir=cache_dir, delay=args.delay, session=session)
        if ok:
            stats.succeeded += 1
        else:
            stats.failed += 1
            logging.warning("char %s failed: %s", ch, err)
        # Compact progress
        if i % 10 == 0:
            elapsed = time.time() - start
            print(f"[bulk] {i}/{len(chars)} ok={stats.succeeded} failed={stats.failed} skipped={stats.skipped} elapsed={elapsed:.1f}s")

    elapsed = time.time() - start
    print(
        f"[bulk] done: attempted={stats.attempted} ok={stats.succeeded} failed={stats.failed} "
        f"skipped={stats.skipped} elapsed={elapsed/60:.1f}m"
    )


if __name__ == "__main__":
    main()


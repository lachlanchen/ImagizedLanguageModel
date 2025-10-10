from __future__ import annotations

import hashlib
import os
import sqlite3
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from ilm.utils.glyphs import make_rgb_token_image, save_rgb_image


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def _sanitize_token(token: str) -> str:
    # Keep alnum and basic separators; replace others with underscore
    safe = []
    for ch in token:
        if ch.isalnum() or ch in ("-", "_", "."):
            safe.append(ch)
        else:
            safe.append("_")
    out = "".join(safe)
    if not out:
        out = "_"
    # limit length to 64 for filesystem
    return out[:64]


def _sha1(data: bytes) -> str:
    return hashlib.sha1(data).hexdigest()


@dataclass
class GlyphRecord:
    lang: str
    token: str
    size: int
    path: str
    checksum: Optional[str]


class GlyphDB:
    """
    SQLite-backed glyph manager.
    Stores metadata (lang, token, size, path, checksum) and ensures glyphs exist on disk.
    """

    def __init__(self, db_path: str, images_root: Optional[str] = None):
        self.db_path = str(db_path)
        self.images_root = str(images_root) if images_root else str(Path(db_path).with_suffix("").parent / "images")
        ensure_dir(Path(self.images_root))
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL;")
        self._conn.execute("PRAGMA synchronous=NORMAL;")
        self._init_schema()

    def _init_schema(self):
        cur = self._conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS glyphs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                lang TEXT NOT NULL,
                token TEXT NOT NULL,
                size INTEGER NOT NULL,
                path TEXT NOT NULL,
                checksum TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(lang, token, size)
            );
            """
        )
        cur.execute("CREATE INDEX IF NOT EXISTS idx_glyphs_lang_token ON glyphs(lang, token);")
        self._conn.commit()

    def close(self):
        try:
            self._conn.close()
        except Exception:
            pass

    def _default_path(self, lang: str, token: str, size: int) -> str:
        safe = _sanitize_token(token)
        sub = Path(self.images_root) / lang / str(size)
        ensure_dir(sub)
        # disambiguate with hash suffix for collisions
        name = f"{safe}.png"
        full = sub / name
        if full.exists():
            # avoid overwriting unrelated different token with same safe name
            # by appending a short hash of original token
            h = hashlib.sha1(token.encode("utf-8")).hexdigest()[:8]
            name = f"{safe}-{h}.png"
            full = sub / name
        return str(full)

    def get(self, lang: str, token: str, size: int) -> Optional[GlyphRecord]:
        cur = self._conn.cursor()
        cur.execute(
            "SELECT lang, token, size, path, checksum FROM glyphs WHERE lang=? AND token=? AND size=?",
            (lang, token, size),
        )
        row = cur.fetchone()
        if not row:
            return None
        return GlyphRecord(*row)

    def upsert(self, rec: GlyphRecord):
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO glyphs (lang, token, size, path, checksum, updated_at)
                VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(lang, token, size)
                DO UPDATE SET path=excluded.path, checksum=excluded.checksum, updated_at=CURRENT_TIMESTAMP
                """,
                (rec.lang, rec.token, rec.size, rec.path, rec.checksum),
            )
            self._conn.commit()

    def ensure_glyph(self, lang: str, token: str, size: int) -> str:
        """Ensure a glyph image exists on disk and in DB. Returns absolute path."""
        rec = self.get(lang, token, size)
        if rec and os.path.exists(rec.path):
            return rec.path
        # render
        rgb = make_rgb_token_image(lang, token, size=size)
        # write
        path = self._default_path(lang, token, size)
        save_rgb_image(path, rgb)
        # checksum
        try:
            with open(path, "rb") as f:
                checksum = _sha1(f.read())
        except Exception:
            checksum = None
        # store
        self.upsert(GlyphRecord(lang=lang, token=token, size=size, path=path, checksum=checksum))
        return path


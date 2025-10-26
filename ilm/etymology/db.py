from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Iterable, Optional


SCHEMA = r"""
PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;

CREATE TABLE IF NOT EXISTS chars (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  char TEXT NOT NULL,
  codepoint TEXT,
  pinyin TEXT,
  main_meaning TEXT,
  importance_freq INTEGER,
  sources TEXT,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  UNIQUE(char)
);

CREATE TABLE IF NOT EXISTS glyphs (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  char_id INTEGER NOT NULL,
  stage TEXT NOT NULL,         -- e.g., oracle, bronze, seal, liushutong
  label TEXT,                  -- site label (e.g., J29285)
  source_site TEXT,            -- hanziyuan, chineseetymology, etc.
  url TEXT,                    -- original URL (if any)
  local_path TEXT NOT NULL,    -- saved image path (svg/png)
  width INTEGER,
  height INTEGER,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  UNIQUE(char_id, stage, label, local_path),
  FOREIGN KEY(char_id) REFERENCES chars(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS variants (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  char_id INTEGER NOT NULL,
  type TEXT,       -- traditional/simplified/older/variant
  label TEXT,
  value TEXT,
  FOREIGN KEY(char_id) REFERENCES chars(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS kv (
  key TEXT PRIMARY KEY,
  value TEXT
);
"""


def connect(db_path: Path) -> sqlite3.Connection:
  db_path.parent.mkdir(parents=True, exist_ok=True)
  conn = sqlite3.connect(str(db_path))
  conn.execute("PRAGMA foreign_keys=ON;")
  return conn


def ensure_schema(conn: sqlite3.Connection) -> None:
  conn.executescript(SCHEMA)
  conn.commit()


def upsert_char(
  conn: sqlite3.Connection,
  char: str,
  *,
  codepoint: Optional[str] = None,
  pinyin: Optional[str] = None,
  main_meaning: Optional[str] = None,
  importance_freq: Optional[int] = None,
  sources: Optional[str] = None,
) -> int:
  cur = conn.cursor()
  cur.execute(
    """
    INSERT INTO chars(char, codepoint, pinyin, main_meaning, importance_freq, sources)
    VALUES(?,?,?,?,?,?)
    ON CONFLICT(char) DO UPDATE SET
      codepoint=COALESCE(excluded.codepoint, chars.codepoint),
      pinyin=COALESCE(excluded.pinyin, chars.pinyin),
      main_meaning=COALESCE(excluded.main_meaning, chars.main_meaning),
      importance_freq=COALESCE(excluded.importance_freq, chars.importance_freq),
      sources=COALESCE(excluded.sources, chars.sources)
    RETURNING id
    """,
    (char, codepoint, pinyin, main_meaning, importance_freq, sources),
  )
  (char_id,) = cur.fetchone()
  conn.commit()
  return char_id


def add_glyph(
  conn: sqlite3.Connection,
  *,
  char_id: int,
  stage: str,
  label: Optional[str],
  source_site: str,
  url: Optional[str],
  local_path: str,
  width: Optional[int],
  height: Optional[int],
) -> None:
  conn.execute(
    """
    INSERT OR IGNORE INTO glyphs(char_id, stage, label, source_site, url, local_path, width, height)
    VALUES(?,?,?,?,?,?,?,?)
    """,
    (char_id, stage, label, source_site, url, local_path, width, height),
  )
  conn.commit()


def add_variants(conn: sqlite3.Connection, char_id: int, rows: Iterable[tuple[str, str]]) -> None:
  conn.executemany(
    "INSERT INTO variants(char_id, type, value) VALUES(?,?,?)",
    ((char_id, t, v) for t, v in rows),
  )
  conn.commit()


Etymology Dataset Toolkit
=========================

This package builds a smooth, multi‑stage Chinese character etymology dataset by:
- Scraping hanziyuan/chineseetymology‑style pages
- Saving per‑stage glyph images (e.g., oracle/bronze/seal/clerical/regular)
- Recording metadata into a SQLite database

Components
----------
- `ilm/etymology/db.py`: SQLite schema and helpers
  - Tables: `chars`, `glyphs`, `variants`, `kv`
- `ilm/etymology/hanziyuan.py`: Scraper and asset saver
- `scripts/ingest_etymology.py`: CLI for ingestion

Data layout
-----------
- Images are written under `data/historic/glyphs/<char>/<stage>/<label>.<ext>`
- SQLite DB defaults to `data/historic/etymology.sqlite3`

Usage
-----
1) Install deps:
   - `pip install requests beautifulsoup4`

2) Ingest a single page:
   - `PYTHONPATH=. python scripts/ingest_etymology.py --url https://example/hanziyuan/车`
   - If the character can’t be detected, pass `--char 车`.

3) Batch ingest from a file (one per line):
   - Lines support `char\turl`, `url`, or `char url`.
   - `PYTHONPATH=. python scripts/ingest_etymology.py --from-file urls.txt`

Notes
-----
- The scraper handles `<img>` sources and CSS `background-image: url(data:...)` SVG/PNG.
- Stage detection is heuristic via nearby headings and known stage terms: 甲骨文/Oracle, 金文/Bronze, 小篆/Seal, 隶书/Clerical, 楷书/Regular, 行书/Running, 草书/Cursive, 六书通/Liushutong.
- Be polite: respect `robots.txt`, site terms, and rate‑limit (`--delay`).
- Large artifacts under `data/` are `.gitignore`’d to keep the repo lean.


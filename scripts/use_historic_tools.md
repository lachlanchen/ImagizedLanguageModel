Using Historic Script Tools
===========================

Cloned (untracked) repos live under `data/historic/repos`:

1) YinQiWenYuan
   - Path: `data/historic/repos/YinQiWenYuan`
   - Entry points: `SearchByText_v*.py`
   - Requirements: `requests`, `pyautogui`, plus site access credentials/session for 殷契文渊.
   - Typical usage:
     - Install deps: `pip install requests pyautogui zhconv`
     - Edit the script `savepath` and any login/session tokens as per README.
     - Run: `python SearchByText_v3.1.py` and follow prompts.

2) Chinese-Etymology-Crawler
   - Path: `data/historic/repos/Chinese-Etymology-Crawler`
   - Entry points: `utils_fetch.py`
   - Requirements: `requests` (and possibly `BeautifulSoup` depending on your extensions).
   - Typical usage:
     - Install deps: `pip install requests beautifulsoup4`
     - Review `utils_fetch.py` and update target lists; run it to fetch images from chineseetymology.org.
     - Respect robots.txt and site licensing.

Large Official Datasets
-----------------------
- Use `scripts/download_historic_datasets.py` to manage downloads to `data/historic/...`.
- Before running, edit the `DEFAULT_MANIFEST` in the script to include the official archive URLs (e.g., HUST‑OBC, Tangut TCD). Then:
  - `PYTHONPATH=. python scripts/download_historic_datasets.py --only hust_obc tangut_tcd`

Caveats
-------
- These third‑party tools depend on external sites that may require login, rate‑limit crawlers, or change HTML structure.
- Downloads can be large (GBs). Keep them in `data/historic/` (ignored by git) to avoid bloating the repo.

Ingest Etymology Pages
----------------------
- Use the bundled CLI to scrape hanziyuan/chineseetymology‑style pages and persist glyphs + metadata.
- Install deps: `pip install requests beautifulsoup4`
- Single page:
  - `PYTHONPATH=. python scripts/ingest_etymology.py --url https://example/hanziyuan/车`
  - If auto‑detection fails, add `--char 车`
- Batch mode from file `urls.txt` (supports lines as `char\turl`, `url`, or `char url`):
  - `PYTHONPATH=. python scripts/ingest_etymology.py --from-file urls.txt`
- Outputs:
  - Images: `data/historic/glyphs/<char>/<stage>/<label>.<ext>`
  - DB: `data/historic/etymology.sqlite3`

Tornado Web UI (optional)
-------------------------
- Quick web front‑end for ad‑hoc ingestion and a demo pinyin mapping (e.g., `zhong` → `中`).
- Install deps: `pip install tornado requests beautifulsoup4`
- Run the server:
  - `PYTHONPATH=. python scripts/serve_etymology.py`
  - Open `http://127.0.0.1:8888` in a browser.
- Usage:
  - Enter a character (e.g., `中`) or demo pinyin (`zhong`), choose a site helper, submit.
  - The app will fetch the page, parse glyphs, save images, and update the DB.

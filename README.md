Imagized Language Model (ILM)
=============================

Language
- English | [简体中文](i18n/README.zh-Hans.md) | [繁體中文](i18n/README.zh-Hant.md) | [日本語](i18n/README.ja.md) | [한국어](i18n/README.ko.md) | [Tiếng Việt](i18n/README.vi.md) | [العربية](i18n/README.ar.md) | [Français](i18n/README.fr.md) | [Español](i18n/README.es.md)

ILM is a research codebase exploring text-as-image generation: it encodes language into compact, image‑like tensors and generates text with diffusion‑style iterative refinement. The representation factors sentences into meta‑elements (grammar, semantics, tone, emotion) and hierarchical, memory‑like codes for words and characters. This unifies ideas from discrete diffusion, superposition/disentanglement, structured embeddings, and glyph‑aware character modeling.

Key links
- Conceptual write‑up: `docs/imagized-language-model.md`
- Code plan and metrics: `docs/ilm-visual-diffusion-code-plan.md`
- Embedding “color” plan: `docs/embedding-color-plan.md`

What’s in this repo
- `ilm/etymology/`: Utilities to ingest historic glyph forms (oracle, bronze, seal, etc.) from hanziyuan/chineseetymology‑style sources.
  - Robust AJAX fetcher for hanziyuan with retries, polite throttling, and caching.
  - HTML/CSS parser that extracts stage‑labeled glyphs (data URIs and image URLs).
- `scripts/`:
  - `ingest_etymology.py`: CLI to ingest pages or single characters into a SQLite DB and filesystem layout.
  - `serve_etymology.py`: Minimal Tornado UI for ad‑hoc ingestion and preview gallery.
  - `use_historic_tools.md`: Notes on external datasets/tools and usage.
- `data/` (git‑ignored): Cached HTML, saved glyph assets, SQLite database.

Quick start
1) Environment
- Python 3.10+
- Install deps (minimal):
  - `pip install requests beautifulsoup4 tornado`

2) Ingest historic glyphs (CLI)
- Hanziyuan (recommended): char‑only AJAX flow
  - `PYTHONPATH=. python scripts/ingest_etymology.py --site hanziyuan --char 中`
- ChineseEtymology (direct URL):
  - `PYTHONPATH=. python scripts/ingest_etymology.py --site chineseetymology --url "https://www.chineseetymology.org/CharacterEtymology.aspx?characterInput=%E4%B8%AD"`
- Batch file (lines can be `char\turl`, `url`, or `char url`):
  - `PYTHONPATH=. python scripts/ingest_etymology.py --from-file urls.txt`

Outputs
- Files: `data/historic/glyphs/<char>/<stage>/<label>.<ext>`
- Cache: `data/historic/cache/*.html`
- DB: `data/historic/etymology.sqlite3`

Web demo (optional)
- `PYTHONPATH=. python scripts/serve_etymology.py`
- Open `http://127.0.0.1:8888`, choose site, enter a character (e.g., `中`).

Polite crawling and site respect
- The fetcher uses per‑host throttling, retries with backoff, and caching.
- Keep delays ≥ 0.5s, avoid bursts, and honor site terms/robots/licensing.
- Do not bypass paywalls or interactive protections; if you see 403/429, slow down and try later.

Project intent and roadmap
- ILM targets practical training/inference on normal computers via compact latent reps, structured embeddings, and controllable meta‑channels.
- See `docs/imagized-language-model.md` for math, references, and staged training plans.

Contributing
- Follow `AGENTS.md` for conventions (atomic commits, push after change, no credentials in code).
- Group related edits in focused commits with conventional messages.

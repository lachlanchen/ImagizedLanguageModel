[English](../README.md) · [العربية](README.ar.md) · [Español](README.es.md) · [Français](README.fr.md) · [日本語](README.ja.md) · [한국어](README.ko.md) · [Tiếng Việt](README.vi.md) · [中文 (简体)](README.zh-Hans.md) · [中文（繁體）](README.zh-Hant.md) · [Deutsch](README.de.md) · [Русский](README.ru.md)


# Imagized Language Model (ILM)

![Python](https://img.shields.io/badge/python-3.10%2B-3776AB?logo=python&logoColor=white)
![Status](https://img.shields.io/badge/status-research-orange)
![Focus](https://img.shields.io/badge/focus-text--as--image-0A7EA4)
![Diffusion](https://img.shields.io/badge/paradigm-diffusion%20%2B%20glyphs-6A5ACD)
![License](https://img.shields.io/badge/license-unspecified-lightgrey)

ILM ist eine Forschungs-Codebasis, die Text-als-Bild-Generierung untersucht: Sprache wird in kompakte, bildähnliche Tensoren kodiert, und Text wird mit diffusion-ähnlicher iterativer Verfeinerung erzeugt. Die Repräsentation zerlegt Sätze in Meta-Elemente (Grammatik, Semantik, Ton, Emotion) sowie in hierarchische, speicherähnliche Codes für Wörter und Zeichen. Damit werden Ideen aus diskreter Diffusion, Superposition/Entflechtung, strukturierten Embeddings und glyphenbewusstem Zeichenmodellieren zusammengeführt.

## Überblick

Dieses Repository enthält derzeit zwei wichtige praktische Schwerpunkte:

1. Ingestion historischer chinesischer Glyphen-Etymologie (Scraping/Parsing/Speicherung/Vorschau).
2. ILM-Glyph/Bild-Modellierungs-Experimente (Token-Glyph-Rendering, Produkt-Codebooks, Frame-Packing, Diffusion/Inpainting, Evaluation/Reporting).

Das aktuelle README in diesem Repo war historisch auf das Etymologie-Toolkit ausgerichtet. Dieser Workflow bleibt unten vollständig dokumentiert und wird als kanonisch beibehalten.

## Wichtige Links

| Bereich | Pfad |
|---|---|
| Konzeptionelle Ausarbeitung | `docs/imagized-language-model.md` |
| Code-Plan und Metriken | `docs/ilm-visual-diffusion-code-plan.md` |
| Embedding-"color"-Plan | `docs/embedding-color-plan.md` |
| Entwicklungsnotizen/-plan | `docs/development-plan.md` |
| Etymologie-Modul-README | `ilm/etymology/README.md` |

## Funktionen

- 🏺 Etymologie-Ingestion aus `hanziyuan`- und `chineseetymology`-artigen Quellen.
- 🌐 Robuster AJAX- + HTML-Ingestion-Pfad mit Retries, Throttling und Cache.
- 🧩 Stage-gelabelte Glyphen-Extraktion inklusive `<img>`- und CSS-`background-image`-Data-URIs.
- 🗃️ SQLite-basierte Speicherung für Zeichen-/Glyphen-Metadaten plus Dateisystem-Asset-Layout.
- 🖥️ Tornado-Web-UI für ad-hoc Ingestion + Galerie-Vorschau.
- 🔤 Glyph-Rendering-Utilities für mehrsprachige Token-Bilder.
- 🧠 Produkt-Code-Embedding-/Codebook-Module.
- 🧱 Satz-Frame-Packing- und Diffusion/Inpainting-Training-/Evaluierungsskripte.
- 📊 Reporting- und Visualisierungsskripte für Embedding- und Pipeline-Inspektion.
- 📄 Publikationsartefakte in LaTeX/PDF unter `publication/`.

## Projektstruktur

```text
.
├── README.md
├── AGENTS.md
├── configs/
│   ├── color.yaml
│   └── diffusion.yaml
├── docs/
├── i18n/
├── ilm/
│   ├── code/
│   ├── data/
│   ├── datasets/
│   ├── db/
│   ├── diffusion/
│   ├── encoders/
│   ├── english_tiles/
│   ├── etymology/
│   ├── frames/
│   ├── models/
│   └── utils/
├── scripts/
├── publication/
├── assets/
├── logs/
└── *.ipynb
```

## Voraussetzungen

| Anforderung | Hinweise |
|---|---|
| Python `3.10+` | Kernlaufzeit |
| `pip` | Paketinstallation |
| Optionale GPU | Hilfreich für PyTorch-CUDA-Trainingsskripte |
| Optionale LaTeX-Toolchain | Erforderlich für Publikations-Builds |

Annahme-Hinweis: Derzeit gibt es keine einzelne root-Abhängigkeitsdatei (`pyproject.toml`, `requirements.txt` usw.), daher werden Abhängigkeiten aus Imports und Skriptnutzung abgeleitet.

## Installation

### Minimal (Etymologie-Toolkit)

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install requests beautifulsoup4 tornado
```

### Erweitert (Modellierungs-/Trainings-Workflows)

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install requests beautifulsoup4 tornado pyyaml numpy pillow matplotlib torch
```

Wenn ein bestimmtes Skript zusätzliche Pakete benötigt, installiere sie anhand der vom Skript gemeldeten Import-Fehler.

## Nutzung

### Schnellstart: Ingestion historischer Glyphen (CLI)

1. Hanziyuan (empfohlen): char-only AJAX flow

```bash
PYTHONPATH=. python scripts/ingest_etymology.py --site hanziyuan --char 中
```

2. ChineseEtymology (direkte URL)

```bash
PYTHONPATH=. python scripts/ingest_etymology.py --site chineseetymology --url "https://www.chineseetymology.org/CharacterEtymology.aspx?characterInput=%E4%B8%AD"
```

3. Ingestion aus Batch-Datei (Zeilen können `char\turl`, `url` oder `char url` sein)

```bash
PYTHONPATH=. python scripts/ingest_etymology.py --from-file urls.txt
```

### Ausgaben

| Ausgabetyp | Speicherort |
|---|---|
| Dateien | `data/historic/glyphs/<char>/<stage>/<label>.<ext>` |
| Cache | `data/historic/cache/*.html` |
| DB | `data/historic/etymology.sqlite3` |

### Web-Demo (optional)

```bash
PYTHONPATH=. python scripts/serve_etymology.py
```

Öffne `http://127.0.0.1:8888`, wähle die Seite und gib ein Zeichen ein (zum Beispiel `中`).

### Höfliches Crawling und Respekt gegenüber Websites

- Der Fetcher nutzt host-spezifisches Throttling, Retries mit Backoff und Caching.
- Halte Delays bei `>= 0.5s`, vermeide Bursts und beachte Nutzungsbedingungen/robots/Lizenzen.
- Umgehe keine Paywalls oder interaktive Schutzmechanismen.
- Bei `403`/`429` langsamer werden und später erneut versuchen.

### Weitere ILM-Workflows

Diese Skripte sind vorhanden und aktiv Teil der Repo-Oberfläche, aber es handelt sich um Forschungs-Workflows, die vorbereitete lokale Datensätze/Checkpoints erfordern können.

1. Daten-Download/-Vorbereitung

```bash
python scripts/download_alpaca.py --outdir data/raw
python scripts/download_corpora.py --out data/raw
python scripts/sample_paragraphs.py --out data/processed/test_100.jsonl
python scripts/build_images_common_freq.py --out data/processed/images_common_freq --size 128 --en 5000 --zh 5000
```

2. Glyph-DB-Lebenszyklus

```bash
python scripts/glyphdb_init.py --db data/glyphdb/glyphs.sqlite3
python scripts/glyphdb_ingest_index.py --db data/glyphdb/glyphs.sqlite3 --index data/processed/images_common_freq/index.tsv
```

3. Code-/Color-Modell-Training

```bash
python scripts/train_color_codes.py --config configs/color.yaml
python scripts/train_codes_from_qa.py --en-json data/raw/alpaca_en.json --zh-json data/raw/alpaca_zh.json --epochs 1
python scripts/train_ilmglyph_codes.py --en data/raw/alpaca_en.json --zh data/raw/alpaca_zh.json --out artifacts/ilm_glyph_train
```

4. Diffusion/Inpainting

```bash
python scripts/train_diffusion.py --config configs/diffusion.yaml
python scripts/train_inpaint_frames.py --ckpt-code artifacts/ilm_glyph_train/ckpt_epoch1.pt --out artifacts/inpaint
```

5. Evaluation/Reporting

```bash
python scripts/eval_color_codes.py --checkpoint artifacts/color_codes_e1.pt
python scripts/eval_diffusion.py --checkpoint artifacts/diffusion_unet.pt
python scripts/eval_qa_retrieval.py --checkpoint artifacts/color_codes_qa.pt
python scripts/report_ilmglyph_pipeline.py --ckpt artifacts/ilm_glyph_train/ckpt_epoch1.pt --lang en --text "hello world"
```

## Konfiguration

Primäre YAML-Konfigurationen:

- `configs/color.yaml`
  - Datenpfad: `data/processed/images_common_freq/index.tsv`
  - Modell-/Code-Parameter: `d_glyph`, `d_code`, `K`, `C`, temperature/anneal
  - Optimierer-/Log-Einstellungen

- `configs/diffusion.yaml`
  - Input-JSONL: `data/processed/test_100.jsonl`
  - Frame/Grid- + Modellgrößen-Einstellungen
  - Train-Mask-Ratio-Bereich und Checkpoint-Einstellungen

Überschreibe Einstellungen über CLI-Flags, wo unterstützt (`--epochs`, `--batch-size`, `--lr` usw.).

## Beispiele

- Erstelle eine einzelne englische Tile-Glyphe:

```bash
python scripts/build_english_tile_glyph.py "language" artifacts/language_tile --save-tensor
```

- Führe die Inpainting-Demo mit trainierten Checkpoints aus:

```bash
python scripts/inpaint_demo.py \
  --ckpt-code artifacts/ilm_glyph_train/ckpt_epoch1.pt \
  --ckpt-inpaint artifacts/inpaint/ckpt_epoch1.pt \
  --lang en \
  --text "the quick brown fox jumps" \
  --mode infill \
  --out artifacts/inpaint_demo
```

- Bulk-Ingestion häufiger Zeichen aus Hanziyuan:

```bash
PYTHONPATH=. python scripts/bulk_ingest_hanziyuan.py --limit 200 --resume
```

## Entwicklungsnotizen

- Dies ist ein Forschungs-Repository mit robusten CLIs und explorativen Artefakten (einschließlich Notebooks und Prototyp-Skripten).
- Generierte große Dateien sind für `data/` und `artifacts/` vorgesehen (beide in `.gitignore` ignoriert).
- Publikationsquellen und PDFs liegen unter `publication/`; Hilfs-Build-Skript: `scripts/latex_build.sh`.
- Kollaborations-/Prozesskonventionen sind in `AGENTS.md` dokumentiert.

## Fehlerbehebung

- `ModuleNotFoundError: ilm...`
  - Skripte vom Repo-Root aus ausführen.
  - `PYTHONPATH=.` für Skripte verwenden, die lokale Paketauflösung erwarten.

- `FileNotFoundError` für Daten/Index/Checkpoints
  - Zuerst erforderliche Daten-/Build-Skripte ausführen.
  - Bestätigen, dass Defaults wie `data/processed/images_common_freq/index.tsv` und `data/processed/test_100.jsonl` vorhanden sind.

- CUDA-/Geräteprobleme
  - Auf CPU wechseln über Skript-Flags/Konfiguration (`device: cpu` oder `--device cpu`).

- Fehlende Paketfehler
  - Erforderliche Abhängigkeit anhand des jeweiligen Skript-Imports installieren (`torch`, `pyyaml`, `Pillow` usw.).

- HTTP-`403` / `429` beim Scraping
  - `--delay` erhöhen, später erneut versuchen und Anfragen höflich halten.

## Roadmap

- Die Text-als-Bild-ILM-Trainings-/Eval-Runbooks über den etymologiezentrierten Schnellstart hinaus weiter ausbauen.
- Reproduzierbarkeit der Umgebung verbessern (eine maßgebliche Abhängigkeits-Spezifikation).
- Test-/CI-Abdeckung für Forschungs-Skripte und Pipeline-Glue erweitern.
- Hierarchische Codebooks, Diffusionsziele und Steuerbarkeitskanäle weiterentwickeln.
- Dokumentation über `docs/`, Skript-Hilfetexte und Publikationsartefakte hinweg konsolidieren.

Für tiefergehende konzeptionelle und stufenweise Planungsdetails siehe:

- `docs/imagized-language-model.md`
- `docs/ilm-visual-diffusion-code-plan.md`
- `docs/development-plan.md`

## Mitwirken

- Folge `AGENTS.md` für Konventionen (atomare Commits, Push nach Änderungen, keine Zugangsdaten im Code).
- Verwandte Änderungen in fokussierten Commits mit konventionellen Messages gruppieren.
- Reproduzierbare Skriptaufrufe mit expliziten Flags und Input-Pfaden bevorzugen.
- Bei scrapingbezogenen Änderungen Throttling-/Cache-Verhalten und Site-Respect-Constraints beibehalten.

## Lizenz

In diesem Repository ist derzeit keine top-level Lizenzdatei vorhanden.

Annahme-Hinweis: Das Projekt als Forschungscode mit nicht spezifizierter Lizenz behandeln, bis eine `LICENSE`-Datei von den Maintainers hinzugefügt wird.

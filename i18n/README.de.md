[English](../README.md) · [العربية](README.ar.md) · [Español](README.es.md) · [Français](README.fr.md) · [日本語](README.ja.md) · [한국어](README.ko.md) · [Tiếng Việt](README.vi.md) · [中文 (简体)](README.zh-Hans.md) · [中文（繁體）](README.zh-Hant.md) · [Deutsch](README.de.md) · [Русский](README.ru.md)


[![LazyingArt banner](https://github.com/lachlanchen/lachlanchen/raw/main/figs/banner.png)](https://github.com/lachlanchen/lachlanchen/blob/main/figs/banner.png)

# Imagized-Sprachmodell (ILM)

![Python](https://img.shields.io/badge/python-3.10%2B-3776AB?logo=python&logoColor=white)
![Status](https://img.shields.io/badge/status-research-orange)
![Focus](https://img.shields.io/badge/focus-text--as--image-0A7EA4)
![Paradigm](https://img.shields.io/badge/paradigm-predictive%20visual%20field-16835B)
![License](https://img.shields.io/badge/license-unspecified-lightgrey)
![Domain](https://img.shields.io/badge/domain-historic%20etymology%20%7C%20glyph%20models-2F80ED?logo=github)

ILM ist eine Forschungscodebasis für **Text-als-Bild-Generierung**: Sie kodiert Sprache in kompakte, bildähnliche Tensoren und erzeugt Text mit iterativer Diffusions-verstärkter Verfeinerung. Die Repräsentation zerlegt Sätze in Meta-Elemente (Grammatik, Semantik, Tonfall, Emotion) sowie hierarchische, speicherartige Codes für Wörter und Zeichen. Das vereint Ideen aus diskreter Diffusion, Superposition/Disentanglement, strukturierten Einbettungen und glyph-bewusster Zeichenerkennung.

## Neuester Prompt-Test: V23-Relationsschaltung bestanden

![Gemessenes V23-Ergebnis: Sechs Bild-Promptframes durchlaufen visuellen Abgleich, Operationsgatter, Quellglyphen-Routing und einen eingefrorenen Bildnormalisierer und erzeugen ein Antwortbild](../publication/ilm-image-native/figures/visual_relation_circuit_v23_result.png)

V23 ist das erste Experiment dieses Repositorys, das die vollständige
Beweiskette von einem Bild-Prompt zu einem Bild-Antwortfeld besteht. Das Modell
erhält nur sechs `32x32`-Schriftbilder und erzeugt ein `32x32`-Antwortbild. Im
Inferenzpfad gibt es keine Strings, Tokens, Unicode-IDs, OCR, Glyphensuche,
Antwortindizes oder externen Sprachmodelle.

In der einzigen autorisierten Frozen-Evaluation mit 98 ungesehenen chinesischen
Zeichen, 1.024 Episoden und 4.096 Promptvarianten erreicht es `0.99829` binäre
Auswahl, `0.99609` Query-Wechsel, `0.99707` Operationswechsel, `0.99463`
Glyphen-Top-1 und `0.78478` Pixel-F1. Query-blinde und operationsblinde
Kontrollen bleiben gegenüber ihrem jeweils unsichtbaren Faktor exakt invariant.

Dies belegt nur visuelles Prompt-Folgen für eine feste Grammatik mit sechs
Rollen, zwei Bindungen und einer Same/Other-Relation, nicht freies
Sprachverständnis. V24 soll feste Framerollen entfernen, einen variabel langen
2D-Schriftbildstrom lesen und nach dem Wiederlesen des ersten Ausgabeframes ein
zweites erzeugen. Siehe den [englischen V23-Bericht](../docs/visual-relation-circuit-v23-result.md).

> Das Repository hält bewusst eine praxisnahe Etymologie-Pipeline und ILM-Experimentieren auf lange Sicht nebeneinander.

## 📌 Überblick

Dieses Repository verfolgt zwei aktive Pfade:

1. Historische chinesische Glyphen-Etymologie-Ingestion (Scraping/Parsing/Speicherung/Vorschau).
2. ILM-Experimente für Glyphen-/Bildmodellierung (Token-Glyph-Rendering, Codebücher, Frame-Packing, Diffusion/Inpainting, Auswertung/Reporting).

Diese README dokumentiert beide Spuren und hält den Etymologie-Workflow als ersten Klassen-Zweig mit reproduzierbarem Pfad.

## 🔗 Wichtige Links

| Bereich | Pfad |
|---|---|
| Konzeptionelles Exposé | `docs/imagized-language-model.md` |
| Codeplan und Metriken | `docs/ilm-visual-diffusion-code-plan.md` |
| Embedding-„Color“-Plan | `docs/embedding-color-plan.md` |
| Entwicklungsnotizen/-plan | `docs/development-plan.md` |
| Etymologie-Modul-README | `ilm/etymology/README.md` |

## ✨ Funktionen

- 🏺 Etymologie-Ingestion aus `hanziyuan`- und `chineseetymology`-ähnlichen Quellen.
- 🌐 Robuster AJAX- + HTML-Ingestion-Pfad mit Wiederholungsversuchen, Drosselung und Cache.
- 🧩 Etappen-etikettierte Glyphen-Extraktion inklusive `<img>`- und CSS-`background-image`-Data-URIs.
- 🗃️ SQLite-basierte Speicherung für Zeichen-/Glyphen-Metadaten plus Dateisystemstruktur für Assets.
- 🖥️ Tornado-Web-UI für ad-hoc-Ingestion + Galerievorschau.
- 🔤 Glyph-Rendering-Werkzeuge für mehrsprachige Token-Bilder.
- 🧠 Embedding-/Codebook-Module im Stil von Product-Codes.
- 🧱 Satz-Frame-Packing sowie Diffusions-/Inpainting-Trainings- und Evaluations-Skripte.
- 📊 Reporting- und Visualisierungs-Skripte für Embeddings und Pipeline-Inspektion.
- 📄 Veröffentlichungsartefakte in LaTeX/PDF unter `publication/`.

## 🧱 Projektstruktur

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

## 🧰 Voraussetzungen

| Anforderung | Hinweise |
|---|---|
| Python `3.10+` | Zentrale Laufzeit |
| `pip` | Paketinstallation |
| Optional GPU | Hilfreich für PyTorch CUDA-Trainingsskripte |
| Optionaler LaTeX-Stack | Für Publication-Builds erforderlich |

Hinweis zur Annahme: Es gibt momentan keine einzelne zentrale Dependency-Sperre/Spezifikation (`pyproject.toml`, `requirements.txt` usw.), daher werden Abhängigkeiten aus Imports und Skript-Nutzung abgeleitet.

## ⚙️ Installation

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

Wenn ein bestimmtes Skript zusätzliche Pakete benötigt, installiere diese anhand der konkreten Import-Fehlermeldung des Skripts.

## 🚀 Verwendung

### Schnellstart: Historische Glyphen-Ingestion (CLI)

1. Hanziyuan (empfohlen): Nur-Zeichen-AJAX-Flow

```bash
PYTHONPATH=. python scripts/ingest_etymology.py --site hanziyuan --char 中
```

2. ChineseEtymology (direkte URL)

```bash
PYTHONPATH=. python scripts/ingest_etymology.py --site chineseetymology --url "https://www.chineseetymology.org/CharacterEtymology.aspx?characterInput=%E4%B8%AD"
```

3. Batch-Datei-Ingestion (Zeilen können `char\turl`, `url` oder `char url` sein)

```bash
PYTHONPATH=. python scripts/ingest_etymology.py --from-file urls.txt
```

### Ausgaben

| Ausgabetyp | Ort |
|---|---|
| Dateien | `data/historic/glyphs/<char>/<stage>/<label>.<ext>` |
| Cache | `data/historic/cache/*.html` |
| Datenbank | `data/historic/etymology.sqlite3` |

### Web-Demo (optional)

```bash
PYTHONPATH=. python scripts/serve_etymology.py
```

Öffne `http://127.0.0.1:8888`, wähle eine Seite und gib ein Zeichen ein (zum Beispiel `中`).

### Höfliches Crawling und Rücksicht auf Webseiten

- Der Fetcher nutzt pro-Host-Drosselung, Wiederholungsversuche mit Backoff und Caching.
- Halte Delays von `>= 0.5s` ein, vermeide Bursts und beachte Nutzungsbedingungen/Robots/Lizenz.
- Umgehe keine Paywalls oder interaktive Schutzmaßnahmen.
- Bei `403`/`429` langsamer machen und später erneut versuchen.

### Zusätzliche ILM-Workflows

Diese Skripte sind vorhanden und aktiv Teil der Repo-Oberfläche, aber Forschungs-Workflows und können vorbereitete lokale Datensätze/Checkpoints erfordern.

1. Daten herunterladen/vorbereiten

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

3. Code-/Farbmodell-Training

```bash
python scripts/train_color_codes.py --config configs/color.yaml
python scripts/train_codes_from_qa.py --en-json data/raw/alpaca_en.json --zh-json data/raw/alpaca_zh.json --epochs 1
python scripts/train_ilmglyph_codes.py --en data/raw/alpaca_en.json --zh data/raw/alpaca_zh.json --out artifacts/ilm_glyph_train
```

4. Diffusions-/Inpainting-Workflows

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

## 🧩 Konfiguration

Primäre YAML-Konfigurationen:

- `configs/color.yaml`
  - Datenpfad: `data/processed/images_common_freq/index.tsv`
  - Modell/Code-Parameter: `d_glyph`, `d_code`, `K`, `C`, temperature/anneal
  - Optimizer-/Logging-Einstellungen

- `configs/diffusion.yaml`
  - Eingabe-JSONL: `data/processed/test_100.jsonl`
  - Frame-/Raster- und Modellgrößeneinstellungen
  - Bereich der Trainings-Maske und Checkpoint-Einstellungen

Überschreibe Einstellungen über CLI-Flags, wo unterstützt (`--epochs`, `--batch-size`, `--lr` usw.).

## 🧪 Beispiele

- Erstelle eine einzelne englische Tile-Glyphe:

```bash
python scripts/build_english_tile_glyph.py "language" artifacts/language_tile --save-tensor
```

- Inpainting-Demo mit trainierten Checkpoints ausführen:

```bash
python scripts/inpaint_demo.py \
  --ckpt-code artifacts/ilm_glyph_train/ckpt_epoch1.pt \
  --ckpt-inpaint artifacts/inpaint/ckpt_epoch1.pt \
  --lang en \
  --text "the quick brown fox jumps" \
  --mode infill \
  --out artifacts/inpaint_demo
```

- Häufige Zeichen massenhaft aus Hanziyuan ingestieren:

```bash
PYTHONPATH=. python scripts/bulk_ingest_hanziyuan.py --limit 200 --resume
```

## 📝 Entwicklungsnotizen

- Dieses ist ein Forschungs-Repository mit robusten CLIs und explorativen Artefakten (inklusive Notebooks und Prototyp-Skripten).
- Große erzeugte Dateien sind für `data/` und `artifacts/` vorgesehen (beides ist in `.gitignore` ausgeschlossen).
- Veröffentlichungssourcen und PDFs liegen unter `publication/`; Hilfsskript: `scripts/latex_build.sh`.
- Zusammenarbeit/Prozesskonventionen sind in `AGENTS.md` dokumentiert.

## 🛠️ Fehlerbehebung

- `ModuleNotFoundError: ilm...`
  - Führe Skripte vom Repo-Root aus.
  - Setze `PYTHONPATH=.` für Skripte, die mit lokaler Paketauflösung arbeiten.

- `FileNotFoundError` bei Daten/Index/Checkpoints
  - Führe zuerst die nötigen Daten-/Build-Skripte aus.
  - Prüfe, dass Standardpfade wie `data/processed/images_common_freq/index.tsv` und `data/processed/test_100.jsonl` existieren.

- CUDA/Geräteprobleme
  - Wechsle mit Skript-Flags/Konfiguration auf CPU (`device: cpu` oder `--device cpu`).

- Fehlende Paketfehler
  - Installiere die benötigte Abhängigkeit aus dem konkreten Importpfad des Skripts (`torch`, `pyyaml`, `Pillow` usw.).

- HTTP `403`/`429` beim Scraping
  - Erhöhe `--delay`, versuche es später erneut und halte Anfragen höflich.

## 🗺️ Roadmap

- Verfeinere die Text-als-Bild-ILM-Trainings-/Eval-Runbooks über den Etymologie-orientierten Quick-Start hinaus.
- Verbessere Reproduzierbarkeit der Umgebung (eine einzige autoritative Dependency-Spezifikation).
- Erweitere Tests/CI-Abdeckung für Forschungsskripte und Pipeline-Klebeschicht.
- Iteriere weiter bei hierarchischen Codebooks, Diffusionszielen und Kontrollierbarkeitskanälen.
- Konsolidiere Dokumentation über `docs/`, Skript-Helptexte und Veröffentlichungspakete.

Für konzeptionell tiefere und stufenweise Planungsdetails siehe:

- `docs/imagized-language-model.md`
- `docs/ilm-visual-diffusion-code-plan.md`
- `docs/development-plan.md`

## 🤝 Mitwirken

- Folge `AGENTS.md` für Konventionen (atomare Commits, Push nach Änderungen, keine Zugangsdaten im Code).
- Fasse verwandte Änderungen in fokussierten Commits mit konventionellen Nachrichten zusammen.
- Bevorzuge reproduzierbare Skriptaufrufe mit expliziten Flags und Eingabepfaden.
- Für Änderungen am Scraping die Drosselung/Cache-Verhalten und Site-Respekt-Kontrollen unverändert erhalten.

## 📄 Lizenz

Eine Top-level-Lizenzdatei ist derzeit im Repository nicht vorhanden.

Annahmehinweis: Betrachte das Projekt bis zum Hinzufügen einer `LICENSE` durch die Maintainer als Forschungs-Code mit nicht spezifizierter Lizenzierung.


## ❤️ Support

| Donate | PayPal | Stripe |
| --- | --- | --- |
| [![Donate](https://camo.githubusercontent.com/24a4914f0b42c6f435f9e101621f1e52535b02c225764b2f6cc99416926004b7/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f446f6e6174652d4c617a79696e674172742d3045413545393f7374796c653d666f722d7468652d6261646765266c6f676f3d6b6f2d6669266c6f676f436f6c6f723d7768697465)](https://chat.lazying.art/donate) | [![PayPal](https://camo.githubusercontent.com/d0f57e8b016517a4b06961b24d0ca87d62fdba16e18bbdb6aba28e978dc0ea21/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f50617950616c2d526f6e677a686f754368656e2d3030343537433f7374796c653d666f722d7468652d6261646765266c6f676f3d70617970616c266c6f676f436f6c6f723d7768697465)](https://paypal.me/RongzhouChen) | [![Stripe](https://camo.githubusercontent.com/1152dfe04b6943afe3a8d2953676749603fb9f95e24088c92c97a01a897b4942/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f5374726970652d446f6e6174652d3633354246463f7374796c653d666f722d7468652d6261646765266c6f676f3d737472697065266c6f676f436f6c6f723d7768697465)](https://buy.stripe.com/aFadR8gIaflgfQV6T4fw400) |

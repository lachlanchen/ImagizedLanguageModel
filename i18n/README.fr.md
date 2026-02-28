[English](../README.md) · [العربية](README.ar.md) · [Español](README.es.md) · [Français](README.fr.md) · [日本語](README.ja.md) · [한국어](README.ko.md) · [Tiếng Việt](README.vi.md) · [中文 (简体)](README.zh-Hans.md) · [中文（繁體）](README.zh-Hant.md) · [Deutsch](README.de.md) · [Русский](README.ru.md)


# Modèle de Langage Imagisé (ILM)

![Python](https://img.shields.io/badge/python-3.10%2B-3776AB?logo=python&logoColor=white)
![Status](https://img.shields.io/badge/status-research-orange)
![Focus](https://img.shields.io/badge/focus-text--as--image-0A7EA4)
![Diffusion](https://img.shields.io/badge/paradigm-diffusion%20%2B%20glyphs-6A5ACD)
![License](https://img.shields.io/badge/license-unspecified-lightgrey)

ILM est une base de code de recherche qui explore la génération de texte comme image : il encode le langage en tenseurs compacts de type image et génère du texte via un raffinement itératif de style diffusion. La représentation factorise les phrases en méta-éléments (grammaire, sémantique, ton, émotion) et en codes hiérarchiques de type mémoire pour les mots et les caractères. Cela unifie des idées issues de la diffusion discrète, de la superposition/désintrication, des embeddings structurés et de la modélisation de caractères sensible aux glyphes.

## Vue d’ensemble

Ce dépôt inclut actuellement deux axes pratiques majeurs :

1. Ingestion d’étymologies de glyphes chinois historiques (scraping/parsing/stockage/aperçu).
2. Expériences de modélisation ILM glyphes/images (rendu de glyphes de tokens, codebooks produits, empaquetage de frames, diffusion/inpainting, évaluation/reporting).

Le README actuel de ce dépôt s’est historiquement concentré sur la boîte à outils d’étymologie. Ce workflow reste entièrement documenté ci-dessous et est conservé comme référence canonique.

## Liens clés

| Domaine | Chemin |
|---|---|
| Présentation conceptuelle | `docs/imagized-language-model.md` |
| Plan de code et métriques | `docs/ilm-visual-diffusion-code-plan.md` |
| Plan de « couleur » d’embedding | `docs/embedding-color-plan.md` |
| Notes/plan de développement | `docs/development-plan.md` |
| README du module d’étymologie | `ilm/etymology/README.md` |

## Fonctionnalités

- 🏺 Ingestion d’étymologies depuis des sources de type `hanziyuan` et `chineseetymology`.
- 🌐 Pipeline d’ingestion AJAX + HTML robuste avec retries, limitation de débit et cache.
- 🧩 Extraction de glyphes étiquetée par étape, incluant les données URI `<img>` et CSS `background-image`.
- 🗃️ Stockage basé sur SQLite pour les métadonnées chars/glyphes, avec organisation des assets sur le système de fichiers.
- 🖥️ Interface web Tornado pour ingestion ad hoc + aperçu en galerie.
- 🔤 Utilitaires de rendu de glyphes pour images de tokens multilingues.
- 🧠 Modules d’embedding/codebook de type product-code.
- 🧱 Scripts d’entraînement/évaluation pour empaquetage de frames de phrases et diffusion/inpainting.
- 📊 Scripts de reporting et de visualisation pour inspecter les embeddings et le pipeline.
- 📄 Artéfacts de publication en LaTeX/PDF sous `publication/`.

## Structure du projet

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

## Prérequis

| Exigence | Notes |
|---|---|
| Python `3.10+` | Runtime principal |
| `pip` | Installation des packages |
| GPU optionnel | Utile pour les scripts d’entraînement PyTorch CUDA |
| Chaîne d’outils LaTeX optionnelle | Nécessaire pour les builds de publication |

Note d’hypothèse : il n’existe actuellement aucun fichier unique de verrouillage/spécification des dépendances à la racine (`pyproject.toml`, `requirements.txt`, etc.), donc les dépendances sont déduites des imports et de l’usage des scripts.

## Installation

### Minimale (boîte à outils d’étymologie)

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install requests beautifulsoup4 tornado
```

### Étendue (workflows de modélisation/entraînement)

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install requests beautifulsoup4 tornado pyyaml numpy pillow matplotlib torch
```

Si un script spécifique nécessite des packages additionnels, installez-les à partir de l’erreur d’import affichée par ce script.

## Utilisation

### Démarrage rapide : ingestion de glyphes historiques (CLI)

1. Hanziyuan (recommandé) : flux AJAX char-only

```bash
PYTHONPATH=. python scripts/ingest_etymology.py --site hanziyuan --char 中
```

2. ChineseEtymology (URL directe)

```bash
PYTHONPATH=. python scripts/ingest_etymology.py --site chineseetymology --url "https://www.chineseetymology.org/CharacterEtymology.aspx?characterInput=%E4%B8%AD"
```

3. Ingestion depuis un fichier batch (les lignes peuvent être `char\turl`, `url`, ou `char url`)

```bash
PYTHONPATH=. python scripts/ingest_etymology.py --from-file urls.txt
```

### Sorties

| Type de sortie | Emplacement |
|---|---|
| Fichiers | `data/historic/glyphs/<char>/<stage>/<label>.<ext>` |
| Cache | `data/historic/cache/*.html` |
| BD | `data/historic/etymology.sqlite3` |

### Démo Web (optionnel)

```bash
PYTHONPATH=. python scripts/serve_etymology.py
```

Ouvrez `http://127.0.0.1:8888`, choisissez le site, puis saisissez un caractère (par exemple `中`).

### Crawling respectueux et respect des sites

- Le fetcher utilise une limitation par hôte, des retries avec backoff et un cache.
- Gardez des délais `>= 0.5s`, évitez les rafales et respectez les conditions/robots/licences des sites.
- Ne contournez pas les paywalls ni les protections interactives.
- Si vous voyez `403`/`429`, ralentissez et réessayez plus tard.

### Workflows ILM supplémentaires

Ces scripts existent et font activement partie de la surface du dépôt, mais ce sont des workflows de recherche pouvant nécessiter des jeux de données/checkpoints locaux préparés.

1. Téléchargement/préparation des données

```bash
python scripts/download_alpaca.py --outdir data/raw
python scripts/download_corpora.py --out data/raw
python scripts/sample_paragraphs.py --out data/processed/test_100.jsonl
python scripts/build_images_common_freq.py --out data/processed/images_common_freq --size 128 --en 5000 --zh 5000
```

2. Cycle de vie de la Glyph DB

```bash
python scripts/glyphdb_init.py --db data/glyphdb/glyphs.sqlite3
python scripts/glyphdb_ingest_index.py --db data/glyphdb/glyphs.sqlite3 --index data/processed/images_common_freq/index.tsv
```

3. Entraînement du modèle code/couleur

```bash
python scripts/train_color_codes.py --config configs/color.yaml
python scripts/train_codes_from_qa.py --en-json data/raw/alpaca_en.json --zh-json data/raw/alpaca_zh.json --epochs 1
python scripts/train_ilmglyph_codes.py --en data/raw/alpaca_en.json --zh data/raw/alpaca_zh.json --out artifacts/ilm_glyph_train
```

4. Diffusion/inpainting

```bash
python scripts/train_diffusion.py --config configs/diffusion.yaml
python scripts/train_inpaint_frames.py --ckpt-code artifacts/ilm_glyph_train/ckpt_epoch1.pt --out artifacts/inpaint
```

5. Évaluation/reporting

```bash
python scripts/eval_color_codes.py --checkpoint artifacts/color_codes_e1.pt
python scripts/eval_diffusion.py --checkpoint artifacts/diffusion_unet.pt
python scripts/eval_qa_retrieval.py --checkpoint artifacts/color_codes_qa.pt
python scripts/report_ilmglyph_pipeline.py --ckpt artifacts/ilm_glyph_train/ckpt_epoch1.pt --lang en --text "hello world"
```

## Configuration

Fichiers YAML principaux :

- `configs/color.yaml`
  - chemin des données : `data/processed/images_common_freq/index.tsv`
  - paramètres modèle/code : `d_glyph`, `d_code`, `K`, `C`, temperature/anneal
  - paramètres optimizer/log

- `configs/diffusion.yaml`
  - JSONL d’entrée : `data/processed/test_100.jsonl`
  - paramètres taille frame/grille + modèle
  - plage de ratio de masquage d’entraînement et paramètres de checkpoints

Surchargez les paramètres via les flags CLI lorsqu’ils sont pris en charge (`--epochs`, `--batch-size`, `--lr`, etc.).

## Exemples

- Construire un glyphe tile anglais unique :

```bash
python scripts/build_english_tile_glyph.py "language" artifacts/language_tile --save-tensor
```

- Exécuter une démo d’inpainting avec des checkpoints entraînés :

```bash
python scripts/inpaint_demo.py \
  --ckpt-code artifacts/ilm_glyph_train/ckpt_epoch1.pt \
  --ckpt-inpaint artifacts/inpaint/ckpt_epoch1.pt \
  --lang en \
  --text "the quick brown fox jumps" \
  --mode infill \
  --out artifacts/inpaint_demo
```

- Ingestion en masse de caractères courants depuis Hanziyuan :

```bash
PYTHONPATH=. python scripts/bulk_ingest_hanziyuan.py --limit 200 --resume
```

## Notes de développement

- Il s’agit d’un dépôt de recherche avec à la fois des CLI robustes et des artéfacts exploratoires (dont des notebooks et scripts prototypes).
- Les gros fichiers générés sont destinés à `data/` et `artifacts/` (tous deux ignorés dans `.gitignore`).
- Les sources de publication et PDF sont sous `publication/` ; script d’aide au build : `scripts/latex_build.sh`.
- Les conventions de collaboration/process sont documentées dans `AGENTS.md`.

## Dépannage

- `ModuleNotFoundError: ilm...`
  - Exécutez les scripts depuis la racine du dépôt.
  - Utilisez `PYTHONPATH=.` pour les scripts qui attendent une résolution de package locale.

- `FileNotFoundError` pour data/index/checkpoints
  - Exécutez d’abord les scripts de préparation des données/build prérequis.
  - Vérifiez que les chemins par défaut comme `data/processed/images_common_freq/index.tsv` et `data/processed/test_100.jsonl` existent.

- Problèmes CUDA/device
  - Basculez sur CPU via les flags/config du script (`device: cpu` ou `--device cpu`).

- Erreurs de package manquant
  - Installez la dépendance requise selon le chemin d’import du script concerné (`torch`, `pyyaml`, `Pillow`, etc.).

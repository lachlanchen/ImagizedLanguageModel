[English](../README.md) · [العربية](README.ar.md) · [Español](README.es.md) · [Français](README.fr.md) · [日本語](README.ja.md) · [한국어](README.ko.md) · [Tiếng Việt](README.vi.md) · [中文 (简体)](README.zh-Hans.md) · [中文（繁體）](README.zh-Hant.md) · [Deutsch](README.de.md) · [Русский](README.ru.md)


[![LazyingArt banner](https://github.com/lachlanchen/lachlanchen/raw/main/figs/banner.png)](https://github.com/lachlanchen/lachlanchen/blob/main/figs/banner.png)

# Modèle de Langage Imagé (ILM)

![Python](https://img.shields.io/badge/python-3.10%2B-3776AB?logo=python&logoColor=white)
![Status](https://img.shields.io/badge/status-research-orange)
![Focus](https://img.shields.io/badge/focus-text--as--image-0A7EA4)
![Diffusion](https://img.shields.io/badge/paradigm-diffusion%20%2B%20glyphs-6A5ACD)
![License](https://img.shields.io/badge/license-unspecified-lightgrey)
![Domain](https://img.shields.io/badge/domain-historic%20etymology%20%7C%20glyph%20models-2F80ED?logo=github)

ILM est une base de code de recherche qui explore la **génération texte-image** : il encode le langage en tenseurs compacts de type image et génère du texte via un raffinement itératif de type diffusion. La représentation factorise les phrases en méta-éléments (grammaire, sémantique, ton, émotion) et en codes hiérarchiques de type mémoire pour les mots et les caractères. Cela unifie des idées issues de la diffusion discrète, de la superposition/désentrelacement, des embeddings structurés et de la modélisation de glyphes sensibles au caractère.

> Le dépôt maintient volontairement une pipeline d’étymologie pratique et des expérimentations ILM à horizon long côte à côte.

## 📌 Aperçu

Ce dépôt suit deux axes actifs :

1. Ingestion d’étymologies de glyphes chinois historiques (scraping/parsing/stockage/aperçu).
2. Expérimentations de modélisation d’ILM glyphes/images (rendu de glyphes de tokens, codebooks, empaquetage de trames, diffusion/inpainting, scripts d’entraînement/évaluation).

Ce README documente également les deux axes et maintient le workflow d’étymologie comme un parcours de référence reproductible.

## 🔗 Liens clés

| Domaine | Chemin |
|---|---|
| Présentation conceptuelle | `docs/imagized-language-model.md` |
| Plan de code et métriques | `docs/ilm-visual-diffusion-code-plan.md` |
| Plan de « couleur » des embeddings | `docs/embedding-color-plan.md` |
| Notes de plan de développement | `docs/development-plan.md` |
| README du module d’étymologie | `ilm/etymology/README.md` |

## ✨ Fonctionnalités

- 🏺 Ingestion d’étymologies depuis des sources de type `hanziyuan` et `chineseetymology`.
- 🌐 Parcours d’ingestion AJAX + HTML robuste avec retries, throttling et cache.
- 🧩 Extraction de glyphes étiquetés par étape incluant les données URI `<img>` et CSS `background-image`.
- 🗃️ Stockage basé sur SQLite pour les métadonnées de caractères/glyphes ainsi que l’organisation des actifs sur le système de fichiers.
- 🖥️ UI web Tornado pour ingestion ad hoc + prévisualisation en galerie.
- 🔤 Utilitaires de rendu de glyphes pour images de tokens multilingues.
- 🧠 Modules d’embedding/codebook de style product-code.
- 🧱 Script de prépa et de calcul : empaquetage de cadres de phrase et entraînement/évaluation diffusion/inpainting.
- 📊 Scripts de reporting et de visualisation pour inspection des embeddings et du pipeline.
- 📄 Artefacts de publication en LaTeX/PDF sous `publication/`.

## 🧱 Structure du projet

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

## 🧰 Prérequis

| Exigence | Notes |
|---|---|
| Python `3.10+` | Runtime principal |
| `pip` | Installation des paquets |
| GPU facultatif | Utile pour les scripts d’entraînement PyTorch CUDA |
| Chaîne d’outils LaTeX facultative | Nécessaire pour les builds de publication |

Note : il n’existe actuellement aucun fichier de verrouillage/spécification de dépendances unique à la racine (`pyproject.toml`, `requirements.txt`, etc.), les dépendances sont donc inférées depuis les imports et l’usage des scripts.

## ⚙️ Installation

### Minimal (boîte à outils d’étymologie)

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install requests beautifulsoup4 tornado
```

### Étendu (workflows de modélisation/entraînement)

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install requests beautifulsoup4 tornado pyyaml numpy pillow matplotlib torch
```

Si un script spécifique nécessite des paquets supplémentaires, installez-les depuis l’erreur d’import affichée par ce script.

## 🚀 Utilisation

### Démarrage rapide : ingestion de glyphes historiques (CLI)

1. Hanziyuan (recommandé) : flux AJAX en caractères seuls

```bash
PYTHONPATH=. python scripts/ingest_etymology.py --site hanziyuan --char 中
```

2. ChineseEtymology (URL directe)

```bash
PYTHONPATH=. python scripts/ingest_etymology.py --site chineseetymology --url "https://www.chineseetymology.org/CharacterEtymology.aspx?characterInput=%E4%B8%AD"
```

3. Ingestion par lot depuis un fichier (les lignes peuvent être `char\turl`, `url` ou `char url`)

```bash
PYTHONPATH=. python scripts/ingest_etymology.py --from-file urls.txt
```

### Sorties

| Type de sortie | Emplacement |
|---|---|
| Fichiers | `data/historic/glyphs/<char>/<stage>/<label>.<ext>` |
| Cache | `data/historic/cache/*.html` |
| Base | `data/historic/etymology.sqlite3` |

### Démo web (facultative)

```bash
PYTHONPATH=. python scripts/serve_etymology.py
```

Ouvrez `http://127.0.0.1:8888`, choisissez le site, saisissez un caractère (par exemple `中`).

### Crawling poli et respect des sites

- Le fetcher applique un throttling par hôte, des tentatives avec backoff et un cache.
- Gardez un délai de `>= 0.5s`, évitez les rafales et respectez les conditions/robots/licences des sites.
- Ne contournez pas les paywalls ni les protections interactives.
- Si vous voyez `403`/`429`, ralentissez et réessayez plus tard.

### Workflows ILM supplémentaires

Ces scripts existent et font partie active du dépôt, mais il s’agit de workflows de recherche qui peuvent nécessiter des jeux de données/checkpoints locaux préparés.

1. Téléchargement/préparation des données

```bash
python scripts/download_alpaca.py --outdir data/raw
python scripts/download_corpora.py --out data/raw
python scripts/sample_paragraphs.py --out data/processed/test_100.jsonl
python scripts/build_images_common_freq.py --out data/processed/images_common_freq --size 128 --en 5000 --zh 5000
```

2. Cycle de vie de la base de glyphes

```bash
python scripts/glyphdb_init.py --db data/glyphdb/glyphs.sqlite3
python scripts/glyphdb_ingest_index.py --db data/glyphdb/glyphs.sqlite3 --index data/processed/images_common_freq/index.tsv
```

3. Entraînement modèle code/couleur

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

## 🧩 Configuration

Configurations YAML principales :

- `configs/color.yaml`
  - chemin des données : `data/processed/images_common_freq/index.tsv`
  - paramètres modèle/code : `d_glyph`, `d_code`, `K`, `C`, temperature/anneal
  - paramètres optimizeur/logs

- `configs/diffusion.yaml`
  - entrée JSONL : `data/processed/test_100.jsonl`
  - réglages de frame/grille et taille de modèle
  - plage du taux de masque d’entraînement et réglages de checkpoint

Remplacez les paramètres via des flags CLI quand pris en charge (`--epochs`, `--batch-size`, `--lr`, etc.).

## 🧪 Exemples

- Construire un glyphe tile anglais unique :

```bash
python scripts/build_english_tile_glyph.py "language" artifacts/language_tile --save-tensor
```

- Exécuter la démo d’inpainting avec des checkpoints entraînés :

```bash
python scripts/inpaint_demo.py \
  --ckpt-code artifacts/ilm_glyph_train/ckpt_epoch1.pt \
  --ckpt-inpaint artifacts/inpaint/ckpt_epoch1.pt \
  --lang en \
  --text "the quick brown fox jumps" \
  --mode infill \
  --out artifacts/inpaint_demo
```

- Ingestion en masse des caractères courants depuis Hanziyuan :

```bash
PYTHONPATH=. python scripts/bulk_ingest_hanziyuan.py --limit 200 --resume
```

## 📝 Notes de développement

- Il s’agit d’un dépôt de recherche avec des CLI robustes et des artefacts exploratoires (dont notebooks et scripts prototypes).
- Les fichiers volumineux générés visent `data/` et `artifacts/` (tous deux ignorés dans `.gitignore`).
- Les sources de publication et PDF sont sous `publication/` ; script de build : `scripts/latex_build.sh`.
- Les conventions de collaboration/processus sont documentées dans `AGENTS.md`.

## 🛠️ Dépannage

- `ModuleNotFoundError: ilm...`
  - Exécutez les scripts depuis la racine du dépôt.
  - Utilisez `PYTHONPATH=.` pour les scripts qui attendent une résolution locale des modules.

- `FileNotFoundError` pour data/index/checkpoints
  - Exécutez d’abord les scripts de préparation des données requis.
  - Vérifiez que des chemins par défaut comme `data/processed/images_common_freq/index.tsv` et `data/processed/test_100.jsonl` existent.

- Problèmes CUDA/device
  - Passez sur le CPU via les flags/configurations du script (`device: cpu` ou `--device cpu`).

- Erreurs de paquets manquants
  - Installez la dépendance requise depuis le chemin d’import indiqué par le script (`torch`, `pyyaml`, `Pillow`, etc.).

## 🗺️ Feuille de route

- Poursuivre le maturage des runbooks d’entraînement/évaluation ILM text-as-image au-delà du démarrage rapide orienté étymologie.
- Améliorer la reproductibilité de l’environnement (spécification unique et autoritaire des dépendances).
- Étendre les tests/CI pour les scripts de recherche et le glue de pipeline.
- Itérer sur les codebooks hiérarchiques, les objectifs de diffusion et les canaux de contrôlabilité.
- Consolider la documentation entre `docs/`, l’aide des scripts et les artefacts de publication.

Pour des détails conceptuels plus profonds et une planification par étapes, voyez :

- `docs/imagized-language-model.md`
- `docs/ilm-visual-diffusion-code-plan.md`
- `docs/development-plan.md`

## 🤝 Contribuer

- Suivez `AGENTS.md` pour les conventions (commits atomiques, push après modification, pas d’identifiants dans le code).
- Regroupez les modifications connexes en commits ciblés avec des messages conventionnels.
- Privilégiez des invocations de scripts reproductibles avec des flags et chemins d’entrée explicites.
- Pour les modifications liées au scraping, conservez le comportement de throttling/cache et le respect des sites.

## 📄 Licence

Aucun fichier de licence de niveau supérieur n’est actuellement présent dans ce dépôt.

Note d’hypothèse : traitez ce projet comme du code de recherche à licence non spécifiée jusqu’à l’ajout d’un fichier `LICENSE` par les mainteneurs.


## ❤️ Support

| Donate | PayPal | Stripe |
| --- | --- | --- |
| [![Donate](https://camo.githubusercontent.com/24a4914f0b42c6f435f9e101621f1e52535b02c225764b2f6cc99416926004b7/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f446f6e6174652d4c617a79696e674172742d3045413545393f7374796c653d666f722d7468652d6261646765266c6f676f3d6b6f2d6669266c6f676f436f6c6f723d7768697465)](https://chat.lazying.art/donate) | [![PayPal](https://camo.githubusercontent.com/d0f57e8b016517a4b06961b24d0ca87d62fdba16e18bbdb6aba28e978dc0ea21/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f50617950616c2d526f6e677a686f754368656e2d3030343537433f7374796c653d666f722d7468652d6261646765266c6f676f3d70617970616c266c6f676f436f6c6f723d7768697465)](https://paypal.me/RongzhouChen) | [![Stripe](https://camo.githubusercontent.com/1152dfe04b6943afe3a8d2953676749603fb9f95e24088c92c97a01a897b4942/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f5374726970652d446f6e6174652d3633354246463f7374796c653d666f722d7468652d6261646765266c6f676f3d737472697065266c6f676f436f6c6f723d7768697465)](https://buy.stripe.com/aFadR8gIaflgfQV6T4fw400) |

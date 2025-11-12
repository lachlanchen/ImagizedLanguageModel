Imagized Language Model (ILM)
=============================

Langues
- [English](../README.md) | [简体中文](README.zh-Hans.md) | [繁體中文](README.zh-Hant.md) | [日本語](README.ja.md) | [한국어](README.ko.md) | [Tiếng Việt](README.vi.md) | [العربية](README.ar.md) | Français | [Español](README.es.md)

Vue d’ensemble
ILM représente le texte sous forme de tenseurs « comme une image » et génère du texte via une débruitage itératif de type diffusion. Une phrase est factorisée en méta‑éléments contrôlables (grammaire, sémantique, ton, émotion) et en codes hiérarchiques « de type mémoire » (mots/graphèmes). Il combine diffusion discrète, superposition/désentrelacement, embeddings structurés et sensibilité aux glyphes.

Liens clés
- Concept: docs/imagized-language-model.md
- Plan de code & métriques: docs/ilm-visual-diffusion-code-plan.md
- Plan « couleur » d’embedding: docs/embedding-color-plan.md

Contenu du dépôt
- ilm/etymology/ : outils pour collecter des glyphes historiques (oracle/bronze/sceau, etc.)
  - Récupération AJAX depuis hanziyuan (retries, throttling, cache)
  - Parsing HTML/CSS pour extraire des images étiquetées par étape (data URI / URL)
- scripts/
  - ingest_etymology.py : CLI pour ingérer → SQLite & fichiers
  - serve_etymology.py : mini‑UI Tornado pour tester/visualiser
  - use_historic_tools.md : notes sur données/outils externes
- data/ (ignoré) : cache HTML, images, base SQLite

Démarrage rapide
- Dépendances : `pip install requests beautifulsoup4 tornado`
- Exemple (AJAX hanziyuan recommandé) :
  - `PYTHONPATH=. python scripts/ingest_etymology.py --site hanziyuan --char 中`
- Démo web :
  - `PYTHONPATH=. python scripts/serve_etymology.py` → http://127.0.0.1:8888

Sorties
- Images : data/historic/glyphs/<caractère>/<étape>/<étiquette>.<ext>
- Cache : data/historic/cache
- Base : data/historic/etymology.sqlite3

Collecte respectueuse
- Limitation par hôte, retries avec backoff, cache activés
- Respectez les conditions/licences. En cas de 403/429, ralentissez et réessayez

Objectif
- Modèle structuré, contrôlable, multilingue, fonctionnant sur des PC courants

Contribuer
- Suivre AGENTS.md (commits atomiques, push après chaque changement, pas d’identifiants dans le dépôt)


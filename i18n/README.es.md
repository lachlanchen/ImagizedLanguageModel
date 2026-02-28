[English](../README.md) · [العربية](README.ar.md) · [Español](README.es.md) · [Français](README.fr.md) · [日本語](README.ja.md) · [한국어](README.ko.md) · [Tiếng Việt](README.vi.md) · [中文 (简体)](README.zh-Hans.md) · [中文（繁體）](README.zh-Hant.md) · [Deutsch](README.de.md) · [Русский](README.ru.md)

# Imagized Language Model (ILM)

![Python](https://img.shields.io/badge/python-3.10%2B-3776AB?logo=python&logoColor=white)
![Status](https://img.shields.io/badge/status-research-orange)
![Focus](https://img.shields.io/badge/focus-text--as--image-0A7EA4)
![Diffusion](https://img.shields.io/badge/paradigm-diffusion%20%2B%20glyphs-6A5ACD)
![License](https://img.shields.io/badge/license-unspecified-lightgrey)

ILM es una base de código de investigación que explora la generación de texto como imagen: codifica el lenguaje en tensores compactos, similares a imágenes, y genera texto mediante refinamiento iterativo de estilo difusión. La representación factoriza las oraciones en meta-elementos (gramática, semántica, tono, emoción) y en códigos jerárquicos tipo memoria para palabras y caracteres. Esto unifica ideas de difusión discreta, superposición/desentrelazado, embeddings estructurados y modelado de caracteres con conciencia de glifos.

## Descripción General

Actualmente, este repositorio incluye dos líneas prácticas principales:

1. Ingesta de etimología de glifos del chino histórico (scraping/parsing/almacenamiento/visualización).
2. Experimentos de modelado de glifos/imágenes de ILM (renderizado de glifos de tokens, codebooks de producto, empaquetado de frames, difusión/inpainting, evaluación/reportes).

El README actual de este repositorio se ha centrado históricamente en el toolkit de etimología. Ese flujo de trabajo sigue completamente documentado abajo y se conserva como canónico.

## Enlaces Clave

| Área | Ruta |
|---|---|
| Documento conceptual | `docs/imagized-language-model.md` |
| Plan de código y métricas | `docs/ilm-visual-diffusion-code-plan.md` |
| Plan de "color" de embeddings | `docs/embedding-color-plan.md` |
| Notas/plan de desarrollo | `docs/development-plan.md` |
| README del módulo de etimología | `ilm/etymology/README.md` |

## Funcionalidades

- 🏺 Ingesta etimológica desde fuentes tipo `hanziyuan` y `chineseetymology`.
- 🌐 Ruta de ingesta robusta AJAX + HTML con reintentos, limitación y caché.
- 🧩 Extracción de glifos etiquetada por etapa, incluyendo `<img>` y datos URI de `background-image` en CSS.
- 🗃️ Almacenamiento sobre SQLite para metadatos de caracteres/glifos y disposición de assets en filesystem.
- 🖥️ UI web con Tornado para ingesta ad-hoc y vista previa en galería.
- 🔤 Utilidades de renderizado de glifos para imágenes de tokens multilingües.
- 🧠 Módulos de embedding/codebook estilo product-code.
- 🧱 Scripts de entrenamiento/evaluación para empaquetado de frames de oraciones y difusión/inpainting.
- 📊 Scripts de reportes y visualización para inspección de embeddings y del pipeline.
- 📄 Artefactos de publicación en LaTeX/PDF bajo `publication/`.

## Estructura del Proyecto

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

## Requisitos Previos

| Requisito | Notas |
|---|---|
| Python `3.10+` | Runtime principal |
| `pip` | Instalación de paquetes |
| GPU opcional | Útil para scripts de entrenamiento CUDA con PyTorch |
| Toolchain LaTeX opcional | Necesario para builds de publicación |

Nota de supuesto: actualmente no existe un único archivo raíz de bloqueo/especificación de dependencias (`pyproject.toml`, `requirements.txt`, etc.), por lo que las dependencias se infieren de los imports y del uso de scripts.

## Instalación

### Mínima (toolkit de etimología)

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install requests beautifulsoup4 tornado
```

### Extendida (flujos de modelado/entrenamiento)

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install requests beautifulsoup4 tornado pyyaml numpy pillow matplotlib torch
```

Si un script específico necesita paquetes adicionales, instálalos según el error de importación que muestre ese script.

## Uso

### Inicio Rápido: Ingesta de Glifos Históricos (CLI)

1. Hanziyuan (recomendado): flujo AJAX solo por carácter

```bash
PYTHONPATH=. python scripts/ingest_etymology.py --site hanziyuan --char 中
```

2. ChineseEtymology (URL directa)

```bash
PYTHONPATH=. python scripts/ingest_etymology.py --site chineseetymology --url "https://www.chineseetymology.org/CharacterEtymology.aspx?characterInput=%E4%B8%AD"
```

3. Ingesta por archivo por lotes (las líneas pueden ser `char\turl`, `url` o `char url`)

```bash
PYTHONPATH=. python scripts/ingest_etymology.py --from-file urls.txt
```

### Salidas

| Tipo de salida | Ubicación |
|---|---|
| Archivos | `data/historic/glyphs/<char>/<stage>/<label>.<ext>` |
| Caché | `data/historic/cache/*.html` |
| DB | `data/historic/etymology.sqlite3` |

### Demo Web (opcional)

```bash
PYTHONPATH=. python scripts/serve_etymology.py
```

Abre `http://127.0.0.1:8888`, elige el sitio e introduce un carácter (por ejemplo `中`).

### Crawling Responsable y Respeto del Sitio

- El fetcher usa limitación por host, reintentos con backoff y caché.
- Mantén demoras `>= 0.5s`, evita ráfagas y respeta términos/robots/licencias del sitio.
- No eludas paywalls ni protecciones interactivas.
- Si ves `403`/`429`, reduce la velocidad y reintenta más tarde.

### Flujos de Trabajo ILM Adicionales

Estos scripts existen y forman parte activa de la superficie del repositorio, pero son flujos de investigación y pueden requerir datasets/checkpoints locales preparados.

1. Descarga/preparación de datos

```bash
python scripts/download_alpaca.py --outdir data/raw
python scripts/download_corpora.py --out data/raw
python scripts/sample_paragraphs.py --out data/processed/test_100.jsonl
python scripts/build_images_common_freq.py --out data/processed/images_common_freq --size 128 --en 5000 --zh 5000
```

2. Ciclo de vida de Glyph DB

```bash
python scripts/glyphdb_init.py --db data/glyphdb/glyphs.sqlite3
python scripts/glyphdb_ingest_index.py --db data/glyphdb/glyphs.sqlite3 --index data/processed/images_common_freq/index.tsv
```

3. Entrenamiento de modelos de código/color

```bash
python scripts/train_color_codes.py --config configs/color.yaml
python scripts/train_codes_from_qa.py --en-json data/raw/alpaca_en.json --zh-json data/raw/alpaca_zh.json --epochs 1
python scripts/train_ilmglyph_codes.py --en data/raw/alpaca_en.json --zh data/raw/alpaca_zh.json --out artifacts/ilm_glyph_train
```

4. Difusión/inpainting

```bash
python scripts/train_diffusion.py --config configs/diffusion.yaml
python scripts/train_inpaint_frames.py --ckpt-code artifacts/ilm_glyph_train/ckpt_epoch1.pt --out artifacts/inpaint
```

5. Evaluación/reportes

```bash
python scripts/eval_color_codes.py --checkpoint artifacts/color_codes_e1.pt
python scripts/eval_diffusion.py --checkpoint artifacts/diffusion_unet.pt
python scripts/eval_qa_retrieval.py --checkpoint artifacts/color_codes_qa.pt
python scripts/report_ilmglyph_pipeline.py --ckpt artifacts/ilm_glyph_train/ckpt_epoch1.pt --lang en --text "hello world"
```

## Configuración

Configs YAML principales:

- `configs/color.yaml`
  - ruta de datos: `data/processed/images_common_freq/index.tsv`
  - parámetros de modelo/código: `d_glyph`, `d_code`, `K`, `C`, temperature/anneal
  - ajustes de optimizador/log

- `configs/diffusion.yaml`
  - JSONL de entrada: `data/processed/test_100.jsonl`
  - ajustes de tamaño de frame/grid + modelo
  - rango de máscara de entrenamiento y ajustes de checkpoints

Sobrescribe ajustes vía flags de CLI donde estén soportados (`--epochs`, `--batch-size`, `--lr`, etc.).

## Ejemplos

- Construir un solo glifo tile en inglés:

```bash
python scripts/build_english_tile_glyph.py "language" artifacts/language_tile --save-tensor
```

- Ejecutar demo de inpainting con checkpoints entrenados:

```bash
python scripts/inpaint_demo.py \
  --ckpt-code artifacts/ilm_glyph_train/ckpt_epoch1.pt \
  --ckpt-inpaint artifacts/inpaint/ckpt_epoch1.pt \
  --lang en \
  --text "the quick brown fox jumps" \
  --mode infill \
  --out artifacts/inpaint_demo
```

- Ingesta masiva de caracteres comunes desde Hanziyuan:

```bash
PYTHONPATH=. python scripts/bulk_ingest_hanziyuan.py --limit 200 --resume
```

## Notas de Desarrollo

- Este es un repositorio de investigación con CLIs robustas y artefactos exploratorios (incluyendo notebooks y scripts prototipo).
- Los archivos grandes generados están pensados para `data/` y `artifacts/` (ambos ignorados en `.gitignore`).
- El código fuente y los PDFs de publicación están en `publication/`; script auxiliar de build: `scripts/latex_build.sh`.
- Las convenciones de colaboración/proceso están documentadas en `AGENTS.md`.

## Resolución de Problemas

- `ModuleNotFoundError: ilm...`
  - Ejecuta los scripts desde la raíz del repositorio.
  - Usa `PYTHONPATH=.` para scripts que esperan resolución de paquetes locales.

- `FileNotFoundError` para data/index/checkpoints
  - Ejecuta primero los scripts previos de datos/build.
  - Confirma que existan los valores por defecto como `data/processed/images_common_freq/index.tsv` y `data/processed/test_100.jsonl`.

- Problemas de CUDA/dispositivo
  - Cambia a CPU con flags/config del script (`device: cpu` o `--device cpu`).

- Errores por paquetes faltantes
  - Instala la dependencia requerida desde la ruta de import del script específico (`torch`, `pyyaml`, `Pillow`, etc.).

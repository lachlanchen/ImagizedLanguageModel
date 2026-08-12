[English](../README.md) · [العربية](README.ar.md) · [Español](README.es.md) · [Français](README.fr.md) · [日本語](README.ja.md) · [한국어](README.ko.md) · [Tiếng Việt](README.vi.md) · [中文 (简体)](README.zh-Hans.md) · [中文（繁體）](README.zh-Hant.md) · [Deutsch](README.de.md) · [Русский](README.ru.md)


[![LazyingArt banner](https://github.com/lachlanchen/lachlanchen/raw/main/figs/banner.png)](https://github.com/lachlanchen/lachlanchen/blob/main/figs/banner.png)

# Modelo de Lenguaje Imaginizado (ILM)

![Python](https://img.shields.io/badge/python-3.10%2B-3776AB?logo=python&logoColor=white)
![Status](https://img.shields.io/badge/status-research-orange)
![Focus](https://img.shields.io/badge/focus-text--as--image-0A7EA4)
![Paradigm](https://img.shields.io/badge/paradigm-predictive%20visual%20field-16835B)
![License](https://img.shields.io/badge/license-unspecified-lightgrey)
![Domain](https://img.shields.io/badge/domain-historic%20etymology%20%7C%20glyph%20models-2F80ED?logo=github)

ILM investiga cómo aprender y generar lenguaje como **escritura visible**: de
imagen a estado visual continuo y de nuevo a imagen, sin un canal simbólico
oculto.

## Última prueba de instrucciones: V23 aprueba el circuito de relación visual

![Resultado medido de V23: seis imágenes de instrucción pasan por comparación visual, compuerta de operación, enrutamiento del glifo fuente y un normalizador de imagen congelado para producir una imagen de respuesta](../publication/ilm-image-native/figures/visual_relation_circuit_v23_result.png)

V23 es el primer experimento de este repositorio que supera toda la cadena de
evidencia de instrucción en imagen a respuesta en imagen. El modelo recibe solo
seis imágenes de escritura `32x32` y produce una respuesta `32x32`. Su ruta de
inferencia no contiene cadenas, tokens, ID Unicode, OCR, consulta de glifos,
índice de respuesta ni un modelo de lenguaje externo.

En la única evaluación congelada autorizada, con 98 caracteres chinos no
vistos, 1.024 episodios y 4.096 variantes de instrucción, alcanza `0.99829` en
elección binaria, `0.99609` al cambiar la consulta, `0.99707` al cambiar la
operación, `0.99463` top-1 del glifo generado y `0.78478` F1 de píxel. Los
controles ciegos a consulta y operación permanecen exactamente invariantes al
factor que no pueden ver.

Este resultado solo demuestra seguimiento visual para una gramática fija de
seis roles, dos asociaciones y relación same/other; no demuestra comprensión
libre del lenguaje. V24 eliminará los roles fijos, leerá un flujo 2D de
escritura de longitud variable y generará un segundo cuadro después de releer
el primero. Véase el [informe V23 en inglés](../docs/visual-relation-circuit-v23-result.md).

## Referencia de investigación anterior: origen del campo visual predictivo

![Diagrama del campo visual predictivo](../publication/ilm-image-native/figures/predictive_visual_field_paradigm.png)

El experimento RFLM V7 mostró que un único flujo de píxeles no debe inferir la
identidad lingüística y dibujar los trazos en la misma operación. V7 elevó el
top-1 de contexto completo de `1,20%` a `2,31%`, por encima de last-only
(`2,02%`) y unigram (`1,86%`), pero lejos de bigram (`13,58%`). La ganancia
normalizada de log-probabilidad mejoró de `-0,9066` a `-0,2155`, pero siguió
siendo negativa, y la salida autónoma continuó ilegible. Por eso **V7 queda
rechazado como modelo de lenguaje**.

El siguiente PVF separa un flujo continuo que predice el próximo estado retinal
de un actuador visual que lo renderiza como tinta y lo vuelve a leer. No hay
búsqueda del carácter más cercano ni tabla de salida. El límite sigue siendo
`píxeles de escritura -> dinámica visual continua -> píxeles de tinta`, sin
tokens, Unicode IDs, OCR, codebook visual ni modelo externo. PVF es una hipótesis
V8 falsable, no una capacidad demostrada.

> El repositorio mantiene de forma deliberada un pipeline práctico de etimología y experimentación ILM a largo plazo lado a lado.

## 📌 Visión general

Este repositorio tiene tres líneas conectadas:

1. Modelado de lenguaje nativo de imagen con flujo retinal y evaluación estricta.
2. Ingesta etimológica de glifos chinos con procedencia conservada.
3. Líneas base anteriores de glifos, codebooks, difusión, folio e InkStream para reproducibilidad.

Este README documenta las tres líneas y conserva el flujo de etimología como una ruta reproducible de primera clase.

## 🔗 Enlaces clave

| Área | Ruta |
|---|---|
| Documento conceptual | `docs/imagized-language-model.md` |
| Objetivo de ingeniería actual | `docs/first-imagized-language-model-goal.md` |
| Dossier de investigación y evidencia | `references/image-native-language-model-research.md` |
| Plan de código y métricas | `docs/ilm-visual-diffusion-code-plan.md` |
| Plan de "color" de embeddings | `docs/embedding-color-plan.md` |
| Notas/plan de desarrollo | `docs/development-plan.md` |
| README del módulo de etimología | `ilm/etymology/README.md` |

## ✨ Características

- 🏺 Ingesta de etimología desde fuentes tipo `hanziyuan` y `chineseetymology`.
- 🌐 Ruta de ingesta robusta AJAX + HTML con reintentos, control de ritmo y caché.
- 🧩 Extracción de glifos marcada por etapas, incluyendo `<img>` y datos URI de `background-image` en CSS.
- 🗃️ Almacenamiento en SQLite para metadatos de caracteres/glifos y organización de recursos en el sistema de archivos.
- 🖥️ UI web con Tornado para ingesta ad hoc y vista previa de galería.
- 🔤 Utilidades de renderizado de glifos para imágenes de tokens multilingües.
- 🧠 Módulos de embedding/codebook con estilo de código de producto.
- 🧱 Scripts de entrenamiento/evaluación de empaquetado de frames de frases y difusión/inpainting.
- 📊 Scripts de reporte y visualización para inspección de embeddings y del pipeline.
- 📄 Artefactos de publicación en LaTeX/PDF en `publication/`.

## 🧱 Estructura del proyecto

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

## 🧰 Requisitos previos

| Requisito | Notas |
|---|---|
| Python `3.10+` | Runtime principal |
| `pip` | Instalación de paquetes |
| GPU opcional | Útil para scripts de entrenamiento CUDA de PyTorch |
| Cadena de herramientas LaTeX opcional | Necesaria para builds de publicación |

Nota de supuestos: actualmente no existe un único archivo raíz de dependencia bloqueada (`pyproject.toml`, `requirements.txt`, etc.), por lo que las dependencias se infieren de los imports y del uso en scripts.

## ⚙️ Instalación

### Mínima (herramientas de etimología)

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

Si un script concreto requiere paquetes adicionales, instálalos según el error de importación mostrado por ese script.

## 🚀 Uso

### Inicio rápido: ingesta de glifos históricos (CLI)

1. Hanziyuan (recomendado): flujo AJAX solo por carácter

```bash
PYTHONPATH=. python scripts/ingest_etymology.py --site hanziyuan --char 中
```

2. ChineseEtymology (URL directa)

```bash
PYTHONPATH=. python scripts/ingest_etymology.py --site chineseetymology --url "https://www.chineseetymology.org/CharacterEtymology.aspx?characterInput=%E4%B8%AD"
```

3. Ingesta por lotes desde archivo (las líneas pueden ser `char\turl`, `url` o `char url`)

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

### Crawling cortés y respeto al sitio

- El colector usa control de ritmo por host, reintentos con backoff y caché.
- Mantén retrasos `>= 0.5s`, evita ráfagas y respeta términos/robots/licencias del sitio.
- No eludas muros de pago ni protecciones interactivas.
- Si ves `403`/`429`, reduce la frecuencia y vuelve a intentarlo más tarde.

### Flujos adicionales de ILM

Estos scripts existen y forman parte activa del repositorio, pero son flujos de investigación y pueden requerir datasets/checkpoints locales preparados.

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

3. Entrenamiento de modelos de códigos/color

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

## 🧩 Configuración

Configuración YAML principal:

- `configs/color.yaml`
  - ruta de datos: `data/processed/images_common_freq/index.tsv`
  - parámetros de modelo/código: `d_glyph`, `d_code`, `K`, `C`, temperature/anneal
  - configuración de optimizador/log

- `configs/diffusion.yaml`
  - JSONL de entrada: `data/processed/test_100.jsonl`
  - ajustes de marco/cuadrícula + tamaño del modelo
  - rango de ratio de máscara de entrenamiento y configuración de checkpoints

Sobrescribe la configuración mediante flags CLI donde se soporte (`--epochs`, `--batch-size`, `--lr`, etc.).

## 🧪 Ejemplos

- Construir un glifo de baldosa en inglés único:

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

## 📝 Notas de desarrollo

- Este es un repositorio de investigación con CLIs robustos y artefactos exploratorios (incluyendo notebooks y scripts prototipo).
- Los archivos grandes generados están destinados a `data/` y `artifacts/` (ambos ignorados por `.gitignore`).
- El código fuente y los PDF de publicación están en `publication/`; script auxiliar de compilación: `scripts/latex_build.sh`.
- Las convenciones de colaboración y proceso están documentadas en `AGENTS.md`.

## 🛠️ Solución de problemas

- `ModuleNotFoundError: ilm...`
  - Ejecuta los scripts desde la raíz del repositorio.
  - Usa `PYTHONPATH=.` para scripts que esperan resolver paquetes locales.

- `FileNotFoundError` para data/index/checkpoints
  - Ejecuta primero los scripts de datos/compilación necesarios.
  - Confirma que existan los valores por defecto como `data/processed/images_common_freq/index.tsv` y `data/processed/test_100.jsonl`.

- Problemas de CUDA/dispositivo
  - Cambia a CPU con flags/config del script (`device: cpu` o `--device cpu`).

- Errores por dependencias faltantes
  - Instala la dependencia requerida según la ruta de import del script concreto (`torch`, `pyyaml`, `Pillow`, `etc.`).

- HTTP `403` / `429` durante scraping
  - Incrementa `--delay`, reintenta más tarde y mantén solicitudes corteses.

## 🗺️ Hoja de ruta

- Seguir madurando los runbooks de entrenamiento/evaluación de ILM text-as-image más allá del inicio rápido centrado en etimología.
- Mejorar la reproducibilidad del entorno (especificación única y autoritativa de dependencias).
- Expandir pruebas/CI para scripts de investigación y lógica de pipeline.
- Iterar en codebooks jerárquicos, objetivos de difusión y canales de controlabilidad.
- Consolidar la documentación entre `docs/`, textos de ayuda de scripts y artefactos de publicación.

Para obtener detalles conceptuales y de planificación por etapas más profundos, consulta:

- `docs/imagized-language-model.md`
- `docs/ilm-visual-diffusion-code-plan.md`
- `docs/development-plan.md`

## 🤝 Contribución

- Sigue `AGENTS.md` para las convenciones (commits atómicos, push tras cambios, sin credenciales en el código).
- Agrupa cambios relacionados en commits enfocados con mensajes convencionales.
- Prefiere invocaciones de scripts reproducibles con flags e inputs explícitos.
- Para cambios relacionados con scraping, conserva el comportamiento de throttling/caché y las restricciones de respeto al sitio.

## 📄 Licencia

No existe actualmente un archivo de licencia de nivel superior en este repositorio.

Nota de supuestos: trata el proyecto como código de investigación con licencia no especificada hasta que los mantenedores añadan un archivo `LICENSE`.


## ❤️ Support

| Donate | PayPal | Stripe |
| --- | --- | --- |
| [![Donate](https://camo.githubusercontent.com/24a4914f0b42c6f435f9e101621f1e52535b02c225764b2f6cc99416926004b7/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f446f6e6174652d4c617a79696e674172742d3045413545393f7374796c653d666f722d7468652d6261646765266c6f676f3d6b6f2d6669266c6f676f436f6c6f723d7768697465)](https://chat.lazying.art/donate) | [![PayPal](https://camo.githubusercontent.com/d0f57e8b016517a4b06961b24d0ca87d62fdba16e18bbdb6aba28e978dc0ea21/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f50617950616c2d526f6e677a686f754368656e2d3030343537433f7374796c653d666f722d7468652d6261646765266c6f676f3d70617970616c266c6f676f436f6c6f723d7768697465)](https://paypal.me/RongzhouChen) | [![Stripe](https://camo.githubusercontent.com/1152dfe04b6943afe3a8d2953676749603fb9f95e24088c92c97a01a897b4942/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f5374726970652d446f6e6174652d3633354246463f7374796c653d666f722d7468652d6261646765266c6f676f3d737472697065266c6f676f436f6c6f723d7768697465)](https://buy.stripe.com/aFadR8gIaflgfQV6T4fw400) |

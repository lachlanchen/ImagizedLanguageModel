Imagized Language Model (ILM)
=============================

Idiomas
- [English](../README.md) | [简体中文](README.zh-Hans.md) | [繁體中文](README.zh-Hant.md) | [日本語](README.ja.md) | [한국어](README.ko.md) | [Tiếng Việt](README.vi.md) | [العربية](README.ar.md) | [Français](README.fr.md) | Español

Resumen
ILM representa el texto como tensores «similares a una imagen» y genera texto mediante un proceso iterativo de eliminación de ruido (difusión). Una oración se descompone en meta‑elementos controlables (gramática, semántica, tono, emoción) y códigos jerárquicos «tipo memoria» (palabras/glifos). Combina difusión discreta, superposición/separación de rasgos, embeddings estructurados y conciencia de glifos.

Enlaces clave
- Concepto: docs/imagized-language-model.md
- Plan de código y métricas: docs/ilm-visual-diffusion-code-plan.md
- Plan de «color» de embedding: docs/embedding-color-plan.md

Contenido del repositorio
- ilm/etymology/: utilidades para recopilar glifos históricos (oráculo/bronce/sello, etc.)
  - Obtención AJAX desde hanziyuan (reintentos, limitación de tasa, caché)
  - Análisis HTML/CSS para extraer imágenes etiquetadas por etapa (data URI/URL)
- scripts/
  - ingest_etymology.py: CLI para ingerir → SQLite y archivos
  - serve_etymology.py: mini UI con Tornado para previsualizar
  - use_historic_tools.md: notas sobre datos/herramientas externas
- data/ (ignorado): caché HTML, imágenes, base SQLite

Inicio rápido
- Dependencias: `pip install requests beautifulsoup4 tornado`
- Ejemplo (recomendado: AJAX hanziyuan):
  - `PYTHONPATH=. python scripts/ingest_etymology.py --site hanziyuan --char 中`
- Demo web:
  - `PYTHONPATH=. python scripts/serve_etymology.py` → http://127.0.0.1:8888

Salidas
- Imágenes: data/historic/glyphs/<caracter>/<etapa>/<etiqueta>.<ext>
- Caché: data/historic/cache
- BD: data/historic/etymology.sqlite3

Rastreo respetuoso
- Limitación por host, reintentos con backoff y caché activados
- Respete términos/licencias. Si recibe 403/429, reduzca la tasa y vuelva a intentar

Objetivo
- Modelo estructurado y controlable, multilingüe, apto para ordenadores comunes

Contribución
- Siga AGENTS.md (commits atómicos, push tras cada cambio, no incluir credenciales)


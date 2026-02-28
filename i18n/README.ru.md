[English](../README.md) · [العربية](README.ar.md) · [Español](README.es.md) · [Français](README.fr.md) · [日本語](README.ja.md) · [한국어](README.ko.md) · [Tiếng Việt](README.vi.md) · [中文 (简体)](README.zh-Hans.md) · [中文（繁體）](README.zh-Hant.md) · [Deutsch](README.de.md) · [Русский](README.ru.md)

# Imagized Language Model (ILM)

![Python](https://img.shields.io/badge/python-3.10%2B-3776AB?logo=python&logoColor=white)
![Status](https://img.shields.io/badge/status-research-orange)
![Focus](https://img.shields.io/badge/focus-text--as--image-0A7EA4)
![Diffusion](https://img.shields.io/badge/paradigm-diffusion%20%2B%20glyphs-6A5ACD)
![License](https://img.shields.io/badge/license-unspecified-lightgrey)

ILM — это исследовательская кодовая база, изучающая генерацию текста как изображения: язык кодируется в компактные тензоры, похожие на изображения, а текст генерируется через итеративное уточнение в стиле diffusion. Представление раскладывает предложения на мета-элементы (грамматика, семантика, тон, эмоция) и иерархические «память-подобные» коды для слов и символов. Это объединяет идеи из discrete diffusion, superposition/disentanglement, структурированных эмбеддингов и glyph-aware моделирования символов.

## Обзор

Сейчас репозиторий включает два крупных практических направления:

1. Инжест исторической этимологии китайских иероглифов (скрейпинг/парсинг/хранение/предпросмотр).
2. Эксперименты ILM по glyph/image-моделированию (рендеринг глифов токенов, product codebooks, упаковка фреймов, diffusion/inpainting, оценка/отчётность).

Текущий README в этом репозитории исторически сосредоточен на toolkit для этимологии. Этот workflow по-прежнему полностью задокументирован ниже и сохранён как канонический.

## Ключевые ссылки

| Область | Путь |
|---|---|
| Концептуальное описание | `docs/imagized-language-model.md` |
| План кода и метрик | `docs/ilm-visual-diffusion-code-plan.md` |
| План «цвета» эмбеддингов | `docs/embedding-color-plan.md` |
| Заметки/план разработки | `docs/development-plan.md` |
| README модуля этимологии | `ilm/etymology/README.md` |

## Возможности

- 🏺 Инжест этимологии из источников стиля `hanziyuan` и `chineseetymology`.
- 🌐 Надёжный путь инжеста через AJAX + HTML с ретраями, троттлингом и кэшем.
- 🧩 Извлечение глифов с маркировкой стадий, включая `<img>` и data URI из CSS `background-image`.
- 🗃️ Хранение в SQLite для метаданных символов/глифов плюс файловая раскладка ассетов.
- 🖥️ Веб-интерфейс Tornado для ad-hoc инжеста и предпросмотра галереи.
- 🔤 Утилиты рендеринга глифов для многоязычных изображений токенов.
- 🧠 Модули эмбеддингов/codebook в стиле product-code.
- 🧱 Скрипты упаковки фреймов предложений и обучения/оценки diffusion/inpainting.
- 📊 Скрипты отчётности и визуализации для инспекции эмбеддингов и pipeline.
- 📄 Артефакты публикации в LaTeX/PDF в `publication/`.

## Структура проекта

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

## Предварительные требования

| Требование | Примечания |
|---|---|
| Python `3.10+` | Основной runtime |
| `pip` | Установка пакетов |
| Опционально GPU | Полезно для скриптов обучения PyTorch CUDA |
| Опционально toolchain LaTeX | Нужен для сборок публикации |

Примечание-предположение: в корне сейчас нет единого lock/spec файла зависимостей (`pyproject.toml`, `requirements.txt` и т. п.), поэтому зависимости выводятся из импортов и использования скриптов.

## Установка

### Минимальная (toolkit этимологии)

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install requests beautifulsoup4 tornado
```

### Расширенная (workflow моделирования/обучения)

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install requests beautifulsoup4 tornado pyyaml numpy pillow matplotlib torch
```

Если конкретному скрипту нужны дополнительные пакеты, установите их по ошибке импорта, которую выводит этот скрипт.

## Использование

### Быстрый старт: инжест исторических глифов (CLI)

1. Hanziyuan (рекомендуется): AJAX-поток только по символу

```bash
PYTHONPATH=. python scripts/ingest_etymology.py --site hanziyuan --char 中
```

2. ChineseEtymology (прямой URL)

```bash
PYTHONPATH=. python scripts/ingest_etymology.py --site chineseetymology --url "https://www.chineseetymology.org/CharacterEtymology.aspx?characterInput=%E4%B8%AD"
```

3. Инжест из batch-файла (строки могут быть `char\turl`, `url` или `char url`)

```bash
PYTHONPATH=. python scripts/ingest_etymology.py --from-file urls.txt
```

### Выходные данные

| Тип вывода | Расположение |
|---|---|
| Файлы | `data/historic/glyphs/<char>/<stage>/<label>.<ext>` |
| Кэш | `data/historic/cache/*.html` |
| БД | `data/historic/etymology.sqlite3` |

### Веб-демо (опционально)

```bash
PYTHONPATH=. python scripts/serve_etymology.py
```

Откройте `http://127.0.0.1:8888`, выберите сайт и введите символ (например, `中`).

### Вежливый краулинг и уважение к сайтам

- Fetcher использует троттлинг по хостам, ретраи с backoff и кэширование.
- Держите задержки `>= 0.5s`, избегайте всплесков и соблюдайте условия сайтов/robots/licensing.
- Не обходите paywall и интерактивные защиты.
- Если видите `403`/`429`, снизьте скорость и повторите позже.

### Дополнительные workflow ILM

Эти скрипты существуют и активно входят в поверхность репозитория, но это исследовательские workflow и им могут требоваться подготовленные локальные датасеты/чекпойнты.

1. Загрузка/подготовка данных

```bash
python scripts/download_alpaca.py --outdir data/raw
python scripts/download_corpora.py --out data/raw
python scripts/sample_paragraphs.py --out data/processed/test_100.jsonl
python scripts/build_images_common_freq.py --out data/processed/images_common_freq --size 128 --en 5000 --zh 5000
```

2. Жизненный цикл Glyph DB

```bash
python scripts/glyphdb_init.py --db data/glyphdb/glyphs.sqlite3
python scripts/glyphdb_ingest_index.py --db data/glyphdb/glyphs.sqlite3 --index data/processed/images_common_freq/index.tsv
```

3. Обучение code/color моделей

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

5. Оценка/отчётность

```bash
python scripts/eval_color_codes.py --checkpoint artifacts/color_codes_e1.pt
python scripts/eval_diffusion.py --checkpoint artifacts/diffusion_unet.pt
python scripts/eval_qa_retrieval.py --checkpoint artifacts/color_codes_qa.pt
python scripts/report_ilmglyph_pipeline.py --ckpt artifacts/ilm_glyph_train/ckpt_epoch1.pt --lang en --text "hello world"
```

## Конфигурация

Основные YAML-конфиги:

- `configs/color.yaml`
  - путь к данным: `data/processed/images_common_freq/index.tsv`
  - параметры модели/кода: `d_glyph`, `d_code`, `K`, `C`, temperature/anneal
  - настройки optimizer/log

- `configs/diffusion.yaml`
  - входной JSONL: `data/processed/test_100.jsonl`
  - настройки размера frame/grid + model
  - диапазон train mask ratio и настройки чекпойнтов

Переопределяйте настройки через CLI-флаги там, где это поддерживается (`--epochs`, `--batch-size`, `--lr` и т. д.).

## Примеры

- Собрать один English tile glyph:

```bash
python scripts/build_english_tile_glyph.py "language" artifacts/language_tile --save-tensor
```

- Запустить демо inpainting с обученными чекпойнтами:

```bash
python scripts/inpaint_demo.py \
  --ckpt-code artifacts/ilm_glyph_train/ckpt_epoch1.pt \
  --ckpt-inpaint artifacts/inpaint/ckpt_epoch1.pt \
  --lang en \
  --text "the quick brown fox jumps" \
  --mode infill \
  --out artifacts/inpaint_demo
```

- Массово инжестить частые символы из Hanziyuan:

```bash
PYTHONPATH=. python scripts/bulk_ingest_hanziyuan.py --limit 200 --resume
```

## Заметки по разработке

- Это исследовательский репозиторий с надёжными CLI и исследовательскими артефактами (включая ноутбуки и прототипные скрипты).
- Сгенерированные крупные файлы предназначены для `data/` и `artifacts/` (оба игнорируются в `.gitignore`).
- Исходники публикации и PDF находятся в `publication/`; вспомогательный скрипт сборки: `scripts/latex_build.sh`.
- Конвенции совместной работы/процесса задокументированы в `AGENTS.md`.

## Решение проблем

- `ModuleNotFoundError: ilm...`
  - Запускайте скрипты из корня репозитория.
  - Используйте `PYTHONPATH=.` для скриптов, которым нужно локальное разрешение пакета.

- `FileNotFoundError` для data/index/checkpoints
  - Сначала запустите prerequisite-скрипты подготовки/сборки данных.
  - Убедитесь, что существуют пути по умолчанию, например `data/processed/images_common_freq/index.tsv` и `data/processed/test_100.jsonl`.

- Проблемы CUDA/device
  - Переключитесь на CPU через флаги/конфиг скрипта (`device: cpu` или `--device cpu`).

- Ошибки отсутствующих пакетов
  - Установите нужную зависимость из import path конкретного скрипта (`torch`, `pyyaml`, `Pillow` и т. д.).

- HTTP `403` / `429` при скрейпинге
  - Увеличьте `--delay`, повторите позже и сохраняйте вежливую частоту запросов.

## Дорожная карта

- Продолжить развивать runbook обучения/оценки text-as-image ILM за рамки быстрого старта, ориентированного на этимологию.
- Улучшить воспроизводимость окружения (единая авторитетная спецификация зависимостей).
- Расширить покрытие тестами/CI для исследовательских скриптов и связки pipeline.
- Итерировать иерархические codebook, diffusion objectives и каналы управляемости.
- Консолидировать документацию между `docs/`, help-текстами скриптов и артефактами публикации.

Для более глубоких концептуальных и поэтапных деталей планирования см.:

- `docs/imagized-language-model.md`
- `docs/ilm-visual-diffusion-code-plan.md`
- `docs/development-plan.md`

## Вклад

- Следуйте `AGENTS.md` для конвенций (атомарные коммиты, push после изменений, без credentials в коде).
- Группируйте связанные правки в сфокусированные коммиты с conventional-сообщениями.
- Предпочитайте воспроизводимые вызовы скриптов с явными флагами и входными путями.
- Для изменений, связанных со скрейпингом, сохраняйте поведение троттлинга/кэша и ограничения уважительного доступа к сайтам.

## Лицензия

В корне этого репозитория сейчас отсутствует файл лицензии верхнего уровня.

Примечание-предположение: рассматривайте проект как исследовательский код с неуточнённым лицензированием, пока сопровождающие не добавят файл `LICENSE`.

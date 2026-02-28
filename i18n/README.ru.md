[English](../README.md) · [العربية](README.ar.md) · [Español](README.es.md) · [Français](README.fr.md) · [日本語](README.ja.md) · [한국어](README.ko.md) · [Tiếng Việt](README.vi.md) · [中文 (简体)](README.zh-Hans.md) · [中文（繁體）](README.zh-Hant.md) · [Deutsch](README.de.md) · [Русский](README.ru.md)


[![LazyingArt banner](https://github.com/lachlanchen/lachlanchen/raw/main/figs/banner.png)](https://github.com/lachlanchen/lachlanchen/blob/main/figs/banner.png)

# Imagized Language Model (ILM)

![Python](https://img.shields.io/badge/python-3.10%2B-3776AB?logo=python&logoColor=white)
![Status](https://img.shields.io/badge/status-research-orange)
![Focus](https://img.shields.io/badge/focus-text--as--image-0A7EA4)
![Diffusion](https://img.shields.io/badge/paradigm-diffusion%20%2B%20glyphs-6A5ACD)
![License](https://img.shields.io/badge/license-unspecified-lightgrey)
![Domain](https://img.shields.io/badge/domain-historic%20etymology%20%7C%20glyph%20models-2F80ED?logo=github)

ILM — это исследовательская кодовая база для **генерации текста как изображения**: она кодирует язык в компактные тензоры, похожие на изображения, и генерирует текст с итеративным уточнением в стиле диффузионных моделей. Представление раскладывает предложения на мета-элементы (грамматика, семантика, тон, эмоция) и иерархические, память-подобные коды для слов и символов. Это объединяет идеи дискретной диффузии, суперпозиции/разделения факторов, структурированных эмбеддингов и glyph-aware моделирования символов.

> Репозиторий сознательно держит практичный пайплайн этимологии и долгосрочные ILM-эксперименты рядом друг с другом.

## 📌 Обзор

В этом репозитории есть два активных направления:

1. Загрузка исторической этимологии китайских иероглифов (scraping/parsing/storage/preview).
2. Эксперименты по glyph/image-моделированию в ILM (рендеринг глифов токенов, codebooks, упаковка фреймов, diffusion/inpainting, оценка и отчётность).

Этот README описывает оба направления и делает workflow по этимологии полноценным воспроизводимым потоком.

## 🔗 Ключевые ссылки

| Область | Путь |
|---|---|
| Концептуальный обзор | `docs/imagized-language-model.md` |
| План кода и метрик | `docs/ilm-visual-diffusion-code-plan.md` |
| План «color» эмбеддингов | `docs/embedding-color-plan.md` |
| Заметки/план разработки | `docs/development-plan.md` |
| Readme модуля этимологии | `ilm/etymology/README.md` |

## ✨ Возможности

- 🏺 Загрузка этимологии из источников формата `hanziyuan` и `chineseetymology`.
- 🌐 Надёжный путь AJAX + HTML ingestion с повторами, троттлингом и кэшем.
- 🧩 Извлечение глифов с пометками этапа, включая `<img>` и data URI из CSS `background-image`.
- 🗃️ Хранение метаданных символов/глифов на SQLite и файловая структура ассетов.
- 🖥️ Tornado UI для ad-hoc ingestion и предпросмотра галереи.
- 🔤 Утилиты рендеринга глифов для мультиязычных токен-изображений.
- 🧠 Embedding/codebook-модули в стиле product-code.
- 🧱 Упаковка фреймов предложений и скрипты обучения/оценки для diffusion/inpainting.
- 📊 Скрипты отчётности и визуализации для проверки эмбеддингов и конвейера.
- 📄 Артефакты публикаций в LaTeX/PDF в `publication/`.

## 🧱 Структура проекта

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

## 🧰 Требования

| Требование | Примечания |
|---|---|
| Python `3.10+` | Основная среда исполнения |
| `pip` | Установка пакетов |
| Опционально GPU | Полезно для скриптов обучения на PyTorch CUDA |
| Опционально LaTeX toolchain | Нужна для сборки публикаций |

Примечание: в корне пока нет единого файла с зафиксированными зависимостями (`pyproject.toml`, `requirements.txt` и т. д.), поэтому зависимости выводятся из импортов и фактического использования скриптов.

## ⚙️ Установка

### Минимальная (этимологический toolkit)

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

Если конкретному скрипту нужны дополнительные зависимости, установите их по сообщению об ошибке импорта этого скрипта.

## 🚀 Использование

### Быстрый старт: загрузка исторических глифов (CLI)

1. Hanziyuan (рекомендуется): поток только по символу через AJAX

```bash
PYTHONPATH=. python scripts/ingest_etymology.py --site hanziyuan --char 中
```

2. ChineseEtymology (прямая ссылка)

```bash
PYTHONPATH=. python scripts/ingest_etymology.py --site chineseetymology --url "https://www.chineseetymology.org/CharacterEtymology.aspx?characterInput=%E4%B8%AD"
```

3. Пакетная загрузка из файла (строки могут быть в форматах `char\turl`, `url` или `char url`)

```bash
PYTHONPATH=. python scripts/ingest_etymology.py --from-file urls.txt
```

### Выходные данные

| Тип результата | Местоположение |
|---|---|
| Файлы | `data/historic/glyphs/<char>/<stage>/<label>.<ext>` |
| Кэш | `data/historic/cache/*.html` |
| БД | `data/historic/etymology.sqlite3` |

### Web Demo (опционально)

```bash
PYTHONPATH=. python scripts/serve_etymology.py
```

Откройте `http://127.0.0.1:8888`, выберите сайт и введите символ (например, `中`).

### Вежливый краулинг и уважение к сайтам

- Fetcher использует лимитирование запросов по хосту, повторы с backoff и кэширование.
- Держите задержки `>= 0.5s`, избегайте всплесков и соблюдайте правила сайтов, robots и лицензии.
- Не обходите paywall и интерактивную защиту.
- Если встречаете `403`/`429`, снизьте частоту и повторите позже.

### Дополнительные ILM workflow

Эти скрипты есть в репозитории и активно используются, но это исследовательские пайплайны; им могут потребоваться подготовленные локальные датасеты и checkpoints.

1. Загрузка и подготовка данных

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

5. Оценка и отчётность

```bash
python scripts/eval_color_codes.py --checkpoint artifacts/color_codes_e1.pt
python scripts/eval_diffusion.py --checkpoint artifacts/diffusion_unet.pt
python scripts/eval_qa_retrieval.py --checkpoint artifacts/color_codes_qa.pt
python scripts/report_ilmglyph_pipeline.py --ckpt artifacts/ilm_glyph_train/ckpt_epoch1.pt --lang en --text "hello world"
```

## 🧩 Конфигурация

Основные конфиги YAML:

- `configs/color.yaml`
  - путь к данным: `data/processed/images_common_freq/index.tsv`
  - параметры модели/кодов: `d_glyph`, `d_code`, `K`, `C`, temperature/anneal
  - настройки оптимизатора и логов

- `configs/diffusion.yaml`
  - входной JSONL: `data/processed/test_100.jsonl`
  - настройки размера кадра/сетки и модели
  - диапазон train mask ratio и параметры чекпойнтов

Перезаписывайте настройки через CLI-флаги, где это поддерживается (`--epochs`, `--batch-size`, `--lr` и т. д.).

## 🧪 Примеры

- Соберите одну английскую tile-glyph:

```bash
python scripts/build_english_tile_glyph.py "language" artifacts/language_tile --save-tensor
```

- Запустите демонстрацию inpainting с обученными чекпойнтами:

```bash
python scripts/inpaint_demo.py \
  --ckpt-code artifacts/ilm_glyph_train/ckpt_epoch1.pt \
  --ckpt-inpaint artifacts/inpaint/ckpt_epoch1.pt \
  --lang en \
  --text "the quick brown fox jumps" \
  --mode infill \
  --out artifacts/inpaint_demo
```

- Пакетно загрузите частые символы из Hanziyuan:

```bash
PYTHONPATH=. python scripts/bulk_ingest_hanziyuan.py --limit 200 --resume
```

## 📝 Заметки по разработке

- Это исследовательский репозиторий с надежными CLI и экспериментальными артефактами (включая ноутбуки и прототипные скрипты).
- Большие сгенерированные файлы предназначены для `data/` и `artifacts/` (оба каталога обычно в `.gitignore`).
- Исходники публикаций и PDF лежат в `publication/`; вспомогательный скрипт сборки: `scripts/latex_build.sh`.
- Процессы и конвенции сотрудничества описаны в `AGENTS.md`.

## 🛠️ Устранение неполадок

- `ModuleNotFoundError: ilm...`
  - Запускайте скрипты из корня репозитория.
  - Используйте `PYTHONPATH=.` для скриптов, которым нужен локальный механизм разрешения пакетов.

- `FileNotFoundError` для data/index/checkpoints
  - Сначала выполните обязательные скрипты подготовки и сборки данных.
  - Проверьте, что существуют пути по умолчанию, например `data/processed/images_common_freq/index.tsv` и `data/processed/test_100.jsonl`.

- Проблемы с CUDA/устройством
  - Переключитесь на CPU через флаги/конфиги скрипта (`device: cpu` или `--device cpu`).

- Ошибки отсутствующих пакетов
  - Установите недостающую зависимость по конкретному пути импорта скрипта (`torch`, `pyyaml`, `Pillow` и др.).

- HTTP `403`/`429` при scraping
  - Увеличьте `--delay`, повторите позже и ведите запросы в вежливом режиме.

## 🗺️ Дорожная карта

- Доработать ILM text-as-image runbooks для обучения и оценки за пределами этнографического quick-start.
- Повысить воспроизводимость окружения (единый авторитетный список зависимостей).
- Расширить тесты и CI для исследовательских скриптов и логики пайплайна.
- Продолжить итерации по иерархическим codebooks, целям диффузии и каналам контролируемости.
- Консолидировать документацию между `docs/`, справкой скриптов и публикационными артефактами.

Для более глубоких концептуальных и пошаговых деталей см.:

- `docs/imagized-language-model.md`
- `docs/ilm-visual-diffusion-code-plan.md`
- `docs/development-plan.md`

## 🤝 Участие

- Следуйте `AGENTS.md` по конвенциям (атомарные коммиты, push после изменений, никаких учётных данных в коде).
- Группируйте связанные изменения в сфокусированные коммиты с конвенционными сообщениями.
- Предпочитайте воспроизводимые вызовы скриптов с явными флагами и входными путями.
- Для изменений, связанных со скрапингом, сохраняйте поведение троттлинга/кэша и соблюдение ограничений сайтов.

## 📄 Лицензия

На уровне верхнего каталога пока отсутствует файл лицензии.

Примечание: считаем проект исследовательским кодом с неуточненной лицензией до тех пор, пока мейнтейнеры не добавят `LICENSE`.


## ❤️ Support

| Donate | PayPal | Stripe |
| --- | --- | --- |
| [![Donate](https://camo.githubusercontent.com/24a4914f0b42c6f435f9e101621f1e52535b02c225764b2f6cc99416926004b7/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f446f6e6174652d4c617a79696e674172742d3045413545393f7374796c653d666f722d7468652d6261646765266c6f676f3d6b6f2d6669266c6f676f436f6c6f723d7768697465)](https://chat.lazying.art/donate) | [![PayPal](https://camo.githubusercontent.com/d0f57e8b016517a4b06961b24d0ca87d62fdba16e18bbdb6aba28e978dc0ea21/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f50617950616c2d526f6e677a686f754368656e2d3030343537433f7374796c653d666f722d7468652d6261646765266c6f676f3d70617970616c266c6f676f436f6c6f723d7768697465)](https://paypal.me/RongzhouChen) | [![Stripe](https://camo.githubusercontent.com/1152dfe04b6943afe3a8d2953676749603fb9f95e24088c92c97a01a897b4942/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f5374726970652d446f6e6174652d3633354246463f7374796c653d666f722d7468652d6261646765266c6f676f3d737472697065266c6f676f436f6c6f723d7768697465)](https://buy.stripe.com/aFadR8gIaflgfQV6T4fw400) |

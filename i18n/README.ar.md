[English](../README.md) · [العربية](README.ar.md) · [Español](README.es.md) · [Français](README.fr.md) · [日本語](README.ja.md) · [한국어](README.ko.md) · [Tiếng Việt](README.vi.md) · [中文 (简体)](README.zh-Hans.md) · [中文（繁體）](README.zh-Hant.md) · [Deutsch](README.de.md) · [Русский](README.ru.md)

# نموذج اللغة المُصوَّر (ILM)

![Python](https://img.shields.io/badge/python-3.10%2B-3776AB?logo=python&logoColor=white)
![Status](https://img.shields.io/badge/status-research-orange)
![Focus](https://img.shields.io/badge/focus-text--as--image-0A7EA4)
![Diffusion](https://img.shields.io/badge/paradigm-diffusion%20%2B%20glyphs-6A5ACD)
![License](https://img.shields.io/badge/license-unspecified-lightgrey)

ILM هو قاعدة شيفرة بحثية تستكشف توليد النص على هيئة صورة: إذ يرمّز اللغة إلى موترات مدمجة شبيهة بالصور، ثم يولّد النص عبر تنقيح تكراري بأسلوب الانتشار (diffusion). يفكك هذا التمثيل الجمل إلى عناصر فوقية (القواعد، الدلالة، النبرة، العاطفة) وإلى أكواد هرمية شبيهة بالذاكرة للكلمات والأحرف. بذلك يجمع بين أفكار الانتشار المتقطع، والتراكب/فك التشابك، والتضمينات المهيكلة، ونمذجة الأحرف الواعية بشكل الـ glyph.

## نظرة عامة

يتضمن هذا المستودع حاليًا مسارين عمليين رئيسيين:

1. إدخال اشتقاقيات/أصول الـ glyph الصيني التاريخي (الاستخلاص/التحليل/التخزين/المعاينة).
2. تجارب نمذجة ILM المعتمدة على glyph/الصورة (تصيير رموز glyph للتوكنات، دفاتر أكواد product codebooks، حزم الإطارات، الانتشار/الاستكمال inpainting، التقييم/التقارير).

ركز README الحالي تاريخيًا على أدوات الاشتقاقيات، وما يزال هذا المسار موثقًا بالكامل أدناه ومحفوظًا كمرجع أساسي.

## روابط أساسية

| المجال | المسار |
|---|---|
| الشرح المفاهيمي | `docs/imagized-language-model.md` |
| خطة الشيفرة والمقاييس | `docs/ilm-visual-diffusion-code-plan.md` |
| خطة "ألوان" التضمين | `docs/embedding-color-plan.md` |
| ملاحظات/خطة التطوير | `docs/development-plan.md` |
| README وحدة الاشتقاقيات | `ilm/etymology/README.md` |

## الميزات

- 🏺 إدخال بيانات الاشتقاقيات من مصادر بنمط `hanziyuan` و `chineseetymology`.
- 🌐 مسار إدخال قوي عبر AJAX + HTML مع إعادة المحاولة، والتحكم في المعدل، والتخزين المؤقت.
- 🧩 استخراج glyph مع وسم المرحلة، بما في ذلك `<img>` وبيانات URI داخل CSS `background-image`.
- 🗃️ تخزين مدعوم بـ SQLite لبيانات المحارف/الـ glyph الوصفية، مع تنظيم الأصول على نظام الملفات.
- 🖥️ واجهة ويب Tornado للإدخال السريع مع معرض معاينة.
- 🔤 أدوات تصيير glyph لصور التوكنات متعددة اللغات.
- 🧠 وحدات تضمين/دفتر أكواد بأسلوب product-code.
- 🧱 نصوص تدريب/تقييم لتعبئة إطارات الجمل والانتشار/الاستكمال inpainting.
- 📊 نصوص تقارير وتصورات لفحص التضمين ومسار المعالجة.
- 📄 مخرجات النشر بصيغة LaTeX/PDF ضمن `publication/`.

## بنية المشروع

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

## المتطلبات المسبقة

| المتطلب | ملاحظات |
|---|---|
| Python `3.10+` | بيئة التشغيل الأساسية |
| `pip` | تثبيت الحزم |
| GPU اختياري | مفيد لنصوص تدريب PyTorch CUDA |
| سلسلة أدوات LaTeX اختيارية | مطلوبة لبناء ملفات النشر |

ملاحظة افتراضية: لا يوجد حاليًا ملف مركزي موحّد لتثبيت الاعتماديات في الجذر (`pyproject.toml` أو `requirements.txt` وما شابه)، لذلك تُستنتج الاعتماديات من الاستيرادات وطريقة استخدام النصوص.

## التثبيت

### الحد الأدنى (أدوات الاشتقاقيات)

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install requests beautifulsoup4 tornado
```

### موسّع (مسارات النمذجة/التدريب)

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install requests beautifulsoup4 tornado pyyaml numpy pillow matplotlib torch
```

إذا احتاج نص معين إلى حزم إضافية، ثبّتها وفق رسالة خطأ الاستيراد التي يعرضها ذلك النص.

## الاستخدام

### بدء سريع: إدخال glyph التاريخي (CLI)

1. Hanziyuan (موصى به): تدفق AJAX خاص بالمحرف فقط

```bash
PYTHONPATH=. python scripts/ingest_etymology.py --site hanziyuan --char 中
```

2. ChineseEtymology (رابط مباشر)

```bash
PYTHONPATH=. python scripts/ingest_etymology.py --site chineseetymology --url "https://www.chineseetymology.org/CharacterEtymology.aspx?characterInput=%E4%B8%AD"
```

3. إدخال دفعي من ملف (الأسطر يمكن أن تكون `char\turl` أو `url` أو `char url`)

```bash
PYTHONPATH=. python scripts/ingest_etymology.py --from-file urls.txt
```

### المخرجات

| نوع المخرجات | الموقع |
|---|---|
| الملفات | `data/historic/glyphs/<char>/<stage>/<label>.<ext>` |
| التخزين المؤقت | `data/historic/cache/*.html` |
| قاعدة البيانات | `data/historic/etymology.sqlite3` |

### عرض الويب (اختياري)

```bash
PYTHONPATH=. python scripts/serve_etymology.py
```

افتح `http://127.0.0.1:8888`، اختر الموقع، ثم أدخل محرفًا (مثل `中`).

### الزحف المهذب واحترام المواقع

- يستخدم الجالب fetcher تنظيمًا للمعدل لكل مضيف، مع إعادة المحاولة وفق backoff والتخزين المؤقت.
- اجعل التأخير `>= 0.5s`، وتجنب الدفقات السريعة، واحترم شروط المواقع/robots/التراخيص.
- لا تتجاوز جدران الدفع أو وسائل الحماية التفاعلية.
- إذا ظهر `403` أو `429`، خفّض السرعة وأعد المحاولة لاحقًا.

### مسارات ILM إضافية

هذه النصوص موجودة وتُعد جزءًا فعليًا من سطح المستودع، لكنها مسارات بحثية وقد تتطلب مجموعات بيانات/نقاط تحقق محلية مجهّزة مسبقًا.

1. تنزيل/تهيئة البيانات

```bash
python scripts/download_alpaca.py --outdir data/raw
python scripts/download_corpora.py --out data/raw
python scripts/sample_paragraphs.py --out data/processed/test_100.jsonl
python scripts/build_images_common_freq.py --out data/processed/images_common_freq --size 128 --en 5000 --zh 5000
```

2. دورة حياة Glyph DB

```bash
python scripts/glyphdb_init.py --db data/glyphdb/glyphs.sqlite3
python scripts/glyphdb_ingest_index.py --db data/glyphdb/glyphs.sqlite3 --index data/processed/images_common_freq/index.tsv
```

3. تدريب نماذج الكود/اللون

```bash
python scripts/train_color_codes.py --config configs/color.yaml
python scripts/train_codes_from_qa.py --en-json data/raw/alpaca_en.json --zh-json data/raw/alpaca_zh.json --epochs 1
python scripts/train_ilmglyph_codes.py --en data/raw/alpaca_en.json --zh data/raw/alpaca_zh.json --out artifacts/ilm_glyph_train
```

4. الانتشار/الاستكمال inpainting

```bash
python scripts/train_diffusion.py --config configs/diffusion.yaml
python scripts/train_inpaint_frames.py --ckpt-code artifacts/ilm_glyph_train/ckpt_epoch1.pt --out artifacts/inpaint
```

5. التقييم/التقارير

```bash
python scripts/eval_color_codes.py --checkpoint artifacts/color_codes_e1.pt
python scripts/eval_diffusion.py --checkpoint artifacts/diffusion_unet.pt
python scripts/eval_qa_retrieval.py --checkpoint artifacts/color_codes_qa.pt
python scripts/report_ilmglyph_pipeline.py --ckpt artifacts/ilm_glyph_train/ckpt_epoch1.pt --lang en --text "hello world"
```

## الإعدادات

ملفات YAML الأساسية:

- `configs/color.yaml`
  - مسار البيانات: `data/processed/images_common_freq/index.tsv`
  - معاملات النموذج/الكود: `d_glyph` و `d_code` و `K` و `C` و temperature/anneal
  - إعدادات optimizer/log

- `configs/diffusion.yaml`
  - JSONL الإدخال: `data/processed/test_100.jsonl`
  - إعدادات حجم الإطار/الشبكة وحجم النموذج
  - نطاق نسبة القناع أثناء التدريب وإعدادات نقاط التحقق

يمكن تجاوز الإعدادات عبر وسائط CLI حيثما كان ذلك مدعومًا (`--epochs` و `--batch-size` و `--lr` وغيرها).

## أمثلة

- بناء glyph بلاطة إنجليزية واحدة:

```bash
python scripts/build_english_tile_glyph.py "language" artifacts/language_tile --save-tensor
```

- تشغيل عرض inpainting تجريبي باستخدام نقاط تحقق مدرَّبة:

```bash
python scripts/inpaint_demo.py \
  --ckpt-code artifacts/ilm_glyph_train/ckpt_epoch1.pt \
  --ckpt-inpaint artifacts/inpaint/ckpt_epoch1.pt \
  --lang en \
  --text "the quick brown fox jumps" \
  --mode infill \
  --out artifacts/inpaint_demo
```

- إدخال دفعي لمحارف شائعة من Hanziyuan:

```bash
PYTHONPATH=. python scripts/bulk_ingest_hanziyuan.py --limit 200 --resume
```

## ملاحظات التطوير

- هذا مستودع بحثي يجمع بين أدوات CLI متينة ومخرجات استكشافية (بما فيها دفاتر ملاحظات ونصوص أولية).
- الملفات الكبيرة المتولدة مخصصة لمساري `data/` و `artifacts/` (وكلاهما متجاهل في `.gitignore`).
- مصادر النشر وملفات PDF تقع ضمن `publication/`، مع نص بناء مساعد: `scripts/latex_build.sh`.
- أعراف التعاون/العملية موثقة في `AGENTS.md`.

## استكشاف الأعطال وإصلاحها

- `ModuleNotFoundError: ilm...`
  - شغّل النصوص من جذر المستودع.
  - استخدم `PYTHONPATH=.` للنصوص التي تتوقع حل الحزم محليًا.

- `FileNotFoundError` لبيانات/فهرس/نقاط تحقق
  - شغّل نصوص البيانات/البناء المطلوبة أولًا.
  - تأكد من وجود المسارات الافتراضية مثل `data/processed/images_common_freq/index.tsv` و `data/processed/test_100.jsonl`.

- مشكلات CUDA/الجهاز
  - انتقل إلى CPU عبر أعلام النص أو الإعداد (`device: cpu` أو `--device cpu`).

- أخطاء حزم مفقودة
  - ثبّت الاعتمادية المطلوبة وفق مسار الاستيراد في النص المحدد (`torch` أو `pyyaml` أو `Pillow` وغيرها).

- HTTP `403` / `429` أثناء الكشط
  - زِد `--delay`، وأعد المحاولة لاحقًا، وحافظ على طلبات مهذبة.

## خارطة الطريق

- مواصلة تطوير أدلة التشغيل للتدريب/التقييم في ILM المعتمد على النص كصورة، بما يتجاوز البدء السريع المرتكز على الاشتقاقيات.
- تحسين قابلية إعادة إنتاج البيئة (مواصفة اعتماديات موحدة ومرجعية واحدة).
- توسيع تغطية الاختبارات/التكامل المستمر لنصوص البحث وروابط خطوط المعالجة.
- التكرار على دفاتر الأكواد الهرمية، وأهداف الانتشار، وقنوات التحكم.
- توحيد التوثيق بين `docs/` ونصوص المساعدة ومخرجات النشر.

للتفاصيل المفاهيمية الأعمق وخطط المراحل، راجع:

- `docs/imagized-language-model.md`
- `docs/ilm-visual-diffusion-code-plan.md`
- `docs/development-plan.md`

## المساهمة

- اتبع `AGENTS.md` فيما يخص الأعراف (commits ذرية، push بعد التغيير، وعدم تضمين بيانات اعتماد في الشيفرة).
- اجمع التعديلات المرتبطة ضمن commits مركزة برسائل تقليدية.
- فضّل أوامر نصوص قابلة لإعادة الإنتاج مع أعلام واضحة ومسارات دخل صريحة.
- في التعديلات المتعلقة بالكشط، حافظ على سلوك تقليل المعدل/التخزين المؤقت وضوابط احترام المواقع.

## الترخيص

لا يوجد حاليًا ملف ترخيص على مستوى الجذر في هذا المستودع.

ملاحظة افتراضية: يُتعامل مع المشروع ككود بحثي بترخيص غير محدد حتى يضيف المشرفون ملف `LICENSE`.

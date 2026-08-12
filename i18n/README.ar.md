[English](../README.md) · [العربية](README.ar.md) · [Español](README.es.md) · [Français](README.fr.md) · [日本語](README.ja.md) · [한국어](README.ko.md) · [Tiếng Việt](README.vi.md) · [中文 (简体)](README.zh-Hans.md) · [中文（繁體）](README.zh-Hant.md) · [Deutsch](README.de.md) · [Русский](README.ru.md)


[![LazyingArt banner](https://github.com/lachlanchen/lachlanchen/raw/main/figs/banner.png)](https://github.com/lachlanchen/lachlanchen/blob/main/figs/banner.png)

# نموذج اللغة المصوّرة (ILM)

![Python](https://img.shields.io/badge/python-3.10%2B-3776AB?logo=python&logoColor=white)
![Status](https://img.shields.io/badge/status-research-orange)
![Focus](https://img.shields.io/badge/focus-text--as--image-0A7EA4)
![Paradigm](https://img.shields.io/badge/paradigm-predictive%20visual%20field-16835B)
![License](https://img.shields.io/badge/license-unspecified-lightgrey)
![Domain](https://img.shields.io/badge/domain-historic%20etymology%20%7C%20glyph%20models-2F80ED?logo=github)

ILM مشروع بحثي لتعلّم اللغة وتوليدها بوصفها **كتابة مرئية**، من الصورة إلى
حالة بصرية مستمرة ثم إلى صورة، من دون مسار رموز نصية مخفي.

## أحدث اختبار للتوجيه: قبول دائرة العلاقات البصرية V23

![نتيجة V23 المقاسة: تمر ست صور للتوجيه عبر المطابقة البصرية وبوابة العملية وتوجيه الرسم المصدر ومطبّع صور مجمّد لإنتاج صورة جواب](../publication/ilm-image-native/figures/visual_relation_circuit_v23_result.png)

V23 هي أول تجربة في هذا المستودع تجتاز سلسلة الإثبات كاملة من توجيه مصوّر إلى
جواب مصوّر. لا يستقبل النموذج سوى ست صور كتابة بقياس `32x32` ويُنتج صورة جواب
واحدة بقياس `32x32`. ولا يحتوي مسار الاستدلال على سلاسل نصية أو tokens أو
معرّفات Unicode أو OCR أو بحث عن glyph أو فهرس جواب أو نموذج لغة خارجي.

في التقييم المجمّد الوحيد المسموح، على 98 محرفاً صينياً غير مرئي سابقاً و1,024
حلقة و4,096 تنويعاً للتوجيه، بلغت دقة الاختيار الثنائي `0.99829`، وتبديل
الاستعلام `0.99609`، وتبديل العملية `0.99707`، وtop-1 للرسم الناتج `0.99463`،
وpixel F1 مقدار `0.78478`. ظل ضابطا حجب الاستعلام وحجب العملية ثابتين تماماً
أمام العامل الذي لا يراه كل منهما.

تثبت هذه النتيجة فقط اتباع توجيه بصري ضمن نحو ثابت ذي ستة أدوار وزوجين من
الارتباط وعلاقة same/other؛ ولا تثبت فهماً حراً للغة. سيزيل V24 أدوار الإطارات
الثابتة، ويقرأ تيار كتابة بصرياً ثنائي الأبعاد متغير الطول، ثم يولد الإطار
الثاني بعد إعادة قراءة الأول. راجع [سجل V23 بالإنجليزية](../docs/visual-relation-circuit-v23-result.md).

## خط الأساس البحثي السابق: بداية المجال البصري التنبؤي

![مخطط المجال البصري التنبؤي](../publication/ilm-image-native/figures/predictive_visual_field_paradigm.png)

أظهرت تجربة RFLM V7 أن جامعاً بصرياً واحداً لا ينبغي أن يستنتج الهوية
اللغوية ويرسم الضربات في العملية نفسها. رفع V7 دقة السياق الكامل من `1.20%`
إلى `2.31%`، متجاوزاً last-only (`2.02%`) وunigram (`1.86%`)، لكنه بقي دون
bigram (`13.58%`). تحسن فرق الاحتمال اللوغاريتمي المطبّع من `-0.9066` إلى
`-0.2155` لكنه ظل سالباً، وبقي الناتج المستقل غير مقروء. لذلك **رُفض V7
كنموذج لغة**.

يفصل نموذج PVF التالي بين تدفق مستمر يتنبأ بالحالة الشبكية التالية ومشغّل
بصري مستقل يحوّل تلك الحالة إلى حبر ثم يعيد قراءتها. لا يوجد بحث عن أقرب حرف
ولا جدول إخراج. الحد الصارم هو: `بكسلات كتابة -> ديناميكيات بصرية مستمرة ->
بكسلات حبر`. لا يتلقى الطالب tokens أو معرفات Unicode أو OCR أو دفتر أكواد
بصرياً أو نموذج لغة خارجياً. PVF فرضية V8 قابلة للاختبار، وليس قدرة مثبتة.

> يحتفظ المستودع عمدًا بمسار عملي لتتبع علم الأصول التاريخي جنبًا إلى جنب مع تجارب ILM طويلة المدى.

## 📌 نظرة عامة

هذا المستودع يضم ثلاثة مسارات مترابطة:

1. نمذجة اللغة بالصور عبر التدفق الشبكي وتقييمها الصارم خارج التدريب.
2. استيعاب أصول glyph الصينية التاريخية مع حفظ المصدر.
3. خطوط أساس أقدم للرموز والانتشار وInkStream محفوظة لإعادة الإنتاج.

يوثّق هذا الـ README المسارات الثلاثة ويجعل سير عمل علم الأصول قابلاً لإعادة التنفيذ.

## 🔗 الروابط الأساسية

| المجال | المسار |
|---|---|
| الموجز المفاهيمي | `docs/imagized-language-model.md` |
| الهدف الهندسي الحالي | `docs/first-imagized-language-model-goal.md` |
| ملف البحث والأدلة | `references/image-native-language-model-research.md` |
| خطة الشيفرة والمقاييس | `docs/ilm-visual-diffusion-code-plan.md` |
| خطة "color" للتضمين | `docs/embedding-color-plan.md` |
| ملاحظات/خطة التطوير | `docs/development-plan.md` |
| دليل وحدة الأصول | `ilm/etymology/README.md` |

## ✨ الميزات

- 🏺 استيعاب الأصول من مصادر من نمط `hanziyuan` و`chineseetymology`.
- 🌐 مسار استيعاب AJAX + HTML قوي مع إعادة المحاولة، وضبط الوتيرة، والتخزين المؤقت.
- 🧩 استخراج glyph موسوم بمراحل يضم بيانات `<img>` وبيانات URI داخل CSS `background-image`.
- 🗃️ تخزين قائم على SQLite لبيانات الـ chars/metadata الخاصة بالـ glyph مع تخطيط الأصول على نظام الملفات.
- 🖥️ واجهة ويب عبر Tornado للاستيعاب اليدوي ومعاينة المعرض.
- 🔤 أدوات عرض glyphs لتوكنات متعددة اللغات.
- 🧠 وحدات embedding/codebook بنمط product-code.
- 🧱 تعبئة إطارات الجمل ونصوص التدريب/التقييم لعمليات diffusion/inpainting.
- 📊 نصوص التقارير والتصور لفحص embedding وسلسلة المعالجة.
- 📄 مخرجات النشر بصيغ LaTeX/PDF داخل `publication/`.

## 🧱 هيكل المشروع

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

## 🧰 المتطلبات المسبقة

| المتطلب | ملاحظات |
|---|---|
| Python `3.10+` | بيئة التشغيل الأساسية |
| `pip` | لتثبيت الحزم |
| GPU اختياري | مفيد لبرامج التدريب باستخدام PyTorch CUDA |
| سلسلة أدوات LaTeX اختيارية | مطلوبة لبناء ملفات النشر |

ملاحظة افتراضية: لا يوجد حاليًا ملف جذر واحد لقفل/تحديد الاعتماديات (`pyproject.toml`, `requirements.txt`, وغيرها)، لذلك تُستنتج الاعتماديات من الاستيرادات واستخدامات السكربتات.

## ⚙️ التثبيت

### الحد الأدنى (أدوات مسار الأصول)

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install requests beautifulsoup4 tornado
```

### موسع (مسارات النمذجة/التدريب)

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install requests beautifulsoup4 tornado pyyaml numpy pillow matplotlib torch
```

إذا احتاج سكربت معيّن حزمًا إضافية، ثبّتها بحسب الخطأ الظاهر في استيراد ذلك السكربت.

## 🚀 الاستخدام

### البدء السريع: استيعاب glyph التاريخية (CLI)

1. Hanziyuan (موصى به): مسار AJAX للحروف فقط

```bash
PYTHONPATH=. python scripts/ingest_etymology.py --site hanziyuan --char 中
```

2. ChineseEtymology (رابط مباشر)

```bash
PYTHONPATH=. python scripts/ingest_etymology.py --site chineseetymology --url "https://www.chineseetymology.org/CharacterEtymology.aspx?characterInput=%E4%B8%AD"
```

3. استيعاب دفعة من ملف (الأسطر قد تكون `char\turl`, `url`, أو `char url`)

```bash
PYTHONPATH=. python scripts/ingest_etymology.py --from-file urls.txt
```

### المخرجات

| نوع المخرج | الموقع |
|---|---|
| الملفات | `data/historic/glyphs/<char>/<stage>/<label>.<ext>` |
| التخزين المؤقت | `data/historic/cache/*.html` |
| قاعدة البيانات | `data/historic/etymology.sqlite3` |

### عرض الويب (اختياري)

```bash
PYTHONPATH=. python scripts/serve_etymology.py
```

افتح `http://127.0.0.1:8888`، واختر الموقع، وأدخل حرفًا (مثل `中`).

### الزحف المهذب واحترام المواقع

- يستخدم أداة الجلب تقييد معدل لكل مضيف مع إعادة المحاولة (backoff) والتخزين المؤقت.
- احرص على فواصل `>= 0.5s`، وتجنب الاندفاع، واطّلع على شروط الموقع/robots/licensing.
- لا تتجاوز الجدران المدفوعة أو الحواجز التفاعلية.
- إذا ظهرت `403`/`429`، خفّف السرعة وحاول لاحقًا.

### مسارات ILM الإضافية

توجد هذه السكربتات حالًا ضمن سطح المستودع، لكنها تُعدّ تدفقًا بحثيًا وقد تتطلب مجموعات بيانات/نقاط تفتيش محضّرة محليًا.

1. تنزيل/إعداد البيانات

```bash
python scripts/download_alpaca.py --outdir data/raw
python scripts/download_corpora.py --out data/raw
python scripts/sample_paragraphs.py --out data/processed/test_100.jsonl
python scripts/build_images_common_freq.py --out data/processed/images_common_freq --size 128 --en 5000 --zh 5000
```

2. دورة حياة قاعدة glyph DB

```bash
python scripts/glyphdb_init.py --db data/glyphdb/glyphs.sqlite3
python scripts/glyphdb_ingest_index.py --db data/glyphdb/glyphs.sqlite3 --index data/processed/images_common_freq/index.tsv
```

3. تدريب كود/ألوان النموذج

```bash
python scripts/train_color_codes.py --config configs/color.yaml
python scripts/train_codes_from_qa.py --en-json data/raw/alpaca_en.json --zh-json data/raw/alpaca_zh.json --epochs 1
python scripts/train_ilmglyph_codes.py --en data/raw/alpaca_en.json --zh data/raw/alpaca_zh.json --out artifacts/ilm_glyph_train
```

4. diffusion/inpainting

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

## 🧩 الإعدادات

إعدادات YAML الأساسية:

- `configs/color.yaml`
  - مسار البيانات: `data/processed/images_common_freq/index.tsv`
  - معاملات النموذج/الكود: `d_glyph`، `d_code`، `K`، `C`، temperature/anneal
  - إعدادات المحسن/السجلات

- `configs/diffusion.yaml`
  - JSONL الإدخال: `data/processed/test_100.jsonl`
  - إعدادات الشبكة/الإطار/النموذج
  - نطاق نسبة القناع أثناء التدريب وإعدادات نقاط التفتيش

يمكن تجاوز الإعدادات عبر أعلام سطر الأوامر حيث يدعم ذلك (`--epochs`, `--batch-size`, `--lr`, ...).

## 🧪 أمثلة

- بناء glyph بلاطة إنجليزية واحدة:

```bash
python scripts/build_english_tile_glyph.py "language" artifacts/language_tile --save-tensor
```

- تشغيل عرض inpainting باستخدام نقاط تفتيش مدربة:

```bash
python scripts/inpaint_demo.py \
  --ckpt-code artifacts/ilm_glyph_train/ckpt_epoch1.pt \
  --ckpt-inpaint artifacts/inpaint/ckpt_epoch1.pt \
  --lang en \
  --text "the quick brown fox jumps" \
  --mode infill \
  --out artifacts/inpaint_demo
```

- استيعاب جماعي للأحرف الشائعة من Hanziyuan:

```bash
PYTHONPATH=. python scripts/bulk_ingest_hanziyuan.py --limit 200 --resume
```

## 📝 ملاحظات التطوير

- هذا مستودع بحثي يجمع بين أدوات CLI المتماسكة والمواد الاستكشافية (بما في ذلك notebooks و prototype scripts).
- الملفات الكبيرة الناتجة تُوجّه إلى `data/` و`artifacts/` (كلاهما مُستثنى في `.gitignore`).
- مصادر وملفات النشر تكون داخل `publication/`؛ سكربت البناء المساعد: `scripts/latex_build.sh`.
- عادات التعاون/العمليات موثقة في `AGENTS.md`.

## 🛠️ استكشاف الأخطاء وإصلاحها

- `ModuleNotFoundError: ilm...`
  - نفّذ السكربتات من جذر المستودع.
  - استخدم `PYTHONPATH=.` للسكربتات التي تتطلب حلًا محليًا للحزم.

- `FileNotFoundError` لملفات البيانات/الفهارس/نقاط التفتيش
  - شغّل سكربتات البيانات/البناء التمهيدي أولًا.
  - تأكد من وجود المسارات الافتراضية مثل `data/processed/images_common_freq/index.tsv` و`data/processed/test_100.jsonl`.



## ❤️ Support

| Donate | PayPal | Stripe |
| --- | --- | --- |
| [![Donate](https://camo.githubusercontent.com/24a4914f0b42c6f435f9e101621f1e52535b02c225764b2f6cc99416926004b7/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f446f6e6174652d4c617a79696e674172742d3045413545393f7374796c653d666f722d7468652d6261646765266c6f676f3d6b6f2d6669266c6f676f436f6c6f723d7768697465)](https://chat.lazying.art/donate) | [![PayPal](https://camo.githubusercontent.com/d0f57e8b016517a4b06961b24d0ca87d62fdba16e18bbdb6aba28e978dc0ea21/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f50617950616c2d526f6e677a686f754368656e2d3030343537433f7374796c653d666f722d7468652d6261646765266c6f676f3d70617970616c266c6f676f436f6c6f723d7768697465)](https://paypal.me/RongzhouChen) | [![Stripe](https://camo.githubusercontent.com/1152dfe04b6943afe3a8d2953676749603fb9f95e24088c92c97a01a897b4942/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f5374726970652d446f6e6174652d3633354246463f7374796c653d666f722d7468652d6261646765266c6f676f3d737472697065266c6f676f436f6c6f723d7768697465)](https://buy.stripe.com/aFadR8gIaflgfQV6T4fw400) |

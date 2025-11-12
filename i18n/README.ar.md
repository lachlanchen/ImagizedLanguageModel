نمذجة اللغة بالصور (ILM)
=========================

اللغات
- [English](../README.md) | [简体中文](README.zh-Hans.md) | [繁體中文](README.zh-Hant.md) | [日本語](README.ja.md) | [한국어](README.ko.md) | [Tiếng Việt](README.vi.md) | العربية | [Français](README.fr.md) | [Español](README.es.md)

نظرة عامة
يعرض ILM النص كموتر “مشابه للصورة” مضغوط، ويولّد النص عبر إزالة الضوضاء تدريجيًا بأسلوب الانتشار (diffusion). يقسّم الجملة إلى عناصر ميتا قابلة للتحكم (القواعد، المعنى، النبرة، العاطفة) مع ترميز هرمي شبيه بالذاكرة (كلمات/أشكال حروف). يدمج الانتشار المتقطع، والتراكب/فكّ الارتباط للميزات، والتمثيلات المهيكلة، والوعي بأشكال الحروف.

روابط أساسية
- الشرح المفاهيمي: docs/imagized-language-model.md
- خطة الشيفرة والمؤشرات: docs/ilm-visual-diffusion-code-plan.md
- خطة “ألوان” التضمين: docs/embedding-color-plan.md

محتويات المستودع
- ilm/etymology/: أدوات لجمع أشكال الحروف التاريخية (عظام السلاحف/البرونز/الخ.
  - جلب AJAX من hanziyuan (مع إعادة المحاولة/تقييد المعدل/التخزين المؤقت)
  - تحليل HTML/CSS لاستخراج الصور المعلّمة بالمراحل (بيانات URI وروابط)
- scripts/
  - ingest_etymology.py: أداة سطر أوامر للحفظ إلى SQLite والملفات
  - serve_etymology.py: واجهة Tornado بسيطة للعرض والتجربة
  - use_historic_tools.md: ملاحظات عن البيانات/الأدوات الخارجية
- data/ (غير مُتعقَّبة): تخزين مؤقت HTML، صور، قاعدة بيانات SQLite

البدء السريع
- المتطلبات: `pip install requests beautifulsoup4 tornado`
- مثال (مستحسن: hanziyuan AJAX):
  - `PYTHONPATH=. python scripts/ingest_etymology.py --site hanziyuan --char 中`
- العرض على الويب:
  - `PYTHONPATH=. python scripts/serve_etymology.py` → http://127.0.0.1:8888

مكان المخرجات
- الصور: data/historic/glyphs/<الحرف>/<المرحلة>/<الوسم>.<الامتداد>
- التخزين المؤقت: data/historic/cache
- قاعدة البيانات: data/historic/etymology.sqlite3

الالتزام بالأدب أثناء الجلب
- تقييد معدل لكل مضيف، مع إعادة المحاولة وتدرّج التراجع، والتخزين المؤقت مفعّلة
- احترم شروط/ترخيص المواقع. عند 403/429 خفّض السرعة وأعد المحاولة لاحقًا

هدف المشروع
- نموذج مُهيكل وقابل للتحكم، يدعم لغات متعددة، ويعمل على الحواسيب العادية

المساهمة
- اتبع AGENTS.md (عمليات commit صغيرة، ادفع push بعد كل تغيير، لا تقم بحفظ بيانات حساسة)


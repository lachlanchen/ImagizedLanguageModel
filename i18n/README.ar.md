[English](../README.md) · [العربية](README.ar.md) · [Español](README.es.md) · [Français](README.fr.md) · [日本語](README.ja.md) · [한국어](README.ko.md) · [Tiếng Việt](README.vi.md) · [中文 (简体)](README.zh-Hans.md) · [中文（繁體）](README.zh-Hant.md) · [Deutsch](README.de.md) · [Русский](README.ru.md)

تعلّم النموذج البصري على بطاقة RTX 4090 D واحدة تصحيح **65 من 96** ترجمة متكررة بين الإنجليزية والصينية. صحّح نموذج المقارنة المدرّب بتعظيم الاحتمال وحده 61 ترجمة. نقل النموذج المعنى كاملاً في **8 من 96** مثالاً آخر من التدريب وفي **مثال واحد من 96** من أربعة كتب جديدة. يعرض الشكل الصور الأصلية لجملة تعلّمها، ورقم شكل ترجمه بصورة صحيحة، وجملة أخفق فيها. [الصور الأصلية والنتائج](../references/visual_sequence_preference_result.md) · [تشغيل النموذج](../models/visual_sequence_preference/README.md)

![النتائج](../publication/ilm-nature/figures/Visual_sequence_preference.png)

في اختبار شمل 96 طلب ترجمة من أربعة كتب أخرى، استرجع النموذج أمثلة ذات صلة من 497,775 زوجا من الجمل ثنائية اللغة. حكم المقيّم بأن المعنى اكتمل في إجابتين، مقابل إجابة واحدة لكل من الاسترجاع العشوائي والنموذج الأصلي، و83 إجابة للنموذج النصي المرجعي. [التجربة وصور المخرجات](../references/retrieved_glyph_writer_result.md)

اختار نموذج دُرّب على بطاقة RTX 4090 D واحدة بين صورتين لترجمتين لكل مقطع كتاب. بعد تصحيح ميله إلى تفضيل صياغات معيّنة، ارتفع عدد الأزواج التي صحّ فيها الاختيار للمقطعين من **21/96 إلى 77/96**. وعند كتابة الترجمة بحرية، زاد توقّع أشكال الكتابة من المصدر عدد الإجابات التي احتفظت بالفعل المطلوب من 17 إلى 43 من أصل 336. لكن إجابة واحدة فقط من 192 نقلت معنى المقطع كاملًا. تعرض الصورة أول إجابة مقبولة في اختبار OPUS المنفصل. [النتائج](../references/source_supported_writing_result.md) · [تشغيل النموذج](../models/source_supported_writing/README.md)

![صورة المصدر والترجمة الناتجة في اختبار OPUS](../publication/ilm-nature/figures/Source_supported_opus.png)

طلبنا من النموذج عرض معنى الحرف وصورته التاريخية بترتيبين متعاكسين من صفحة المعجم نفسها. ارتفع عدد الأزواج التي نجح فيها الطلبان معًا من 138 إلى 332 من أصل 608. وفي اختبار منفصل للترجمة الحرة، نقل نموذج الصور المستقل المعنى كاملًا في حالة واحدة من أصل 192، مقابل 154 حالة لنموذج Qwen3-8B مع مدخل نصي. [البحث في ست صفحات](../publication/ilm-nature/manuscript.pdf) · [تشغيل مثال المعجم](../models/origin_instructions/README.md)

### تجارب سابقة

عندما سُئل النموذج عمّن صمّم قاطرة Salamanca بعد عرض صفحة عن السكك الحديدية، كتب **Matthew Murray designed Salamanca.** وعندما تغيّر الاسم وحده في صورة الصفحة إلى Daniel Harper، تغيّر الفاعل في الجواب أيضاً. قرأ النموذج الإجابة القصيرة الصحيحة في 25 صفحة من أصل 96 صفحة جديدة. وفي مثال آخر قرأ سنة تأسيس مدرسة ثم أسقطها عند كتابة الجملة. تعرض الصفحة الأولى من الورقة صور الإجابات الفعلية. [PDF](../publication/ilm-nature/visual_language_paper.pdf) · [PNG](../models/visual_statements/examples/intervention)

اختبرنا النموذج على 256 صفحة جديدة بالإنجليزية والصينية، بسؤالين كتبهما أشخاص لكل صفحة. أجاب ILM بعد التدريب التبايني عن السؤالين إجابة صحيحة في 34 صفحة، مقابل 101 لنموذج Qwen3-8B المكمّم عند قراءة النص. وعند عرض الصور نفسها على النموذجين في عينة من 64 صفحة، نجح ILM في 6 صفحات وQwen3-VL-8B في 21. [PDF](../publication/ilm-nature/visual_language_paper.pdf) · [PNG](../models/competing_facts/examples)

[![Salamanca](../publication/ilm-nature/figures/Figure_150_optical_reading_and_writing.png)](../publication/ilm-nature/visual_language_paper.pdf)

عند عرض صفحة صينية عن سمكة، كتب النموذج اسمها الشائع أو تصنيفها بحسب السؤال. تدرّب النموذج البصري، الذي يضم 2.44 مليون معامل، على بطاقة RTX 4090 D واحدة. أجاب عن السؤالين إجابة صحيحة في 44 من أصل 256 مقطعاً جديداً بالصينية والإنجليزية. كانت نتيجة النموذج البصري السابق صفراً، وبلغت نتيجة نموذج المقارنة الذي يستقبل البايتات 52 باستهلاك طاقة مماثل. ولم ينجح النموذج البصري في كتابة الأشكال التاريخية للحرف 言. [تشغيل النموذج](../models/paired_evidence/README.md) · [البحث](../publication/ilm-nature/visual_language_paper.pdf)

[![Figure 145](../publication/ilm-nature/figures/Figure_145_paired_evidence.png)](https://github.com/lachlanchen/ImagizedLanguageModel/blob/203f40812f03afbdfa55cf957a56ee8cfe0f944c/publication/ilm-nature/visual_language_paper.pdf#page=2)

تعلّم نموذج يضم 2.12 مليون مُعامِل، دُرِّب على بطاقة RTX 4090 D واحدة، توقّع أشكال الحروف التالية في نصوص إنجليزية وصينية جديدة. وعندما سُئل عن طريقة لتوفير الطاقة، كرّر جزءًا من السؤال. بعد إضافة قارئ بصري، كتب النموذج الإجابة المرجعية في 11 سؤالًا من أصل 384. ولم يُجب إجابة صحيحة عن السؤالين معًا في أيّ من المقاطع الـ192. يعرض الشكل 144 صور المخرجات الفعلية. [قراءة البحث](../publication/ilm-nature/visual_language_paper.pdf) · [تشغيل النموذج](../models/visual_cursor/README.md)

[![Figure 144](../publication/ilm-nature/figures/Figure_144_visual_discourse.png)](https://github.com/lachlanchen/ImagizedLanguageModel/blob/203f40812f03afbdfa55cf957a56ee8cfe0f944c/publication/ilm-nature/visual_language_paper.pdf#page=4)

يقرأ نموذج من 1.18 مليون معامل حرفاً محدداً في صورة نص صيني قديم ويكتب معناه الإنجليزي في صورة. كتب speak للحرف الأخير في 言及之而不言 وكتب speech للحرف 言 في 有是言也. وأصاب في المعنيين معاً في 213 من 566 زوجاً من «المحاورات» و82 من 403 أزواج من «أغاني تشو». وأصابت طريقة تعتمد على تواتر الحروف المجاورة في 211 و104 أزواج على التوالي. [تشغيل النموذج](../models/visual_word_senses/README.md) · [النتائج](../references/visual_word_sense_result.md)

[![119 · ILM](../publication/ilm-nature/figures/Figure_119_visual_word_senses.png)](../publication/ilm-nature/manuscript.pdf#page=160)

قرأ النموذج أسئلة صينية ومقاطع روائية على هيئة صور، وكتب الإجابة في صورة تضم عبارتين. طابقت الإجابات الصورة المرجعية في 34 من 48 حالة لرواية «رحلة إلى الغرب» وفي 42 من 48 لرواية «حلم الغرفة الحمراء». وفي 9 من أصل 60 عائلة من الحروف المخصصة للتقييم، كتب النموذج شكلي الحرف الموثقين على العظام والبرونز بصورة صحيحة في جميع الترتيبات. وعند طرح الأسئلة بالإنجليزية انخفض العدد إلى 2 و10 من أصل 48 على التوالي. [تشغيل النموذج](../models/grounded_ink_editor/README.md) · [النتائج والأمثلة](../references/grounded_ink_writing_result.md)

[![118 · ILM](../publication/ilm-nature/figures/Figure_118_grounded_ink_editor.png)](../publication/ilm-nature/manuscript.pdf#page=159)

أجاب قارئ بصري يضم 86 مليون معامل عن 38 من 56 سؤالاً صينياً مصوّراً حول شخصيات كتاب «خطابات الممالك» (國語). وعندما أُعيد طرح السؤالين بعد تبديل اسمي الشخصيتين، أصاب في الإجابات الأربع في 4 من 14 مقطعاً، مقابل 3 مقاطع للقارئ الأصلي ومقطعين للنموذج الذي أعطى الاسمين وزناً متساوياً أثناء التدريب. وفي الإنجليزية أجاب عن 16 سؤالاً إجابة صحيحة، ولم يُجب عن الأسئلة الأربعة كلها في أي مقطع. يعيد المثال صورتي الاسمين 密康公 و恭王 من الصفحة الأصلية. [Guoyu · ILM](../models/optical_role_reader/README.md) · [38/56](../references/optical_role_adapter_result.md)

[![116 · ILM](../publication/ilm-nature/figures/Figure_116_optical_role_adapter.png)](../publication/ilm-nature/manuscript.pdf#page=157)

يقرأ نموذج يضم 64,666 مُعاملاً صورة السؤال وصورة المقطع الصيني، ثم يعيد الجزء الذي اختاره من صورة النص الأصلي. بعد التدريب على 6,439 مقطعاً، أجاب إجابة صحيحة عن 104 من 452 سؤالاً بالصينية و45 من 452 سؤالاً بالإنجليزية حول روايتين خُصّصتا للتقييم. وعند التدريب على إجابات البشر وحدها، بلغ العدد 81 و48 على التوالي. وما زال من النادر أن يجيب النموذج عن سؤالين مختلفين حول المقطع نفسه إجابة صحيحة. [ILM · 64k](../models/book_span_distillation/README.md) · [104/452](../references/book_span_distillation_result.md)

[![113 · ILM](../publication/ilm-nature/figures/Figure_113_book_span_distillation.png)](../publication/ilm-nature/manuscript.pdf#page=154)

تتبّع النموذج البصري تبديل الأدوار بين شخصين في صور قوائم المشاركين في الأعمال في 175 مقارنة من أصل 256، مقابل 19 للنموذج الضابط. وفي اثنتي عشرة قصة جديدة لم ينجح أيّ منهما في مقارنة كاملة. وعند تقديم سؤالين مصوّرين مع القسم المناسب من الفهرس لكل سؤال، أعاد النموذج إنتاج أربعة أشكال تاريخية للحرف 言 في محاولتين من أصل ثمانٍ. [↗](../references/individual_clause_writer_result.md)

[![111 · ILM](../publication/ilm-nature/figures/Figure_111_individual_clause_writer.png)](../publication/ilm-nature/manuscript.pdf#page=152)

عند تبديل اسمين في صورة تعرض أسماء المشاركين في عمل فني، أجاب النموذج ذو 87 مليون معامل إجابة صحيحة في كلا ترتيبي السؤالين في 165 مقارنة من أصل 256. نجح النموذج الضابط الذي تدرب على الأمثلة نفسها في خمس مقارنات. وفي قصص كُتبت بصورة مستقلة كانت نتيجة كلا النموذجين صفراً من أصل 24. [النتائج وصور المخرجات الأصلية](../references/role_change_writer_result.md) · [تشغيل النموذج](../models/role_change_writer/README.md)

بعد استبعاد مواد DRCD المحجوزة وإعادة التدريب، طابق القارئ ذو قائمة الإجابات الأوسع الصورة المرجعية في 234 من 359 سؤالاً في مجموعة التطوير، مقابل 242 للقارئ العام. استُخدمت في تقييمات DRCD السابقة فقرات كانت محجوزة للاختبار. [النتائج وتصحيح مصدر البيانات](../references/span_list_result.md)

[![97 · 战国史](../publication/ilm-nature/figures/Figure_97_span_list.png)](../publication/ilm-nature/manuscript.pdf#page=138)

في تقييم جديد شمل 384 سؤالاً لفهم نصوص صينية، طابقت إجابات النموذج صورة الإجابة المرجعية في 285 سؤالاً بعد جمع ترجيحات أربعة نماذج، وهي نتيجة القارئ العام أيضاً. وفي اختبارات تبادل الأسماء نجح في جميع أسئلة 63 مجموعة من أصل 128، مقابل 79 للقارئ المتخصص. [النتائج وأوزان النموذج](../references/span_choice_result.md)

[![道地的 · 96](../publication/ilm-nature/figures/Figure_96_span_choice.png)](../publication/ilm-nature/manuscript.pdf#page=137)

عند تبادل اسمين في مقطع صيني مصوّر، أجاب النموذج المدرَّب بتعظيم الإمكان عن جميع الأسئلة بالترتيبين المطلوبين للكلمات في 79 من 128 مقطعاً، مقابل 29 سابقاً. أمّا في الأسئلة التي كتبها أشخاص، فانخفض عدد الإجابات الصحيحة بالترتيبين من 34 إلى 28 من أصل 48. [النتائج والنماذج](../references/participant_margin_result.md)

[![松窗 · 95](../publication/ilm-nature/figures/Figure_95_participant_margin.png)](../publication/ilm-nature/manuscript.pdf#page=136)

كتب النموذج المعلومة الصحيحة بالترتيبين المطلوبين للكلمات في 34 من أصل 48 سؤالاً صينياً كتبها أشخاص. وذكر ملحّن أغنية «秋潮» على نحو صحيح، لكنه أخطأ في اسم الملحّن عندما تبدّل دورا الملحّن وكاتب الكلمات في أغنية أخرى. [النتائج والنموذج](../references/pictured_relation_result.md)

[![秋潮 · 94](../publication/ilm-nature/figures/Figure_94_pictured_relation.png)](../publication/ilm-nature/manuscript.pdf#page=135)

تدرّب القارئ على 26,890 مثالاً صينياً مصوّراً، وكتب إجابات تامة وصحيحة عن 33 من 48 سؤالاً صاغها أشخاص. وكتب نموذج منفصل يولّد أشكال الحروف تباعاً الحقيقة نفسها بالترتيبين المطلوبين في 31 سؤالاً. في مثال الفيلم حدّد المخرج وكاتب السيناريو في النص الأصلي، ثم نسخ جزءاً زائداً من النص بعد تبديل الاسمين. [النتائج والنموذج](../references/role_binding_result.md)

[![93 · 雨巷来信](../publication/ilm-nature/figures/Figure_93_role_binding.png)](../publication/ilm-nature/manuscript.pdf#page=134)

ماذا يحدث عند تبديل اسمين في صورة نص؟ في 136 سؤالاً عن أزواج من الأدوار استُبعدت من تدريب الإجابات ذات الجملتين، قسّم النموذج الطلب وكتب الجملتين بالترتيب الصحيح في 70 إجابة. أصاب النموذج الذي يكتب الجملتين دفعة واحدة في ثلاث فقط. وفي مثال عن فيلم، كتب جملتين مقروءتين لكنه نسب الإخراج وكتابة السيناريو إلى الشخص نفسه. [النتائج والنموذج](../references/grounded_proposition_result.md)

[![图 92 · Film credits](../publication/ilm-nature/figures/Figure_92_grounded_proposition.png)](../publication/ilm-nature/manuscript.pdf#page=133)

من رسم الخريطة ومن احتاج إلى السؤال عن الطريق؟ أعاد النموذج بعد تعديله اسمين مختلفين، وأعطى الاسم الصحيح في 42 من 64 سؤالاً مصوراً جديداً. واستطاع كتابة شكل الحرف 言 في نقوش العظام وفي نقوش البرونز بالترتيب المطلوب. وفي 128 سؤالاً مفتوحاً جديداً تتطلب أشكالاً كتابية غائبة عن المدخلات، لم تتطابق أي إجابة تماماً مع الأجوبة المرجعية. [النتائج والنموذج](../references/pointing_causal_writer_result.md)

[![言 · 91](../publication/ilm-nature/figures/Figure_91_pointing_causal_writer.png)](../publication/ilm-nature/manuscript.pdf#page=132)

تذكر الفقرة المصوّرة مَن رسم الخريطة ومَن ضلّ الطريق. حدّد النموذج رسّام الخريطة، لكنه كرّر الحديث عن الرسم عندما سُئل عمّن احتاج إلى السؤال عن الطريق. في 300 سؤال تطلّبت إجابتها شكلاً كتابياً غائباً عن المدخلات، اختار النموذج أول شكل غائب اختياراً صحيحاً 18 مرة فقط، حتى عند تزويده بالجزء الصحيح السابق من الإجابة. [النتائج وصور الإجابات الأصلية](../references/causal_glyph_writer_result.md)

[![سؤالان عن خريطة واحدة](../publication/ilm-nature/figures/Figure_90_causal_glyph_writer.png)](../publication/ilm-nature/manuscript.pdf#page=131)

يستطيع النموذج كتابة الشكلين التاريخيين للحرف 言 بالترتيب المطلوب في السؤال المصوّر. في أمثلة خارج التدريب نجح في الترتيبين معًا في 94 من 96 مقارنة بصياغة مألوفة، وفي 16 من 96 على بطاقات أخرى بصياغة جديدة. [النتائج وصور الإجابات](../references/ordered_page_writer_result.md)

[![言 · 89](../publication/ilm-nature/figures/Figure_89_ordered_page_writer.png)](../publication/ilm-nature/manuscript.pdf#page=130)

**قراءة صور الحروف والإجابة عنها.** أجاب النموذج إجابة صحيحة عن 46 سؤالًا من أصل 64 عن شخصين بالخط الصيني العادي، وعن 41 بالخط العريض. وأجاب Qwen3-8B المحلي عن 55 سؤالًا إجابة صحيحة عند إدخال النص. وعند استخدام خط كاي لم يسبق للنموذج التدرّب عليه، أجاب عن سؤال واحد فقط إجابة صحيحة. [تشغيل النموذج](../docs/visual-page-responder.md)

[![الشكل 88. صور الأسئلة والإجابات الفعلية](../publication/ilm-nature/figures/Figure_88_font_invariant_reader.png)](../publication/ilm-nature/manuscript.pdf#page=129)

**هل يميّز النموذج بين شخصين في جملة واحدة؟** كتب النموذج اسم المتبرع لمبنى غيتس واسم المهندس المعماري بشكل صحيح. ومن أصل 864 زوجًا من الأسئلة أجاب عن السؤالين معًا إجابة صحيحة في 100 زوج. وعند تبديل الأسماء في 24 موقفًا جديدًا لم يُجب عن كل أسئلة أي موقف، بينما نجح نموذج Qwen3-8B المحلي في 20 موقفًا. [النتائج وطريقة التشغيل](../references/shared_sentence_writer_result.md)

[![الشكل 86. سؤالان عن الجملة نفسها وصور الإجابات](../publication/ilm-nature/figures/Figure_86_shared_sentence_writer.png)](../publication/ilm-nature/manuscript.pdf#page=127)

**لماذا يختار النموذج الاسم الخطأ؟** في 387 سؤالًا صينيًا عن هوية شخص ارتفع عدد الإجابات الصحيحة بالكامل من 62 إلى 116 عند اختيار جملة بحسب أشكال الحروف المشتركة مع السؤال. وبالطريقة نفسها أجاب نموذج المقارنة ذو التعيين العشوائي في التدريب الإضافي عن 127 سؤالًا إجابة صحيحة. يوضح الشكل 85 كيف كتب النموذج اسم مكان حين سُئل عن اسم مخترع. [النتائج وطريقة التشغيل](../references/inverse_role_writer_result.md)

[![الشكل 85. اختيار الجملة وصور الإجابات](../publication/ilm-nature/figures/Figure_85_inverse_role_writer.png)](../publication/ilm-nature/manuscript.pdf#page=126)

**عمّن تتحدث الجملة؟.** كتب النموذج 368 جملة صينية كاملة بصورة صحيحة من أصل 404 عند استخدام أسماء وتتمات جديدة. وعندما نسخ اسمي تشين وتشو من صور نقوش برونزية، أصاب في 24 جملة من أصل 32. وعندما عُرضت عليه الصفة وحدها وطُلب منه تحديد الاسم، لم يُرجع أي اسم برونزي صحيح. [النتائج وطريقة التشغيل](../references/visual_role_writer_result.md)

[![الشكل 84. فقرتان تتضمنان نقوشاً برونزية وصورتا الإجابة الفعلية](../publication/ilm-nature/figures/Figure_84_visual_role_writer.png)](../publication/ilm-nature/manuscript.pdf#page=125)

**ماذا يقرأ النموذج من صور الكتابة؟** عُرضت على النموذج صورة الجملة الصينية «她喝了一口□。» (أخذت رشفة من □). عندما تغيّر الشراب في الفقرة المرافقة من الماء إلى الشاي، تغيّرت الإجابة من 水 إلى 茶. واستعاد النموذج، بعد الجمع بين الشبكة العصبية وتكرارات المحارف المتجاورة، النص الأصلي في 651 موضعاً من أصل 2,947. وعند تبادل شخصين دوريهما في 48 زوجاً من الأمثلة، لم يصدر النموذج الحكم الصحيح على النسختين معاً في أي زوج. [النتائج وأوامر التشغيل](../references/visual_inference_writer_result.md)

[![الشكل 83. فقرتان وصورة الجملة نفسها وصورتا الإجابة](../publication/ilm-nature/figures/Figure_83_visual_inference_writer.png)](../publication/ilm-nature/manuscript.pdf#page=124)

<!-- paired-writing-status:start -->
لم تحقق مقارنة إزالة الهوامش البيضاء، بعد 512 خطوة تحديث، نسخاً موثوقاً: التطابق التام عبر OCR هو 6/32 للتدريب و10/32 للتحقق، مقابل 27/32 و28/32 للصور المرجعية الواضحة. تظهر المخرجات الأصلية حذفاً وتكراراً للأحرف. انخفض وقت التدريب من 795.624 إلى 287.182 ثانية، لكن فهم اللغة والكفاءة عند جودة عملية لم يثبتا بعد. [المخرجات الأصلية الـ64](../references/paired_padding_result.md).
<!-- paired-writing-status:end -->

الحالة الحالية: لم تثبت بعد صلاحية توليد اللغة على هيئة صور للاستخدام الفعلي. في أحدث مقارنة لدوال خسارة المفكّك على 32 سؤالاً سبق استخدامها في التدريب، حقق كلا النموذجين 0/32 إجابات مستقلة مطابقة بكسلاً ببكسل. أدّت الخسارة المراعية لحساسية المفكّك إلى خفض F1 لضربات الكتابة من 85.984% في النموذج الضابط إلى 70.009%. وحتى مع إعطاء السياق السابق الصحيح للإجابة، لم يتجاوز F1 نسبة 92.430% و75.116%؛ أما إعطاء رموز صورة الإجابة الصحيحة مباشرة فحقق إعادة بناء بنسبة 99.9953% في الحالتين. فشلت معايير الإصلاح الأربعة، ولذلك رُفض توسيع نطاق هذه الوصفة التدريبية. هذه مقاييس تطابق صور وليست دقة دلالية، ولا توجد بعد نقطة تحقق لنموذج ILM ثبتت صلاحيتها للاستخدام. [النتيجة وكل الصور الأصلية](../references/decoder_metric_v82_result.md). تتضمن المخطوطة الآن الملاحظة التكميلية 39 والشكل S16 القابل للتحرير. حقق خط أساس إضافي لمولّد صور خارجي، دون تدريب محلي، تطابق OCR تاماً في 0/12 من طلبات النسخ و0/8 من الأسئلة، مع معايرة OCR للصور المرجعية بلغت 18/20؛ حُفظت المخرجات الأصلية الأربعون. [FLUX.2-klein-4B](../references/image_edit_baseline_result.md).



[![LazyingArt banner — apps and four buddies](https://lazying.art/media/brand/lazyingart-intro.gif)](https://lazying.art)

# نموذج اللغة المصوّرة (ILM)

عند عرض السطر نفسه 舉杯邀明月، يكتب النموذج **cup** عندما يُسأل عمّا رُفع، و**moon** عندما يُسأل عمّن دُعي. ظهر هذا البيت في تدريب سابق. وفي اختبار منفصل حدّد النموذج كل الأفعال ومفاعيلها وكتب معنى كل مفعول بالإنجليزية في 247 جملة من أصل 971، مقابل 11 عند تبديل المفاعيل أثناء التدريب. ولم يُجب عن كل الأسئلة في أي من الجمل الـ21 التي كانت فيها جميع الصياغات الطبيعية جديدة. [تشغيل النموذج](../models/questioned_events/README.md) · [النتائج](../references/questioned_events_result.md)

[![126 · ILM](../publication/ilm-nature/figures/Figure_126_questioned_events.png)](../publication/ilm-nature/manuscript.pdf#page=167)

عندما وصف السؤال الفعل بالإنجليزية، حدّد النموذج البصري موضعي المفعولين وكتب معناهما بالإنجليزية في 44 زوجًا من أصل 268 زوجًا من الجمل الصينية الكلاسيكية. وأصاب النموذج السابق في 11 زوجًا. وفي عشرة أزواج من قصص بو سونغ لينغ اختار النموذج السطرين الصحيحين في أربعة أزواج، ولم يكتب الإجابتين الصحيحتين في أيّ زوج. يعرض الشكل 125 إجابته الفعلية عن سؤال يتعلق بثمرة خوخ. [تشغيل النموذج](../models/visual_action_meaning/README.md) · [النتائج](../references/visual_action_meaning_result.md)

[![125 · ILM](../publication/ilm-nature/figures/Figure_125_visual_action_meaning.png)](../publication/ilm-nature/manuscript.pdf#page=166)

تجمع صفحة واحدة بين 焉得仁 و策其馬. حين يُسأل النموذج بالإنجليزية عمّا ينالونه يكتب **benevolence**، وحين يُسأل عمّا يضربونه بالسوط يكتب **horse**. حدّد المفعول به وكتبه في الجملتين على 703 من أصل 1691 صفحة اختبار مع تبديل ترتيب السطرين وصيغة السؤال. وعند إعادة صياغة الفعل على الصفحات الـ188 نفسها، انخفض عدد الصفحات الصحيحة بالكامل من 95 إلى سبع. [تشغيل النموذج](../models/pictured_action_lookup/README.md) · [النتائج](../references/pictured_action_lookup_result.md)

[![124 · ILM](../publication/ilm-nature/figures/Figure_124_pictured_action_lookup.png)](../publication/ilm-nature/manuscript.pdf#page=165)

يقرأ النموذج المقطع الصيني الكلاسيكي والسؤال في صورتين، ثم يكتب الإجابة الإنجليزية في صورة. في المقطع 還其珠 يكتب **return** أو **pearl** بحسب السؤال. نجح النموذج، بحجمه البالغ 1.40 مليون معامل، في كتابة القراءتين *it* و*go* للرمز 之 في 31 من 40 زوجًا من السياقات المحجوزة للاختبار، مقابل أربعة أزواج عند خلط إجابات التدريب. وفي كتابين آخرين أجاب عن الأسئلة الأربعة كلها إجابة صحيحة في مقطع واحد من أصل 29. [تشغيل النموذج](../models/pronoun_writer/README.md) · [النتائج](../references/pronoun_writer_result.md)

[![122 · ILM](../publication/ilm-nature/figures/Figure_122_pronoun_writer.png)](../publication/ilm-nature/manuscript.pdf#page=163)

عند عرض 執大象 وسؤال مصوّر، يرسم النموذج كلمة **hold** للفعل وكلمة **image** للمفعول به. أجاب عن الأسئلة الثمانية بالصينية والإنجليزية إجابة صحيحة في 230 من 539 مقطعًا، وارتفع العدد إلى 247 عند تقييد الإجابات بالمعاني التي ظهرت في التدريب. ما زالت صياغات الأسئلة اليومية صعبة. [تشغيل النموذج](../models/asked_gloss/README.md) · [نتائج التجربة](../references/asked_gloss_result.md)

[![120 · ILM](../publication/ilm-nature/figures/Figure_120_asked_gloss.png)](../publication/ilm-nature/manuscript.pdf#page=161)

زادت الإجابات المقتبسة بدقة من 17 إلى 41 في 412 سؤالًا إنجليزيًا من MLQA باستخدام تمثيل صورة السؤال. وفي 824 سؤالًا باللغتين، أنتج نموذج الكتابة 111 إجابة دقيقة توقف فيها تلقائيًا عند تزويده بالمقتطف، مقابل 90 عند تزويده بالصفحة كاملة. [↗](../references/question_evidence_transfer_result.md)

[![108 · ILM](../publication/ilm-nature/figures/Figure_108_question_evidence_transfer.png)](../publication/ilm-nature/manuscript.pdf#page=149)

طابق المُرمّز البصري الأسئلة الإنجليزية بنظيراتها الصينية في 364 من أصل 374 مقطعًا روائيًا. وفي 92 سؤالًا عن خمسة أعمال أخرى، اختار نموذج القراءة 31 مقتطفًا صحيحًا، وقدّم نموذج الكتابة 19 إجابة دقيقة عند تزويده بالمقتطفات. [↗](../references/bilingual_book_intervals_result.md)

[![107 · ILM](../publication/ilm-nature/figures/Figure_107_bilingual_book_intervals.png)](../publication/ilm-nature/manuscript.pdf#page=148)

بعد التدريب على 642 إجابة أخطأ فيها سابقًا، أعطى النموذج البصري 31 إجابة مصوّرة مطابقة من أصل 144 سؤالًا عن ستة مؤلفات صينية أخرى. وأجاب عن السؤالين بالصينية والإنجليزية إجابة صحيحة في 0 من أصل 36 مقطعًا. [↗](../references/stopped_book_preference_result.md)

[![106 · ILM](../publication/ilm-nature/figures/Figure_106_stopped_book_preference.png)](../publication/ilm-nature/manuscript.pdf#page=147)

في اختبار شمل 34 مقطعاً من أربعة كتب صينية أخرى، أعطى النموذج الذي زيد وزن تدريب الحرف الأول فيه 31 إجابة مطابقة من أصل 136 سؤالاً بالصينية والإنجليزية. ولم يُجب عن الأسئلة الأربعة كلها إجابة صحيحة في أي مقطع. [↗](../references/first_glyph_book_result.md)

[![105 · ILM](../publication/ilm-nature/figures/Figure_105_first_glyph_book.png)](../publication/ilm-nature/manuscript.pdf#page=146)

طُرح سؤالان عن كل مقطع روائي صيني، مرة بالصينية ومرة بالإنجليزية. أجاب النموذج الثنائي اللغة عن الطلبات الأربعة بدقة في 3 من 374 مقطعاً، مقابل 0 للنموذج الضابط الذي بُدّلت أسئلته الإنجليزية أثناء التدريب. وفي اختبار XQuAD أجاب النموذج الثنائي بدقة عن الأسئلة الأربعة في 1 من 227 مقطعاً. [النتائج وصور الإجابات](../references/bilingual_novel_result.md)

[![104 · BiPaR](../publication/ilm-nature/figures/Figure_104_bilingual_novel.png)](../publication/ilm-nature/manuscript.pdf#page=145)

رفع تغيير سماكة الخط ونسب الحروف أثناء التدريب عدد الإجابات الصحيحة في أسئلة القراءة الصينية بخط HanaMin من 15 إلى 43 من أصل 394. أجاب النموذج عن السؤالين معاً في 6 من 197 مقطعاً، وفي اختبار XQuAD مستقل بالخط نفسه لم يُجب عن السؤالين معاً في أي من المقاطع الـ227. [النتائج وصور الإجابات](../references/stroke_scale_result.md)

[![103 · HanaMin](../publication/ilm-nature/figures/Figure_103_stroke_scale.png)](../publication/ilm-nature/manuscript.pdf#page=144)

في مقاطع صينية جديدة طُبعت بخط HanaMin المستبعد من التدريب، أعطى النموذج 15 إجابة مطابقة من أصل 394، مقابل 3 عند الاكتفاء بالتدريب على خطوط إضافية. وأجاب عن السؤالين معاً في 0 مقطعاً من أصل 197. يتيح الفهرس المحلي طلب الأشكال التاريخية لرمز مثل 中 وتلقّي الإجابة في صورة. [تشغيل النموذج والفهرس](../models/typeface_writer/README.md)

[![102 · 中](../publication/ilm-nature/figures/Figure_102_typeface_writer.png)](../publication/ilm-nature/manuscript.pdf#page=143)

أجاب النموذج المدرَّب بالتقطير بدقة عن 72 سؤالاً من أصل 196 حول صور 98 مقطعاً صينياً، مقابل 63 مع التدريب المعتاد. وأجاب كل من النموذجين عن السؤالين معاً في 11 مقطعاً. وعندما تلقّى Qwen3-8B المقاطع نفسها في صورة نص، أجاب عن السؤالين معاً في 55 مقطعاً. [النتائج والنموذج](../references/interval_teacher_result.md)

[![101 · 72/196](../publication/ilm-nature/figures/Figure_101_interval_teacher.png)](../publication/ilm-nature/manuscript.pdf#page=142)

أجاب النموذج نفسه بدقة عن 114 سؤالاً من أصل 359 سؤال فهم صيني، واستخرج من بطاقة مصوّرة شكلي الرمز 言 في نقوش العظام والبرونز وكتبهما بالترتيب المطلوب. وفي 128 مقطعاً إضافياً لكل منها سؤالان، أجاب عن كليهما في 15 مقطعاً، مقابل 49 لنموذج يستخرج الإجابة من النص. [تشغيل النموذج](../models/mixed_visual_answers/README.md)

[![100 · 言](../publication/ilm-nature/figures/Figure_100_mixed_visual_answers.png)](../publication/ilm-nature/manuscript.pdf#page=141)

قرأ النموذج نصوصًا صينية مصوّرة وكتب الاسم والسنة الصحيحين بعد كل تغيير وبثلاث صيغ جديدة للسؤال، في 34 من 64 حالة بفقرة واحدة و8 من 48 حالة بفقرتين. وطابقت إجاباته المرجع في سبعة من 359 سؤال فهم أوسع نطاقًا كتبها أشخاص.

[![98 · 1993 → 1994](../publication/ilm-nature/figures/Figure_98_page_evidence.png)](../publication/ilm-nature/manuscript.pdf#page=139)

![Python](https://img.shields.io/badge/python-3.10%2B-3776AB?logo=python&logoColor=white)
![Status](https://img.shields.io/badge/status-research-orange)
![Focus](https://img.shields.io/badge/focus-text--as--image-0A7EA4)
![Paradigm](https://img.shields.io/badge/paradigm-predictive%20visual%20field-16835B)
![License](https://img.shields.io/badge/license-unspecified-lightgrey)
![Domain](https://img.shields.io/badge/domain-historic%20etymology%20%7C%20glyph%20models-2F80ED?logo=github)

ILM مشروع بحثي لتعلّم اللغة وتوليدها بوصفها **كتابة مرئية**، من الصورة إلى
حالة بصرية مستمرة ثم إلى صورة، من دون مسار رموز نصية مخفي.


فحص الواجهة الأحدث: يجيب النموذج المسبق المجمد إجابة مطابقة في 11/32 حالة إنجليزية بعد تغيير حجم الصفحات نفسها من 1024 إلى 432، مقابل 0/32 بالحجم الأصلي. يحقق كل من المحولين المدرَّبين 5/32 فقط بالحجم 432. لم يتحقق بعد نموذج ILM مستقل يولّد الإجابات كصور.

[2026-11-07: English result and full evidence](../references/native_reader_geometry_result.md)

نتيجة 2026-11-08: اكتمل تقييم التدريب ذي 128 تحديثاً، وفُحصت صور الإجابات الأصلية الـ384. يحقق النموذجان الأولي والنهائي 0/32 إجابة مطابقة وفق OCR في كل من الأسئلة التي دُرّب عليها النموذج والأسئلة المحجوبة، حتى مع بدائل المصدر. انخفض النسخ المطابق في المجموعة المحجوبة من 19/32 إلى 16/32. أخفقت الشروط الستة المحددة مسبقاً؛ ولم يُثبت تعلّم اللغة بعد.

[النتيجة وحدود الدليل](../references/native_paired_qa_learning_result.md)

## قراءة جملة صينية كلاسيكية من صورة وكتابتها بالصينية الحديثة

نسخ النموذج، الذي يضم 2.03 مليون مُعامِل، المقاطع الـ 2,045 القابلة للرسم كاملة في فصول خُصصت للاختبار. وعند طلب الترجمة إلى الصينية الحديثة، طابقت 30 ترجمة من أصل 2,048 الصورة المرجعية تمامًا. وفي إحدى الجمل نسخ تسعة رموز ورسم رمزين جديدين، فاستبدل 薨 بـ 去世. وظل أسلوب التحرير البسيط المعتمد على السياق القريب أقل خطأً في مجمل النتائج. [تشغيل النموذج](../models/visual_edit_zh/README.md) · [صور المخرجات والمراجع](../publication/ilm-nature/figures/Figure_48_visual_edit.png)

## قراءة الأشكال التاريخية

أعاد النموذج الصور التاريخية بالترتيب المطلوب في 9,565 من أصل 9,600 سؤال بالصينية والإنجليزية عن عائلات محارف لم تدخل التدريب. تأتي الصور من جدول معروض من أربعة صفوف. [تشغيل النموذج](../models/pictured_idiom_zh/README.md) · [Fig. 42](../publication/ilm-nature/figures/Figure_42_historical_plates.png)

## قراءة النص المصوّر وكتابة التعبير الصيني الناقص

يقرأ نموذج يضم 2.12 مليون مُعامِل صور النص على جانبي الفراغ ثم يرسم أحرف التعبير الصيني الناقص. طابقت إجاباته الأصل تمامًا في 134 حالة من أصل 22,907 حالات في مجموعة التحقق. وفي 100 حالة من هذه الإجابات الصحيحة، لم يظهر أيّ من الأحرف الأربعة في النص المعروض. وأعطت طريقة مرجعية تعتمد على التكرار في السياق القريب 1,012 إجابة صحيحة.

[تشغيل النموذج](../models/pictured_idiom_zh/README.md) · [الشكل 40](../publication/ilm-nature/figures/Figure_40_pictured_idioms.png)

عند استخدام تنبؤات نموذج القراءة لتوجيه التدريب الإضافي، بلغ عدد الإجابات المطابقة 378 من أصل 22,907، مقابل 292 عند التدريب للمدة نفسها من حيث عدد التحديثات على الإجابات الأصلية وحدها. وعلى 9,135 مقطعًا لم تُستخدم سابقًا، بلغ العدد 163 و125 على التوالي. [الشكل 41](../publication/ilm-nature/figures/Figure_41_writer_transfer.png)

بعد تدريب القارئ والكاتب البصريين معًا، أنتج النموذج الإجابة الصحيحة في 433 من 17,555 مقطعًا صينيًا جديدًا، مقابل 276 للنموذج السابق و789 لطريقة العدّ في السياق القريب. [Fig. 43](../publication/ilm-nature/figures/Figure_43_expression_preference.png)

## اختيار الإجابة من صور الحروف الصينية

تدرّب نموذج القراءة الصغير في 64 ثانية باستخدام خيطين من المعالج المركزي. وفي فقرات صينية حُجزت للتقييم، اختار صورة الإجابة المطابقة تمامًا في 269 سؤالًا من أصل 1,788. وعند إخفاء السؤال انخفض العدد إلى 18. أما نموذج الكتابة الحرة المنفصل فلم ينتج شرحًا مقروءًا لتطوّر الحرف 言. وتظلّ القدرة على الكتابة العامة وزيادة كفاءة التدريب ألف ضعف هدفين قيد الاختبار.

![اختيار الإجابة من صور الحروف الصينية](../publication/ilm-nature/figures/Figure_34_visual_reading_writing.png)

[تشغيل نموذج القراءة](../references/visual_extractive_reading_result.md) · [البحث](../publication/ilm-nature/manuscript.pdf)

اختار نموذج القراءة المقطع الصحيح في 565 سؤالًا من أصل 3,170 سؤالًا صينيًا. وأعاد نموذج الكتابة رسم 124 إجابة كاملة بخط مختلف مع تطابق كل بكسل. أجاب عن سؤالين حول الفرقة نفسها بعبارتي 前卫金属 و1996年.

![كتابة الإجابة بخط مختلف](../publication/ilm-nature/figures/Figure_38_grounded_detail.png)

[كتابة الإجابة بخط مختلف](../references/grounded_detail_writing_result.md)

## رسم الحرف الصيني التالي من السياق

في 668 سياقًا من مجموعة التطوير، أدى توقّع التوزيع العام للحبر قبل رسم التفاصيل إلى زيادة عدد الصور المطابقة بكسلًا ببكسل لأشكال كتابية معروفة غير فارغة من 198 إلى 649. رسم كلّ من النموذجين الحرف التالي الوارد في النص الأصلي 18 مرة. وعند السؤال عن أصل الحرف 言، كرّر النموذج الجديد شكلًا يشبه الصفر.

![رسم الحرف الصيني التالي من السياق](../publication/ilm-nature/figures/Figure_37_shared_layout.png)

[نتائج التجربة والنموذج المدرّب](../references/shared_layout_writing_result.md)

## تجربة إكمال النص السابقة

2026-11-13

شملت تجربة التسلسلات الكاملة **13,766 تسلسلاً و41,298 تحديثاً و6,108,741 موضعاً مستهدفاً لأشكال الكتابة**. روجعت **صور المخرجات الـ732 جميعها** قبل التقييم بالإجابات المرجعية. مع المدخلات ذات التنضيد النسبي، حقق النموذج النهائي **7/32 إجابات كاملة مطابقة بالمعيار الصارم، و0/16 زوجاً صحيح الإجابتين، و14/32 إجابة مقروءة، و0/12 متابعة طويلة للكتب متماسكة ومكتملة**. لم يجتز المعايير المسجلة مسبقاً ولا التدقيق المستقل لحالات محو السؤال أو الفقرة الأربع.

استُخدمت بطاقة RTX 4090 واحدة، وبلغت ذروة ذاكرة VRAM المخصصة **18.568 GiB**. يحتوي النموذج المنشور للتشغيل على **7,578,390,003 معامل** بما فيها أوزان الأساس الموروثة، بينما حُدّث **20,212,225 معامل** في هذا التدريب. استُخدمت مجموعة تطوير معاد استخدامها ومراجع نموذجي واحد، دون تحكيم بشري مستقل. لم تثبت بعد صلاحية ILM خفيف للاستخدام أو أفضلية كفاءته عند جودة مماثلة.

[النتائج الكاملة](../references/visual_continuation_learning_result.md) · [الهدف الهندسي](../docs/first-imagized-language-model-goal.md) · [البروتوكول الثابت](../references/visual_continuation_full_protocol.md)

## نتائج V35 السابقة

![نتيجة V35 المقاسة: يولد نموذج كامل بمدخلات ومخرجات نقطية علامات تتأثر بالتوجيه، لكنه يفشل في النسخ وربط التعليمات](../publication/ilm-image-native/figures/causal_glyph_flow_v35_result.png)

V35 نموذج لغة بصرية صيني يضم `129,092,738` معلّمة. أكمل جميع تحديثات BF16
البالغ عددها `22,000` خلال `2.770` ساعة على بطاقة RTX 4090 واحدة، وبذروة ذاكرة
GPU قدرها `2.899 GiB`. لا يستقبل مسار التشغيل إلا بكسلات الكتابة وقناعاً
بصرياً، ثم يولد الصورة النقطية ويعيد قراءتها مباشرة. ولا يستخدم أثناء التشغيل
tokenizer أو معرفات Unicode/الأحرف أو OCR أو الاسترجاع أو codebook بصرياً أو
نموذجاً معلماً.

القرار المقاس هو **`not-qualified`**. دقة النسخ مع التوجيه الصحيح `0.3125%`،
ودقة التعليمات `0.1116%` وهي أقل من حالة التوجيه المخلوط. نجح المسار في `5/12`
فقط من بوابات السببية البصرية و`5/9` من بوابات الدلالة النقطية. المخرجات غير
فارغة وتتغير مع التوجيه، لكن علاماتها مشوهة وغير مرتبطة بالإجابة المطلوبة؛ لذلك
لم تُفتح مجموعة الاختبار المختومة.

يعتمد V35 على نقل التعلم. مُرمّز الكتابة المستمر V34 هو من عمل المشروع، أما
النواة السببية فمهيأة من نقطة PIXAR الخارجية. يُنسب PIXAR بوضوح بوصفه أساساً
خارجياً ولا يُدّعى أنه مساهمة من المشروع. وتعني «الاستقلالية» هنا حزمة تشغيل
ذاتية الاكتفاء، لا تدريب كل المكونات من الصفر. راجع [التقرير المقاس بالإنجليزية](../references/causal_glyph_flow_v35_result.md)،
[دليل التنفيذ](../docs/causal-glyph-flow-v35.md)، و[البحث](../publication/ilm-image-native/ilm-image-native.pdf).

## اختبار توجيه محدود سابق: قبول دائرة العلاقات البصرية V23

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

<!-- support:start -->
[![GitHub Sponsors](https://img.shields.io/badge/GitHub-Sponsors-EA4AAA?style=for-the-badge&logo=githubsponsors&logoColor=white)](https://github.com/sponsors/lachlanchen)
[![LazyingArt Support](https://img.shields.io/badge/LazyingArt-Support-0EA5E9?style=for-the-badge)](https://chat.lazying.art/donate)
[![PayPal Support](https://img.shields.io/badge/PayPal-Support-00457C?style=for-the-badge&logo=paypal&logoColor=white)](https://paypal.me/RongzhouChen)
[![Stripe Support](https://img.shields.io/badge/Stripe-Support-635BFF?style=for-the-badge&logo=stripe&logoColor=white)](https://buy.stripe.com/aFadR8gIaflgfQV6T4fw400)
<!-- support:end -->

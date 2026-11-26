# Imagized Language Model: Engineering Goal

## What qualifies as a working visual language model

The target is an independent, trainable reader and writer on one RTX 4090-class GPU. A user can present a pictured question together with a book page, or render a typed prompt into the same image stream. The model must produce a legible, grounded answer image and stop. Test reading, composition, free continuation, instruction changes and meaning-preserving changes of font and layout in English and Chinese. For etymology, test retrieval of the correct attested source, distinction of historical forms, their order, and the modern-language explanation on unseen headwords. Measure raw-page segmentation separately from results on aligned cells. Report the complete system needed at deployment, including externally trained dependencies. An optional transcription may follow the emitted image.

**Primary goal, 26 November 2026.** Establish that an independent reader can preserve a proposition and write a complete answer from pictured language on one RTX 4090-class card. Prioritize semantic accuracy and transfer over power and energy. Keep original input and answer PNGs, show both correct and incorrect cases, and report the source of every generated glyph. Establish any efficiency advantage at matched language quality.

**Current model decision, 26 November 2026, writing the relation between words.** A greater variety of pictured passages preserved 108/118 grammatical families against 69 for repetition, yet both writers received 1/96 whole-meaning credits. Explicit source contrast gained one credit on a separate panel; associative recall tied its ordinary and shuffled controls at 3/96. The nearest actual writing states could supply only 27/2,348 four-Han continuations, while 1,560 were available as one or two fragments across the full library. Prioritize how the writer binds a subject to its predicate and object while composing a fresh answer. The existing trilingual annotations align exactly with the training images on both sides. Evaluate real role supervision against matched disrupted alignment and ordinary likelihood, with inference from the picture alone, raw outputs, distinct works and complete-meaning scores. Keep model and data costs explicit. [Results](../references/visual_library_writing_result.md)

**Preceding model decision, 26 November 2026, transfer to unseen works.** Continue toward an independent glyph-image reader and writer capable of faithful natural answers. With supervision and source images matched, the distilled writer earned 0/96 registered and 3/96 source-only semantic credits, compared with 0 and 2 for its start and 58 and 88 for the quantized text teacher. In a final-pass diagnostic it reproduced 1,117/2,614 training translations. The selected writer preserved 113/118 controlled role families. Keep this measured failure and its genuine successes. Test a larger number of distinct natural propositions at a matched training budget on work-disjoint evaluation. The next corpus already has 28,342 aligned training pairs from 21 new works and six separate evaluation works. Retain raw pixels, both grading procedures and comparative whole-system cost at matched quality. [Result and Lean diagnostic](../references/complete_teacher_translation_result.md)

**Preceding model decision, 26 November 2026, natural writing.** On one RTX 4090 D, training on the starting writer's mistaken glyphs retained 111/118 exact controlled role families across two fonts but earned 0/144 fresh whole-translation credits. The quantized text comparator earned 96. Increasing the matched reference-prefix writer from 1.69M to 14.90M parameters gave 0/64 on 32 further reserved articles, against Qwen’s 49. Both widths gave only one credited translation on 96 fitted natural requests. Preserve the visual models and negative results. Prioritize instruction in complete meanings by an offline bilingual teacher. A training-only pilot found 56/96 teacher replies writable, stopped and accepted by both reference-informed and reference-blind assessments. Obtain independent bilingual checks given the shared model family, then compare teacher versus corpus targets on the same pictured sources with matched student resources. Require independently written, stopped and semantically correct images on new document families, the controlled role retention criterion and inspection of original PNGs. Continue training when semantic development gains justify the cost. Capacity alone and the edit-recovery theorem establish no efficiency multiplier. [Matched prefix comparison and proof](../references/visual_prefix_recovery_result.md) · [Width trial and successor design](../references/visual_writer_width_result.md)

**Preceding model decision, 26 November 2026.** On the same 96 fresh literary and encyclopaedic requests the rehearsal writer ranked the true source first for 90 supplied target images, preferred the true source order to reversed order on 93, and earned zero credits for independently written translations. Qwen3-8B Q4 earned 58. The learned optical adapter carried 110 complete role families across a change to Noto Serif and all 96 fresh answers were exactly the same as in the original font. Retain the writer as a controlled-grammar baseline and the adapter for aligned known-repertoire printed input. The next learning comparison should target autonomous generation of a complete proposition from its own preceding output glyphs. Use a separate development set of natural passages to compare training on model-produced prefixes, with teacher correction where available, against equal-exposure reference-prefix training. Reserve new source documents before fitting. Require better free writing, a matched text comparator, and retention of role reversals and optical transfer. Source ranking and glyph fidelity are secondary outcomes. [Raw outputs, tests and limits](../references/book_rehearsal_result.md) · [Optical stability theorem](../formal/span_list/InkSeparation.lean)

**Preceding empirical decision, 26 November 2026, bilingual books.** A matched single-card continuation on 231,337 distinct pictured pairs produced 5/144 automated complete-meaning credits against zero for a 31,918-pair replay. Three responses reproduce the full reference image and stop. One graded credit drops a location. Quantized Qwen received 86. Both visual continuations lost the previously measured passive construction (422/472 exact to zero). Neither translated the 24 held classical passages into a credited English reply. Preserve both checkpoints and original PNGs. The next intervention is generative rehearsal during broad training. Match total updates, glyph exposure and starting weights against continued broad reading. Train on separate human role/voice examples or distil the earlier writer on disjoint prompts, reserve unseen complete propositions, and test both new book meanings and retention on role-swapped images. Require simultaneous progress on the two outcomes. If rehearsal only restores the old construction, change capacity or teacher transfer as a separate experiment. Whole-page inference, historical explanations, independent human review and a matched-quality resource advantage remain open. [Data, math and measured results](../references/book_breadth_result.md)

**Preceding empirical decision, 26 November 2026.** On 48 new bilingual families a learned optical writer distinguished all eleven translation pairs whose sources became identical when glyph order was removed. It nevertheless earned zero whole-meaning credits across 192 freely written answers. Adding the two-assignment learning objective left recognition at 75/96 and worsened the average likelihood of the held correct translations relative to the starting writer. An order-sensitive decoder changed 192 outputs without improving complete meanings. Preserve these controls. The next comparison should train on a broader collection of clean generative English–Chinese sentence and passage targets, with the existing writer and disjoint development families used to choose stopping. Hold the architecture and starting weights fixed when testing the effect of corpus breadth. Reserve unseen document families, counterfactual role swaps, freely written PNGs, independently reviewed semantic credits and a matched text comparator. If broader targets fail to raise development and held semantic accuracy, test decoder capacity or transfer from a trained teacher as separate interventions. Evidence of language requires faithful new answers across wording and document changes. [Experiments and proofs](../references/paired_ink_result.md)

**Prior criteria, 25 November 2026.** The primary outcome is a complete, grounded answer drawn freely from a pictured question and page. Require unseen bilingual document families, same-page questions with different answers, blinded semantic judgment with human and foil calibration, and actual stopped output images. A Lean theorem must state a quantitative bound or an experimentally distinguishable prediction, identify the deployed quantities it applies to and survive a check on real outputs. Extend training over weeks or months while independent development shows progress on composition and visual transfer. Measure data preparation, antecedent training, memory, time and energy at matched answer quality before claiming an efficiency multiplier.

**Preceding empirical decision, 25 November 2026.** Preserve the compact aligned optical encoder for bilingual image retrieval and the varied-source checkpoint for controlled role-preserving writing. On 944 previously unexamined English requests, canonical and varied training both drew 471/472 Chinese targets from active sources. Varied training drew 422/472 from passive sources, compared with zero for its matched control. Across 128 natural pairs the aligned reader retrieved 109 correct Chinese and 112 correct English images. Both final writers earned 0/64 new PAWS-X meanings while quantized Qwen earned 46/64. Prioritize writing the full human event from a pictured source. For the next mechanism, distinguish a correct broad sentence match from preservation of two argument roles within the same sentence. Require freely generated answers, matched exposure, role-swapped human sentences, independent document families, independent bilingual review of a sealed sample and a competing text model on the same requests. Retain earlier image and manuscript evidence. General reading, etymological explanation and lower compute at matched quality remain outstanding. [Three matched comparisons and formal bounds](../references/visual_proposition_translation_result.md) · [Executable PNG translation](../models/translation_paraphrases/README.md)

**Preceding translation decision, 25 November 2026.** Preserve full-resolution addressable specimens and the finer source positions for ordinary translation. A formal substitution law predicts when grammar can be shared across opaque specimens. In practice strict paired transport with the same 1.77M weights rose from 5/70 to 56/70, yet only two full outputs matched their reference, on two assignments of one sentence. A finer 1.69M translator scored 4/64 conversational meanings to an 8B Q4 model’s 61/64. Adding 61,448 distinct short editorial pairs yielded 0/32 on separate article translations; the local comparator scored 26/32. The next semantic comparison should contrast human bilingual propositions differing in argument order, polarity or quantity, keep the identical-exposure likelihood control, and judge independent complete image replies in both directions. Preserve free reading, historical explanations, font transfer and matched-quality whole-system costs as separate requirements. [Theory, protocols and measured limits](../references/ordered_visual_translation_result.md) · [Lean 4 proofs](../formal/span_list/OrderedTranslation.lean)

**Earlier historical selection decision, 25 November 2026.** An independent 1.18M-parameter optical system learned conditional historical form selection on one RTX 4090 D. Counterbalancing language and row order yielded 63/64 sealed families passing all 16 requests; equal-exposure fixed continuation gave 0/64. On 64 additional families new paraphrases scored 261/512 and a simple visible-label rule scored 512/512. On the four-specimen 言 page the eight category queries passed but the request for an explanation yielded one glyph. On new English MCTest and Chinese C³ pages the whole-page writer freely matched only 1/64 human answers in each language. Preserve all four fitted weights, both contrasts, and all images. Distinguish template-conditioned selection, lexical matching, grounded composition, and cost at equal quality. Prioritize a held-source comparison requiring the right relation and an appropriate explanation with an attested image, with diverse question wordings, negative requests, a printed-label baseline, full modern-language replay and freely generated replies. Keep the sealed questions outside training. Data quality, independently judged meanings, raw-page transfer and comparisons with mature models remain acceptance criteria. [Result and controls](../references/visual_story_writer_result.md) · [MiniMind investigation](../references/minimind_visual_language_study_zh.md)

**Previous experiment on evidence selection.** Four trained, released selectors test whether better evidence selection improves the complete drawn answer. In the narrative experiment, training on the stopped human answer reduced held negative log likelihood by 0.400 nats (95% paired interval 0.157–0.636). Across 64 questions the parent, single-path, marginal and quantized vision comparator earned 14, 11, 13 and 41 semantic credits. On 92 other story questions, no available evidence strip let the frozen writer draw a verbatim reference. Preserve these weights and images. The next learning target is the writer's ability to state the requested relation from the visual question and the page. Start with full human replies, including punctuation and stopping; compare the frozen writer with matched writers trained using the selected strip alone or access to the whole pictured section. Use matched Chinese replay, new held document families, paired questions and explicit fact substitutions. Require a replicated gain in complete grounded answers and retention of Chinese performance before scaling the training duration. The independent reading/writing, historical-form and matched-efficiency goals remain active. [Measured answers](../references/story_answer_fit_result.md) · [Exhaustive selection test](../references/optical_competition_result.md)

**Preceding decision on evidence selection.** The directional reader provides a positive mechanism result. Exact English evidence improved from 37 to 43 of 192 against an equal-capacity unsigned control. Exchanging the directional features reduced it to 21. In Chinese, supplying the correct visible strip raised exact free-writing-stage answers from 13 to 181 of 189. With the ordinary page-only interface, the signed model earned 8/48 semantic credits to a quantized 8B vision model's 33. Keep the existing general baseline and publish the specialist and its actual answer images. The next bounded learning comparison is question-conditioned discrimination among competing candidates. On exactly 131 English and 77 Chinese questions the 32 candidates already contain an accepted strip, yet only 43 and 14 were selected. Hold the candidate list and writer fixed, compare equal exposure, and require gains in independently generated bilingual answers on fresh documents. English formulation and off-page language remain separate open requirements. [Measured evidence and decision](../references/directional_reading_result.md) · [Checked finite candidate ceiling](../formal/span_list/OpticalCandidateCeiling.lean)

**Previous objective and decision, 25 November 2026.** Build a reader that identifies the evidence asked for and a writer that preserves who did what to whom in a complete, freely generated image. On 384 new English and Chinese questions the retained model wrote 12/192 and 50/192 exact replies. Training two matched writers on 133,048 reference answers gave 6/40 and 5/38 English/Chinese successes. The second assigned 95/96 human English reference pairs to their questions with both choices supplied, yet freely answered both questions on 0/96 pages. An untrained count of shared individual glyphs also paired 95/96. Require an explicit overlap control before interpreting recognition as semantics. Retain the earlier system. On the preceding cohort a search bound certified 310 reference mismatches as maximum-probability stopped images under the frozen model and budget. Nine of thirteen English questions with exactly the annotated fact still had a reference mismatch at the global mode. All 39 Chinese questions with the exactly annotated fact had a matching mode. Include equivalent phrasings when assessing meaning. [Full evidence, weights and actual outputs](../references/visual_sentence_context_result.md)

**Next training decision.** On 48 new Chinese pages chosen to remove the shared-glyph cue, the retained model freely matched 19/96 answers; providing the correct fact raised this to 94/96. On 85 more English pages the shared-glyph count solved 84 assignments, and the model freely answered both questions correctly on zero. No remaining English page pair from this source met the predeclared overlap challenge. Put the next training comparison on question-conditioned evidence selection, preserving the current unrestricted writer and end-to-end tests. Use nonidentical relations on the same page and require the reader to distinguish a fact about a named entity from a fact about another entity. For English, source independent relational questions and independently judged paraphrases beyond QA2D. Treat a returned date, organization or person of the wrong relation as a reading error even when visually flawless. Measure the value added by any teacher before using it for sustained distillation. [Fresh bilingual tests](../references/evidence/visual_lexical_controls_v1/test.json) · [Natural intake and missing English challenge coverage](../references/evidence/visual_lexical_controls_v1/reservation.json)

**Conditions for the next model.** Learn to select the correct argument for a pictured question, preserve its grammatical role when composing, and produce both short and extended responses. Train on substantial, source-disjoint human questions, books and complete responses. Offline teachers may augment supervision. Keep the deployed reader and writer independent, with typed input rendered into the same image interface. Compare the current released system, each independently changed component and the complete candidate on the same fresh pages. Change the question on a fixed page and the evidence under a fixed question. Score the visible fact, role and proposition; distinguish faithful paraphrases from wrong participants. Stratify by actual and human-supplied evidence. Include authentic historical images for unseen headwords, new print styles, long book continuations and off-page instructions. Score complete replies and stopping, retaining original pixels, training curves, data lineage and judge calibration.

**Formal and resource criteria.** For each new objective, state a falsifiable hypothesis in Lean 4, prove its conditional prediction, and name which assumptions can only be checked experimentally. For instance, nonincreasing conditional probability gives a bound on every continuation of a discarded prefix. A wrong certified mode then directs attention to the learned score or supplied evidence. The proof does not establish that an annotation is semantically unique. A finite corpus result establishes performance only on its measured population. Single-GPU training may run for weeks or longer when separate development curves support continuation. Save restartable checkpoints and report examples, decisions, GPU-hours, peak memory and joules for training and inference. Compare with quantized Qwen on identical inputs under the same semantic acceptance and stopping criteria. Treat 10×, 100× and 1,000× as hypotheses until measured at matched accuracy. Maintain the English research article, original output figures and editable charts throughout. [Lean search theorem](../formal/span_list/StoppedSentenceSearch.lean) · [Frozen scores, bounds and replay discrepancies](../references/evidence/visual_sentence_search_v1/development.json)

**Earlier decision, 25 November 2026.** Focus the next fit on generating the proposition around a visually read fact. In a fourth cohort, scoring an entire evidence span raised complete Chinese replies from 31 to 43 of 192. All three methods gave the same five exact English statements. With a human fact supplied the writer gave 44/192 English. Even a reference-informed replacement of an arbitrary interval with the correct fact could repair only 19 of 113 failed English cases with a once-occurring reference fact. Hold this experiment as a Chinese short-reading baseline. Test a writer that learns the subject, predicate and placement of a variable from real sentence structure. Compare equally exposed direct writing, keep unseen multi-question passages and arbitrary visual values, and require real autonomous whole-answer gains in both languages, correct changed facts and retention outside short QA before promotion. Evaluate books, realistic glyph variation, historical requests and matched quality with an 8B comparator separately. [Full experiment and original pixels](../references/fact_span_ranking_result.md)

**Requirements for longer instruction answers.** Use complete visual reference answers and report how many the writer can physically express before interpreting learning curves. The original training set contains 43,779 English sentences with a single visible fact and additional words, including 2,258,954 surrounding glyphs. All 12,101 Chinese exam training targets consist solely of the fact. Include Chinese composition when testing writing. In a held optical inventory test on 9,712 Chinese Alpaca responses, lengthening the budget from 256 to 1,024 decisions and fitting the output inventory on training images increased physically eligible references from 4,095 to 9,546. Correct use of these images remains to be learned. Use the matched whole-answer and surrounding-text comparison with a fresh family-held evaluation, teacher-free deployment and durable checkpoints for sustained single-GPU training. [Support theorem, corpus measurements and training comparison](../references/visual_answer_support_result.md)

**Preceding model decision, 25 November 2026.** Learn to choose a complete span of visible evidence for a specific question and to write a grounded, stopped answer using that span. Hold the present 982,152-parameter reader–writer as a baseline. On 384 new real questions, autonomous fact-conditioned writing raised exact English answers from 4 to 10 and Chinese answers from 2 to 29 per 192. It passed neither the paired-language promotion rule nor historical and book transfer. A capacity experiment executed the fixed writer at all 12,288 candidate positions. Even an ideal selector could obtain at most 35 English and 115 Chinese exact answers. Of the respective failures, 25 and 86 could be resolved by better selection within the list; 157 and 77 require a change elsewhere. The checked Lean error identity separates these two terms. [Results and actual images](../references/fact_channel_result.md) · [Formal statement](../formal/span_list/FactAnswerCapacity.lean)

**Design of the completed span-ranking comparison.** Compare a question-conditioned scorer of entire candidate spans with the frozen first choice and an equal-exposure boundary-scoring control. Keep the writer frozen for this comparison, predeclare duration and cost, and reserve a fresh family-disjoint cohort before training. Count exact and semantically grounded whole replies, both questions on a passage, changed facts, book continuation and correct historical specimens. For English also measure sentence formulation with the right fact supplied. The next comparison has 192 fresh English and 192 fresh Chinese questions, paired within 96 passages per language. All eligible English SQuAD-derived QA2D passages had been used, so this reservation uses independently matched QAMR passages and human QA2D statements. Chinese uses CMRC. Publication of the reservation precedes optimization. [Protocol](../references/fact_span_reranking_protocol.md) · [Cohort digest](../references/evidence/fact_span_ranking_v1/reservation.json)

A reader improvement only earns an end-to-end claim when its actual answer images improve. If training extends for weeks or months, choose duration on separate development pages and record device-hours, full training data and energy. Derive a falsifiable hypothesis for each model change in Lean 4, map every assumption to a neural or data measurement, and test the conclusion on freely generated pixels. Extend benchmarks to altered typefaces, handwriting, original page layouts, non-Unicode specimens and multilingual book-length questions as the held performance supports them. An efficiency factor requires comparison at matched semantic quality and measured whole-system cost. Maintain the English Nature-style article with original output pixels, failures, measured learning curves, an editable deck and reproducible weights. Preserve separate measures of glyph fidelity, evidence selection, sentence formation and domain transfer.

**Preceding decision after two controlled image-writer fits, 25 November 2026.** The next model must learn to read the requested fact and state its relation to the question in an autonomous, stopped answer image. In 192 unseen English questions, supplying the human-marked region raised the all-path writer from ten to 11 exact answers. Supplying only the human fact yielded five. A specialist explicitly trained to formulate a sentence from that fact reached 34 under the same supplied inputs, and scored zero against the original sentence after switching to its sibling question. Its ordinary pipeline with predicted evidence gave 11 normalized exact answers. Test whether fact-conditioned training retains its advantage when its own reader supplies the evidence. Keep the whole page and question as the only release inputs. Include both English and Chinese, unseen factual substitutions and arbitrary visible historical glyphs. [Experiment and actual answer images](../references/optical_phrase_learning_result.md)

**How to qualify the next model.** Before fitting, register an independently reserved set of document families, two distinct natural questions per page, full-answer references, the direct equal-exposure control, stopping and physical-output rules, and both a development-duration rule and a device budget. Compare training with human pictured facts, predictions from a reader trained on disjoint families, and equal-cost direct writing. At evaluation cross the original, substituted and sibling questions with self-produced or supplied evidence. Test a formal hypothesis in Lean with measurable assumptions about where factual and formulation errors enter the final answer. Use a trusted numerical implementation and actual output images to check the neural computation. Scale to weeks or longer when development improves on complete grounded answers and the measured marginal value of another training pass justifies the cost. Keep frozen confirmation data out of duration selection. Seek independent bilingual assessment of the complete claim whenever automated judges fail calibration. A claimed 10×, 100× or 1,000× efficiency gain needs a task-matched quality level and whole-system cost on the same inputs. The current 16-image comparison gave the local Q4 vision baseline 12 requested facts and the all-path ILM one. Maintain the publication with preserved positive and negative answer images, uncertainty and matched baselines.

**Working criterion for question-conditioned writing.** Change one fact while keeping the pictured question fixed, then ask two different real questions about the same passage. The model must independently write the requested answers and stop. When a dictionary entry moves, the answer must follow the requested period and its specimen across positions. Count distinct, grounded multi-glyph answers that cannot be copied verbatim, evaluate meaningful paraphrases with a calibrated blind reading, and retain the non-etymology book and instruction tasks. A system that succeeds on one form, period, font or template earns credit for that scope. General language and deployment require separate positive evidence.

**Single-glyph result, 25 November 2026.** A 913,127-parameter reader and writer trained from scratch on 89,263 full human responses produced exact stopped human images on 6/96 held English statements and 33/96 Chinese catalogue questions. The exact English answers survived a strict input-copy exclusion. All exact Chinese answers reused a training frame around a held value. Before and after changing a fact, the model retained both values on 5/24 English and 84/96 Chinese pairs. In a calibrated blind reading it earned 0/16 English semantic credits. Source-absent conversation, ordinary book continuation and historical-form transfer still fail. Retain the strongest earlier models in each domain. [Measured outputs and data](../references/region_writer_result.md) · [Released weights and instructions](../models/region_writer/README.md)

**Next model requirement.** Train for whole propositions with correct subjects, actions and factual values on new English and Chinese pages. The checked mixture inequality and exact probability replay show that on 66/90 failed English statements, adjusting the copying probability at the first wrong glyph cannot recover an accepted next glyph with fixed component distributions. Change the visual question-and-evidence representations feeding both components. On 34 of these 66 first divergences, a complete multi-glyph reference word was present as consecutive input pixels. Compare learning an entire visible span with one-glyph writing, retaining the ability to generate new wording and to stop. Compare against the complete-answer objective at equal exposure. Hold out document families, score both versions of a fact edit, pair different questions about a page and preserve book, typeface and historical transfer. Register and check the new mechanism in Lean, connect its assumptions to emitted pixels and retain a decision for either result. Extend training over days or weeks when independent development shows continuing improvement in complete grounded propositions. Compare with quantized text and vision models at matched capability before claiming gains in compute. [Checked mixture theorem](../formal/span_list/OpticalRouteMargin.lean) · [Exact replay](../references/evidence/region_writer_v1/route-native.json)

**Whole-story baseline, 25 November 2026.** English full-story training raised both-correct reading from 16/498 to 63/498 stories. Extra human-highlight views gave 51/498. On 116 questions where every accepted multiword answer was absent from both the visible query and the story as normalized text and as consecutive glyph bitmaps, no writer produced an exact answer. Giving the human-marked evidence still yielded zero. The local semantic judge failed its human calibration. The training material contains 9,547 strictly source-absent English replies. Use the whole-story checkpoint as an English comparison and retain the existing Chinese specialist. The present checkpoints do not meet the bilingual deployment criterion. [Experiment and data](../references/evidence_views_result.md) · [Checked transfer theorem](../formal/span_list/EvidenceViewTransfer.lean)

**Acceptance test for the composition programme.** Give the reader a pictured page, pictured question and, for follow-ups, only the available dialogue. It must independently express the relevant proposition in a stopped glyph stream. Compare direct writing with an image-only intermediate record of the requested fact followed by writing in new words. Train on real, distinct human phrasings of grounded facts; separate the benefit of a supplied record from the benefit of one generated by the model. Keep a matched direct writer with equal target exposure, initialization and measured device budget. Require new grounded answers on sealed English and Chinese document families, changes after fact substitutions, meaningful crossed-period answers and retention on ordinary reading. Record exact physical copy eligibility, human-validated alternative phrasing, and complete book continuations separately. Specify fixed evaluation and an uncertainty estimate before a sustained fit. Use an explicit Lean hypothesis for the composition mechanism, test every assumption on the deployed computation and verify its implications with actual self-generated outputs. Train on complete wordings and provision the output length from the observed data. A training-only inventory finds 7,290 English fact groups with distinct phrasings. Only 450 fit two phrasings within 64 output decisions, versus 7,286 at 256. The corresponding Chinese sample contains 846 groups. Use this domain-specific material for formulation training, with generalization tested independently on books and real questions. Longer training is warranted while held-development semantic and grounding measures improve. Quantify efficiency only against a baseline reaching the same quality. [Failure and next comparison](../references/evidence_views_result.md)

**Historical-form experiment, 25 November 2026.** The goal is autonomous visual language learning on a single consumer GPU. The latest 2.46M-parameter writer improved the selection of visible historical forms in Chinese. It answered both period requests on 14/64 original cards and neither writer solved both requests under every layout on a single card. It wrote 23/288 exact new Chinese exam answers, with both requests correct on the same one of 80 pages as the control. English whole-page answers fell to 21/398 from 108 before Chinese fitting. Even when shown only the human-selected English evidence, it matched an answer on just 1/47 questions whose references were all absent from that evidence. Retain the stronger bilingual model. [Research record and checkpoints](../references/branch_ink_result.md)

**Design of the completed evidence comparison.** Train on human evidence and independently written answers with bilingual replay. Compare auxiliary training on the correct supporting image with equal-exposure full-answer training, keeping the page/question-only release interface. Use dialogue history where the source requires it. Preserve a document-disjoint, untouched confirmation set and all pretraining inputs' provenance. Evaluate evidence location and autonomous paraphrase separately on two questions about the same page, exchanged facts, source-absent multiword answers, long book continuations and crossed historical forms. Score both Chinese and English against their strongest applicable prior readers before promotion. An initial design test may be short. Allocate sustained training against held development learning curves and measure matched-quality cost against the local quantized 8B baselines. Require a checked Lean theorem for each new core mechanism, its assumptions, and a numerical correspondence test on real data. Publish original unguided answer images and their failures in the Nature-style article. [Checked first-glyph bound](../formal/span_list/BranchInkMass.lean) · [Usable evidence and answer supervision](../references/evidence/branch_ink_v1/training-evidence-supply.json)

**Earlier result and first-glyph intervention (25 November 2026).** A 2,462,192-parameter optical writer trained on 11,865 natural Chinese reading questions wrote 22/384 held answers, versus 16 after question exchange and ten before fitting. All three solved zero of 128 paired pages. All five exact source-absent multi-cell replies were the same verdict. A historical reading check yielded 33/128 correct Chinese specimens; every success was in position two. Exchanging labels and specimens independently yielded 0/256 correct in position one and 150/256 in position two. Do not promote either new writer. The next falsifiable intervention is to train the conditional probability of the first differing output glyph against the full set of possible next glyphs on genuine sibling questions, while retaining the likelihood of the complete stopped answer. Keep the existing writer as an equal-exposure control. Shared target-prefix ink is training supervision only. For a new model require both correct *free* answers on previously unseen paired documents to improve, transfer to new factual substitutions and diversified source-absent phrases, and check retention on ordinary book reading and historical position exchanges. Prefer longer training when independently reserved documents show continuing improvement. Keep test sets out of model and duration selection. [Exam and historical evidence](../references/open_exam_result.md) · [Exact greedy-prefix criterion](../formal/span_list/GreedyInkPrefix.lean) · [Whole-page control](../references/page_competition_result.md)

**Formal design constraint.** Every new central learning or decoding mechanism needs an explicit theorem statement in Lean, the hypotheses under which it applies, a checked proof and a mapping from its quantities to executable measurements. State when finite, quantized or exact-arithmetic hypotheses differ from a neural implementation. Use learning theory or mathematical physics when it gives an experimentally distinguishable prediction. Record a counterexample or a test that could refute the architectural interpretation. Separate the guarantee about the computation from evidence that training learned the intended language. The two-question calibration theorem is one example. [Lean source](../formal/span_list/CompetingFacts.lean)

**Language and efficiency milestones.** Evaluate on sealed, deduplicated book families and human questions, from image input to self-generated answer. Keep blind semantic judgments, whole-answer scoring, evidence interventions and an audit of factual errors. Compare locally measured quantized models at the same task, crop and question, and use identical image inputs for the visual comparison. Track general dialogue and book writing separately from short extractive QA. Count external teacher and inherited training costs where known. Report elapsed time, energy, peak memory, data exposure and achieved quality. A ten-, hundred- or thousand-fold efficiency claim requires both systems to attain the same prespecified capability. Establishing parity with a useful local quantized model is an intermediate milestone.

**Training horizon.** Use the available books, open corpora and independently checked instruction data in enough quantity to test generalization. Teacher-generated supervision can be used during fitting. Extend real runs over days or weeks, and longer where justified by the measured trajectory, on the same device. Each sustained fit should be restartable with model, optimizer, random state, corpus identity and draw position preserved. Retain predetermined evaluation snapshots, independent test data and quality-versus-compute curves. A short fit can validate a component; its duration cannot establish a mature language model. Maintain an English Nature/CVPR-style manuscript with legible original outputs, controls, exclusions, mathematical assumptions, architecture and failure cases.

**Decision from the visual list experiments, 25 November 2026.** Measure success by the answer image a reader writes from an unseen pictured question and page. In Chinese plural QA, training against incomplete lists raised fixed-reference hits from 23 to 62 of 996 parts, yet yielded zero complete answers on 96 pages. A separately trained two-mode reader distinguished both supplied correct lists from exchanges on 20/64 additional pages. Free writing gave 0/64 and replacing the question still gave 16/64 correct pairings. Both model fits reproduced byte-identical final weights. Preserve the stronger deployed reader. [Whole-answer result](../references/complete_fact_sets_result.md) · [Binding experiment](../references/fact_binding_result.md) · [Exact scope of the Lean result](../formal/span_list/FactSetBinding.lean)

**Next training target.** On pictured training pages compare the best complete supported reference placement with the reader's highest-scoring unrestricted output, retaining likelihood supervision and matched resource controls. The unrestricted optimum beat both evaluated reference placements on every one of 64 held pages. A training-only census now finds 52,795 of 67,405 Chinese pages and 4,813 of 5,230 English pages permit a complete literal placement. On 8,701 Chinese pages only an alternative reference makes the page usable. These are counts before question-family and optical screening. Preserve all acceptable alternatives during training. [Source counts](../references/evidence/fact_binding_v1/training-visibility.json) · [Checked placement rule](../formal/span_list/VisibleFactPlacement.lean) Account for repeated placements, legitimate alternative answers and the finite span cap. Keep test families sealed until selection is over. Qualify an update through autonomous exact answers, blinded calibrated meaning judgments, and question/source exchanges at comparable accuracy and cost. Long training should build toward independently composed bilingual prose, book continuation and explanations of attested historical forms. Retain a separately reported test whose answer cannot be copied from the prompt or page. A successful list reader alone cannot meet the project goal. [Evidence for the choice of training target](../references/evidence/fact_binding_v1/paired-binding.json)

**Current measured distance to a useful local reader.** The new 256-page comparison measured 34 both-correct pages for contrast ILM, 32 for its original reader, and 101 for quantized Qwen3-8B reading the text. On a balanced 64-page subset ILM solved six and quantized Qwen3-VL-8B solved 21 from the exact same PNGs. Lean established which two-fact margin a shared bias can attain. Training improved that preference while missing the adoption criterion on autonomous bilingual answers. A second Lean result identified the lack of a length-corrected endpoint interaction; its direct architectural test solved 18/128 additional pages, tied with its matched control, below the original reader's 20. Keep the original default. [Original comparison and energy](../references/competing_facts_result.md) · [Independent endpoint comparison](../models/endpoint_interaction/README.md)

**Latest production test.** On a new 96-page English multi-fact set the visual reader wrote three complete answers; its matched classifier wrote one. Both scored zero on 96 Chinese pages. On 116 Chinese questions with a fully visible first reference containing two to twelve parts, the reader always selected just one. A replay of the draw schedule found that only 24,476 of 43,087 available Chinese questions had been presented, including 9,775 distinct multi-part questions. The count head itself predicted one on 102/116. Giving the true count recovered four complete lists. Two separate Lean proofs establish the conditional likelihood and the integer path search used by this reader. These guarantees locate the training and selection problem precisely. They do not certify semantic generalization. [Complete bilingual results and images](../references/multiple_ink_runs_result.md) · [Conditional probability proof](../formal/span_list/InkRunSurvival.lean)

**Next learning experiment.** Test a question-conditioned whole-set training objective against the fixed conditional reader. Contrast complete human answers with plausible proper subsets and wrong lists of the same size on the identical page. Give the matched arms the same full Chinese corpus coverage and report distinct examples as well as repeated presentations. Preserve natural Chinese singleton questions; reserve previously unused pages and require the appropriate answer to change after substituting evidence or exchanging real questions about the same passage. Score all parts, factual grounding, a stopped independent sentence and the cost of matching a quantized baseline. Progress toward an etymology chapter also requires a held headword with the attested visual forms in order and a grounded explanation. Keep the book continuation and raw-page tests. Use extra training time where the independent evaluation shows continued improvement.

**Previously demonstrated sentence generation.** With the question image held fixed, replacing Matthew Murray by Daniel Harper on an unseen railway page changed the independently written English statement from “Matthew Murray designed Salamanca.” to “Daniel Harper designed Salamanca.” The complete 2,787,303-parameter model runs from pictured inputs without an external teacher. On 96 fresh passage families the fact reader matched a human short answer on 25. Across the first 24, an exploratory project review credited four grounded, grammatical complete statements. The same writer read a changed school year correctly twice and omitted it from both sentences. The cumulative deployed GPU fits cost about 506 seconds and 67.1 kJ on one 4090 D. [Actual output images](../models/visual_statements/examples/intervention) · [Full comparison](../references/visual_statement_result.md)

**Earlier intervention design.** Learn to express facts, relations and historical distinctions from glyphs, with the question and output both in image space. Test the same pictured question with different evidence and different questions about the same evidence. Study the two measured failure modes separately. Improve selection against plausible distractors from the same page with the sentence writer fixed. With the reader fixed, train on edited facts and demand complete grounded answers before and after an edit. Use new page families, matched byte and source-copy controls, both English and Chinese, and report end-to-end compute at common capability. Keep historical explanation, unfamiliar ink, free prose and stopping as independent acceptance tests. The current English two-stage result does not establish general visual language, etymology or a computational advantage.

**Preceding recurrent writing experiment.** Adding dynamic visual copy runs helped on a fossil-location question while reducing both-correct modern passage pairs from 44/256 to 38/256. The recurrent writer gave zero exact references on 192 source-absent questions. A finite-path likelihood identity and gradient/decoder checks validate the loss; learning the intended relation is an empirical condition. [Results](../references/recurrent_evidence_result.md)

**Earlier paired reading result.** On 256 newly reserved passages the independent 2.44M-parameter image reader and writer gave both human answers on 44, compared with zero for the preceding optical model. It passed both versions of 20/28 altered-number questions and reproduced unfamiliar digit-shaped marks on seven. Independent CPU PNG replay matched all sixteen archived answers. It failed all 120 external Winograd pairs and all 64 historical headword pairs in both languages, including a separate demonstration on 言. Both credited replies on the 32 source-absent English dialogue questions were “no”. A constant “yes” matches 12. Success on evidence selection has yet to transfer to composing an answer. Matched bytes solved 52/256 pairs at comparable energy. [Evidence and limits](../references/paired_evidence_result.md)

**Preceding composition objective.** Demonstrate generalizable composition from pictured evidence. Require self-generated, stopped, factually grounded answers to questions whose wording cannot be copied in one interval. Test changed facts and questions on fresh pages. Preserve ordinary English and Chinese prompting, open continuations and authentic historical ink. Measure the complete deployed model, reference-copy and matched-byte comparisons on one consumer GPU. The ability to write a useful etymology entry from a pictured book, and any general language or efficiency advantage, remain open.

**Prior open-writing result.** A 2.12M-parameter neural writer trained from scratch on one 4090 D reads and emits aligned glyph images. Held English fiction improved from 5.66 to 2.64 bits per glyph, with 2.83 after a prompt exchange. Chinese fiction improved from 10.68 to 7.40, with 7.67 after exchange. Free answers to ordinary instructions remain repetitive. Both variants were written correctly on 0/120 independent external Winograd contrasts. In an additional 528k-parameter visual reader, new training with genuine English and Chinese questions yielded 11/384 complete answers with persistent source positions and 12/384 without. Neither answered both questions on any of 192 passages. [Open generation](../references/visual_discourse_result.md) · [Human questions and controlled diagnosis](../references/visual_cursor_result.md)

**Preceding experimental decision.** Require a causal response to the pictured question under identical evidence. Giving two candidate answers only for diagnosis, the persistent reader ranked both in the correct direction on 0/192 pairs. Both references are in the visible passage on all 128 Chinese pairs. In the next learning comparison, expose the writer to alternate real questions on the same training passage and teach where the relevant evidence is, using the human annotations only during fitting. Compare at equal exposures with the independent-position arm, hold out whole passages, and preserve open free generation, diverse question types, fiction, instructions, prompt erasure, unfamiliar-ink copying and total device cost. Require improvement on new complete autonomous answers and independent questions before claiming language transfer. Evaluate the next direction on fresh passage families with changed facts, where the same question requires answering differently. Loss, image fidelity and forced assignment cannot qualify a general language claim.

**Preceding marked-reference result.** A 567,759-parameter optical system reads prepared Chinese venue questions and writes answer images assigning a stated property to the former or latter of two supplied names. Under crossed name and order changes, 365/536 groups pass all four questions, compared with zero for the preceding writer. On 41 screened natural English and Chinese ordinal expressions, its 38 correct decisions match an explicit ordering rule; matched bytes solve 39. On 120 Winograd contrast pairs it gives both correct names only once. General language and a compute advantage remain open. [Measured results](../references/story_mentions_result.md)

**Earlier contrast decision.** Train the response to a change of meaning. Keep both candidate spellings and the pictured query fixed, and change a different word or clause that reverses the correct referent. Compare a paired contrast objective with the same model trained on individual examples, using the same unique pages, updates and hardware. Hold out passage families and wording before fitting. Report both directions together, novelty, bytes, recency, copy quality and full cost. The frozen reader could observe some difference in 114/120 Winograd pairs, yet solved both directions on only one. Wider context alone cannot explain that result. Keep independent book continuation, general questions and efficiency claims subject to their own evidence.

**Preceding venue attribution result.** A 486,442-parameter reader and writer answers short pictured Chinese questions about a named venue in a continuous pictured paragraph. On a real two-restaurant recommendation the same question elicited two different correctly attributed rating images. With the same fitted writer, supplying the pictured target-name position raised four-way attribution from 0/539 to 472/539 controlled comparisons using joined held conversations. The matched byte system passed 487/539. On separately authored two-venue prose the visual system passed 5/12. Among ten archived external admission-price questions it returned a compatible core on six, including one with conflicting references. The four required GPU fitting stages consumed 5.72 kJ on one RTX 4090 D. General visual language and an efficiency advantage at matched capability remain open. [Results and counterexamples](../references/prose_fields_result.md) · [Runnable reader](../models/prose_fields/README.md)

**Earlier reference-learning decision.** Learn to link an explicit name to a later mention whose appearance differs, and link that mention to its own value. Start with a reader that can distinguish two people or places and two same-unit quantities in a paragraph when neither proximity nor order of mention answers the question. Test implicit references, including former and latter, separately from explicit names. Reserve natural paragraphs and complete construction families before training. Keep both target exchanges on each page, a name-erased control, a proximity baseline, matched bytes and the current optical reader. Report exact source evidence, both complete answer images, abstentions, external transfer and total cost. Track ancillary fees and qualifiers separately. This directly tests the exposed failure without rewarding a fixed cue or changing the old scores.

**Earlier formatted-card result.** A 333,850-parameter model trained from scratch on one RTX 4090 D reads human Chinese questions on formatted venue pages and writes one or two clauses as images. Making the current property visible to the writer raised clean answers with both values attached to suitable words from 24/37 to 36/37 under the same visual reader. The final model gave clean answers on 515/515 single-field turns and 12/14 additional two-field turns with new wording. The small byte-input control achieved 37/37 and 12/14. The held 37 informed development. Domain and field labels are fixed and one venue is supplied. The optical model failed all 18 paired exchanges under a delayed-order instruction and omitted expenditure when a guest requested three attributes. Its observed inference savings against a general local 8B model apply only to this workload. [Model](../models/chinese_two_fact/README.md) · [Observed outputs, costs and formal guarantees](../references/chinese_two_fact_result.md)

The preceding one-field model gave 507/515 clean answers and 237/515 after question erasure. [Preceding comparison](../references/chinese_card_reply_result.md) · [Historical-name transfer](../references/variable_ink_result.md)

**Current evidence and acceptance criterion.** A usable visual model must express the requested meaning in a complete generated answer when the facts, question or evidence change. Measure independently legible glyphs, factual coverage, natural wording, appropriate stopping, unsupported claims and end-to-end cost. A finite bilingual input domain is an admissible first demonstration. Claims about Chinese writing, unfamiliar scripts, free-form pages and general conversation each require held language evidence.

One 1.59-million-parameter image writer plus visual listener now produces English descriptions from printed Chinese and English restaurant cards. On 1,847 unseen Chinese-card meanings, 657 answers recover every literal detected fact and 1,606 pass after merging price and rating aliases. The matched byte result is 665 and 1,610. Of the 241 merged-rule failures, 179 lacked any passing candidate and 62 had one the selector missed. Both polarities are correct for 1,286/1,291 edits. A separate mixed-script set passes on 1,304/1,484 after merging. Independent source-grounded historical lookup also works on 4,268/6,360 fresh two-field requests. The next model comparison should improve candidate coverage for names and unsupported assertions with a source-image copy option, then test human-written Chinese answers on independent material. Keep free generation and dictionary retrieval distinct. The trained systems do not yet qualify for general-language or compute-advantage claims. [Measured outputs and theory](../references/semantic_page_result.md) · [Actual historical forms](../references/glyph_atlas_result.md)

The goal is an independent model that learns to read a pictured passage and question and writes a useful reply as glyphs on one consumer GPU. The efficiency hypothesis compares complete energy to a common held-out language quality, with inherited training and input preparation accounted for. Tenfold, hundredfold and thousandfold savings remain hypotheses. A cold 547,125-parameter image writer now answers 1,574/5,588 development questions. Its matched byte writer answers 1,558. Both cross 20% accuracy on both held splits at step 2,400, at 5.20 and 4.86 kJ respectively. Neither matches a human answer on the 83 turns requiring absent multiword wording in all four references. [Learning curves](../references/dialogue_efficiency_result.md) · [Models](../models/dialogue_surfaces/README.md) · [Lean and its assumptions](../formal/span_list/VisualTokenRisk.lean)

On 3,854 reserved Song pages a Tang-trained image writer fitted with contrastive phrase supervision assigned supplied endings correctly on 3,020/3,781 comparisons but restored only 8/3,854 two-cell gaps. The matched image control assigned 2,970 and restored 12. All eight exact repairs were shared. Even when given the recorded first glyph, the second ranked first on only 151 pages. Keep the existing writer default. Test the useful book-reading task directly next: locate an entry in a larger pictured collection from the user's question, then render the requested historical forms. Use unseen headword families, competing entries and alternate order requests. Score page selection, assembled answer, stopping, actual inference and full resource cost, with the supplied-plate reader as an upper control. A stored source must remain distinguishable from independent historical recall. [Song comparison](../references/verse_contrast_result.md) · [Existing supplied-plate reader](../references/historical_plate_result.md)

A 3,282,782-parameter independent writer reads paired Chinese verse as ink. Across 6,362 fresh line pairs, direct visible-glyph supervision improved blank-line recovery from 1,281 to 1,392 of 37,442 positions. A source exchange lowered this to 340. It completed 17 two-cell repairs and zero full blank-line references. Its unseen-face identification gain (1,651 to 5,519 of 8,414) produced no blank-line gain in that face (857 in both conditions). The image writer emitted a fitting-attested pair on 5,656 pages versus 4,459 such reference pairs. Repeated generic phrases remain a bottleneck. For the next comparison, score competing complete expressions against the same surviving ink, train on authentic versus source-exchanged phrases from fitting poets, and test source-dependent exact writing on fresh works. Include an order-sensitive contextual scorer, the frozen identity-trained reader, physical counts and matched bytes. Quality at equal compute, unconstrained replies and new-script generation remain open. [Experiments and proof](../references/verse_canvas_result.md) · [Restore an actual page](../models/verse_canvas/README.md)

A causal image writer retaining its original 32 glyphs recovered 839 of 28,880 positions during free continuation on two reserved classical Chinese books. Exchanging the distant prefix reduced this to 426. With correct prior glyphs it recovered 2,744 positions, ahead of the fitted image reader alone (2,613) and suffix counts alone (2,354). The byte mixture recovered 2,809. No writer produced the first eight reference glyphs on any of 1,805 passages. Only 532 positions were outside the image palette and visible prefix. Test whether the final reader prefers the genuine source-to-continuation assignment within a single work when the recent glyphs match. Preserve a separate free-writing test and require matched language quality before claiming energy savings. [Chinese book results](../references/book_prefix_result.md) · [Runnable writer](../models/book_prefix/README.md) · [Support theorem](../formal/span_list/BookWriterSupport.lean)

An independent 896,983-parameter writer learned measurable local language regularities from aligned pictured English. On 50,012 new passage positions it restored 4,863 missing images, falling to 1,010 with exchanged context; matched count backoff restored 6,715. On source-absent QA words, direct contextual learning raised mean reciprocal rank from 0.116 to 0.219, but complete difficult replies remained at one of 265. The first 147 eight-word self-fed continuations yielded 36 distinct streams and zero reference matches. The next comparison should preserve the original visual prefix throughout causal writing and learn to combine longer matching ink sequences with a neural predictor. Use independent book splits including Chinese, measure true-history and free generation, and keep the whole-answer and matched-cost requirements active. [Results](../references/context_cloze_result.md) · [Released models](../models/context_cloze/README.md) · [Checked state limit](../formal/span_list/ClozeStateRisk.lean)

On 147 additional reserved stories, a frozen visual reader with 2,048 learned word-image retrieval keys produced 502/2,238 exact replies, compared with 502 using the same decoder with retrieval disabled. All three visual conditions returned zero complete matches on the 259 difficult turns. The target ranked first in memory on only 3/108 concealed words known to be stored and absent from the inputs. The next experiment should target conditional discrimination of missing words, retain a frequency prior and the frozen-reader control, and evaluate the passage and question exchanges. The active goal still requires independently generated complete visual language and measured end-to-end cost at a shared quality threshold. [Retrieval trial](../references/word_memory_result.md) · [Released writer](../models/word_memory/README.md)

The four-pass masked image writer and the matched left-to-right writer have now run on 200 additional reserved stories. The masked writer returned 693/3,155 normalized exact answers and zero on 443 difficult turns. Its causal control returned one source-sensitive difficult answer. Giving the masked writer the true length yielded one normalized exact in 435 eligible difficult cases. Its first pass fixed an incorrect literal image in 393. With every other answer word supplied, only 677/1,937 available word images and 0/327 absent images were correct. Of the latter, 245 had appeared in fitting answers. The next comparison should learn a conditional distribution over word images previously observed during fitting alongside source copying and open spelling, then test autonomous composition on independently reserved stories. Count all retrieval/storage costs and compare with matched bytes. Both accurate source selection and unseen-word writing remain necessary. [Whole-answer experiment](../references/word_canvas_result.md) · [Lean commitment condition](../formal/span_list/IrrevocableDraft.lean)

A visual writer can now assemble five pictured words into a correct list answer. With list training it produced 49/501 complete held list replies, including 10/396 difficult answers that lost credit under both question and passage exchange. On 407 difficult conversation turns from 200 additional held stories, all six visual and byte writers scored zero exact. When supplied two human answers, the visual list writer assigned 139/158 pairs correctly with empty history, falling to 117 after passage exchange. With no candidates supplied, just 1/239 copy-assemblable conversational references appeared in its 32 proposals. These outcomes locate an obstacle in generating candidate replies. The next learning comparison targets a whole answer draft conditioned on the pictured passage and question, with independently reserved passages, an unchanged writer and a matched byte control. Measure free-running stopped exact answers, both input exchanges, proposal coverage, word F1 and complete costs. Keep the continuity of actual glyph output, including historical Chinese, as a separate acceptance criterion. [Composition results](../references/compose_dialogue_result.md) · [Code and weights](../models/compose_dialogue/README.md) · [Checked spelling bound](../formal/span_list/RequiredSpelling.lean)

A matched continuation experiment reached 270/758 any-reference exact answers from pictured passages and questions, falling to 29 after exchanging question/history and 74 after exchanging the passage. Human evidence supervision changed F1 by +0.39 percentage points on 47 stories and +0.27 on eight longer stories across four paired seeds. Both 95% intervals include zero. The Qwen3-4B prompt variants missed the registered label-quality threshold. All eight visual fits gave zero exact on 27 questions requiring new multiword phrasing. An exact action search found that 18 can be answered by reordering visible word images and the other nine need one spelled word. The next learning test targets the selection and ordering of words in complete, source-sensitive replies on unrehearsed stories. Retain matched byte controls, original failures, inherited costs, and a common quality threshold. Independent Chinese historical-form generation, general prose and an efficiency advantage remain active goals. [Results and model](../references/dialogue_evidence_result.md) · [Lean matched-quality certificate](../formal/span_list/MatchedReadingCost.lean)

A 2,136,181-parameter reader now extracts whole answers from pictured English paragraphs and pictured questions. On 20 reserved SQuAD articles it returned 872/1,957 exact normalized answers; the first disjoint answer pairs scored 108/484 versus zero with the questions exchanged. The eligible official development subset scored 3,114/6,389 and 351/1,285 pairs versus two. A clean single-GPU refit reproduced all released decisions and example pixels. This establishes question-sensitive reading of aligned print. The extractive decoder cannot produce an answer absent from the source. In 7,199 first-turn CoQA training examples, 765 human multiword references cannot be copied even after lowercasing and whitespace normalization. The next comparison must learn a conditional glyph writer from natural free-form answers, with unseen passages, a strictly nonextractive multiword subset and source/question exchange controls. Retain the extractive reader as a frozen comparator, verify source-sensitive wording and stopping, and keep historical Chinese and matched-quality compute claims open. [Reading results](../references/optical_passage_qa_result.md) · [Independent released model](../models/optical_passage_qa/README.md) · [Lean proof of the copy limit](../formal/span_list/OpticalPassageQA.lean)

The independent 3.59-million-parameter reader can now answer about two different events in the same line. On 971 held multi-event clauses it located every verb and object and wrote every accepted meaning on 247 clauses (51 with frozen conditional scores and 11 with exchanged objects). Extra English description training raised literal success to 257, but gave zero written successes on 21 wholly unseen-question clauses and zero of ten screened Pu Songling extracts. Supplying valid event candidates recovered three of the 21 for event selection alone. The next test should measure whether a small semantic teacher can match natural questions to the correct actions on fresh, independently sourced paraphrases, then fit and test a visual-only student on disjoint language. Preserve whole-system cost, open-ended generation and historical forms as independent requirements. [New result and derivation](../references/questioned_events_result.md)

A 3,526,604-parameter image reader completed 44/268 held action-description pairs, compared with 11 for its parent and six for exchanged descriptions. It selected both lines on four of ten screened Pu Songling pairs but wrote both answers on none. The inherited writer completed two pairs given the correct object positions. The next comparison should test question-dependent selection of verb and object positions within a real line containing multiple actions, with frozen literary passages and an exchanged-binding control. Keep general language generation and matched-quality compute as separate acceptance criteria. [Results, model and derivation](../references/visual_action_meaning_result.md)

The 1,551,989-parameter image reader can now select a Chinese line from an action named in a pictured English question and write its object. It completed 703/1,691 paired pages across all eight requests (650 for the supplied-name glyph-count baseline). On the same 188 pages, replacing the action names with alternate English expressions reduced complete writing from 95 to seven. The next acceptance test is transfer to independently sourced descriptions of the same action in fresh, linguistically checked passages. Require novel descriptions, distinct event contexts, correct source positions, a complete stopped written answer, a shuffled-supervision comparison and retention on the original benchmark. The broader independent language and matched-quality compute goals remain active. [Model and scores](../references/pictured_action_lookup_result.md) · [Margin derivation and next comparison](../references/pictured_action_semantic_transfer_direction.md)

The joint visual reader and writer completes 110/429 same-glyph contrasts across two grammatical roles. It completes four of 18 opposite-order pairs in independent books and misses three of twelve new question wordings. The next experiment asks which of two Chinese lines contains an action named in a pictured English question. Require both line orders, distinct questions, correct source positions and complete written answers. [Results](../references/visual_clause_roles_result.md) · [Two-line experiment](../references/pictured_action_lookup_protocol.md)

The independent image-to-image path now has 1,396,977 parameters with an optional learned writing residual. The broader pictured-question reader answered both Chinese and both English classroom questions on 98/539 previously held passages, compared with zero for the preceding reader, and 1/29 on two fresh classical books. Restoring person annotations enabled the writer to give both *go* and *it* for the same printed 之 in 31/40 reserved-section pairs versus 4/40 with exchanged answers. On eight pairs from held works the counts were four and two. With the new writer the classroom score is 101/539 and 1/29. Supplying the correct source positions raises the latter to 10/29. [Question transfer](../references/everyday_gloss_result.md) · [Grammatical sense and actual images](../references/pronoun_writer_result.md) · [Run the complete release](../models/pronoun_writer/README.md)

The next acceptance test is source selection when a classical object comes before its verb. The classroom reader located both roles across both languages on zero of 14 such new-book passages. Reserve fresh works, contrast fronted and postposed instances of the same words, and compare a syntax-balanced reader against unchanged and within-glyph shuffled supervision. Count exact locations, contextually accepted written meanings and complete two-language pages separately. Preserve prior results and the known motion losses. General dialogue, connected book generation, historical-script generation and the proposed thousandfold efficiency advantage require their own evidence.


A 1,345,457-parameter independent model now reads an unmarked classical Chinese clause and a pictured question, locates the requested word, and draws an English gloss. On 539 held clauses it produced both readings and located both words across eight contrastive instructions on 230 clauses, or 247 with a training-only physical gloss bank. The recurrent reader's matched pooled and exchanged-answer controls scored zero. On 執大象 it drew *hold* or *image* according to the question. The conditional attention was fitted on one RTX 4090 D using only answer images as its supervisory signal. Four everyday paraphrase pairs gave 0, 33, 0 and 2 complete pages, and object-first coverage was sparse. [Source-level results](../references/asked_gloss_result.md) · [Runnable reader and writer](../models/asked_gloss/README.md) · [Lean identifiability proof](../formal/span_list/LatentGlossFocus.lean)

The next acceptance test is transfer to independently phrased natural questions about the same passages, with both expected roles, held paraphrase authors or sources, object-first examples, and held books. Fix the questions and acceptable readings before fitting broader paraphrases. Count source discovery, stopped meaning, order sensitivity and novel-language transfer separately. Keep the original held sources, authentic historical forms, whole-system compute and open-ended language as ongoing requirements.

The new 1.18-million-parameter writer learns from Chinese source images and English target images on one RTX 4090 D. It draws distinct content words absent from the page. For 言 it wrote *speak* and *speech* in two real passages. In 566 and 403 same-headword contrasts from two held works, it completed 213 and 82 pairs, versus 4 and 5 under within-headword shuffling. Local context counts completed 211 and 104. Its only five-cell contrast success reproduces an inconsistent annotation of 樂. The single-device fit cost 38.75 s plus 0.41 s for the glyph motor. Of 69 instances needing an unseen gloss string, none was exact. These limits leave general language and matched-quality compute savings open. Optical source matching substantially recovers serif reading and leaves printed Kai difficult. [New evidence](../references/visual_word_sense_result.md) · [Complete independent writer](../models/visual_word_senses/README.md) · [Proof of the context-blind limit](../formal/span_list/SameInkMeaning.lean)

The next test should join pictured question following to this demonstrated ability to generate content words. Reserve natural passages in a fresh literary source, with two legitimate word-meaning questions per page, independently supported readings and multiple question wordings. Present the image of the page and the image of the question. Score autonomous focus selection separately from writing with the true location supplied. Include a quoted-headword alignment control, a local count baseline and cases whose interpretation cannot be decided by adjacent characters. Keep printed variants, true manuscript handwriting, etymological knowledge without source plates and open-ended explanation as distinct validation targets.

The editor now writes short Chinese replies from pictured questions and a pictured passage. Across two reserved novels, it wrote 34/48 and 42/48 exact two-clause replies, including a generated connective. The matched randomized-order control gave 6/48 and 4/48. On 60 held-out historical families, the original optical reader and editor reproduced both attested forms across all four arrangements for nine families. With both source locations supplied, the editor succeeded for all 60. English reading, source-free etymological explanations, lexical generation beyond learned connective patterns, whole-model training costs and matched-quality speedups remain open. The next writing comparison must require contextually correct *content words absent from the page*. Reserve paired contexts and acceptable alternative phrasings before fitting, compare against the frozen compositor and a shuffled-gloss student, and measure source grounding, lexical meaning, stopping and whole-system resource use separately. Retain the 60-family bronze-location benchmark as the independent reading control. [Experiments and limits](../references/grounded_ink_writing_result.md) · [Run the model](../models/grounded_ink_editor/README.md) · [Formal boundary of copying](../formal/span_list/GeneratedConnective.lean)

On 14 new passages from the *Discourses of the States*, our independent optical reader returned 38/56 exact names for Chinese requests and 16/56 for English. It followed both questions before and after exchanging the names on 4/14 Chinese passages; the frozen reader completed three and the equal-label control two. No visual arm completed an English exchange. The aligned update used 50,690 trainable parameters of an 86-million-parameter contextual reader, 1.33 GiB peak allocation and 24.6 seconds on one RTX 4090 D. Parent pretraining is outside this measurement. Keep the general visual language goal active. The next discriminating test keeps both names in place and changes which action belongs to each. Preserve the free writer and its original outputs until independent evidence links this source-reading ability to composed visual replies. [Three measured comparisons and decision](../references/optical_role_adapter_result.md) · [Fixed-position action comparison](../references/fixed_order_action_direction.md) · [Lean separation result](../formal/span_list/FixedOrderRoles.lean) · [Working reader](../models/optical_role_reader/README.md)

An independent 64,666-parameter visual reader now returns short source images from questions about two held-out novels. With 394 training passages it identified both supplied people on 62/226 Chinese passage pairs, but autonomously found only 23/452 answers. A fit on 6,439 passages recovered 104/452 Chinese and 45/452 English answers autonomously. The same-data human-only fit recovered 81 and 48; the exchanged-teacher fit recovered 88 and 59. The specific teacher alignment effect is unresolved in both languages. The matched student answered two different pictured questions correctly on eight Chinese passages and zero English passages. An 87.1M visual writer remains available for composed clauses. Its inherited and fresh capabilities have separate evaluations. [Training, proofs and comparisons](../references/book_span_distillation_result.md) · [Runnable checkpoint](../models/book_span_distillation/README.md)

The immediate learning target is to identify which action each participant performed when two plausible people occur in the same pictured passage. Train and test question-dependent bindings with both plausible names and competing relation contexts, preserve the all-span objective and evaluate unrestricted bitmaps. Reserve a new work or independently authored population before tuning, with both questions and both assignments fixed. Use a same-compute human-only control, a question-ablated control and the unchanged small reader. A reproducible improvement in correct paired responses on unseen content precedes any new compositional writing or efficiency claim. The final goal remains independent visual reading and writing, including historical source forms, on one consumer GPU.

The independent 87.1M image writer now generates one stopped clause per pictured question. Aligned action supervision improved complete edited-credit comparisons to 175/256 from 19/256 with independent labels. Both scored zero on 24 new-story comparisons. The writer gave three supported replies in 48 readings, against 38 for local text Qwen3-8B, and preferred the correct supplied clause only 53/96 times on those stories. Four distinct catalogued forms of 言 were reproduced on two of eight source-assisted black-on-white requests. Averaging the scores under a complete exchange of written names left the writer at one of 24 new-story comparisons even with both candidate answers supplied. Preserve the release and its failures. Test whether a visual reader learns action–participant compatibility on new predicates and works before another free-writing scale-up. The next fresh reader has 394 two-person-question passages for training and 226 passages from two disjoint works reserved. [Corpus and decision](../references/book_participant_split.md). Matched-quality efficiency remains open. [Experiment, Lean proof and original pixels](../references/individual_clause_writer_result.md) · [Run the checkpoint](../models/individual_clause_writer/README.md)

Aligned assignment supervision raised correct four-request contrasts on edited credits from 5/256 to 165/256. It yielded 0/24 on separately worded stories and 4/48 supported complete statements, versus 39/48 for local text Qwen3-8B. A supplied-answer probability comparison was correct on 998/1,024 credit requests and 49/96 story requests. Even recasting both stories and questions in the training style recovered only four contrasts, all in Chinese. Keep this 87.1M writer as a runnable domain experiment. Next test binding each of two pictured actions to its participant separately before changing the multi-clause decoder or scaling training. Keep open-ended language, new glyph synthesis, response to rephrased historical requests and matched-quality compute savings as separate goals. [Edited passages and original outputs](../references/role_change_writer_result.md) · [Lean derivation](../formal/span_list/SourceEditSupervision.lean)

Two-question visual writing now has an explicit test for factual roles and order. On 96 novel passages the direct writer passed both request orders on zero of 192 passage–language pairs; a writer trained to generate wording around selected source ink passed 15. On 48 newly authored requests each writer produced four complete supported statements in an arm-blind reading, versus 34 for local text Qwen3-8B. The frame writer learned some surrounding language and obtains answer order from the supplied question slots. The next comparison targets role, time and negation swaps with matched names and fresh passage-level testing. Independent general language and a compute advantage remain open. [Model, Lean proofs, outputs and reproduction](../references/ordered_book_reply_result.md)

A fresh-weight model of 848,961 parameters learned a bilingual relation from printed question images, matching 364/374 pairs of Chinese and English questions on novel passages. Feeding its visual representation into the independent reader increased exact English-request excerpts from 17 to 41 of 412 on MLQA and preserved 124 exact Chinese-request excerpts. The reader and causal writer produced 111 exact stopped replies in 824 trials, versus 90 from the full-page writer. Keep the qualified excerpt reader available and keep open composition, historical forms, stopping and total compute as separate criteria. The four measured reply strategies all miss 610/824 requests. On 326 of those misses the correct spelling is already among the first 32 distinct predicted excerpts. Next test learned selection and composition with multiple predicted excerpts and the full original page, against within-page distractors, on new two-fact answers. [Finite learnability proof and measured coverage](../references/visual_reply_learnability.md) [Lean, actual images, controls and resource costs](../references/question_evidence_transfer_result.md)

An earlier single-4090 comparison improved supervised whole-reply ordering from 55/642 to 582/642. On 36 passages from six separate works, the trained image writer answered 31/144 questions exactly, versus 32 for matched replay and 32 for the release. No image writer completed a four-request passage. Text Qwen3-8B completed 32/36. The visual span reader gave 45/72 Chinese answers but only 1/72 English answers. Preserve the released writer. The next comparison transfers this stronger source reading to aligned bilingual pictured questions, using 11,439 prepared pairs. Judge free language, novel glyphs and matched-quality single-GPU costs separately. [Experiment and original replies](../references/stopped_book_preference_result.md) · [Bilingual evidence comparison](../references/bilingual_book_interval_direction.md)

Two matched 49,283-example fits tested whether the weight on the first answer glyph explains poor book reading. On 34 passages from four additional literary works, first-glyph weighting gave 31/136 exact stopped replies, ordinary training 30 and later-action weighting 29. All three completed zero four-request passages. Text Qwen3-8B completed 21/34. In 94 of 104 wrong stopped replies from the retained ordinary writer its emitted string outranked every accepted answer. Preserve the bilingual release and use complete training-answer likelihood against validated wrong model replies for the next comparison. Reserve new works for qualification. The goal remains a useful, independent image-in/image-out language model on one consumer GPU, with open composition and matched-quality efficiency demonstrated separately. [Measured comparison](../references/first_glyph_book_result.md) · [Next training decision](../references/whole_reply_book_direction.md)

The selected bilingual research writer reads a Chinese passage as a glyph stream and emits stopped answer images to pictured Chinese or English questions. With aligned English supervision it gave 121/748 exact English-request replies on Chinese novels versus 54 with the same English questions exchanged on the same pages. It gave all four answers on 3/374 book and 1/227 regular-print XQuAD passages, with zero in both controls. Familiar historical cards passed 99/100 pairs, while none of 300 canonical open answers requiring an absent content glyph matched. The most frequent measured failure is selecting the first answer glyph (573 of 627 English misses). At the first divergent action, a third action beats both annotated book answers on 536/748 English questions. Preserve this single-GPU bilingual checkpoint, its offline catalogue and the earlier writers. The next controlled intervention raises the weight of the unrestricted first-glyph distribution and compares it with equal supervision on a later action. General book reading, writing absent content and matched-quality efficiency remain open. [Results and original pixels](../references/bilingual_novel_result.md) · [Next comparison](../references/first_glyph_book_direction.md)

Updated: 2026-11-24. **Active; usable visual language remains unproved.**

Training on changes in stroke thickness and proportions increased exact HanaMin trial replies from 15 to 43 of 394 and raised complete two-question passages from zero to six of 197. The one-writer optical router recovered all 14 regular-print CMRC pairs and preserved the gains in both unfamiliar fonts. On a separate XQuAD evaluation in HanaMin the routed writer answered 22/454 Chinese requests and 2/454 equivalent English requests; neither language passed a complete pair in 227 passages. On the first 128 questions in each language local Qwen3-8B from text answered 85 and 74 versus 8 and 2 from images. Keep the earlier released writer as default. Matched-quality efficiency, open composition and general visual language remain open. [Figure, physical replies, costs and both failed selection decisions](../references/stroke_scale_result.md)

The next language comparison uses human questions in two languages about the same pictured novel passages. The prepared BiPaR corpus provides 11,526 Chinese--English question pairs on 2,910 training passages and 1,486 on 374 separate validation passages. Compare correctly paired bilingual supervision with same-page exchanged English questions and an equal-exposure Chinese-only control. Each validation passage must support distinct answers when the reader sees a new question. Historical requests, regular and unfamiliar print, open writing and end-to-end resources retain separate criteria. [Language comparison and corpus provenance](../references/bilingual_novel_protocol.md)

The image writer can now answer requests for two attested forms from the offline Hanziyuan catalogue. On 100 additional headwords it returned both requested orders exactly on every card with familiar Chinese wording. With paraphrased requests it did so on nine cards, and with English requests on none. The catalogue retrieves a page using the supplied headword. The writer receives the pictured page and question and returns an image. Its archive runs on CPU or a single GPU. [Examples, controls and model](../references/typeface_writer_result.md)

Across typefaces the same trained writer answered 72/192 questions in UKai and 13/196 in the withheld HanaMin family. On a second set of 394 questions, HanaMin gave 15 correct and no complete two-question passages. Providing the correct regular-print identity where available raised the same writer to 95 correct and 13 complete passages. Regular print gave 97 correct and 14 complete passages. Local text Qwen3-8B gave 142 correct and 33 complete passages. Keep the current default and retain the mapped writer as a runnable research checkpoint. Authentic copies of source inscriptions, response to changes in the question, writing absent content and matched-quality compute savings have separate evidence requirements. [Typeface study and measured costs](../references/typeface_writer_result.md) · [Checked mathematics](../formal/span_list/TypefaceCopy.lean)

The preceding optical comparison held the language writer fixed and varied the training images of its correspondence model. The downstream test is whether recognizing an unfamiliar printed page improves both answers when the question changes. Keep an additional font family separate for that comparison. In parallel, the general language criterion continues to require meaningful responses when no passage span can supply the complete answer. The present writer gave zero exact answers on 300 questions whose canonical answers require an absent content glyph. [Decision and evaluation](../references/typeface_writer_decision.md)

The visual interval-teaching comparison is complete. On 196 human questions the distilled writer gave 72 exact stopped images, ordinary replay gave 63, the visual span reader gave 113 and local Qwen3-8B from text gave 144. Both distilled and replay writers answered both questions on 11 of 98 passages. Keep the earlier mixed model as default and the distilled model as a runnable research checkpoint. It passed all 192 familiar historical requests, 56/192 rephrased requests and 0/300 canonical open replies needing a content glyph absent from the input. Its 46 ms warm response time on one RTX 4090 D excludes rendering and accompanies substantially lower accuracy than Qwen. The interval probabilities have been derived and checked in Lean. General language learning and matched-quality efficiency remain active goals. [Results, original pixels and model](../references/interval_teacher_result.md)

Changing the typeface of both inputs to Kai took the distilled writer from 72/192 exact replies to zero on identical questions. Restoring only the page recovered one of the 72 original successes, and restoring only the question recovered none. The interval study motivated teaching the same visual representation on both pictured inputs while retaining the raw source ink for copying. Compare a learned mapping to equal-exposure typeface replay from the same parent. Reserve an entire additional font family from all training and selection. Require improvement on complete question pairs in new print, preserve regular-print answers and independently score historical shapes, abstention on unrecognized ink and answers requiring original prose. Measure all additional training and inference costs. A future claim of general language learning still requires supported free replies and transfer to new sources. [Typefaces and controls](../references/interval_teacher_typeface_intervention.md)

The preceding mixed image writer answers 114/359 ordinary Chinese questions and 15/128 additional two-question CMRC passages completely. With the question erased it passes zero pairs; the span reader passes 49. Its historical-form scores are 190/192 and 54/192 with familiar and rephrased instructions. On two pictured paragraphs it passes all twelve date prompts on 23/48 distinct-subject and 14/34 shared-subject boards after learned selection. The from-scratch selector helps with shared-subject relations, while the 86.6M writer inherits text pretraining. Preserve the completed fits and both strengths. The next matched intervention should teach the independent writer where an answer starts and ends on an ordinary pictured page, using source-only alignment and a frozen visual span reader during training. Compare to equal whole-answer replay at equal exposure. Check the 98 reader-only and 21 writer-only outcomes, both-question performance on new sources, changed dates, inscriptions, stopping, teacher-free deployment and full costs. No claim of arbitrary new glyph synthesis or a 1,000-fold improvement follows from these experiments. [Paragraph comparison](../references/paragraph_evidence_result.md) · [Mixed training and physical outputs](../references/mixed_visual_answers_result.md)


A causal writer reading the actual page and question now produces both a named fact and its changed year. With varied instructions it answered every formulation and edit on 34/64 newly screened single passages and 8/48 two-paragraph boards. It often selects a date from the wrong paragraph, returns 7/359 exact broader human answers and fails both ordinary factual retention criteria. The one-instruction checkpoint retains the actual 言 demonstration in both orders. Preserve both checkpoints. The next intervention should train an explicit question-conditioned choice of evidence paragraph while allowing the writer to see the whole original request and source. Compare learned selection with full-page writing and a supplied correct paragraph. Require changed and unchanged distractors, natural human answers, stopping and historical-form retention. Measure selective quotation on competing passages; a whole-page copy already gives 125/128 on the single-page clause set. The general and efficiency goals remain open. [Model, Lean proof, outputs and source controls](../references/page_evidence_writer_result.md)

The screened wider reader returned 234/359 exact direct answers versus 242 for the generic reader. It recovered 12 lower-ranked references but missed all replacement performance thresholds. The official DRCD test reservation was exposed in earlier evaluations; see the correction under criterion 3. In 32 controlled year edits, identical factual fields forced identical writer output in every pair. The next intervention should give the writing model the original pictured request and selected evidence from the page, including the requested detail. Compare that change to the fixed-fields writer on new source-disjoint counterfactuals, both factual word orders, and open source-supported replies. Preserve correct simple answers and independent stopping tests. [Results, proof and original output pixels](../references/span_list_result.md)

In the earlier sample subsequently found to include reserved DRCD material, pooling four visual span readers matched the generic reader on 285/384 human questions but lost complete name exchanges (63/128 versus 79) and historical substitutions (11/32 versus 17 with generic object selection). The four-output experiment is complete. Wider span lists expose 46 of 84 previously absent exact answers. Test whether a question-conditioned scorer can actually recover these outputs using a larger physical candidate set. Select on natural paragraphs and genuinely unseen exchange subjects outside the frozen specialists' training sources. Freeze the rule before a second fresh human test and evaluate complete stopped replies, historical forms, and open writing independently. Keep generic and specialist models available. [Lean proof, actual images and diagnostic](../references/span_choice_result.md)

Conditioning the copied name on the pictured role raises complete generative name-exchange tests from 29 to 70 of 128 unseen DuIE passages. The same supervision with likelihood alone passes 79/128. Reversing clause order as well raises complete tests from 13 to 36 of 48 with the extra margin, or 37 with likelihood alone. On broadly sampled human questions direct answers fall from 147 to 121 of 256. The trained scorer is optional. Future work should train on naturally phrased, diverse questions with exact reply boundaries and evaluate the same counterfactuals alongside broader open answers. Preserve the previous reader and all failed transfers. The exchange tests establish adaptation to pictured relationships. From-scratch language acquisition, unrestricted writing and a measured matched-quality efficiency advantage remain open. [Lean result, measured transfer and weights](../references/participant_margin_result.md)

Reading the role named in the pictured question raises complete statements to 36/48 original human questions and 930/1,200 independent DuIE passages. The image writer generates both requested word orders on 34/48 and 892/1,200. A new four-way song-credit test succeeds on three inputs. After exchanging the artists, the reader sometimes copies the lyricist despite identifying the requested role as composer. The next participant-binding comparison should test the same two names across both roles, source orders and new phrasings, with predictions and control scores reported for all four cases. Keep broader questions, unseen typefaces and freely generated details as independent tests. [Pictured questions and emitted answers](../references/pictured_relation_result.md)

A visual fact reader trained on 26,890 pictured examples now writes 33/48 complete correct Chinese statements on original human questions, up from three in the preceding evidence-conditioned fit. A separate autoregressive writer expresses the same predicted fact in both subject-first and object-first sentences on 31/48 human questions and 810/1,200 held-out DuIE passages. Five Lean-checked results identify the assumptions under which a writer can generalize to new names. Actual historical-glyph substitution obeyed the expected exchange law on 14/32 examples. The system transfers pretrained text-model knowledge, covers twelve roles and emits unsupported details in freer prose. Across broadly sampled human questions it lost 40 of the frozen baseline's aggregate 187 exact direct answers. [Proof, output strips and transfer results](../references/role_binding_result.md) · [Runnable local model](../models/role_binding/README.md)

The next test should preserve enough source evidence for every new claim and measure whether an answer stays within that evidence when two passages have the same subject, relation and object but differ on a further detail. Keep a held-out corpus of human questions, unseen script substitutions, direct-answer controls and both requested word orders. Evaluate contradictions, unsupported additions, stopping, new glyphs and complete correct replies separately. Compare a learned choice of the general reader or factual writer against both fixed routes on new questions. General visual language acquisition and matched-quality end-to-end efficiency remain open.

An ordered image decoder now returns the attested oracle and bronze forms of 言 in either requested order. It learned to continue beyond one source span on one RTX 4090 D. With familiar wording the varied writer returned both orders correctly in 94/96 held-out comparisons. Rephrasing gave 36/96, and new cards and wording gave 16/96. Both fall below the 48/96 criterion. It recovered both orders on 4/128 human compound-question pairs and zero exact name-only answers on the 64 role questions. Some longer replies contain a correct proposition. All 37 exact open answers were single spans. Of 300 references requiring absent content ink, 235 were fully representable using the page and writing memory; none was returned in full. The earlier single-question reader remains available. [Original outputs, source coverage and controls](../references/ordered_page_writer_result.md) · [Ordered decoding checkpoint](../models/ordered_page_writer/README.md)

The ordered-writer study proposed conditional language generation on naturally written answers. Give the image writer a generative next-glyph prior trained on connected text or transferred from a Chinese generative model. Compare with matched compute and scrambled context, retain the strong single-span reading control, and measure complete new-content replies on previously unseen sources. Test whether a source change changes the answer while grammar and glyph legibility survive. Count valid alternative phrasings separately from exact images and explicit copying. Report training and storage for the complete deployed system, including inherited weights. An isolated historical plate does not establish open language generation or a compute advantage.

A continuous visual interface now transfers an existing Chinese language reader into a source-image responder. On 64 freely phrased role questions the combined model gives 46 correct answers in regular print, 41 in bold and one in an unseen Kai family. Native CKIP gives 49 and local text Qwen3-8B gives 55. The learned output decision retains 344/404 familiar composed statements and 30/32 historical descriptions. It reads 263/387 natural person answers, but unrestricted DuReader writing remains 0/571. On a real Hanziyuan plate it reads five of eight individual forms and none of four ordered two-form requests. Both cross-font acceptance criteria failed. Contextual transfer is established on these measured tasks. General generation, from-scratch visual language learning and a matched-quality compute advantage remain open. [Measured outputs and limitations](../references/continuous_interface_result.md) · [Usable image responder](visual-page-responder.md)

The ordered-image comparison above tested the preceding decision to select multiple source regions and vary the request order. The earlier Kai result remains an independent typeface failure.

A 782,171-parameter writer combines independent glyph generation with a learned contiguous source-reading action. On 864 pairs of human questions receiving the same sentence, the matched head answered both correctly on 100 versus 72 with randomized competing assignments and 58 for the frozen writer. Only one of 27 person pairs passed. A source review found both relations explicitly stated in just one of those 27. On 24 fresh, fully specified role exchanges, the calibrated writer gave 21/96 answers and no complete scene; local text Qwen3-8B gave 90/96 and 20 complete scenes. Calibrating the choice of copying or composing restored 344/404 familiar reordered statements and 30/32 bronze descriptions while retaining 148/387 person answers with automatic sentence selection. Open writing and a matched-quality compute advantage remain unproved. [Source-run model, original outputs and comparisons](../references/shared_sentence_writer_result.md)

A frozen CPU replay reproduced all 96 explicit-role answer images. Of these, 61 matched neither name and 14 matched the other person. Even when supplied with both name boundaries, the trained scores preferred the requested person in 53/96 questions and in all four versions of zero scenes. Both answer location and sensitivity to the question need improvement. [Frozen role diagnosis](../references/shared_sentence_role_diagnosis.md)

The next learning comparison must use source-supported roles, change the role while preserving the names, and separate position and verb memorization from language transfer. A pinned Chinese treebank now provides 872 training, 113 development and 106 test predicate instances with both a grammatical subject and an object rendered in pixels. Augment its canonical word order before claiming that the model learned the underlying relation. Keep the independent explicit-role and natural passage controls, original composition tasks and historical shapes. [Corpus preparation](../scripts/prepare_pictured_grammatical_roles.py)

A matched participant-exchange fit now writes 368/404 complete reordered statements for names and complements outside the role-training pool, versus 186 with randomized assignment and zero for the parent. All four source–question combinations pass for 80/101 situations versus zero controls. With two attested bronze shapes outside the fixed memory it writes 24/32 complete source-grounded descriptions. Both earlier and new independently worded role tests remain weak (5/48 and 1/32 both-correct), and ordinary OCNLI judgments fall to 1,533/2,947. Requests naming only the property recover zero exact participants in print or bronze. Even scoring the two supplied names plus stop gives both orders on only 9/16 printed and 8/16 bronze pairs. The earlier spelling selector and supported passage writer remain separate. [Model comparison and portable inference](../references/visual_role_writer_result.md)

The inverse participant experiment is complete. On 387 human “who” questions the matched adaptation gives 62 exact stopped answers versus 64 with randomized competing assignments and 95 for the deployed passage reader. In 314 of its 325 errors it copies a contiguous passage fragment; 301 miss the first glyph. Selecting one sentence by pictured glyph overlap gives 116, 127 and 100, respectively. An annotated evidence sentence gives the matched writer 141, leaving 246 errors even with evidence localized. It returns the first bronze name on 27/32 requests and follows the correct participant through only 7/16 exchanges. Keep the standalone optional sentence selector and the original controls. [Measured failure, correction and operating instructions](../references/inverse_role_writer_result.md)

The shared-sentence comparison above followed the earlier decision to test which name within a sentence bears the requested relation. Use naturally occurring competing answers and evaluate paired questions where sentence selection alone cannot distinguish them. Compare against the identical frozen overlap selector, the randomized weights and the released passage reader. Preserve independent short and multi-sentence cases, new historic shapes, free writing and end-to-end costs. Copying a source glyph remains a distinct claim from understanding an etymology or writing unsupported content.

The 694,152-parameter writer now learns from 33,441 original Chinese OCNLI premise–claim pairs and reconstructs missing writing. Paired evidence raises exact official relation sentences to 1,610/2,947 versus 1,514 after premise reassignment and exact stopped restorations to 524 versus 133. A visual neighbor-count selector calibrated on separate premises raises restorations to 651, with 170/1,858 requiring absent ink versus 31 for the neural writer alone and 196 for simple counts. This is short-span completion with fixed writing memory. On 105 identical cases, local Qwen3-8B gives 73 correct relations against 59 for the visual writer; combined restoration is 26 versus 18 with an interval crossing zero. The model reverses both decisions correctly on zero of 48 controlled changes of subject, loses ordinary passage writing and gives zero whole DuReader answers. Preserve both controls and the original shipped passage reader. [Measured evidence, commands and costs](../references/visual_inference_writer_result.md)

The local nine-cell question encoder has a constructive language limit. Two training comparisons and a frozen cue exchange are now complete. On 96 independent English dictionary requests, genuine matching raised complete images from 19 to 48, versus 21 under shuffling. Yet a local writer gave the identical response to four semantically distinct questions with identical nine-cell windows in every tested group. A 694,152-parameter writer with relative reading-distance attention completed all four in 1,060/1,136 headword–layout groups with familiar clause cues, versus zero with unordered global attention, and gave all 16 responses for 264/284 held-out headwords. It gave zero complete groups on unfamiliar cue wording. Changing only “I already have” to “I can supply” reduced the score to 16; changing only the requested clause cue gave 246. Independent English reached 60/96, independent Chinese fell to 56/96 and natural absent-content writing remained zero of 300. The ordered fit took 82.9 seconds and 0.43 GiB allocated on one 4090 D after substantial inherited training. It is a research checkpoint. Preserve the stronger released passage reader and previous complete outputs. [English supervision](../references/english_request_writer_result.md) · [Position, counterexamples and images](../references/scoped_request_writer_result.md)

The natural opposed-evidence study is complete. On 18 official question pairs the adapted writer gave both opposing human positions correctly for 10 questions, versus one under reassignment and six for lexical markers. It wrote 11/36 complete stopped answers, nine quoting an entire reference-selected window. Its paired validation advantage over the marker included zero. All 144 validation source records appeared in earlier answer pretraining. Erasing the question left the position unchanged on 30/36 official examples. On 24 new passages with two questions requiring opposing stances the matched writer kept its stance in every case. Preserve both fits and their complete outputs. The next comparison must train and test the dependence of output on the claim as well as on evidence, include independent single-input controls, and preserve the separate criteria for open writing, transfer, historical ink and total cost. [Opposing evidence and question switches](../references/paired_stance_writer_result.md)

Two further fits separate choosing an answer, locating its source and writing it. The same-passage writer recognized both supplied answers on 522/1,042 nontrivial DRCD pairs, yet wrote both from an empty output on only 21. In 1,203 stopped errors the reference outranked the model’s actual response by mean log probability just 30 times. A separate dictionary fit scored 284/1,136 whole standard answers. All responses opened with the requested headword, but 568 used the other headword’s shape from the right stage. Giving only the correct record recovered 1,104 answers. Automatically locating an explicitly named pictured headword recovered 1,103, including 819 additional correct answers and no losses. Independently worded Chinese improved from 19/96 to 60/96 with location. English located the correct entry 1,136/1,136 times but wrote only 249 exact answers. Both tested 言 prompts used the right fact and attested shape. These are layout-specific source-grounding gains. The mixed writer retained just 12/1,042 natural DRCD pairs, 15/571 canonical DuReader answers and zero of 300 references requiring new content ink. Retain the stronger previously released passage reader. [Paired natural questions](../references/same_passage_writer_result.md) · [Attribution, source-selection intervention and actual images](../references/general_entry_writer_result.md)

The next question is whether a learned evidence locator can read across visual changes in a headword and use clause context when the headword is absent or repeated. A complementary controlled language test should change only the lexical field or script in independently authored questions. After correct English entry location, 631 errors exactly answered a different question about that headword. Use frozen, family-separated rerenderings and intact clean-image controls. Compare learned location with exact physical matching and supplied source locations, and report full independently written answers and unmarked natural passage performance. Keep natural composition a separate requirement. Use real first errors on passages with verified answer evidence to test a writing objective that competes with the writer’s own mistakes. The 7,715-image memory permits copying unseen visible forms but limits independent drawing. Record all initialization, training and inference costs before making any comparative efficiency claim.

The matched question–answer experiment is complete. On 191,007 real pictured training triples, both from-scratch writers learned to reproduce legible ink. Correctly paired training yielded 32/571 exact stopped responses versus 37 with reassigned questions and 66 for a question-blind first sentence. Both learned writers gave 0/300 input-absent answers. On a separate two-candidate diagnostic the correct pairing improved both-order preferences to 25/285 versus 1/285, while direct bitmap overlap reached 231/285. The published probes reproduced the entire premise. Preserve the released passage reader. This result motivated the comparison teaching the same writer to select different facts from one pictured passage when the question changes, with evaluation cases on which raw glyph overlap ties or favours the wrong answer. A newly prepared DRCD subset has 7,108 training pairs and 1,042 held-out pairs across 521 development passages. Its answers are extractive. Use source selection and free DuReader composition as separate outcomes, with independent new-shape and resource criteria still open. [Training and failure analysis](../references/grounded_glyph_writer_result.md) · [Follow-up cohort](../references/evidence/grounded_glyph_writer_20261119/prepared_followup.json)

The first autonomous DuReader composition comparison failed. All 25 of 571 exact outputs were contiguous and a question-blind first-sentence baseline gave 66. A pixel correction damaged 106 correct missing glyphs and repaired one. An independent ordered-image spelling prior trained on 192,230 additional answers raised supplied-history exact missing cells from 162 to 431 of 1,525 (shuffled 121). Autonomous input-absent answers stayed at zero of 300; 228 outputs diverged before the first missing glyph. Preserve the old released passage reader and all failures. This result motivated using the actual paired question images for those 192,230 answers to learn the first warranted answer segment and subsequent actions conditioned on the request and evidence. Compare with identical question-erased or rotated supervision, a question-blind first-sentence baseline and the fixed ordered speller. Keep full autonomous answers, paired question sensitivity and externally supported content as the central outcomes. Preserve unseen-shape emission as a distinct requirement. [Composition and damaged glyphs](../references/composed_writer_result.md) · [Conditional spelling and failures](../references/bitmap_history_writing_result.md)

Additional passage supervision has now passed its matched comparison. The mixed 101,144-parameter reader gives 1,270/3,505 exact DRCD singles versus 380 under replay and 500 before training, and 1,111/3,170 on official CMRC versus 999 before training. It recovers 613/4,638 DRCD pairs in both orders with a short instruction and only 77 with the longer traditional cue. Use the mixed checkpoint as the current passage baseline. The next language test requires naturally composed answers that cannot be extracted from one span, including answers with content glyphs absent from both prompt and passage. Preserve the single-question and exchanged-order comparisons and distinguish finding source evidence from independently writing a faithful answer. [Comparative results](../references/traditional_passage_training_result.md)

The prepared natural writing test contains 2,565 training questions and 571 held-out questions. Three hundred held-out references require a content glyph absent from the question and evidence, giving a strict upper bound of 271 exact references for any writer that only rearranges input cells. Evaluate the next writer on these independently checked inputs against a source-copy baseline. Measure exact newly written glyphs, complete answers, stopping and factual support separately. A train-answer bitmap memory with perfect context choices could reproduce 289 of the 300 input-absent references. Compare that finite writer with the open raster writer and score correctness under changed prompts. [Pictured writing corpus](../references/composed_chinese_answers_preparation.md) · [Bitmap support and writing objective](../references/composed_answer_bitmap_support.md)

The paired-passage trial is complete. A 101,144-parameter system answers in source pixels, reads the complete prompt, and can exchange two distinct answers when the questions change order. On 836 official CMRC development passages the same varied planner gave zero of 4,800 both-correct pairs on a short Chinese instruction with independent cell decisions and 505 with a contiguous-question decoder. It gave 518 on the familiar instruction and 999 of 3,170 exact single answers. Retain the constrained reader as the passage baseline. The next comparison must improve evidence selection on natural questions, preserve these instruction and single-question tests, and measure a source-disjoint traditional-Chinese collection. A separate traditional-Chinese training corpus now supplies 21,934 DRCD questions and 3,505 development questions with verified physical answer intervals. Test additional data against equal-update replay and measure both CMRC retention and DRCD generalization. Track source copying, freely written answers and full training cost independently. [Prepared corpus](../references/traditional_passage_preparation.md) [Measured results](../references/paired_passage_result.md)

A shared-update reader recovered 66/96 independently authored pictured dictionary requests, versus 18/96 after equal-compute replay and 52/96 with a candidate-specific update. It learned the familiar before-field construction (4,540/4,544) but solved only 4/1,136 distinct corrections and zero complete groups using former/latter references. Keep these data and both failures. Move the central language test beyond framed dictionary requests: a prepared corpus of human-written Chinese passage questions has 24,846 training and 6,086 development requests, each requiring two complete answers in order. Only two development targets appear as a single span. A supplied-split composer reaches 440/6,086. Compare a learned visual writer with that composer and an order-erased control, verify every output and its order, and keep single-question retention and independent dictionary answering as separate measurements. Open-book translation, multiple scripts and the full compute claim remain active. [Independent author and controlled fits](../references/independent_entry_result.md) · [Paired passages](../references/paired_passage_questions_preparation.md)

A 51,776-parameter ordered visual reader chose the correct fact and historic glyph on 4,542/4,544 family-disjoint Chinese requests, versus 1,136 for its sorted control. All four answers were exact on 1,135/1,136 groups; reserved expressions gave 1,134 complete groups. Both arms retained all 1,136 simple requests. Moving exactly the same status expressions before their fields reduced accuracy to 356/4,544 and zero complete groups. The comparison is complete. The next test must distinguish grammatical roles across different placements in independently phrased Chinese as well as authored controls. Keep the source and target images matched, reserve entire constructions and distinct families before fitting, and count every complete answer and all role switches. Restore a separate natural-passage measure before extending the claim to general language or efficiency. [Measured result](../references/entry_question_reader_result.md)

A 28,032-parameter recurrent writer trained on complete pictured questions and answer images. It gave 1,134/1,136 exact sentences on 284 held-out Hanziyuan families and 1,132 after rephrasing while retaining the printed field names. Deranged supervision gave zero exact sentences, and the authored parser reached 1,136. Swapping source values changed the answers appropriately on 1,134 requests. All eight requests were correct on 141/142 original cards. An equal-length synonym audit identified an architectural limit: all 568 pairs requiring distinct answers are represented identically. No further tuning of these weights can recover that erased distinction. Carry the whole pictured question into the next writer, compare matched cue supervision, and require complete correct answers on unseen families and formulations. Maintain the separate broad translation and efficiency criteria. [Evidence and standalone inference](../references/pictured_entry_writer_result.md)

A separate 2.33-million-parameter translation model learned 39/39 familiar contrasts, reached 1/15 on new pairs and conveyed 0/18 complete unseen book passages. The source-card result is a different task. [Translation result](../references/context_phrase_trial_result.md)

The registered supervision comparison is complete. All 1,024 source-only translations were reviewed, yielding 841 eligible passages and a nested set of 108. The two single-GPU writers had identical book replay and matched supplemental exposures. Both scored 0/18 on complete unfamiliar translations and 4/8 on paired quantities. The broad writer learned a supervised biography and reduced glyph edits on 192 fresh passages, while adding an age unit to a celestial count. The text teacher scored 12/18. A matched visual-language efficiency advantage remains unproved. Preserve both checkpoints and the frozen assessments. The next question is whether the writer uses contextual evidence to select a meaning when the same visible span occurs in different roles. Compare matched context-answer swaps, including roles, polarity and quantities, against copying and the current writer. Require gains in autonomous whole answers on untouched chapters before scaling. [Result and evidence](../references/visual_translation_coverage_result.md)

A training-only search found 10,244 pairs where two source images share an answer prefix and imply different recorded next glyphs, both absent from the sources. Only two of twelve inspected pairs provisionally distinguished source-supported meanings. Grade competing phrases and remove synonyms, extra context and unrecoverable dates before using this material as supervision. [Pair inventory and editorial pilot](../references/visual_context_fork_preparation.md)

The preceding matched target experiment reproduced thirteen of eighteen numeral changes on familiar frames; a one-cell substitution control solved all eighteen. It remains a separate positive control for local transfer. [Figure 67](../references/screened_visual_targets_result.md)

The ordered-copy writer completed six passes over 55,872 paired passages on one RTX 4090 D. On 512 passages from unseen chapters it recovers 84.4% of reference copy positions, but its 30,884 glyph edits exceed the composer's 18,419. Complete faithfulness is 2/24 versus 3/24. A finite gap model removes the repetition loop, reaches 18,873 edits and still scores 2/24. Only four of the 24 passages have any credited source or generated choice across the tested writers. Both neural successes are changed outputs absent even as contiguous strips in all training targets. Keep the shipped writers experimental. The next controlled learning change is source-supported supervision: compare original and verified modern targets for the same pictured source passages, preserving compute and independently testing meaning on new chapters. A training-only local Qwen3-8B pilot yielded twelve reviewed pictured pairs from 32 drafts. Expand reviewed coverage before fitting. [Figure 66, evidence and decision](../references/anchor_gap_result.md)

The source-boundary comparison repairs all three earlier distance-expression failures and two fresh authored examples. Both matched writers give one fully faithful natural pair in six and zero faithful diverse passages in twelve. Even an ideal single three-cell substitution can reproduce only 60 of 1,024 recorded reference glosses under the fixed surrounding composer. This motivated learning to copy and rewrite several parts of a sentence from images. An initial training-only preparation covers 55,872 passages and 356,806 changed regions. A frozen 512-passage cohort covers 390 further chapters, with 24 complete answers selected for reading before fitting. Require independent full answers, with true correspondences reserved for a diagnostic. [Figure 65 and results](../references/phrase_boundary_result.md) · [Corpus and next experiment](../references/ordered_copy_anchor_protocol.md)

The conditional verb comparison is complete. A full-corpus neural contrast fit produced 0/15 complete both-sense pairs; supervised phrase emission conveyed both senses on 7/10 further pairs but both complete propositions on 0/10. A supplied-cue baseline achieved 1/10. Five published examples reproduced through independent CPU image inference. The next bounded comparison should learn the source-span boundaries as well as the target phrase, test previously untreated constructions on fresh chapters, and retain complete autonomous answers, an explicit cue comparator and the broad regression. The current phrase writer does not qualify as a general language model. [Results and Figure 64](../references/predicate_context_result.md)

The participant-exchange experiment reached 4/12 faithful printed pairs versus 2/12 with replay. Reversible assignment of unfamiliar historical ink preserved both names and rewrote both directions on 14/20 pairs. Whole-meaning reading credited 2/8 mixed-script pairs and 6/10 controlled answers with fixed Qin and Chu images. The exchange writer still scored 0/12 on diverse passages. The next language test holds participant images and the predicate fixed and varies context to distinguish two readings of the same predicate. Require improvement on unseen complete answers against shuffled context, literal copying and a phrase composer. Track regression on diverse held-out chapters and publish original failures alongside successes. [Experiment and Figure 63](../references/visual_event_roles_result.md)

The paired teacher-label comparison is complete. Both independent students gave zero faithful answers among eighteen. Pooling routes to the same visible ink reduced 2,545 complete-answer edits but still yielded zero faithful translations. The completed full-corpus comparison trains visible-image likelihood on all 55,872 real sentence pairs. [Paired study](../references/paired_student_result.md) · [Full-corpus comparison](../references/visible_image_likelihood_protocol.md)

The all-path writer preserved the speaker and the permission/refusal distinction in all eight prompts crossing four speakers and two signs. Seven complete outputs were acceptable, with no complete source or neural output present in training. The same checkpoint produced zero faithful answers among eighteen diverse new passages. Treat the four-speaker result as a narrow positive control. The next test is role binding in naturally occurring transitive clauses. Use paired changes of agent, recipient and destination with a common event vocabulary. Score full answers and preservation of both roles, compare with the stronger phrase composer, and reserve unseen predicates and chapters. Keep the default unchanged until a replication improves whole meaning on diverse passages. [Measured output and limitations](../references/visible_likelihood_result.md)

## Objective and deployment boundary

Build a useful, independently running language model that reads visible writing,
understands the supplied prompt and generates readable answer images on one
RTX 4090. Its causal history must include its own emitted pixels, and it must
learn when to stop. Begin with modern-Chinese questions, passages and book
continuation; this development stage does not complete the full objective.

The full target includes simplified and traditional Chinese, historical forms,
handwriting and writing absent from Unicode. The eventual word-origin application
accepts Chinese or English questions and produces explanatory writing with
attested historical forms and provenance. Generalize the visual representation
to other writing systems, including cuneiform, after useful modern-Chinese
language is established. Raw pages, three-dimensional glyph streams, movies and
audio remain later extensions of this language objective.

Typed prompts may be rendered into images at the input boundary. Offline
teachers, OCR, text corpora and inherited foundation weights are allowed for
preparation and training; disclose their contribution and cost. Deployed answers
must come from the trained model: no live teacher, tokenizer, lexical output
head, answer lookup, OCR or font renderer in answer generation. Optional
transcription may read a finished image without changing its pixels. Transferring
inherited language capability and learning language from images alone are
separate claims.

## Current language experiment, 18 November 2026

The source-only causal visual writer can modernize a short military statement while preserving its participants. The matched clean-history and error-trained checkpoints both read 尝攻襄州，为孔勍所败。 and write 曾经进攻襄州，被孔勍打败。 Each earned only one acceptable gloss in the eighteen-passage reading. On 1,024 fresh passages, error training increased complete physical edit error from 98.85% to 194.69%. Preserve both experiments. Wider exact phrase search also failed to rescue any of twelve reviewed failures with acceptable references.

The source-conditioned reviser now changes 189 of 1,024 fresh drafts. Its 49-edit physical gain costs two acceptable drafts among eighteen changed passages, with no newly accepted answer. On eighteen source-stratified passages it receives zero acceptances against ten for a local Qwen3 8B text baseline. A separate Qwen3-VL 8B checkpoint receives twelve with typed sources and eight with pictured sources, despite transcribing sixteen exactly. The next learning comparison has 511 usable offline teacher glosses from 512 training chapters matched to the original inputs and archive controls. Six teacher-only gains and six teacher-only losses in a 24-passage paired reading justify retaining both targets. These preparations do not measure student performance. Evaluate autonomous source-sensitive meaning, retention of identities and dates, learned stopping, and all preparation and training costs. The general visual language and matched-quality efficiency objectives remain open. [Revision experiment](../references/pictured_revision_result.md) · [Teacher-target protocol](../references/teacher_gloss_preparation.md)

## Independent generation of absent Han characters

The [candidate-free idiom trial](../references/pictured_idiom_result.md) provides a [standalone 2.12-million-parameter model](../models/pictured_idiom_zh/README.md) that reads pictured prose and writes the missing expression. It exactly completed 134 of 22,907 official validation gaps, including 100 in which all four target glyphs were absent from the input window. Whole-answer alignment raised the original 35 exact answers to 134. Exchanging the contexts gave 23. The local count baseline gave 1,012. Of the 930 passages solved when the reader could choose among pictured training expressions, the independent writer solved 79. It solved none of 38 whole-expression novelties and both members in zero of 8,900 matched-context pairs. Figure 40 shows a correct answer and a failure. Extra sampled proposals failed the registered improvement criterion. Repeating that search is unwarranted.

The [matched reader-transfer comparison](../references/visual_writer_transfer_result.md) is complete. The independent writer produced 378 exact development expressions, up from 134 originally and 292 with equal extra training on the source targets. It recovered 190 of the 851 previously lost reader successes. On 9,135 newly held-out passages the corresponding counts were 163, 60 and 125. Eight nearest input glyphs alone gave 126, outer glyphs alone seven and the local count baseline 406. None of 38 development or 18 fresh cases requiring an unseen whole expression was solved. Figure 41 shows a new success and a remaining error. The runnable physical writer uses a 1.1 MiB update and the original base weights, with no teacher or expression inventory at inference.

The [bilingual source-plate comparison](../references/historical_plate_result.md) now qualifies an integrated reader. It selected and stopped on the exact requested historical images in 9,565 of 9,600 questions under each of three frames, gave the same score after relocation, and retained every original modern stream. The 249 development families are disjoint from training. All 35 errors involved 文 as both a competing headword and part of a stage name. The public executable accepts typed or pictured questions and a supplied four-row plate. Figure 42 reports the physical outputs, including an error.

The [local teaching and whole-answer comparison](../references/expression_preference_result.md) is complete. Local hints did not reliably improve on the guided writer. A preference loss improved supplied pair choice but reduced independent completion relative to jointly training the reader and writer on the observed answer. That joint reference fit is independently confirmed and released. Across 17,555 previously unused gaps it produces 433 exact stopped answers, versus 276 for the guided writer and 789 for local counts. There are 309 all-four-absent successes, 4/6,646 both-correct pairs and 0/21 exact novel expressions. All 70,220 emitted cells match known glyph bitmaps. The historical path preserves every intact and relocated stream. The fitted runtime remains independent of a teacher, answer inventory, tokenizer and output font.

The released writer's ten most frequent outputs accounted for 54.6% of 17,555 answers. A [matched comparison of passage diversity](../references/passage_diversity_result.md) is now complete. The two fits each saw 571,512 examples. On 17,862 fresh gaps the starting writer, repetition and distinct passages recovered 410, 516 and 550. The 95% interval for diverse versus repeated data touches zero and its top-ten share is higher. On 17,962 further gaps inverse-square-root frequency weighting recovered two rare cases but reduced complete answers from 521 to 427 and increased the top-ten share to 52.5%. Both comparisons failed release. Keep all final weights and original streams. The independent joint-reference checkpoint remains the deployed model. A local baseline returned 974 exact answers on the last holdout, including 739 the uniform writer missed. The writer alone solved 286.

The [ordered-neighborhood and writing-control experiments](../references/ordered_neighborhood_result.md) are complete. The local reader solved 801 of 26,213 new idiom gaps, against 797 for the original reader, 805 for shuffled views and 1,504 for local counts. Giving the writer the true target images yielded 2,759 of 3,845 exact training expressions and zero complete new combinations of the same glyphs. New-combination prefix counts were 3,514, 159, 6 and 0. With true landmarks, the renderer could draw all four glyphs in 3,833 of these strings. The biggest loss occurs at the first glyph boundary. Keep the independent joint-reference checkpoint deployed.

The matched boundary reset is complete. With length supplied and identical within-glyph decoding, clearing drawing history raised complete unfamiliar strings from 0 to 1,450/3,845. Observed expressions fell from 2,484 to 1,564. No weights changed. The [independent landmark-planner trial](../references/glyph_planner_result.md) is complete. On 17,686 new idiom gaps, local commitment yielded 136 exact stopped answers, four-history search yielded 176 and the existing uniform writer yielded 547. In a separate set of 17,723 gaps, the four-history planner improved from 191 to 435 after four more passes over real answers. Equal teacher-guided training yielded 383. The uniform teacher solved 540 and local counts solved 977. No new deployment qualified.

Before further fixed-length idiom optimization, address the transfer to variable-length passage questions. Among the 1,788 held-out CMRC answers, 680 fit the eight-glyph cap and only 183 also reproduce exactly from their true 48-bit ink measurements through the current renderer. Combining both training inventories produces 3,975 physical forms and 14 ambiguous landmark signatures. A training-majority bitmap lookup supplied with the correct target landmarks recovers 1,612 complete held-out answers, and every one of the 1,788 references occurs verbatim as pictured cells in its source. Test a question-conditioned choice between copying a pictured span and generating absent glyphs, with the whole passage available and no answer-conditioned crops. Train and score a length decision on real variable-length targets; keep the full test denominator, including long and unencoded writing. A canonical renderer test cannot establish prompt understanding. Independent answers and matched-quality compute remain the release criteria.

## Question-dependent reading and the length of independent writing

The [mixed-reader experiment](../references/mixed_ink_reading_result.md) trained a 61,972-parameter visual question reader beside the frozen physical writer. On 3,219 official CMRC questions the standalone CPU executable returned 1,000 complete accepted answers. The earlier reader returned 565. Exchanging questions gave seven on 3,158 eligible cases. On 17,600 additional idiom gaps the writer produced 441 exact missing expressions, including 330 whose four glyphs were all absent. The separate historical-plate route retained every intact and relocated output. The initial default-precision retention failure and the exact historical-precision replay both remain recorded. Figure 47 shows original answers and a failure.

On 47,944 of 53,585 pictured adjacent book segments the reference exceeds the eight-cell drawing limit and is unavailable as a contiguous source span. Training the present architecture longer cannot reproduce those references. The next bounded model comparison should learn a recurrent choice to copy a source image, draw a new glyph, or stop after each emitted glyph. Use real mixed-origin training targets, full variable-length answers, frozen-source controls and a held-out passage split. Require retention of natural reading and evidence of correct composition beyond the existing expressivity bound. General book language, attested bilingual origin explanations and matched-quality compute remain open.

## Editing a sentence with visible and newly drawn writing

The [classical Chinese visual editing trial](../references/visual_edit_translation_result.md) removes the earlier whole-answer length restriction. A [2.03M-parameter research checkpoint](../models/visual_edit_zh/README.md) trained on 48,000 sentence pairs copied 2,045/2,045 pictured passages and reproduced 30/2,048 modern translations on separate chapters. Three of 1,964 longer mixed sentences were exact, including a two-glyph modernization of the honorific for death. Overall translation failed the local-edit control. An exact ink-sum audit also found three real source-order collisions, including ages 65 and 56. Keep position when arranging glyphs on a page or compressing their stream.

With correct preceding history the writer chose the reference drawing on 3,816/37,451 draw positions. The same renderer recovered 37,427/37,451 given the reference measurements. The [matched whole-glyph comparison](../references/edit_contrast_result.md) is complete. The likelihood control used source order. Reversing the source while keeping the cursor glyph fixed reduced exact drawings from 5,633 to 4,695 of 55,758. Whole-glyph contrast increased fresh translation error from 67.90% to 69.92%. Correcting for sampled glyph frequency partly recovered fidelity but failed the local and equal-training comparisons on both newly reserved chapter sets. Preserve the stable likelihood objective. Test additional distinct parallel passages against repeated training with matched exposure, independent complete changed sentences, local edit counts, retained quotation and fresh chapters. The goal of useful autonomous visual language and its matched compute advantage remains active.

The [sentence-diversity experiment](../references/edit_diversity_result.md) compared 192,000 distinct new translations with four passes over 48,000. Across two held-out chapter cohorts, the diverse editor wrote six exact changed sentences in 4,096 requests. Its total glyph error increased slightly. Modal insertion allowances made 3,845/4,085 renderable reference lengths unreachable. A fixed upper-quantile allowance reduced that count to 94 but increased actual edits from 76,869 to 92,944. The [released decoder and example](../models/visual_edit_zh/README.md) reproduce original output pixels. Figure 50 shows a successful calendrical modernization beside a legible semantic failure.

The [path-alignment comparison](../references/visual_span_result.md) trained the new writer on 16,384 pictured Chinese pairs with a shared output-prefix state. Summing paths increased correctly recovered inserted spans to 484/10,789 (365 with a fixed path objective). It preferred the correct natural phrase assignment on 213/293 pairs, versus 161 after source reversal. With subsequent modern writing withheld the result was 210/293 versus 148. More capable selection among supplied phrases did not yield reliable autonomous translations. Greedy actions could repeat until the cap; summing cursors restored stopping on every renderable request but achieved zero exact changed translations. Minimum estimated edit risk also achieved zero; all 65,248 sampled translations stopped, yet only two distinct inputs had any exact changed translation in 32 samples. Figure 51 reports original output. The training and evaluation code and [CPU-runnable research update](../models/visual_edit_zh/README.md) are available.

The [whole-phrase trial](../references/visual_phrase_result.md) is complete. Given the actual boundary and preceding modern writing, the source-conditioned writer generated 605/10,789 whole phrases; the prefix-only writer generated 307, the original writer 417 and local counts 1,676. It recovered none of 3,281 unseen insertion instances. It drew an incorrect first glyph at 9,516 sites. The model's exact physical measurement probability favored the emitted first glyph over the recorded one in 9,453 of those cases. The renderer could reproduce 10,783 full phrases when supplied the correct landmarks. Keep these fitted models and original images as controls. The next comparison should estimate the conditional first-glyph ink distribution from local source and modern context with statistical backoff and no complete glyph or phrase inventory. Keep the remainder of the writer fixed, require a replicated advantage over local controls in independently produced phrases, measure parameter or count storage and compute, and retain unseen-phrase and stopping criteria before integrated sentence testing. The full objective remains useful independent visual language generation at matched quality and measured cost.

The [local visual context experiment](../references/local_ink_result.md) raised whole-phrase generation on 5,987 fresh insertion sites from 187 to 405. Local whole-phrase counts recovered 651. A study-involved blinded reader accepted 8/48 generated insertions and 45/48 recorded translations. The writer drew 35,279/35,290 known bitmaps yet solved none of 2,433 instances of unseen phrases. Perfect stopping alone is bounded at 590 exact answers. A descriptive recount found 534 sites with the same local observations and competing recorded phrases. Next, test one earlier within-glyph pattern in the local probability correction, preserve the full sentence state and score competing contexts. If phrase selection stalls despite a likelihood gain, change the sentence-level learning signal. Keep the supplied boundary and prefix explicit. Whole-sentence translation and the matched-quality resource target remain open.

The [within-glyph transition comparison](../references/conditional_ink_result.md) raised four-cohort exact stopped insertions from 2,027 to 2,117 of 23,211. Training-frequency counts produced 2,997. The sole novel complete insertion, 所以来, occurs within six modern training sentences. On new chapters, both neural writers received 13/48 acceptable judgments. Swapping full sentences while preserving local evidence reduced independent exact insertions only from 158 to 151 of 1,572. Stop tuning stroke transitions as the main language intervention. The [sentence-context comparison](../references/sentence_choice_protocol.md) now has 120,628 training pairs and 6,720 calibration pairs from 240,000 deduplicated sources. Train the same source-boundary residual under matched positive likelihood and competing-source supervision. Qualify on independently drawn, stopped and contextually reviewed answers, including new sentences and phrases. General-purpose language and efficiency at matched quality remain open.

The [matched sentence-learning comparison](../references/sentence_choice_result.md) is complete. Exact stopped phrases rose from 2,143 to 2,529 of 24,048, a total reproduced by changing only the ending computation. Multi-glyph exact answers fell from 138 to 60, and the paired learner solved zero of 7,504 unseen insertion instances. In 1,093 of 1,100 incorrect first images the model assigns greater probability to the emitted signature than to the recorded one. The [next comparison](../references/first_emission_learning_protocol.md) trains directly on the competing first images actually emitted on training passages, while keeping the parent's later drawing and ending decisions fixed. Verify source-dependent first choices, multi-glyph language and new chapters before expanding the fit or making efficiency claims.

The [first-glyph comparison](../references/first_emission_result.md) is complete. Preference over actual previous model errors increased correct first images from 4,920 to 5,090 and exact stopped insertions from 2,143 to 2,280 of 24,048. Whole-phrase counts recover 3,554. With neither boundary nor prefix supplied, neither writer reproduced any of 84 complete translations. Of 62 preference outputs capped within a phrase, 52 ended in a threefold visual repetition. In matched choice calibration, source exchange changed the compact writer from 33 to 31 correct sites out of 64. A local 4B teacher showed a larger source effect but missed the registered accuracy threshold even after strict response-format resolution. Prioritize [complete pictured sentence meaning and independent clause generation](../references/full_sentence_meaning_protocol.md). Keep the trained residual as a control. Additional data and compute should target measurable transfer beyond local visual counts and source copying. A new calibration packet supplies 64 within-chapter translation pairs from 64 distinct chapters, balanced on unigram, bigram and length advantages. Verify linguistic identifiability before training on this type of comparison.


The [source-only comparison](../references/source_only_visual_translation.md) trained a 1.70M-parameter bitmap writer and two phrase composers. The neural writer used source order but lost to copying and received 0/24 acceptable complete glosses. The smaller composer produced five changed exact sentences absent from training and reduced glyph error by 2.97 percentage points from copying. Adding 23,104 pairs made little further difference on 1,097 additional passages. Keep copying, local transfer and changed complete answers as separate controls. The next language decision is whether correct clause meaning survives among independently proposed bitmap phrases, especially for names, predicates and negation. Require fresh chapter-disjoint evaluation and blinded whole-answer gains before scaling a new scorer.

## Learning speed on visible Chinese passages

The [512-article successor comparison](../references/visual_successor_result.md) measures how often a reader assigns different glyphs to contexts sharing their final cell. A compact visual model consumed 1,047,284 training targets in 19.46 seconds on two CPU threads. The matched character-indexed model took 18.73 seconds. They scored 55.1% and 56.2% on supplied-candidate assignment. The visual advantage over rotated supervision remained uncertain. The native writer preferred the same candidate for both contexts in 509/512 articles. Preserve these results when evaluating a change to the writing objective. Both individual choices and independently generated answers matter to the deployment goal. General-language efficiency at matched quality remains open.

The [unrestricted drawing comparison](../references/context_contrast_drawing_result.md) tests the same writer without supplied successor images. Subtracting last-glyph density increased the average change between two contexts from 7.08 to 60.18 pixels, with zero exact successors across 1,024 drawings in every arm. Both new complete answers about 言 remained unreadable. Retire this decoding intervention.

The [physical patch writer](../references/patch_count_writing_result.md) now provides a trained generative reference. Across 334 further articles, ordered supervision draws 18/668 exact successors spanning 14 characters, versus zero after rotation of training successors. The targets are absent from both input histories. Ordered writing still produces 348 blanks and no article pair with both independent successors correct. All four ordered free completions reach 128 cells and yield no coherent continuation or etymology. Preserve the successful images alongside the blanks and repeated outputs. This supports learning short-range succession from ink and leaves useful independent language and the matched-cost goal open. A [complete-bitmap probability recount](../references/evidence/patch_count_writing_20261118/blank_probability.json) finds that the learned distribution favors the blank over the recorded successor on 329 of its 348 blank outputs. A change to the learned distribution is needed to make those specific successors win.

The [shared-layout writer](../references/shared_layout_writing_result.md) drew 649/668 known nonempty glyphs and one blank on the same development contexts. It still drew only 18/668 exact successors. The full 言 answer fell into a repeated oval. Enumerating every layout shows that any planner using this fixed greedy writer could reproduce at most 242/668 recorded successors. A language model using only the last two coarse images also loses distinctions between questions with the same ending. Preserve this measured physical improvement; evaluate longer pictured context and distinguishable fine detail for the next model. The [question-state bound](../references/visual_question_information_bound.md) proves that on the 1,788 formatted passage questions any writer retaining only the last two visual signatures can match at most six complete reference answers. Stripping the common answer cue raises the ceiling to 184. Preserve ordered prompt memory and a fine-detail channel in subsequent comparisons. The evidence does not qualify autonomous language or a 1,000-fold saving.

The [grounded two-typeface trial](../references/grounded_detail_writing_result.md) couples the trained full-question reader with a local ink writer. On 3,170 official development questions it selected 565 exact source spans and wrote 124 complete correct Serif answers from Sans inputs. It scored nine exact on 3,158 exchanged questions, and seven both-correct pairs out of 3,100. Answer content and length still depend on selecting a visible span. A 10,833-parameter neural fallback raised withheld shape F1 from .660 to .751. Its advantage over copying (.0271) fell below the required .03 and yielded no additional exact answers. Keep this as a grounded writing reference and preserve the full free-writing, stopping, historical transfer and matched-quality cost criteria.

The [pictured title experiment](../references/visual_title_transducer_protocol.md) tests independent selection, composition and stopping. Of 834 eligible development titles, 46 require joining visible glyphs and 64 require drawing a glyph absent from the full prompt. The [completed title comparison](../references/visual_title_result.md) gives 629/834 correct, all extractive, versus 630 with a contiguous restriction and 580 with a learned opening rule. With genuine answer history, the ink head draws 85/143 absent targets exactly, all punctuation or spaces. Treat the title fit as terminal. Preserve the earlier model and test independent Han drawing and semantic composition as distinct requirements. Both native free-writing arms have now failed [complete blind review](../references/native_continuous_blind_result.md).

## Current model decision: full-context recipe rejected

The [complete full-context comparison](../references/trainable_visual_context_result_20261117.md)
is terminal and fails. Both arms complete all 13,785 records and 8,172 updates;
all 792 prescribed streams, 180 native boards and 144 normal-book visible
prefixes have actual review receipts. All readings were frozen before reference
scoring. Final learned and frozen context each give **0/32 strict complete QA,
0/16 both-correct pairs, 0/32 readable complete QA and 0/12 credited coherent
complete long-book continuations**. Every prescribed snapshot has zero strict,
paired and readable normal QA. Do not choose an intermediate checkpoint.

Treatment's twelve normal long books stop naturally after 209–415 emitted glyph
positions; the control stops on two blank books and caps on ten. This establishes
a stopping difference without usable complete writing. Overall review retains
741 uncertain streams and 51 blank streams. Unreadability prevents credit and
assessment of latent semantic correctness; it is not evidence of a particular
wrong semantic answer. One model reviewer had prior development familiarity;
there was no independent human adjudication. Never repair frozen readings from
references, source weights, OCR, fonts or answer images.

Both original CPU fit audits and both separate strict cold replays pass. Each
cold replay checks the same one fixed final book input, outside the 792 language
streams, with exact pixels, stopping and feedback and no undeclared artifact,
font/tokenizer/corpus or network access. Neither execution nor natural stopping
qualifies language. Preserve the earlier failed admission and cold executions.
The original immutable controller status predates the separate completed review;
current authority belongs in the private handoff.

The whole deployed model has 1,465,381,142 parameters, including 1,409,410,048
inherited contextual parameters. Treatment fit takes 60,127.268 monitored seconds
with 11.647 GiB peak allocation; control takes 4,309.029 seconds with 7.200 GiB.
These fit scopes exclude inherited pretraining, earlier teacher preparation,
all generation and failed executions. Both arms fail useful quality; no
matched-quality efficiency or language-from-images-alone claim follows.

Use the [public hash-bound evidence](../references/evidence/trainable_visual_context_result_20261117.json)
and [failure decision](../references/trainable_visual_context_failure_decision_20261117.md).
Do not restart completed training, generation, audits, cold checks or visual
review; do not relax thresholds or repeat unchanged earlier reader, codec,
writer, native-history or sampling diagnoses. All old failures and successful
cases remain evidence. The preceding goal revision is preserved at Git commit
`6390e80e7ee3e113419279702c2f872448d8c88d`.

## Next work: useful conditional image selection

The [gradient diagnostic](../references/visual_context_gradient_result_20261117.md)
is now complete on all three whole training records in both final actors. The
original host-gradient attempt stops at its pressure-wait ceiling with no complete
case. A separately bound device-copy recovery preserves all original arithmetic,
records and guards, completes in 183.602 seconds at 16.687 GiB peak allocation,
and verifies both full models and training-only projections unchanged. It makes
no optimizer updates or answer generations. Every combined retained-moment
AdamW direction has a negative inner product with the flow gradient; the
registered opposing-direction criterion is not met. Do not select new loss
weights from this result. First-order directions are not actual updated loss or
language measurements, and three records do not establish a corpus-wide cause.

The [saved-output analysis](../references/visual_context_saved_dependence_result_20261117.md)
checks all 792 streams and 488 comparisons without new neural calls or reading
repair. Every final normal QA stream is distinct. Literal prompt independence
is not established. Yet the learned-context actor has only fourteen distinct
first tiles across 32 normal questions, with one repeated ten times; the frozen
control has 32. Same-passage question pairs with matched first-span noise shapes
have median first-tile distance one pixel in treatment (16 pairs), versus sixteen
pixels in control (14 pairs). These are descriptive pixel effects, not semantic
understanding, and the matched subsets differ. Preserve the complete results.

The next algorithm work must address useful conditional image selection and
writing rather than assume opposing auxiliary gradients or repeat unchanged
training. Before selecting a conditional discrimination objective or output-law
change, read the existing failed contrastive, joint-density and writer results
and explain precisely how the proposed intervention differs. Choose one bounded
change with an unchanged control and training-only update/reset support on
complete authentic records. Require actual conditional writing improvement and
fixed complete-language evaluation before a full-corpus claim. No successor
full fit is currently admitted.

The [predictive-state comparison](../references/predictive_state_distillation_result_20261118.md)
is now terminal and fails all support checks. Both arms complete six real
updates on the same three complete training records. Across all 552 conditional
target glyph positions, F1 is .350346 initially, .336480 for cosine control and
.307754 for the KL replacement; false blanks are zero, two and five. Candidate
QA KL falls but book KL increases from 6.805294 to 14.433991. All trial model and
projection updates reset exactly; the frozen training-only readout is unchanged.
No successor is retained, autonomous language is unmeasured and no full fit is
admitted. Preserve all 1,656 actual output tiles and the failed preflight.

The [complete native-feedback calibration](../references/nonlinear_feedback_alignment_result_20261118.md)
fails its fixed reader gate after twenty epochs and 2,660 updates per arm on all
33,941 derived rows, weighted from every original training source. Error is
.578608694 affine and .542496895 nonlinear, from .639791851 initially. No
language fit is admitted. Do not repeat the fit or select an intermediate head.

The [separate weighted ridge refit](../references/feedback_ridge_refit_result_20261118.md)
keeps both final feature maps fixed and improves actual nonlinear error to
.271038527, versus .536556525 affine. Its ideal matrix error is .269287063;
actual stored-weight/BF16 execution remains separate. All six complete-record
trial solves improve and reset; original source head files/features are
unchanged. The 27.085-second, 1.135-GiB head-only run does not qualify a small
foundation or language efficiency. Final error and every complete-record limit
still fail. No full language fit, writer change or autonomous score follows.

The [saved-query identity analysis](../references/frozen_query_identifiability_result_20261118.md)
finds 2,682 conflicting duplicate groups on the same complete support, imposing
an exact-arithmetic .016774741 free-function floor. Most are same-image cross-
slot states: one-key cross-attention loses the query's slot identity when no
query residual is carried forward. This explains a real restriction, not the
whole .271 error. It does not rule out the unchanged .10 gate. Preserve all
conflicts, original image aliases and source labels; do not repeat their audits.

The [next bounded design](../references/profiled_visual_feedback_direction_20261118.md)
is feature learning with a solved output layer (variable projection), using
these same raw queries and source masses. Qualify its fixed-ridge envelope
gradient, source support, fitting/reset and numerical execution before a new
registered head comparison. Keep a common precision-matched fixed-feature
control; distinguish ideal arithmetic from actual weights. Do not silently add
slot residuals, change the pixel/query source, tune lambda or add old epochs.
The [implementation and separate preflight protocol](../references/profiled_feedback_preflight_protocol_20261118.md)
have four passing CPU numerical tests and a passing
[real complete-record preflight](../references/profiled_feedback_preflight_result_20261118.md).
All three trial steps improve and reset exactly. The book's re-solved finite
source derivative differs from the partial gradient by 3.20e-5 relative.
The 25.903-second, 1.355-GiB preflight qualified the separately registered
[twelve-step full-support comparison](../references/profiled_feedback_fit_result_20261118.md),
which is now terminal. Actual error falls from the common FP32 control's
.269500554 to .228074131; the .10 absolute error, .10 gain and all three
complete-record limits fail. All twelve first proposals reduce the fixed-ridge
objective, but numerical optimization is not useful visual language. Preserve
both final heads and every trial; do not extend this fit or choose an
intermediate. The 75.782-second, 1.246-GiB head-only result does not establish
whole-model efficiency or a trained successor.

The [read-only image consequence](../references/reader_writing_consequence_result_20261118.md)
is terminal. With identical noise, correct source-image histories and supplied
lengths, pooled F1 on all 552 targets per reader is .353468785 for control and
.353352368 for the learned head. Both have nine exact masks, one false blank
and four false-ink outputs. All 1,104 generated tiles are independently
recounted and exported losslessly. The unchanged actor and both heads pass
tensor-identity checks; no autonomous language is measured. Coordinate
improvement does not establish an actor repair. Do not continue this head fit.

The [writer-layer conditioning comparison](../references/writer_layer_conditioning_result_20261118.md)
is also terminal and fails. Both arms complete twelve whole-record optimizer
updates, with matched noise and exactly identical initial outputs. Pooled
physical F1 changes from .348600259 to .342911305 for unchanged conditioning
and .339850462 for modulation. Short-QA F1 improves to .876016260, but its
source masks remain nonexact and both book/long-QA case comparisons deteriorate.
Retain that limited positive effect alongside the failed pooled support gate.
Every trial resets exactly, all frozen tensors and stopping remain unchanged,
and all 2,208 generated tiles are independently recounted and exported losslessly.
The run takes 129.921 monitored seconds at 6.957 GiB peak allocation. It proves
execution of a physical learning comparison, not autonomous language.

Do not extend this conditioning trial, select an intermediate, change its
support thresholds or install the failed reader. No new algorithm is qualified
or selected for a full fit. The next substantive proposal must address complete
conditional language-to-writing learning, distinguish its mechanism from the
completed flow, mixture, joint-density, native-state, predictive-KL and modulation
trials, and retain the original language/confirmation criteria. A lower auxiliary
loss, an easy fitted short answer or another representational name is insufficient.
Use the existing positive complete-answer/counterfactual cases as retained
constraints without repeating their completed tests or repairing frozen readings.
Before any new execution, bind one failure-specific intervention, unchanged
control, complete training-only support, real gradients/reset and resources.
No deployment, language score or full-fit permission follows from the new code.

A new [lossless occupancy-density candidate](../references/occupancy_visual_density_direction_20261118.md)
is implemented, with eight passing CPU structure/probability/gradient tests.
It predicts a learned empty/nonempty root and generates physical quadrant masks
in a prefix-determined tree. The equal-parameter flat control has the same root
and a normalized nonempty branch. No glyph identity, font, reference topology
or lexical inventory enters generation. This is a different physical output
factorization, not a qualified language algorithm or a repetition of old
blank margins. Root errors and unreadable nonempty images remain possible.
The [registered CPU structure audit](../references/occupancy_density_structure_result_20261118.md)
now verifies all 68,948 saved-support cells exactly in 6.521 seconds. Both modes
preserve all packed bytes and 3,206 empty cells. On this saved support, tree
refinement averages 97.302 events versus 245.096 for the equal-parameter flat
control; both components have 2,261,074 parameters. No neural call or optimizer
update occurs in the audit. The event ratio .396997 is not a throughput,
compression-byte or language-efficiency measurement; the dense-image support
still permits 342 events. These cells are not corpus exposures or held-out language.

The [registered real-image preflight](../references/occupancy_density_preflight_result_20261118.md)
is now terminal. Both modes complete 96 whole-record updates and exact reset;
all 2,208 actual generated images/traces are independently recounted. Tree
F1 improves from .056570593 to .455979811, versus final flat .004023445, but
three of six fixed support conditions fail. Both have zero exact source masks
and ink at all four true spaces. All-event cached probability replay passes
within 9.54e-7 nat. Keep the finite contextual/root/mask gradients and limited
conditional NLL effect separate from readable or correct language. The run
uses .221 GiB peak allocation and 162.945 monitored seconds, excluding its
inherited 1,465,381,142-parameter actor and earlier cache preparation.

A [saved-array ink diagnosis](../references/evidence/occupancy_density_ink_mass_20261118.json)
finds opposite density errors: source mean 164.824 pixels, flat 1.772 and tree
418.797. Of 548 nonempty source positions, 547 flat outputs contain at most
four ink pixels; 366 tree outputs have more than twice their source ink.
These are descriptive post-hoc bins, not new rules or language scores. Do not
extend this failed fit, choose another snapshot or treat root nonemptiness
as successful writing. The lossless event savings remain a component result.

The [whole-form supervision candidate](../references/global_form_preflight_result_20261118.md)
is now terminal and fails seven of nine fixed conditions after all 96 updates.
Exact training-image equivalence yields 9,581 forms and covers all 552 targets.
Source-form helper accuracy rises from zero to 79/552, but physical F1 is
.455463432 versus unchanged tree .455979811, with zero exact masks and ink at
all four true spaces. Even short-QA helper accuracy 4/5 does not yield an exact
emitted image. Pooled rotated-minus-correct NLL .395939 fails its .5 condition.
Preserve both the classification improvement and the failed drawing.

All initial images match the saved control; no old control is re-executed.
All three complete-record gradient probes pass, every actual physical event's
probability replay is within 1.073e-6 nat, and all physical/helper updates reset
exactly. All 1,104 images, causal traces and helper decisions are independently
recounted. The helper is absent from generation. The 59.973-second, .233-GiB
head-only execution excludes the reused 1,465,381,142-parameter source actor
and earlier preparation, so it proves no whole-model efficiency. No successor,
autonomous score or full fit follows. Do not extend this failed trial.

The next algorithm decision must make a complete visual writing choice usable
by its actual generator, rather than repeat an auxiliary-label fit whose decision
is discarded. Keep global form selection, physical rendering and semantic
learning distinct. Read the completed continuous-codec, spatial-plan, prior-
guidance and exact-pixel generalization results before selecting another route.
A more compact physical representation is a hypothesis about modeling cost,
not proof of language or consumer-GPU foundation parity. Any new route needs a
bounded architectural change, real image support and unchanged controls before
its own prospective fit. The full language criteria below remain unchanged.

The next [implemented factorization](../references/binary_visual_plan_direction_20261118.md)
generates a 64-bit global visual plan, decodes it with a learned neural image
function, then generates a normalized physical XOR correction. The plan is
part of inference, unlike the discarded auxiliary classifier. For each plan,
XOR is bijective, retaining full binary-image support; training a fixed
posterior branch upper-bounds marginal image NLL rather than equalling it.
Quantization is explicit and is not a Unicode/glyph inventory. The 6,536,211-
parameter writer is untrained, excludes the contextual actor and establishes
no whole-model efficiency. Six CPU tests pass; the earlier signature-error
check is preserved. Existing model/data sources remain unchanged.

Next prospectively bind source preparation and a matched continuous/binary
64-dimensional codec comparison. Group exact image aliases in one physical
partition, retain all original bytes and leave reserved language data untouched.
Require faithful actual decoded images and lower actual refinement work before
registering a conditional prior/correction fit. Supplied-image compression,
mathematical normalization and numerical tests are not autonomous language.
Four additional CPU checks now qualify the matched continuous/binary training
operator, per-image objective and alias-preserving split. The
[source-only preparation](../references/binary_codec_preparation_protocol_20261118.md)
is complete: all 68,948 cells remain, with 64,041 training cells (8,629
forms), 4,907 physical-validation cells (952 forms), all aliases grouped and
3,206 empty training images. Two independent CPU metric checks also pass.
The [matched codec comparison](../references/binary_codec_comparison_result_20261118.md)
is terminal and fails six of nine conditions after all 6,000 updates per arm.
Validation F1 is .932736083 continuous and .810400756 binary; binary has zero
exact validation cells and loses .122335327 F1. The continuous branch also fails
its .98 ceiling. Both have zero source false blanks/ink. Binary source plan/
correction mean 30.015026 events passes the source event gate against 97.302489,
but poor per-form and held-mask fidelity rejects the recipe. Event counts are
not bits, throughput or language efficiency.

All complete-training-support preflight updates reset exactly, both actual
optimizer replays pass, and all 275,792 source reconstructions, correction
events and 12,000 retained updates are independently recounted. Both final
checkpoints and all source bytes remain. The 256.499-second, .593-GiB neural
execution loads no contextual actor and proves no whole-model advantage. Do
not extend this global 64-bit fit or select a snapshot. No conditional language
fit is admitted. A new bounded architectural decision must address the measured
common generalization loss as well as quantization; neither is a unique-cause
diagnosis or an impossibility proof. All original language criteria remain.

A [saved-plan information analysis](../references/evidence/binary_plan_information_20261118.json)
finds all 9,581 physical forms have distinct final 64-bit codes, with no code
conflicts or unstable aliases. All bits vary and empirical H(form|code)=0.
This rules out an exact-code collision floor on this finite support, not a
learning or universal-capacity limit. A [saved-frequency analysis](../references/evidence/binary_codec_frequency_20261118.json)
finds 110 common forms take 54.6% of training cells and reconstruct exactly,
while 5,821 once-occurring forms have binary macro F1 .823330. These descriptive
bins do not prove causality or change any rule.

The [equal-form codec comparison](../references/balanced_codec_comparison_result_20261118.md)
is now terminal after 6,000 updates per arm and fails seven of ten conditions.
It preserves initial weights, architecture, 64 bits and optimizer budget.
Binary validation F1 rises from .810400756 to .820592454, below the fixed .02
gain and every absolute fidelity requirement, with zero exact validation cells.
Continuous training macro F1 improves to .996662318 but validation macro F1
remains .903684546. Preserve the limited allocation effect and the remaining
generalization failure together; neither collisions nor word-frequency
imbalance alone explains the result. No unchanged extension is admitted.

All weighted complete-source preflight updates reset exactly; optimizer replay
passes and all 275,792 reconstructions/events and 12,000 updates are independently
recounted. Natural-frequency controls remain unchanged and are not rerun. The
257.158-second, .593-GiB physical run establishes no contextual language or
whole-model efficiency. The mask partition is reused development. A new
architectural proposal may test spatial representation/decoder geometry with
all original absolute criteria, rather than add discarded auxiliary labels,
unmeasured code width or more unchanged exposure. No conditional fit is admitted.

The [spatial 64-bit comparison](../references/spatial_codec_comparison_result_20261118.md)
is terminal after 6,000 updates per arm. Binary validation F1 improves from
.820592454 to .847344074, passing the fixed .02 gain, but six of ten conditions
fail with zero exact binary validation images. Continuous validation F1 is
.929688379. Spatial structure is helpful in this bounded comparison and remains
insufficient for faithful drawing. Both 1,767,365-parameter codecs preserve
all 1,766,209 common CNN initial values and the original balanced source order;
old global controls are not rerun. Every original threshold remains.

All real preflight updates reset, optimizer replay passes and all 275,792
reconstructions/events and 12,000 updates are independently recounted. The
261.088-second, .524-GiB run loads no contextual actor and proves no language
or whole-model efficiency. Keep the limited architectural gain and the failed
absolute quality together. Do not extend this recipe or select a snapshot.
A later spatial capacity intervention must keep geometry, allocation, loss
and budget fixed, declare increased plan events and preserve every original
quality/resource condition. No conditional language fit is admitted.

The [spatial capacity comparison](../references/spatial_capacity_comparison_result_20261118.md)
is terminal after 6,000 updates per arm. At 256 values/bits, continuous physical
validation F1 reaches .996423850 and macro F1 .993422244, with 316/952 exact
forms; binary F1 reaches .943690071 and macro F1 .928693592, with only three
exact forms. The common CNN initialization, data/order, loss and resources stay
fixed. Each codec has 1,770,449 parameters. Binary training F1 .981661645 passes,
but four of ten conditions fail: held pooled/macro F1, macro exactness and
continuous retention. Thirty-two plan events plus actual XOR refinement average
49.475474 on the complete source, passing the original total event condition.

All complete-source preflight updates reset, optimizer replay passes and all
275,792 saved reconstructions/events and 12,000 updates are independently
recounted. Saved spatial-64 controls are unchanged and not rerun. The
256.635-second, .524-GiB physical run loads no contextual actor. The continuous
motor can reconstruct new forms accurately on this reused development split;
hard quantization still loses detail. Neither observation proves language or
whole-model efficiency. No conditional language fit is admitted. Preserve the
failed absolute rules and test a bounded representation change before any
successor fit; do not extend unchanged epochs or select a snapshot.

The [equal-bit scalar-plan comparison](../references/scalar_codec_interrupted_result_20261118.md)
is interrupted and unqualified. Its 128-value continuous control completes all
6,000 updates but reaches only .978102816 validation F1, below the unchanged
.98 ceiling. The hard arm stops before any update because all initial features
round to zero and the preflight requires a nonzero expansion-weight gradient,
which is mathematically zero for a zero input. This does not establish zero
encoder/total decoder gradient or inability to learn. Hard-code quality remains
unmeasured; no unchanged recovery is selected. All 137,896 saved control images
and 6,000 retained updates are independently audited. The original failure
checkpoint's stale 6,000 counter belongs to the previous continuous arm; actual
hard weights equal initialization and no hard update ledger exists. Preserve
the file, corrected interpretation and earlier CPU-test expectation failure.
No language fit is admitted, and every original criterion remains unchanged.

The selected next [physical intervention](../references/residual_plan_motor_direction_20261118.md)
trains the previously untrained normalized XOR correction while freezing the
actual spatial-256 codec. Its condition contains generated plan bits and the
frozen base pixels, all available at deployment. Greedy identity initialization
matches the unchanged base for every possible plan. The implemented motor has
2,385,763 parameters including its frozen 1,770,449-parameter codec; 615,314 are
trainable correction parameters. This excludes all contextual language weights
and was initially untrained. Three CPU checks pass for initial writing, likelihood,
actual gradients/update/reset, fixed-codec identity and target-encoder exclusion.

Its [complete source preparation](../references/evidence/residual_motor_preparation_result_20261118.json)
retains all 68,948 saved plans/bases and original target bytes with the unchanged
physical partition. It takes 1.575 seconds with no neural calls or training.
True residual targets are not generated corrections. The separate
[learning comparison](../references/residual_motor_comparison_protocol_20261118.md)
is now prospectively registered: complete-source real gradient/update/reset,
6,000 updates, unchanged sampling/budget, all-source initial/final actual
generation and every emitted-prefix probability replay. The new admission unit
is the composite motor D(B) XOR R, with every original absolute fidelity,
exactness, empty-image and total-event condition retained; the failed base stays
failed. Actual generated correction work and remaining error complexity are
reported separately. No conditional language fit, whole-model efficiency or
autonomous score follows from preparation or registration.

The [first correction-fit execution](../references/evidence/residual_motor_admission_failure_20261118.json)
ends at the 48-GiB host entry guard, constructing no model and making zero
updates. No drawing or language quality is measured. Preserve its original
directory. A separate [admission recovery](../references/residual_motor_admission_recovery_protocol_20261118.md)
retains every model, source, schedule, quality and resource rule, with a new
output directory and one worker maximum. It first waits for two stable
50-GiB host observations ten seconds apart; the original worker's 48-GiB guard
remains. Scheduling wait is at most 300 seconds and included in a separately
enforced 2,400-second total ceiling. Do not duplicate its owner, reduce reserves,
kill foreign services or retry a worker exception automatically.

That [scheduler is terminal](../references/evidence/residual_motor_scheduling_failure_20261118.json)
after 291.299 seconds with no worker or model. Both reserve-only attempts remain
preserved. The current separately registered [physical-head resource amendment](../references/residual_motor_head_resource_protocol_20261118.md)
uses 40-GiB entry for this 2,385,763-parameter motor only: 24 GiB unchanged live
host reserve + 8 GiB unchanged maximum process RSS + 8 GiB additional entry
margin. It retains all GPU, disk, pressure, mutex, time and thread limits. The
original full-context admission source remains unchanged; no contextual model
is loaded. This is an explicit resource change before any model execution,
not a change to quality, mathematical learning rules or failed model scores.
Every source, initialization, update, generation/probability test and physical/
language criterion remains fixed. Launch at most once, qualify real complete-
source memory/gradient/update/reset before retained fitting, and preserve any
resource violation. No automatic retry or language admission follows.

The [first head-resource execution](../references/evidence/residual_motor_loading_failure_20261118.json)
passes admission but stops at serialized-plan loading, before constructing a
model or making any update. The uint8 plan bytes need widening before the
shared validator's scalar-256 comparison. The original component is unchanged.
The [registered CPU loader qualification](../references/evidence/residual_motor_loader_result_20261118.json)
recovers all 2,206,336 bytes / 17,650,688 bits exactly across all 68,948 plans
in .490 seconds, with no neural call or source change. A separate unit check
covers every byte value. Preserve this loading failure and the earlier resource
attempts, all of which leave physical quality unmeasured.

The [loaded comparison](../references/residual_motor_loaded_comparison_protocol_20261118.md)
is now separately registered. Its training body changes only the explicit
integer widening and the already selected physical-resource context. Model,
source pixels, frozen codec, initialization, objective, sampling, 6,000 updates,
actual generation/probability tests and every quality criterion stay fixed.
Use its one serial owner and new output path. Do not alter bound sources or
restart any earlier run. Only actual complete-source update/reset and final
generated-image qualification can establish a trained motor; neither code nor
loading proves language.

The loaded trial is now [terminal at optimizer replay](../references/evidence/residual_motor_replay_failure_20261118.json).
It passes the real complete-source update/reset path and generates all 68,948
initial images equal to the unchanged base, then retains two updates and fails
the exact third-update replay. No final trained-image result exists. The saved
failure state includes an unaccepted replay attempt; do not install it or call
it a qualified two-update model. The combined check did not retain which
loss/gradient/model/optimizer component differs, so the cause is unproved.

The [registered numerical diagnostic](../references/residual_motor_replay_diagnostic_protocol_20261118.md)
compares the original global deterministic-algorithm flag with strict true,
keeping actual initial tensors, source batches, loss and optimizer fixed.
It reconstructs two updates and repeats the actual third four times per mode,
recording every gradient/model/optimizer and RNG digest. All twelve trial
updates are discarded and reset. cuDNN flags alone do not cover all CUDA
operations, but neither attention nor embedding reductions are yet an isolated
cause. Require an observed unequal original replay and exact candidate replay
before a separately registered fresh full physical fit. No language, final
motor quality or efficiency follows from numerical reproducibility.

The [numerical diagnostic is complete](../references/residual_motor_replay_diagnostic_result_20261118.md).
Original-mode repeats have identical loss, norm and model weights but depth/
row/column embedding gradients differ by up to 5.820766091346741e-11, changing
six optimizer-moment tensors. Strict global deterministic mode makes all four
repeats exact, including every gradient and optimizer tensor. RNGs do not change;
all twelve trial updates reset. The 1.674-second, .310-GiB execution qualifies
this setting, not physical writing or language. Tiny numerical differences do
not establish a cause of earlier unreadable language.

A separately registered [fresh deterministic fit](../references/residual_motor_deterministic_comparison_protocol_20261118.md)
uses the committed loaded worker with this one global execution flag and new
output path. All model, data, quality, source partition, initialization,
6,000-update schedule, actual-generation/probability and resource rules stay
fixed. Do not resume the failed weights or select a checkpoint. Observe its
single serial owner and independently audit all actual final images before
admitting any conditional-language work. This fit is now terminal and audited.

The [complete correction-motor result](../references/residual_plan_motor_comparison_result_20261118.md)
fails seven of nine fixed conditions after all 6,000 updates and 768,000 training
draws. Validation F1 declines from .943690071 to .933584204, macro F1 from
.928693592 to .916099307, and exact cells from 94/4,907 to zero. Every one of
the 3,206 true spaces gains ink. Actual plan-plus-generated work averages
45.998129 events; this passes its event condition while physical quality fails.
All 137,896 initial/final images, emitted actions, complete causal traces and
6,000 retained updates are independently recounted. The strict GPU probability
and optimizer replay checks pass. Numerical correctness does not repair writing.
The monitored fit takes 434.507 seconds, including 140.133 seconds of pressure
wait, at .485 GiB peak allocation. These physical-only costs exclude preparation,
earlier codecs and any contextual foundation. Preserve the fixed base, failed
motor, every prior attempt and original criteria. No unchanged extension,
intermediate selection or conditional-language admission follows. Autonomous
language remains unmeasured for this failed physical actor.

The next [native continuous-density candidate](../references/continuous_autoregressive_language_direction_20261118.md)
returns to causal language selection using the already qualified V34 motor,
without installing the failed binary or correction actors. A newly initialized
causal image reader predicts 48 groups of continuous writing coordinates, each
with a normalized shared-component Gaussian mixture. Compare one versus eight
components, with identical common reader/local initial weights. Actual generated
coordinates feed the neural decoder; the next history contains the emitted
pixels re-encoded, never the unrendered coordinate proposal. A learned stop head
receives no target length. No inherited language body, teacher state, lexical
head or glyph bank is used. Code density is not exact pixel likelihood.

Three CPU tests cover distribution scoring, both causal levels, cached attention,
actual updates/reset, frozen codec, reproducible sampling and emitted-image
feedback. The [registered GPU preflight now passes](../references/continuous_autoregressive_preflight_result_20261118.md)
on all three original complete training cases in each arm. All six actual
trial updates reset, the fixed codec/common initialization checks pass, and an
independent audit recounts every one of the 2,208 generated images. Execution
takes 11.962 seconds at 1.336 GiB peak allocation. The candidate has 23,549,194
whole parameters, including the previously trained 7,423,361-parameter codec;
no language weights are inherited. Code NLL falls slightly in all six cases,
but updated supplied-history F1 is only about .22–.23 and most loss reduction
comes from stopping. This is execution support, not trained physical quality
or language. Do not repeat this completed preflight. Its pass permits only a
separately bound complete-corpus comparison.
Keep the original complete-answer, book, retention, erased-input, stopping,
untouched-confirmation and whole-model cost criteria. The requested 1,000-fold
efficiency remains an unproved target, not a property inferred from this design.

The [complete native comparison](../references/continuous_autoregressive_full_comparison_protocol_20261118.md)
is now implemented and separately registered. Its two new CPU execution checks
pass on the authentic source: all 13,785 complete training records remain,
with 2,571,477 next-image targets per arm. This excludes 4,762 first book images
used as causal input anchors from the earlier 2,576,239 source-position count.
The longest record has 1,101 images and no training truncation is needed.
One shared shuffled order gives 13,785 updates per arm; retain all five fixed
snapshots and the 792-stream fixed-cell autonomous plan. An independent CPU
completion audit must precede generation. The full proportional, earlier-
success, cold-independent and untouched-confirmation requirements remain later
requirements even if this development comparison passes. Use the one serial
owner in the private handoff. Do not edit bound sources, duplicate its owner,
restart on observation timeout, add unchanged epochs or infer language from loss.
Quality, complete language and efficiency remain unmeasured before evaluation.

Both native fits and the independent CPU schedule audit are complete.
The [fit receipt](../references/evidence/continuous_image_training_completed_20261118.json)
records 27,570 updates, 5,142,954 next-image exposures, 5,170,524 stop
decisions and ten snapshots. Monitored training totals 1,306.37 seconds
on one RTX 4090 D with 1.416 GiB peak allocated memory. The [completed blind evaluation](../references/native_continuous_blind_result.md) covers all 792 original outputs. Both final arms give zero of 32 readable or correct answers and zero of 12 coherent long continuations. The eight-component model reached the cap on every long normal book. Preserve the frozen failure. A training-only frozen encoder cache gives
1.53–2.86× faster warm forward/backward on three complete CPU cases with
identical measured codes, losses and gradients. This establishes neither GPU
training acceleration nor a matched-quality advantage over a text model.
The registered fits used the ordinary encoder.

The [visual word-order evaluation](../references/visual_word_order_result.md)
compares the same 552 Mandarin sentence pairs at all five native K=8 checkpoints.
Final grammatical preference is 284/552 (51.4%) versus 299/552 (54.2%) before
training and 57.2% for a matched visual bigram counting adjacent bitmaps.
Paradigm-level intervals include chance and zero improvement. Preserve all
paired margins, checkpoints and this negative result. Avoid treating a decrease
in writing-code loss as evidence that the reader learned syntax. The active
792-stream autonomous qualification is now complete and failed.
The next learning decision should account for both visual ordering and the
eventual complete-answer evidence before another heavy fit.

The [matched training-source comparison](../references/visual_passage_register_result.md)
recovers 2,127 unique modern passages with 1,047,284 direct next-image targets.
Two visual bigrams trained on equal numbers of transitions score 59.5% for
modern passage ink and 57.6% for historical book ink on 6,062 additional
pairs. The paired construction interval for the 1.9-point difference spans
−9.0 to +12.7 points. The data adapter is verified; changing register alone
has not passed the criterion for a follow-on neural fit.

The [actual 言 question](../references/native_yan_inference_result.md) produced
identical 128-cell greedy outputs from typed and pictured prompts. Greedy and
sampled output were both unreadable. At every position in both answers the
writing code changed more after drawing and re-reading than the largest change
among 9,581 distinct source images. In the sampled answer, mean and scale
explain only 0.39% of squared discrepancy. Preserve the original pixels and
directions. A scalar normalization remedy misses its prespecified criterion.
This one question is separate from the complete autonomous evaluation.

A separate [CPU passage reader](../references/visual_extractive_reading_result.md)
recovers 269/1,788 complete held-out answer rasters from the supplied passages.
Question erasure gives 18/1,788 and same-passage exchange gives 11/1,783.
This passes the registered question-dependence test for extractive reading.
Rotating answer assignments among training questions about the same passage
reduces exact answers to 59/1,788. A whole-passage interval puts the advantage
of correct supervision at 10.1--13.5 percentage points.
Preserve it as a compact reference for selecting evidence from a supplied
page. Its scope is a contiguous passage span; standalone writing about an
absent fact, historical glyph synthesis and matched-quality cost remain open.

Historical diagnostic registrations, all failed executions and frozen readings
remain protected. No development answer, reserved DRCD record, rejected reward
judge or policy fit may enter the new training objective. Source-conditioned
contrast or image likelihood is training supervision, not evidence of language
until autonomous complete answers and books meet the original criteria.

The earlier white-target comparison, both reward judges, isolated-cell teacher,
fixed encoder metric, label uniqueness and preprocessing audits remain terminal.
Do not recover old evaluators, tune rejected thresholds or reuse development
outputs as fitting data. Physical aliases and invisible source symbols remain
preserved; an exact pointwise Unicode condition's failure is not proof that
visual language is impossible. The reserved DRCD collection remains unrendered,
unscored and evaluation-only.

Every successor fit requires its own failure-specific, prospectively bound
training-only support/gradient/resource preflight on a fresh clone with all trial
updates reset. Bind complete sources, one intervention, unchanged control,
initialization, schedule, checkpoints, full evaluation, resource ceilings and
actions for either result. Direct supervised proposals need no reward judge;
policy proposals also need qualified complete-action rewards and exact emitted-
pixel probability replay. More unchanged epochs are not a selected intervention.

An ordered visual volume remains a representation hypothesis. Its completed
invertible CPU operator preserves order but does not establish compression,
faster language learning or coherent generation. Any future block must predict
future visible spans without receiving its own target block as context. Do not
replace language learning with supplied-answer reconstruction.

## Acceptance criteria

The unit of success is a **complete correct answer or coherent continuation**.
Copying, glyph fidelity, canonical bitmaps, loss, stopping, cold execution and
resource savings remain distinct claims.

1. Demonstrate complete modern-Chinese answers and coherent book continuation
   on held-out works and passage groups, using actual emitted pixels as history.
   Ask different questions about the same passage and retain question-, passage-
   and prefix-erased controls. Preserve raw outputs, natural stops, caps and
   uncertain readings; freeze review rules before generation and readings before
   new reference scoring. Canonical mismatch alone does not establish falsity.
2. Preserve the original complete-response pilot: at least 12/32 strict
   proportional answers, 4/16 both-correct pairs, 24/32 readable answers and 4/12
   coherent complete long books; all eight proportional and two fixed earlier
   successes; each QA-erasure arm at most two exact answers and a dependence gap
   of eight. Keep caps 128/640, emitted-image feedback and natural stopping.
   A physical failure leaves autonomous language unmeasured, not zero.
3. Confirm development improvements on untouched source groups. The original
   DRCD confirmation pool contained 998 paired passages and 3,485 questions.
   Later evaluations used at least 529 of its paragraphs. Its original untouched
   status is lost. Preserve the original evidence and keep that pool excluded from
   further training and development scoring. Separate screened source groups and
   a precommitted confirmation are still required after a qualified development
   comparison. Development results cannot discharge this criterion.
   [Exposure audit and correction](../references/drcd_source_exposure_correction.md)
4. Reproduce training and standalone inference within one 24-GiB GPU. Account for
   the entire model, inherited weights, preparation, teacher cost, wall time,
   peak VRAM and latency. Compare useful matched quality against a named local
   Qwen baseline before claiming efficiency or large-model parity. A prototype
   attached to a 7.522B inherited body is not a compact foundation.
5. Extend demonstrated language to held-out writing styles and attested
   historical or unencoded forms. Evaluate explanation, historical stage,
   glyph fidelity and provenance separately; label reconstructed forms.
   Demonstrate Chinese/English word-origin questions as part of the full goal.
6. Maintain Nature-style TeX/PDF and editable native PowerPoint figures from
   actual measurements, preserving original images and negative results. These
   artifacts support the evidence; publication does not qualify the model.

A development pilot alone does not complete this objective. Novelty, human-like
reading and broad language capability require their own evidence. The requested 1,000-fold training-efficiency gain is a stretch target.
Define the ratio as the baseline compute required to reach a fixed held-out
quality divided by the ILM compute required to reach that same quality. Use
the same task, source groups, output grading and stopping criteria. Report
wall time on identical hardware, data exposure and whole-system energy where
metered. Charge corpus preparation, visual codec fitting and teacher work
explicitly. Report inference latency per complete correct answer separately.
An existing pretrained Qwen can establish a quality and serving comparison;
estimating its training cost requires documented comparable information.
A smaller checkpoint or faster span extraction does not measure the target
ratio for a general language model.

## Completed constraints and data

The last [autonomous continuation result](../references/visual_continuation_learning_result.md)
is 7/32 strict proportional answers, 0/16 both-correct pairs, 14/32 readable
answers and no credited coherent complete long-book continuation. All 732 streams,
183 boards and 96 normal-book prefixes are reviewed. Both the original scorer and
independent four-QA-erasure audit reject the trial. Preserve all five checkpoints;
the better intermediate result does not replace its prescribed final result.

Reuse the completed [clean-history](../references/continuation_clean_history_result.md),
[trajectory](../references/visual_continuation_trajectory_result.md),
[joint-writer](../references/joint_writer_physical_result.md),
[context-gradient](../references/joint_context_physical_result.md),
[nonempty-ranking](../references/global_glyph_ranking_physical_result.md),
[whitespace](../references/ranked_writer_whitespace_result.md) and
[saved-score](../references/ranked_glyph_score_gap_result.md) findings. Books fail
with correct history, so a feedback-only explanation is insufficient. Earlier
native-reader, teacher-state, question-assignment, recognition-loss, copying,
writer, codec and released MIXAR controls are terminal; do not repeat unchanged
fits, evaluator recovery or diagnoses. Preserve contradictory readings and
original scores. The preceding full goal text is retained at Git revision
`2dda87d203ce8e137a0c5450fa9efd9c28c1f302`; historical directions are indexed in
[the decision archive](ilm-goal-decision-history.md).

Use the authenticated corpus and whole-work/passage-group splits. The original
13,785 training sequences contain 2,576,239 target glyph positions; the complete
matched continuation comparisons retain 13,766 eligible sequences. Exposures are
not unique facts or foundation-scale pretraining. Add data only with a declared
coverage or supervision reason, provenance and group splits. Preparation is not
student-language evidence.

The current manuscript includes paired source-only translation, composition and data scaling. Figure 57 on page 77 shows complete original outputs and a failed translation. Figure 54 uses the corrected exchange measurements. The reading, review and cumulative redline editions contain 100, 161, 164 pages, respectively. [Manuscript](../publication/ilm-nature/manuscript.pdf) · [Verified figures and editions](../publication/ilm-nature/evidence/block_translation_article_build.json)

## Work-unit and resource rules

Read this goal and `artifacts/glyph_tail_current_handoff.json` at each model work
unit. Verify actual results and live process identity before acting. Record the
new evidence and next decision at the end. Keep private runtime details out of
commits and historical result files out of live scheduling authority.

Run at most one heavy project job under `artifacts/ilm_gpu0.lock`. Before a heavy
launch inspect host RAM, swap, pressure, owned processes, tmux and ports. The
existing full-reader entry requires 48 GiB available host RAM and 20 GiB free
GPU; retain 24 GiB host RAM, 48 GiB disk and at most 22 GiB allocated VRAM, or
stricter registered ceilings. Use two CPU threads. Protect foreign projects,
reuse shared weights and stop obsolete owned runtimes after evidence capture.
An observation timeout or serial phase transition does not authorize a duplicate
or restart.

Prioritize the failure-driven model comparison and complete-language evidence
over additional source preparation, copying repairs, covers or manuscript
polish. Commit and push related source/document changes while preserving ignored
private artifacts and unrelated work. Keep the session goal active until every
language, independence, resource and written-form requirement is proved.

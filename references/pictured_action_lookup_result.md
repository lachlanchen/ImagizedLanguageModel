# Answering different questions about one pictured page

Given the two lines 焉得仁 and 策其馬, the model writes *benevolence* when asked what they get and *horse* when asked what they whip. The input consists of the same two-line image and an independently rendered English question. Each answer is a stream of generated bitmaps. A text transcription is optional. [Source image and actual replies](evidence/pictured_action_lookup_v1/samples/) · [Run or train the model](../models/pictured_action_lookup/README.md)

The evaluation juxtaposed two real clauses from the same book on each page. The clauses have different English action and object annotations. Every page was tested with two actions, two held English question frames and both line orders. Complete writing requires the correct row, the registered object position and a stopped answer image matching an accepted English gloss throughout.

| Reader | Complete row selection, 1,691 pages | Complete object writing, 1,691 pages |
|---|---:|---:|
| Exchanged action supervision | 0 | 0 |
| Training counts for the predicted Chinese glyph, English action identity supplied | 1,121 | 650 |
| Exact visual match of a drawn verb within the question | 946 | 546 |
| Learned visual similarity | 1,234 | 667 |
| Unique visual match, otherwise learned similarity | 1,306 | 703 |

With the correct row supplied, the frozen source reader and writer could complete 818 pages. Supplying the correct object location as well raised this to 1,260. The combined model wrote both answers on 165/487 pages from held works and 538/1,204 from held sections. In a paired comparison with the English-name count control, 57 pages passed only for the combined model and four passed only for the count control. These are glossary-based outcomes on short printed passages.

**Changing the question wording exposes a large loss.** On 188 of the same pages, both actions had eligible editorial alternatives, such as *obtain* for *get*. Replacing the names in the frozen model's questions reduced complete line selection from 151 to 11 and complete written answers from 95 to seven. Both actual line orders were executed on CPU for 3,008 image responses. This additional experiment was defined after the literal-name scores and before any alternative-name scores. [Wording test and image-level recount](evidence/pictured_action_lookup_v1/audit.json)

**Two successes on the nine same-predicate contrasts need particular care.** Both involve 復, both come from reserved sections, and the first includes 致粟帛多者復其身. Its registered verb is *repeat*. Wang Shumin interprets the expression as exemption from labour service. We retain the original score as a measure of reproducing the annotation. These cases provide weak evidence about philological interpretation. [Academia Sinica, commentary on the *Shiji*, chapter 68](https://www11.ihp.sinica.edu.tw/storage/w2_file/3806EaymdsB.pdf)

## Training and mathematical scope

A second fit from the published sentence identifiers produced byte-identical aligned and exchanged weight files, the same loss traces, and identical held scores and saved wording responses. A separate isolated invocation reconstructed all training rows and drew the expected answer with only released assets and the pinned treebank. [Independent refit receipt](evidence/pictured_action_lookup_v1/clean_refit.json)

A 150,016-parameter network learns to compare an English question image with a contextual state at a Chinese predicate. Two convolutions over question cells give each location a view of its neighbouring letters. The model uses the largest cosine similarity between the projected source state and the visible question locations. A bidirectional contrastive loss aggregates all batch positives for the same registered action. Sampling includes instances of the same Chinese glyph with different glosses. Training used 16,632 clauses, 507 actions and ten question frames. No answer text, action label, OCR or external model is supplied to the released predictor. Input cells and output writing are calibrated to the training typeface.

The aligned fit took 7.39 seconds over 1,740 updates on one RTX 4090 D. The complete system has 1,551,989 parameters and includes the earlier independent image writer and clause locator. Corpus preparation, the earlier fits, teacher data preparation and evaluation are outside this incremental timing. A full-system, quality-matched comparison with a language model remains open. [Fit record](evidence/pictured_action_lookup_v1/aligned_fit.json)

Lean verifies that an unchanged ranking of two rows cannot satisfy two distinct requested answers. With unequal row scores, reversing the physical page preserves the winning source. It also bounds the number of wrong or tied comparisons by their measured total hinge penalty divided by a positive margin, for integer-valued comparison scores. A singleton positive carrying over half the weight outranks each distractor. With multiple positives, pooling their weights can obscure an individual error. None of these results supplies a held-book generalization guarantee. [Lean derivation](../formal/span_list/PicturedActionSelection.lean) · [Training protocol](pictured_action_lookup_protocol.md)

The next test should use independently sourced English descriptions of actions and human-checked Chinese readings that convey the same event without sharing the training action name. Compare aligned and exchanged semantic supervision on the same optical reader. Keep held works, exact object writing and original literal-name accuracy visible throughout. [All recorded comparisons](evidence/pictured_action_lookup_v1/result.json)

The clause images and reference glosses derive from the Kyoto Classical Chinese treebank, release 2.16, by Koichi Yasuoka and collaborators, under CC BY-SA 4.0. Sentences were filtered, paired and rendered, and selected person glosses were normalized. The treebank-derived data in the evidence directory retain that licence. [Corpus and attribution](https://github.com/UniversalDependencies/UD_Classical_Chinese-Kyoto/tree/r2.16) · [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/)

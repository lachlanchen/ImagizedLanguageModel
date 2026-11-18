# Which fact does the pictured question ask for?

The following two requests have identical inventories of visible glyphs. They were presented as images alongside the same card for 耳.

- 关于耳：本义我想问，读音我已知；金文我想看，六書通我已见。 The writer gave the gloss *ear* and the bronze form B16241.
- 关于耳：本义我已知，读音我想问；金文我已见，六書通我想看。 The writer gave the pronunciation *ěr* and the Liushutong form L13682.

Each answer matched the entire expected raster and ended autonomously. The source supplies the knowledge and historical pixels. The writer chooses which spans to use, generates the connecting glyphs and selects the stopping point. [Actual questions and answers](../publication/ilm-nature/figures/Figure_71_question_order.png) · [Run these images on CPU](../models/entry_question_reader/README.md)

| Development condition | Requests | Ordered reader exact | Sorted reader exact | Ordered groups with all four correct | Sorted groups with all four correct |
|---|---:|---:|---:|---:|---:|
| Training expressions on unfamiliar families | 4,544 | 4,542 | 1,136 | 1,135 / 1,136 | 0 / 1,136 |
| Reserved expressions and reversed clause order | 4,544 | 4,540 | 1,136 | 1,134 / 1,136 | 0 / 1,136 |
| Familiar expressions moved before each field | 4,544 | 356 | unmeasured | 0 / 1,136 | unmeasured |

Both writers reproduced all 1,136 simple requests from the preceding study. The primary comparison passed its registered complete-answer, group, control and retention criteria. The reserved wording passed its separately defined transfer criteria. That wording retains the placement of the status after its field. In the additional cue-placement experiment, the same expressions, characters, target answers and source cards were preserved. On the illustrated example the model returned *ěr* and the Liushutong form where *ear* and the bronze form were requested. The sharp reversal on this test identifies a dependence on local placement. [Registered comparison](pictured_entry_request_order_protocol.md) · [Saved results](evidence/entry_question_reader_20261119/evaluation.json) · [Additional test and CPU replay](evidence/entry_question_reader_20261119/recount.json)

The cue-placement test distinguishes choosing the wrong stated evidence from writing something outside the four intended answers. Of the 4,544 outputs, 1,234 matched the other fact with the requested form, 354 matched the other form with the requested fact, and 603 matched the other fact and form together. The remaining 1,997 matched none of the four complete targets. These counts compare complete images and stops, and preserve the 356 original exact answers. [Error counts](evidence/entry_question_reader_20261119/cue_errors.json)

## What the pixels contribute

Within a group, four distinct answer images are equiprobable. The histogram of visible question glyphs is exactly constant, so an order-invariant reader can achieve at most one complete correct answer per group. A deterministic generator receiving identical state, document pixels and actions recurses identically, however long it runs. The earlier reader also retained some position features, but these collapsed 1,135 of 1,136 groups. More observations of those collapsed inputs could estimate answer frequencies more precisely without revealing which of the four was requested. The ordered reader restored this missing information by carrying a sequence of glyph images through two short convolutions. It shares the recurrent copy-and-write mechanism and its initial weights with the sorted control. [Corpus and information bound](pictured_entry_request_order_protocol.md)

Both 51,776-parameter models trained their full weights on 35,680 compound questions and 8,920 replayed simple questions for eight passes. Each fit made 2,792 updates on a single RTX 4090 D. Timed optimization and checkpoint writing took 13.87 seconds ordered and 15.00 seconds sorted. Shared training tensor preparation took 3.35 seconds. The measurements exclude rendering the corpus, learning the initial writer, compiling the manuscript and qualification. The ordered run peaked at 128.43 MiB allocated by PyTorch. An independent CPU replay reproduced all 10,224 ordered streams and their termination decisions across the three registered conditions. All 20,448 archived streams and stops were independently rescored. The order test addresses a local grammatical distinction in one visual dictionary layout. The authors supplied clause patterns, a single typeface and visible record delimiters. Free book translation and a matched language-quality efficiency advantage remain open. [Ordered fit](evidence/entry_question_reader_20261119/ordered_fit.json) · [Sorted fit](evidence/entry_question_reader_20261119/canonical_fit.json) · [Book translation](context_phrase_trial_result.md)

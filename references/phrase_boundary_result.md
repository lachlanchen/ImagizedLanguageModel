# A phrase boundary can repair a local translation

The source 两营相去九里。 says that two camps were nine li apart. A writer that always replaces one glyph produced 两营相距离九里。 A writer trained to select adjacent source cells and a modern pictured expression produced 两营相距九里。 The same repair held for the second authored distance clause and the three earlier development errors. Five complete source and answer images in Figure 65 were reproduced byte for byte through the released CPU image interface. The incorrect readings are reproducible too. The felling clause 戍卒入谷伐榆。 still gives 戍卒进入谷攻打榆。, in which soldiers attack an elm. [Figure 65](../publication/ilm-nature/figures/Figure_65_phrase_boundary.png) · [Reproduction receipts](evidence/phrase_boundary_20261119/reproduction.json)

| Complete answers accepted | Matched single-cell writer | Joint source-span writer | Explicit cue rule |
|---|---:|---:|---:|
| Both clauses in a natural pair, out of 6 | 1 | 1 | 1 |
| Individual clauses from those pairs, out of 12 | 3 | 3 | 3 |
| Individual authored clauses, out of 8 | 1 | 3 | 0 |
| Both clauses in an authored pair, out of 4 | 0 | 0 | 0 |
| Additional archive clauses, out of 17 | 7 | 7 | 7 |
| Diverse passages, out of 12 | 0 | 0 | — |

One study-involved reader examined shuffled distinct complete outputs. The reader had prior familiarity with the task and the twelve diverse prompts. Natural clauses came from 29 different chapters. None was an earlier training chapter. An audit found one complete source clause in an older training passage from another chapter. Eleven chapters had appeared in an early held-out set and eight in its confirmation set. One source and one target clause matched strings in the early held-out set. These overlaps limit any claim of complete historical novelty. Copying conveyed a recognizable sense in 36 of the 37 targeted prompts yet gave zero accepted complete modern translations. Sense counts can credit context already legible in the input. The phrase rule changed ten targeted answers relative to single-cell writing. Two changes replaced 相距离 with 相距 and improved the complete reading. The other eight chose lexical alternatives for 打 or 离开 without a change in whole-answer credit. Both writers lost the command relation in 戍卒奉命伐燕. In the natural clause 吾举厥也而中, both retained 厥 but left the meaning of the recommendation unclear. The natural-pair improvement criterion failed. [Frozen study](phrase_boundary_comparison.md) · [Output images and method-blind choices](evidence/phrase_boundary_20261119/review_pixels.npz) · [Readings](evidence/phrase_boundary_20261119/reading.tsv) · [Scores](evidence/phrase_boundary_20261119/scores.json)

The two writers share their learned verb decisions. Training used 870 positive clauses and 377 natural pass-through examples. Nineteen of 56 candidate alignments for 谢 and 举 were excluded after whole-clause reading. Aligned glyphs occupy 32 × 32 pixels. Training the two heads together took 6.7 seconds on CPU using an existing 2.25-million-parameter visual encoder and previously fitted composer. The saved sense weights occupy 39,963 bytes and the additional boundary table occupies 185,290 bytes. Corpus preparation, inherited components and evaluation are outside this incremental timing. On 1,024 diverse passages the four centre glyphs occurred in only 37 sources. Exactly one output changed between the matched writers, reducing the total reference edits from 15,561 to 15,560. The comparison measures neither general writing ability nor a compute advantage over a text model. [Weights and training counts](evidence/phrase_boundary_20261119/fit.json) · [Inference and evaluation](evidence/phrase_boundary_20261119/evaluation.json)

## What can one perfect edit accomplish?

For every nonempty source interval, we independently generated both surrounding passages with the unchanged composer. An ideal replacement can give the exact reference if the two exteriors match its beginning and end without overlap. The middle can then be filled with the correct target images of any length. Six unmodified translations were exact. Allowing up to one, three and six source cells for the ideal edit makes 29, 60 and 182 references reachable, including those already exact. A separate exhaustive enumeration reproduced the minimum widths on the first twenty passages. This bound applies to exact reproduction with one edit. A meaningful paraphrase can use different images. [All 1,024 witnesses](evidence/phrase_boundary_20261119/reach.json)

For the next comparison we prepared all 55,872 training passages as ordered input and output images with candidate places to preserve visible writing and compose multiple intervening regions. The archive contains 52,459 training pairs with at least two changed regions under this alignment. Local repair covers a small fraction of the language task. Whether an independent writer can infer useful alignments and translate the intervening material remains an experimental question. [Corpus preparation and proposed comparison](ordered_copy_anchor_protocol.md)

## Reproducing the comparison

Run the four scripts in order with the repository on PYTHONPATH. The corpus and previous model artifacts follow the paths in the preceding predicate study. Preparation and admission create fresh versioned files and stop on an existing output. Fit and evaluation use only CPU. The public inference command reads a supplied image and the packaged weights.

~~~bash
export PYTHONPATH=.
export OPENBLAS_NUM_THREADS=2
python scripts/prepare_phrase_boundary_trial.py
python scripts/admit_phrase_boundary_trial.py
python scripts/train_phrase_boundary_trial.py
python scripts/evaluate_phrase_boundary_trial.py
~~~

The archived review table preserves every distinct choice. The scoring script joins the frozen readings with the concealed method labels. The separate reach calculation enumerates possible edit intervals. Five complete CPU replays compare both the image file and the unfurled bitmap bytes with the archived predictions. [Public inference instructions](../models/visual_edit_zh/README.md) · [Early-roster audit](evidence/phrase_boundary_early_roster_audit_20261119.json)

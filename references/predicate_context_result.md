# Reading distance and departure from the same glyph

In 去寿春三十里, the experimental phrase writer produces 距离寿春三十里。 In 我不能舍仲宝去也, it writes 我不能舍仲宝离开。 In both prompts the pixels of 去 are identical. Moving to a different context changes the meaning the writer expresses. Its failure on 既而段君去 is equally informative. It writes 不久段君距离。, assigning distance to a person who has departed. Figure 64 contains the actual source and output images from all three cases. [Figure 64](../publication/ilm-nature/figures/Figure_64_predicate_context.png) · [Editable figure](../publication/ilm-nature/figures/Predicate_context.pptx)

## How much survives in a whole answer?

| Writer | Intended sense / 20 | Whole answer faithful / 20 | Both answers faithful / 10 pairs | Edits on 1,024 diverse passages |
|---|---:|---:|---:|---:|
| Phrase composer | 9 | 5 | 0 | 15,564 |
| First-glyph contrast | 15 | 6 | 0 | 18,521 |
| Most frequent reading with the same phrase composition | 10 | 4 | 0 | 15,570 |
| Learned contextual phrase | 17 | 6 | 0 | 15,561 |
| Explicit cue rule with the tested predicate supplied | 19 | 7 | 1 | — |

Six pairs came from twelve fresh archive chapters. For these the learned writer gave ten recognizable senses and three faithful answers. Four authored pairs held the subject and position of 伐 fixed across a timber and a military use. Seven senses and three whole answers were credited. No further natural felling clause qualified after duplicate and chapter screening. Seven pairs conveyed the intended sense on both sides, yet every pair had at least one incomplete or inaccurate answer. The explicit cue comparator performed better. None of the learned writer's twelve independently selected diverse outputs was credited as faithful. One study-involved reader graded shuffled options. Scores are descriptive, and the small, balanced samples target these two verbs. [Frozen full readings](evidence/predicate_context_20261119/readings.json) · [Independent chapter selection](evidence/predicate_context_20261119/span_admission.json)

## Why predicting the first glyph fell short

We exposed two independent copies of the same visual writer to 55,872 ordinary paired passages and 4,656 presentations of 114 curated sense clauses apiece. Each fit processed 1,811,103 target cells in 582 optimizer steps on one RTX 4090 D. The comparison arm added a loss on the relative probabilities of the two candidate initials at their position in the pictured gloss. The additional training loss fell from 0.440 to 0.00044. The uninterrupted free writer still translated only three of thirty original held-out clauses faithfully, against four for the matched replay. Both scored zero fully correct pairs out of fifteen.

With the preceding answer pixels supplied, the comparison writer preferred the correct first glyph in 16/30 cases. Substituting an opposite-sense source of equal length while keeping the answer prefix identical gave 15/30. Median combined probability for both choices under copying and exemplar emission was 8.8% with the actual source. A relative preference can become very sharp in training even when the free decoder has many other emissions to choose from. Scoring conditional discrimination and grading complete autonomous output answer different empirical questions. [Actual probabilities at every source intervention](evidence/predicate_context_20261119/neural_forced.json) · [Matched fits](evidence/predicate_context_20261119/neural_fit.json)

## How a complete phrase is emitted

The second intervention selected a phrase of observed ink using neighbours of the pictured verb and the frozen visual encoder's contextual state. Class-balanced ridge regression learned from 833 sense examples and 192 pass-through examples. The extra 719 sense alignments came only from earlier training chapters and were selected automatically. The independent writer scans aligned 32 × 32 cells, identifies its learned centres by their pixels and emits the selected bitmap sequence. The shared phrase composer writes the material before and after each selected span. Inference uses the packaged weights and pictures. The clause order, position matching, stopping rule and single printing font limit the present application.

The fitted head occupies 22,246 bytes. It reuses the 2,251,925-parameter writer's source encoder and the existing 55,872-pair phrase composer. The two neural fits took 13.4 and 20.5 seconds on a 4090 D and the later phrase fit took 4.3 seconds on CPU. All times exclude training the shared components. Performance at matched general language quality remains unmeasured. Prior work found that contextual encoder states can carry information useful for word sense disambiguation in translation. Here the held-source intervention asks whether specific pictured contexts actually change the completion, and complete free answers test how far that information travels into output. [Tang, Sennrich and Nivre](https://aclanthology.org/D19-1149/) · [Writer and phrase fit](evidence/predicate_context_20261119/span_fit.json) · [Standalone pixel reproduction](evidence/predicate_context_cpu_replay_20261119.json)

A recurring failure is visible in 相去五里. The surrounding composer retains 相 while the selector writes 距离, yielding the awkward 相距离五里。 The fitted training data already contain 27 distance examples with 相去. Fifteen gloss it with 相距, nine with 相隔 and three with 距离. None contains the emitted sequence 相距离. The canonical sense label discarded this distinction before generation. Other errors come from mistranslating surrounding source words. The next comparison should learn how many neighbouring **input** cells jointly realize a phrase as well as which **output** cells express it. Freeze a fresh test with contextual constructions outside the two intervention verbs, keep a cue comparator, and retain the broad passage regression. More epochs on the saturated two-choice loss would not address the observed boundary error.

## Reproducing the trials

From the repository root, set `PYTHONPATH=.`. The following scripts use the pinned local NiuTrans archive, the previous committed model weights and the paired translation rosters. Each study writes to a fresh versioned directory under `artifacts/`. The public image inference command needs only the runtime dependencies and packaged model files.

| Experiment | Prepare and qualify | Fit | Generate and score |
|---|---|---|---|
| Neural first-glyph contrast | `prepare_predicate_likelihood.py`, `admit_predicate_likelihood.py` | `train_predicate_likelihood.py` | `evaluate_predicate_likelihood.py`, `measure_predicate_source_dependence.py` |
| Contextual phrase writer | `prepare_contextual_phrase_realization.py`, `admit_contextual_phrase_realization.py` | `train_contextual_phrase_realization.py` | `evaluate_contextual_phrase_realization.py` |

All named scripts are in `scripts/`. Model generation and the masked reading are separate. Both frozen reviews include the actual packed pixels of every answer choice, including failures. In each compressed NumPy file, `review_choice` gives zero-based review and choice numbers, `offsets` indexes rows of `packed`, and each 128-byte row is one 32 × 32 glyph in big-endian bit order with ink equal to one. The neighboring JSON and TSV files contain the method mapping and recorded readings. [Neural output pixels](evidence/predicate_context_20261119/neural_review_pixels.npz) · [Phrase experiment output pixels](evidence/predicate_context_20261119/span_review_pixels.npz)

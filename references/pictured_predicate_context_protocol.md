# Can the same pictured verb take two meanings?

The exchange writer can carry an unfamiliar participant image through a sentence. A stronger test keeps the central bitmap fixed and changes the intended meaning. In 去国, 去 describes leaving a country. In 去寿阳三十里, the same character denotes a distance from Shouyang. Similarly, 伐 can describe felling a tree or attacking a state. A reader of the single isolated cell receives identical physical evidence in both senses. On a balanced two-sense set, any decision based solely on that cell has expected accuracy one half. More examples of the isolated cell cannot remove the ambiguity. The surrounding pictured text can.

## Data available before evaluation

A scan of the existing 55,872 training pairs found the following unique pictured source clauses. Each pair contains exactly one instance of the pictured predicate, one selected modern expression in the parallel target, and matching clause ordinals after a punctuation split. Neither contrastive modern expression occurs verbatim in the source. This is a training inventory. The search reads no held-out inputs and runs no optimizer. [Reproducible selection](../scripts/survey_pictured_predicate_contexts.py) · [Complete training candidates](evidence/pictured_predicate_training_candidates_20261119.json)

| Visible predicate | Modern reading | Candidate clauses |
|---|---|---:|
| 去 | leave, 离开 | 80 |
| 去 | be distant from, 距离 | 13 |
| 伐 | fell, 砍伐 | 7 |
| 伐 | attack, 攻打 | 38 |

A reading of all 20 minority-sense pairs confirmed the targeted correspondence, including temporal and generational distance. The full translation sometimes supplies context implicit in the short source. The final felling clause names its tree only in the gloss. One attack gloss preserves the attack but contains 水灵军队 where the source describes leading an army. A verified target sense alone does not certify a clean whole-answer training label.

## Matched test

Use entire pictured translations as supervision and the existing independent writer as the common initialization. Select one instance per physical source, freeze split membership by chapter, and exclude earlier training and evaluation images in either role before constructing the blind test. Curate both senses on training data first. Record differences between strict chapter novelty, absence of complete source images, and recurrence of short phrases. Cap any candidate fit at the registered event-fit GPU resource envelope. Retain an untreated parent, an equal-exposure original-training replay and literal copying. Evaluate the same candidate with full input, the predicate alone, and semantically different context matched for length. Report the number and origin of all accepted and rejected pairs before fitting.

Score the sense of the emitted predicate and the complete modern proposition separately. Include base-rate and simple cue controls, especially the distance unit 里 and the tree component 木. Hold out familiar local cue phrases and report temporal-distance and military cases separately. Report both orders of each context contrast, changes in free-written meaning, and the existing diverse-chapter regression. New sense transfer requires an advantage in both-sense complete answers over replay and over the strongest cue-only comparator. If only the lexical gloss improves, preserve that result as restricted disambiguation. Expand the corpus only after the sense and full-clause outcomes are known.

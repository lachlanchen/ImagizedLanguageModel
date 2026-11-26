# Finding the participants of a pictured verb

A 312,244-parameter visual reader found both correct head words in **901 of 1,196** eligible English clauses from two official treebank tests. The input gives the pictured sentence and identifies the verb. The reader searches all available ink spans for both answers. Its matched position control found 48 pairs. On a third treebank the reader found 347/478 pairs, compared with 12/478 for the control. Each response consists of two actual source-image crops. [Original predictions](../publication/ilm-nature/evidence/visual_role_localization.zip) · [Run from a PNG or a typed prompt](../models/visual_role_locator/README.md)

| Held text | Clauses | Both spans exact | Position control |
| --- | ---: | ---: | ---: |
| EWT, active | 618 | 470 | 27 |
| EWT, explicit-agent passive | 14 | 6 | 0 |
| GUM, active | 546 | 420 | 21 |
| GUM, explicit-agent passive | 18 | 5 | 0 |
| PUD, active | 478 | 347 | 12 |

The metric requires the start and end of each head word at the correct occurrence, in the correct role. One additional GUM active and one additional PUD active produced both correct bitmaps at different text positions. Only active PUD clauses met the registered dependency-label rule. Relative clauses without an overt agent and sentences with ambiguous attachments fall outside the study. [Full counts](evidence/natural_pictured_roles_v1/localization.json) · [Eligibility and corpus hashes](evidence/natural_pictured_roles_v1/corpus.json)

## What the models read

All models learned from the same 8,973 human-annotated EWT and GUM training clauses, including 270 explicit-agent passives. No complete official test source image occurs in the training set. We reconciled the 1,196 official gold labels with the original dependency arcs. The model reads aligned 32-pixel glyph cells. A convolutional encoder and two attention layers predict the start and last cell of each answer. Inference maximizes their summed scores over two disjoint ink spans outside the marked predicate. The position control receives the same predicate geometry, occupied length and legal spans. Only the context model sees the source glyph features. Both models used 153,600 training presentations on one RTX 4090 D. The context fit took 26.7 seconds with 0.386 GiB peak CUDA allocation. GPU time excludes corpus preparation, evaluation and the preceding independent experiments. [Fixed comparison](visual_role_localization_protocol.md) · [Context weights and training history](../models/visual_role_locator/context/fit.json)

On 2,346 evaluation cases, a second CPU pass reproduced all 9,384 output PNGs from the two models. Exhaustive enumeration on twelve short held clauses verified the maximizing predicted spans for both models. [Replay and enumeration](evidence/natural_pictured_roles_v1/locator-replay.json)

## How much do supplied spans help?

Giving a separate reader the two participant locations reduced the task to choosing their order. It chose correctly on all 1,196 official clauses and all 478 eligible PUD clauses. The layout control, which saw the two participant crops and mark geometry, reached balanced accuracies of 80.5% on EWT and 76.0% on GUM. These scores assume a prior step has already identified the three marked words. Only the open locator in the table finds the participant words. [Marked reader](evidence/natural_pictured_roles_v1/assessment.json) · [Third-treebank check](evidence/natural_pictured_roles_v1/pud.json)

Typographic spacing exposed a shortcut in the marked reader. With three extra spaces after the participle, it missed all 84 active cases. Compacting consecutive unmarked paper cells restored all 84 and preserved every official choice. In a separate test, an active honorific and a passive preposition occupied the same two cells. The contextual reader selected both correct roles in all 84 layout-matched pairs, while the geometry control did so in zero. The Lean theorem establishes invariance of the compact stream to an extra blank in an existing run. It makes no claim about how an arbitrary reader interprets the remaining ink. [Raw spacing challenges](../publication/ilm-nature/evidence/natural_roles_spacing.zip) · [Counts](evidence/natural_pictured_roles_v1/spacing.json) · [Lean proof](../formal/span_list/OpticalPaperSpacing.lean)

## What remains difficult

With only the verb marked, the open locator found both names in 83/84 active and 77/84 passive controlled sentences. It scored zero on both 84-case cleft constructions when the requested answers were the named referents. In all 84 subject clefts it selected the relative pronoun *who* as actor. For *It is Anna who helps David*, the response crops read *who* and *David*. This is a plausible local grammatical analysis and an incomplete answer to the referent question. In object clefts it selected *whom* as undergoer in 25/84. Reference resolution and discovery of a predicate from a full prompt remain open. The English model has been evaluated on one rendered print face and annotated transitive constructions. Free bilingual writing and novel historical glyph synthesis require separate evidence. [Raw construction and official results](../publication/ilm-nature/evidence/visual_role_localization.zip) · [Free-writing comparison](visual_strip_restoration_result.md)

The annotations and source texts retain their corpus-specific terms. [Attribution and source revisions](../publication/ilm-nature/evidence/natural_roles_attribution.md)

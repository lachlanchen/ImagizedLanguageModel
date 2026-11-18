# How much of a Chinese translation survives visual generation?

Two matched writers translated **尝攻襄州，为孔勍所败。** as **曾经进攻襄州，被孔勍打败。** from source pixels alone. The actors and direction of defeat survive the change into modern prose. This was the sole acceptable answer from each writer in the final eighteen-passage reading. In another passage both lost the person predicted to die. The two complete examples appear as the original ink in Figure 59. A reproduction on CPU produced identical pixels from the released checkpoint for both examples. The successful typed input also reproduced the image-input answer. [Input, weights and output hashes](evidence/visual_feedback_20261118/inference.json)

## Coverage of learned phrases

The larger composer was trained on 55,872 classical–modern pairs. Among 1,024 previously unused passages from 235 chapters, its ordinary decoder reproduced ten complete corpus glosses, four requiring a change. Although the six highest-scoring streams always contained its ordinary output, exact edit-distance search over every legal segmentation and translation reached the recorded gloss for only 31 passages. The best possible physical error with this inventory was 9,397 edits in 23,147 target cells. Some perfectly reasonable translations can differ from the archive.

In a reading of all six alternatives for 18 complete passages, the composer offered one acceptable gloss. None of its alternatives rescued the twelve other cases with an acceptable reference. In one, 卒 means “finally” in the context of dispatching an envoy. The expression learned for death produced a misleading gloss. Figure 58 shows the actual strips. The protocol distinguishes a source-only answer from selecting an alternative with access to the recorded gloss. [Protocol](pictured_clause_support_protocol.md) · [Scores](evidence/clause_support_20261118/summary.json) · [Blinded readings](evidence/clause_support_20261118/grades.tsv)

## A recurrent visual writer

On one RTX 4090 D, an encoder–decoder with 2,251,925 parameters learned from the same paired images for eight passes. A bidirectional source encoder and a causal history encoder select the next action from the visible writing. The model can copy the source, select one of 2,048 frequent observed pictures, draw new landmarks, or stop. In the first source-only evaluation it chose the drawing route zero times. It made 21,073 edits over 22,655 reference cells on 1,024 new passages, compared with 11,799 for phrase composition. Reversal raised the neural error count to 84,055. One of 18 neural outputs and 17 of the reference translations passed the reading. The fit took 93.4 seconds and peaked at 349 MiB of allocated tensors, excluding data preparation and the inherited drawing motor. [Model protocol](causal_glyph_translator_protocol.md) · [Initial scores](evidence/visual_feedback_20261118/initial/summary.json)

Supplying the correct previous pictures made 8,840 of 22,655 next positions exact. At the first actual error, changing only that picture reduced correct predictions of its successor from 442 to 133 among 1,019 cases. There were 341 losses and 32 gains. One batched versus stepwise replay had a differing image before the first divergence. [History counts](evidence/visual_feedback_20261118/history.json)

## Does exposure to errors improve writing?

We initialized two continuations from the same trained checkpoint and used identical minibatches and updates. One received clean preceding images. The other received 1,431,375 substitutions sampled from the parent's own wrong glyphs in otherwise supervised history. Each saw four more passes. The new evaluation comprised 1,024 passages from 724 chapters and 23,507 target cells.

| Writer | Glyph edits | Generated cells | Learned stops | Changed exact | Faithful / 18 |
|---|---:|---:|---:|---:|---:|
| Phrase composer | 12,666 | 16,152 | — | 6 | 0 |
| Initial neural | 23,092 | 28,853 | 993 | 1 | 0 |
| Clean-history continuation | 23,236 | 30,134 | 989 | 1 | 1 |
| Error-history continuation | 45,765 | 52,267 | 674 | 1 | 1 |

The chapter-resampled interval for the increase in error was 87.08–104.30 percentage points. The error-trained continuation overproduced familiar glyphs and reached the 96-cell limit on 350 passages. Edit distance counts insertions, so its value can exceed the reference length. The published recovery checkpoint serves to reproduce this experiment. The stronger phrase model remains the default. [Matched protocol](visual_feedback_recovery_protocol.md) · [Scores](evidence/visual_feedback_20261118/matched/summary.json) · [Frozen readings](evidence/visual_feedback_20261118/matched/grades.tsv) · [CPU inference](../models/visual_edit_zh/README.md)

All three sets used presegmented, canonical 32 × 32 Chinese cells. Each reading packet covered eighteen distinct chapters with automatic markers for negation, speech or events. Dates can trigger those markers accidentally. A single study-involved model reader graded randomized, deduplicated strings transcribed from the canonical pictures before method assignments were scored. Independent human adjudication would address a material source of uncertainty. The final packet contained twelve fully acceptable reference glosses. Performance on natural scans, free bilingual questions, new historical forms and general conversation remains open.

## What the evidence suggests

Giving a causal writer the wrong previous image and supervising the original next image can introduce conflicting histories. This offers a possible explanation for the observed repetition. The experiment directly establishes the increase in length and edit error. A draft-conditioned intervention would let a model retain source-grounded ink and revise only uncertain spans. Its optimizer, controls and evaluation must include the full cost of both draft and revision. [Draft preparation and evaluation design](pictured_revision_protocol.md)

Autoregressive generation of rendered language already has published precedent in PIXAR. A November 2026 preprint also reports better pixel-space reasoning when models write intermediate visual states on controlled tasks. This motivates testing an explicit, revisable pictured state. It does not establish unrestricted Chinese translation or a computational advantage over Qwen. The copy pathway has a separate antecedent in pointer–generator models. [PIXAR, ACL Findings 2024](https://aclanthology.org/2024.findings-acl.874/) · [Chu et al., 2026 preprint](https://arxiv.org/abs/2611.22820) · [See et al., ACL 2017](https://aclanthology.org/P17-1099/)

## Reproduce the experiments

The following scripts use the pinned aligned archive and ink motor described in the source-only translation experiment. Fit and evaluation commands create new experiment directories. The reporting scripts use the frozen readings after model generation. [Corpus preparation](source_only_visual_translation.md)

```bash
PYTHONPATH=. python scripts/measure_pictured_clause_support.py
PYTHONPATH=. python scripts/train_causal_glyph_translator.py
PYTHONPATH=. python scripts/evaluate_causal_glyph_translator.py
PYTHONPATH=. python scripts/diagnose_causal_glyph_history.py
PYTHONPATH=. python scripts/train_visual_feedback_recovery.py
PYTHONPATH=. python scripts/evaluate_causal_glyph_translator.py --recovery
PYTHONPATH=. python scripts/prepare_pictured_revision.py
```

Packed per-passage images, rows and checkpoints reside under `artifacts/pictured_clause_support_v1/`, `artifacts/causal_glyph_translation_v1/` and `artifacts/visual_feedback_recovery_v1/`. The public evidence directories retain both reading packets, their scores, the displayed original strips and the hashes of the complete local generations.

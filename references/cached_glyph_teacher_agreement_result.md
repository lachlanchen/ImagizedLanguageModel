# Native-head agreement along gold image histories

An offline audit of previously saved native-head labels finds substantial
reference agreement after glyph-image feedback, with a dip immediately after
the first glyph. It does not establish the cause of the remaining language
failures or measure autonomous answer accuracy.

The [complete audit](evidence/cached_glyph_teacher_audit_v1/report.json) covers
all 4,363 cached states from the original 508 supported training questions.
Every state and reference label is checked against its original causal cache.
The native-head winners were computed in the earlier readout-calibration run;
this audit performs no neural forward, optimizer update, GPU operation or new
teacher call. It took 1.09 seconds on CPU. No development case enters it.

| State | Reference-form matches / states | Agreement | Native terminal winners | Unsupported winners |
| --- | ---: | ---: | ---: | ---: |
| First glyph, no output-image prefix | 368 / 508 | 72.44% | 0 | 0 |
| Continuation glyph, gold-image prefix | 2,632 / 3,347 | 78.64% | 97 | 190 |
| Expected terminal state | 434 / 508 | 85.43% | 434 | 0 |

Agreement is with the prepared next form in one training answer. The native
winner is mapped to its first complete glyph, a terminal label, or an
unsupported label. Alternative valid wording, formatting, source-unit
segmentation and image-embedding effects can all cause disagreement. The saved
mapped labels do not distinguish the causes of an unsupported winner.

Prefix length is not a matched intervention: later positions represent
different subsets of questions and different target forms. The first-state
column below reports only questions represented in that row, making the
change in composition visible.

| Gold prefix glyphs | Next-form matches / states | Agreement | First-state agreement in the same question subset |
| --- | ---: | ---: | ---: |
| 0 | 368 / 508 | 72.44% | 72.44% |
| 1 | 328 / 507 | 64.69% | 72.39% |
| 2–3 | 756 / 891 | 84.85% | 70.70% |
| 4–7 | 939 / 1,166 | 80.53% | 66.76% |
| 8 or more | 609 / 783 | 77.78% | 57.37% |

Among questions whose first winner agrees with the reference, continuation
agreement is 1,702 / 2,149 states (79.20%). In 138 / 508 gold-history sequences,
every mapped next-form winner and the terminal winner agrees. Those sequences
were supplied their correct previous glyph images; they are not 138 observed
autonomous answers. All [per-state rows](evidence/cached_glyph_teacher_audit_v1/records.json)
and [per-question summaries](evidence/cached_glyph_teacher_audit_v1/cases.json)
are retained, including disagreements.

The results do not support treating cumulative image-feedback damage as an
already established, sufficient explanation for the failures. They also do
not prove that the encoder is reliable: the absent counterfactual would compare
the same questions and prefixes under matched exact-source embeddings and
learned image embeddings. No such causal comparison is made here. The first
128 primitive glyphs' mean semantic cosine of 0.77735 and their near-exact
pixel reconstruction remain distinct measurements.

The next experiment remains the genuine visual-copy curriculum, with the
readout trained on broader real image states and original language histories.
The present audit changes no weights, gates, runtime or default checkpoint.
It provides no reason to replace that experiment with another image renderer
or an encoder-only repair before its physical and autonomous results exist.

Reproduce from the existing authenticated calibration and causal cache:

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=. python scripts/audit_cached_glyph_teacher_labels.py \
  --output artifacts/cached_glyph_teacher_audit_v2
```

The output directory must be new. No model or source vocabulary table is loaded
by this command; it reads the previously computed evaluator labels only.

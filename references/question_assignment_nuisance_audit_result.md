# Coarse visual cues in the frozen V63 training pairs

Completed 5 November 2026 under the
[plan written before the diagnostic](question_assignment_nuisance_audit_plan.md).
The active V63 training run, its admission/sampling rules and its autonomous
language gates are unchanged. This is CPU data analysis, with no actor or
teacher call, development image, or sealed-test example.

All 3,793 admitted pairs, 3,016 distinct questions, 939 paragraphs and 939
article titles are included. Five folds isolate original training article titles. Each prediction
uses only the other four folds, including feature normalization and target
centering. The fixed ridge penalty is one; no feature or hyperparameter is
selected from the results. Each accessed native packed image and the frozen
source/pair/cache reports are hash-verified.

| Probe predicting physical answer width | Correct / wrong / tie | Pair-weighted accuracy | Context-weighted accuracy |
| --- | ---: | ---: | ---: |
| Constant prediction | 0 / 0 / 3,793 | 50.00% | 50.00% |
| Question width | 1,641 / 1,500 / 652 | 51.86% | 51.50% |
| Question width and coarse ink moments | 1,830 / 1,590 / 373 | 53.16% | 53.31% |

Ties receive one half. Context weighting matches the production sampler's
uniform paragraph then uniform pair weighting. Ink features are fraction,
normalized horizontal/vertical centroid and horizontal/vertical variance;
no glyph identities, answer locations, text embedding or image encoder is a
predictor. These features can still contain task information. No inferential
confidence interval or claim of statistical significance is made.

Only 373 pairs (9.83%) have identical unpadded physical answer width;
659 (17.37%) have identical scored native suffix width, including the
termination blocks. All 373 equal-answer-width pairs also have equal suffix
width. A width-prediction assignment probe necessarily ties equal-width
answers, regardless of its training quality. These counts do not change which
pairs enter the active experiment.

This audit establishes modest associations in coarse training-image
statistics. It does not establish that V63 uses them, measure semantic
understanding, estimate Bayesian susceptibility, or predict the final
autonomous-writing result. The live comparison must still generate correct
answer images under its registered question controls.

Reproduce from the repository root:

```bash
CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=. \
artifacts/external/mixar_env_py312/bin/python \
  scripts/audit_question_assignment_nuisance.py --out artifacts/assignment_nuisance_reproduction
```

The completed run is `artifacts/question_assignment_nuisance_20261105/`:
`report.json` pins inputs and outputs, `questions.jsonl` retains every feature,
fold and out-of-fold prediction, and `pair_scores.jsonl` retains every paired
margin. It takes 2.20 seconds on CPU; no GPU allocation is made. A compact
[tracked receipt](evidence/question_assignment_nuisance_audit.json) records
the result and full-report hash. Eight fixed question-image crops match an
independent rendering exactly. Two numerical tests pass: held-fold target
changes cannot affect that fold's predictions, and exhaustive two-way
assignment costs have the expected orientation and constant-predictor ties.

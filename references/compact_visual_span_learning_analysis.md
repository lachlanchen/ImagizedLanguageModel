# Complete-answer learning analysis

Analyze the five fixed checkpoints from the compact full-sequence fit using
the registered [held-out evaluation](compact_visual_span_evaluation_protocol.md).
The checkpoint sequence, normal cases, erased-input controls and three displayed
examples are fixed before fitting. They remain unchanged by this analysis.

Recompute the original scores from their authenticated readings. Verify all
generated PNGs, underlying tiles, input bindings, feedback hashes and checkpoint
identities through the existing reviewer. Require the full 120 cases at both
endpoints and the 44 normal cases at each intermediate checkpoint. Missing
questions, unreadable outputs, premature stops and cap hits remain in their
original denominators. The analyzer refuses an unfinished fit or review.

Report complete exact answers out of 32 questions and both-correct pairs out
of 16 contexts. Obtain pointwise 95% percentile intervals by sampling the sixteen
passage groups with replacement, preserving both questions together. Use 10,000
replicates and seed 202611154. Resample the same groups for paired final-minus-
initial and normal-minus-erased-input differences. Keep the question-erased and
passage-erased comparisons separate. No new selection threshold is introduced.

These intervals describe a small development cohort. They have no simultaneous
coverage guarantee across checkpoints or comparisons and cannot account for
unknown inherited pretraining exposure. An all-zero sample produces a zero-width
bootstrap interval; this cannot establish zero population success. Report book
coherence and exact continuation as counts, with two held-out source works and
twelve prefixes. Do not infer a general book-level confidence interval from
those two works.

Export the language measurements as JSON and CSV for the manuscript and native
PowerPoint charts. Copy the original PNG bytes for `qa_01_a_normal`,
`qa_01_b_normal` and `book_01_normal` at every step, including blank display
placeholders for zero-glyph responses. Keep the raw-output and input hashes.
Numerical losses and pixel reconstruction measures have their own plots and
units. The analyzer makes no model qualification decision.

After the pipeline completes and every blind reading is recorded and scored:

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=. python scripts/summarize_compact_visual_span_learning.py \
  --pipeline artifacts/compact_visual_span_pipeline_v3 \
  --readings artifacts/compact_visual_span_pipeline_v3/review/readings.json \
  --output artifacts/compact_visual_span_learning_v1
```

The visual trajectory has not run. Its fit, five checkpoint generations, image
review and trained cold replay remain pending. Export code and input checks
are not a measured learning curve.

## Measured native-text reference

The same analyzer recomputed the existing Qwen3 1.7B native-text scores from all
44 authenticated raw responses. Its unchanged 8/32 exact QA answers give a
group-bootstrap interval of 9.375–43.75%; its 2/16 both-correct pairs give
0–31.25%. Book coherence remains unrated. These CPU reference results do not
establish visual-model quality or same-GPU efficiency. See the
[evidence](evidence/compact_visual_span_qwen_uncertainty_20261111.json).

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=. python scripts/summarize_compact_visual_span_learning.py \
  --baseline artifacts/compact_visual_span_qwen_reference_v1/generation \
  --output artifacts/compact_visual_span_qwen_uncertainty_reproduction
```

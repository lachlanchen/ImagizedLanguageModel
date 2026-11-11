# Identical pixels in the complete feedback schedule

The compact span encoder predicts its continuous updates and update count
from the glyph pixels of the span. When two source spellings have identical
pixels, this encoder produces the same prediction. The prepared corpus
contains nine such groups among 29,740 distinct visual spans. They account
for 27,676 of 5,749,578 span appearances. The largest group combines line
breaks and several kinds of blank space. It appears 17,478 times.

The audit authenticates all training shards and uses the actual 8,172-batch
schedule with a 1,024-position budget. It reads the same BF16 source vectors
as the compact trainer. It performs no optimization or language inference.

| Auxiliary objective | Minimum imposed by exact pixel aliases |
| --- | ---: |
| Mean relative squared feedback error | 0.0007492351 |
| Update-count cross entropy, nats | 0.0001318350 |
| Update-count classification error | 0.0000220569 |

The weights follow the trainer's reductions within batches. The vector
objective averages over 5,844,979 source update positions. The count objective
averages over visual spans. Both objectives then average equally over the
scheduled batches. The loop verifies that their total weights equal one.

For an appearance \(a\) in batch \(b\), let \(v_{aj}\) be the source vector
for update slot \(j\), \(U_b\) the number of source updates in that batch,
and \(B\) the number of batches. For an identical-image group \(g\), define

\[
  w_{aj}=\frac{1}{B U_b\max(\|v_{aj}\|^2,10^{-8})},
  \qquad
  \mu_{gj}=\frac{\sum_{a\in g}w_{aj}v_{aj}}{\sum_{a\in g}w_{aj}}.
\]

Only appearances containing slot \(j\) enter these sums. The minimum
feedback loss over a free output vector for each group and slot is

\[
 L_{\min}=\sum_{g,j}\left[
  \sum_{a\in g}w_{aj}\|v_{aj}\|^2
  -\frac{\|\sum_{a\in g}w_{aj}v_{aj}\|^2}{\sum_{a\in g}w_{aj}}
 \right].
\]

The program independently evaluates the weighted residuals at
\(\mu_{gj}\) and checks agreement with this expression. Conditional entropy
gives the corresponding minimum count cross entropy. Subtracting each
group's largest count mass from its total gives the classification-error
minimum. These bounds use free predictions per exact image group. Parameter
sharing and generalization can impose additional error on a neural encoder.

The main complete-sequence recipe proceeds to its scheduled fit. This
measurement records a small unavoidable component of its feedback objective.
Answer quality still requires the registered complete-image evaluation.
See the [evidence](evidence/visual_feedback_identifiability_20261111.json)
and [span representation analysis](visual_chunk_coverage_result.md).

Reproduce the calculation from the prepared corpus and shared source archive.

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=. python scripts/audit_visual_feedback_identifiability.py \
  --output artifacts/visual_feedback_identifiability_reproduction
```

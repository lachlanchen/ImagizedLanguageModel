# Compact visual-language failure occurs before generated feedback

Both completed compact fits fail conditional writing even when supplied with
correct reference-image history and the next span's length. In the registered
32-record diagnostic, the first target span gives zero exact glyph matches
among twenty seen-training and twenty-six development glyphs in each model.
Generated-image feedback has not yet occurred at those positions. A
feedback-only explanation is therefore insufficient for these failures.

The codec control reconstructs the same targets nearly exactly. Increasing
the flow sampler from eight to 64 steps does not repair conditional writing.
The encoder also has a measured restriction when aligning to native embedding
targets. These observations reject continuing the unchanged compact joint
recipe or choosing a sampler-only repair. They do not identify a single cause
of every failed language answer.

## Fixed interventions and original outputs

The [registered cached comparison](evidence/compact_visual_span_diagnosis_cached_registration_20261111.json)
uses both final 8,172-update checkpoints. It selects eight QA and eight book
records in each of the training and development partitions, taking source
groups in deterministic hash order. The two development books supply four
records each. It tests at most eight following spans per record: 201 spans,
122 training glyphs and 131 development glyphs in total. Cases were fixed
before neural execution. These selected development records are not an
untouched confirmation set.

All six writer arms receive the reference next-span length. They use identical
initial-noise hashes for each case/span. Correct-image history and emitted-image
history follow the same prefix prefill and cached-update schedule. Correct
reference images and update counts are explicitly supplied where indicated.
The native arm bypasses the image encoder with source embeddings; its input
distribution differs from the learned image states used to fit these writers.
It is a diagnostic bypass, not deployed visual inference.

| Arm | V1 training F1 | V1 development F1 | Wider training F1 | Wider development F1 |
| --- | ---: | ---: | ---: | ---: |
| Direct reference-codec reconstruction | 0.999885 | 0.999872 | 0.999885 | 0.999872 |
| Correct image history, 8 steps | 0.380124 | 0.347704 | 0.372034 | 0.351765 |
| Correct image history, 64 steps | 0.374644 | 0.342647 | 0.372581 | 0.350661 |
| Native reference history, 8 steps | 0.367306 | 0.343895 | 0.360783 | 0.332995 |
| Wrong reference context, 8 steps | 0.345125 | 0.311861 | 0.340855 | 0.317660 |
| Generated history, supplied counts | 0.347256 | 0.321425 | 0.337839 | 0.303559 |
| Learned prompt grouping and generated history | 0.351208 | 0.317971 | 0.339821 | 0.298761 |

F1 pools foreground pixels against the supplied target images. It measures
physical reconstruction, not answer meaning or readability. All sixteen
records in each split remain in the denominators. Correct-image-history
writing has zero exact target glyphs in V1; the wider writer has zero on
training records and one on development records. Direct codec reconstruction
has 117/122 and 128/131 exact glyphs, respectively. Native-bypass exact
matches are reported separately from blank targets in the receipt; they do
not qualify language. There is no diagnostic semantic grading or confidence
interval for this selected sample.

The [measured plot](evidence/compact_visual_span_diagnosis_20261111/conditional_writing.png)
shows all arms. The [original image examples](evidence/compact_visual_span_diagnosis_20261111/raw_examples/)
retain every arm, codec result, target and input for the first registered case
of each split/kind, in both models: 72 PNGs. That deterministic display rule
was added during execution; it was not a prospectively registered example
selection. The full 64 case reports and all raw ink/latent endpoints remain
local and hash-bound.

The analyzer independently recomputes every pixel metric from the registered
targets and saved binary outputs, verifies PNG/array equality, trace coverage,
initial-noise matching and emitted-pixel hashes, and authenticates every case.
Target codec latents were not separately persisted, so latent-error values
come from the bound neural execution and are not independently recomputed.
The [complete receipt](evidence/compact_visual_span_diagnosis_20261111.json)
and editable [pixel](evidence/compact_visual_span_diagnosis_20261111/pixel_metrics.csv)
and [reader](evidence/compact_visual_span_diagnosis_20261111/reader_metrics.csv)
measurements preserve these distinctions.

## Reader restriction and numerical execution

Both trained feedback maps have 383 varying affine directions for 2,048 output
coordinates. With final LayerNorm, the fixed head takes values in
\(a+\operatorname{col}(A)\), where

\[
 A=D W\operatorname{diag}(\gamma)C,\qquad
 a=\mu+D(W\beta+b),\qquad C=I-\mathbf1\mathbf1^\top/384.
\]

The closest point in that family gives a lower bound for the head's relative
squared error, even after relaxing all preceding image-network constraints.
On uniformly weighted selected source positions, the minimum is 0.570531
(training) and 0.576817 (development) for V1, and 0.568472 and 0.575389 for
the wider-writer fit. Achieved image-encoder errors are around 0.75.
This is an exact-arithmetic target-alignment bound, not the full training
objective's batch weighting or a language-quality bound. See the
[alignment plot](evidence/compact_visual_span_diagnosis_20261111/reader_alignment.png).

Local update-count accuracy remains high, but none of the 32 complete prompt
groupings exactly matches the reference grouping in either model. Learned
versus reference-grouped contextual cosines are around 0.996--0.998. Native
versus image-context cosines are lower. These are representation diagnostics;
they do not establish understanding or show that grouping is the dominant
remaining cause.

The initial diagnostic stopped at its fixed prefix consistency check. A
[separate precision probe](compact_visual_prefix_precision_result.md) found a
5% relative RMS discrepancy between full and truncated BF16 evaluation on
one prefix, almost removed by FP32 arithmetic on the same rounded weights.
A same-shape future-input erasure is bit-equal at that prefix. The corrected
comparison uses matching cached execution and retains the original tolerance;
its first reference/generated-history outputs have identical pixel counts
and errors. The original stopped attempt is preserved.

The completed two-model neural diagnostic takes 260.16 seconds, including
20.02 seconds waiting on pressure, and peaks at 3.401 GiB allocated GPU memory.
Including the stopped attempt, precision probe and two CPU preparations gives
467.24 seconds across those registered stages. The stopped attempt's peak
allocation was not retained. These costs are additional to corpus preparation,
teacher extraction, training and autonomous evaluation; they are not an
end-to-end language-learning efficiency result. No optimizer update is made.

## Next decision

Preserve both failed compact fits. Their joint adaptation has not established
readable conditional writing. Before another fit, evaluate the earlier
[model with readable image answers](glyph_tail_paired_answers_result.md) on
the current fixed cohort. Its 16/32 clear complete answers came from another
cohort and metric and cannot be compared numerically with the compact exact
scores. Audit its training/source-group overlap before calling the current
cohort held out for that model. Retain its larger inherited contextual-body cost. A matched-cohort
control will test which demonstrated capability was lost during compaction.

Then choose one bounded reader/writer intervention. Fitting conditional
writing from native source states remains a possible diagnostic training
comparison; it would not satisfy visual-input inference or historical-form
transfer by itself. Do not repeat it without checking the earlier
[separately trained semantic glyph loop](semantic_glyph_language_result.md),
which already produced limited readable strings but remained unreliable.
New model names or another width increase do not constitute a solution.

The full objective remains an independent image-input, image-output language
model on one 4090 with held-out question-dependent answers, coherent
continuation, historical/unencoded-form transfer and measured resource use.
The compact autonomous results remain 0/32 exact QA answers; this diagnostic
adds no autonomous language success claim.

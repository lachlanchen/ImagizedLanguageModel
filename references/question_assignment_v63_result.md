# Question assignment: completed matched image-language experiment

5 November 2026. Both 2,000-update arms, standalone offline replay and the
registered development audit are complete. The decision is
`reject_assignment_intervention`. Neither actor passes the bounded reading
requirements; useful general image-language generation remains unqualified.

| Measurement | Control, assignment weight 0 | Aligned, assignment weight 1 |
| --- | ---: | ---: |
| Fully trained parameters | 489,998,208 | 489,998,208 |
| Training through final optimizer update | 7,355.156 s | 6,737.336 s |
| Peak allocated training memory | 10.301 GiB | 10.301 GiB |
| All-development answer/termination-pixel NLL | 0.113901 | 0.114186 |
| Exact primary answers | 2/256 | 1/256 |
| Strict-raster multiset F1 | 3.229% | 2.417% |
| Aligned foreground F1 | 20.227% | 17.219% |
| Both questions answered exactly in a paragraph pair | 0/393 | 0/393 |
| Primary outputs reaching the generation cap | 0/256 | 0/256 |

Aligned minus control raster F1 is **-0.812 percentage points**, article-grouped
95% bootstrap interval **[-1.504, -0.202]**. The foreground difference is
-3.008 points [-3.978, -2.014]. Both show resolved question dependence:
normal minus blank-question raster F1 is 1.765 points [0.504, 3.251] for
control and 1.677 points [0.492, 3.033] for aligned. Question use therefore
cannot be dismissed, but its presence does not establish correct answers.

Each actor passes seven of ten reading gates. Exact answers, raster F1 and
paired answers fail in both. Requirements remain 10% exact answers, 25% raster
F1, 2% fully correct pairs, resolved question dependence, fewer than 20% capped
outputs and all implementation checks. No threshold, decoder, final-EMA
selection or primary case changes after the audit.

The separate supplied-answer diagnostic assigns the accepted images to their
questions correctly in 324/393 pairs for control and 330/393 for aligned.
Both individual energy ranks are correct in 82/393 and 109/393 respectively.
These differences are descriptive, without registered confidence intervals.
Accepted answer images are supplied to this diagnostic and are unavailable to
autonomous generation. Higher reference compatibility is insufficient evidence
of a useful writer.

Both arms start from the identical V62 eight-expert EMA and consume exactly
the same 8,000 pair draws, with four conditional likelihood streams per draw.
All 2,000 sample-journal rows match byte for byte. The preflight verifies exact
optimizer, EMA and RNG resume, full-backbone and assignment-only gradients,
and future-pixel causality. Both standalone bundles reproduce output tensors,
PNG bytes and stopping events with networking disabled and empty source/model
caches. The audit takes 582.104 s excluding loading, at 1.904 GiB peak allocated
memory. All GPU phases use physical GPU 0. The coordinator, trainer, audit and
owned tmux session have exited normally.

The first two fixed primary outputs, `DEV_1659_QUERY_3` and
`DEV_257_QUERY_1`, were inspected without repair or reselection. They contain
some recognizable digits alongside incomplete forms and do not answer their
questions. Strict-raster scores can penalize readable alternative renderings;
this qualitative inspection does not replace the fixed metrics with an
unregistered OCR score.

External MIXAR pretraining remains part of the provenance. This is one seed
on reused development data, with a fixed Noto-26 interface. Training timing
excludes loading, final serialization and external foundation pretraining.
The timing difference does not establish algorithmic efficiency. Sealed final
partitions remain unopened.

Checkpoint SHA-256:

- Control: `293abac3ef6ac376d22ff72083bf3997b1dde863e7bc28e7205c8d9227864ac8`
- Aligned: `783922cc917162b3ff72058b80528c67b0aa77fe642c1c457a2d02e5abbe6334`

The [protocol](question_assignment_v63_protocol.md) and
[guide](../docs/question-assignment-language.md) describe the frozen comparison.
The [compact receipt](evidence/question_assignment_v63_result.json) binds the
full reports, raw score tables, journals and final models. Full artifacts remain
in `artifacts/question_assignment_v63_20261105/`.

Recheck the saved-score aggregation, registered bootstrap and artifact hashes
without loading a model or using CUDA:

```bash
CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 \
  python scripts/verify_question_assignment_result_v63.py \
  --out references/evidence/question_assignment_v63_result.json
```

The verifier preserves an existing receipt and rejects changed content. It
does not rerun generation or reimplement the original image-scoring function.

This rejects this assignment intervention at the tested budget. It does not
prove image-native language impossible. The next candidate must improve
coherent, correct free writing; another reference-ranking gain alone would
not address the measured failure.

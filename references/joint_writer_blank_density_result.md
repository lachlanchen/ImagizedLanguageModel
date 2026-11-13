# Empty-image preference is not explained by greedy search alone

The registered final-head probability diagnosis completes all 124 cases and
12,597 target positions. It reuses authenticated states and emitted pixels;
there is no reader replay, optimizer update, new image generation or change to
the frozen physical grades. The
[independently recounted aggregate](evidence/joint_writer_blank_density_result_20261113.json)
binds every saved probability array and the original registration.

| Supplied-history group | False blank outputs | Source preferred over blank | Blank mode supported by prefix bound |
|---|---:|---:|---:|
| Training QA, fixed | 13 | 1 | 4 |
| Training QA, proportional | 0 | 0 | 0 |
| Training books, fixed | 137 | 7 | 22 |
| Training books, proportional | 145 | 5 | 22 |
| Development QA, fixed | 11 | 2 | 0 |
| Development QA, proportional | 1 | 0 | 1 |
| Development books, fixed | 42 | 1 | 15 |

A false blank has zero emitted ink while its source contains ink. The source
comparison uses complete-image log likelihood and the fixed 0.001-nat margin.
The prefix bound upper-bounds every nonempty image by the probability of its
first nonzero prefix. A smaller bound than the complete empty-image probability
supports the empty image as a global mode; the finite-precision calculation is
not a formal numerical certificate. A bound that does not decide the question
is inconclusive. A source image is a diagnostic comparison, not a runtime
candidate or a repair for an answer.

Across all positions, only 16/349 false blanks prefer the source image. Across
the selected physical-review positions, only one of fourteen development-book
false blanks and no training-fixed-QA false blank prefer their source. The
prospective search-only criterion requires seven and one, respectively, and
fails. All saved empty outputs agree with parallel zero-history argmax checks.
A better search might change some outputs, but these observations do not
justify treating search alone as the next repair.

The public probability arrays and JSON contain fourteen selected development-book
false blanks. An earlier prose denominator of thirteen was incorrect: three,
rather than all four, blank source positions coincide with the seventeen blank
outputs. The minimum-seven criterion, original scores and failed decision are
unchanged by this descriptive correction.

The script reports 9.369 seconds including its neural imports; its resource
monitor reports 7.933 seconds, no pressure waiting and 0.094 GiB peak allocated
VRAM. This is a small-head probability evaluation, not a full-reader inference
benchmark. Complete-language quality remains unmeasured because the original
physical prerequisite failed.

Reproduce the independent CPU recount, without a model call:

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  PYTHONPATH=. python scripts/summarize_joint_writer_blank_density.py --verify
```

The next [controlled change](joint_writer_context_learning_decision.md) lets
complete-image likelihood train the continuation adapters and query, instead
of fitting only the writer to states learned under the earlier pixel loss.
Its exact memory-bounded state-gradient implementation passes an independent
unsplit-autograd comparison. The real-source GPU preflight and full fitted
language comparison are still required; these checks do not establish language.

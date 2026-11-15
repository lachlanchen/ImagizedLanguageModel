# White-target ranking passes complete-source execution checks

The [registered six-case preflight](white_target_preflight_protocol.md) is
complete. Both original QA layouts, both complete 513-glyph books and both
layouts of the complete 314-glyph mixed-white book pass. All 131 learned
context/head tensors receive finite nonzero gradients. Four actual book
optimizer steps are verified and fully reset to the identical initial tensor
inventory. The new objective remains **untrained** for quality: this is a
computational prerequisite, not a writing or language result.

| Complete source | Layout | Target glyphs | True-white targets | Actual steps | Added white-target loss |
|---|---|---:|---:|---:|---:|
| Original QA | Fixed | 5 | 0 | 0 | 0 |
| Original QA | Proportional | 5 | 0 | 0 | 0 |
| Original maximal book | Fixed | 513 | 0 | 1, reset | 0 |
| Original maximal book | Proportional | 513 | 0 | 1, reset | 0 |
| First qualification mixed-white book | Fixed | 314 | 2 | 1, reset | 0.0381981 |
| First qualification mixed-white book | Proportional | 314 | 2 | 1, reset | 0.0381551 |

The mixed-white source is index 4,447, selected by the first qualifying training
book in the unchanged qualification order. Actual complete targets agree with
the bound original pixels. No convenient span was cropped, no blank was
inserted and no cached predictor state substituted. The displayed loss is the
additional source-averaged training term, not an accuracy or loss-improvement
measurement.

Both actual QA cases compare all 131 gradients against independent ordinary
unsplit autograd. Maximum relative L2 error is 2.900e-6; maximum absolute error
is 4.172e-7. Both full-source loss comparisons satisfy the fixed 2e-5 tolerance.
These QA sources contain no whitespace. The new white-branch gradient is
checked in the separate CPU fixtures; actual full-body mixed-white books check
finite gradients, complete execution, updates and reset rather than an unsplit
GPU comparison of that branch.

Nine CPU checks pass. They include mixed, all-white and all-nonempty complete
physical-writer gradients, non-divisible batches, exact recovery of existing
ranking gradients at zero added strength, exhaustive first-nonzero bounds with
tied probabilities, invalid inputs and missing-binding rejection before neural
import. A scalar-conversion warning in the initial eight-check run was removed
before registration; the final nine-check run passes without warnings.

Frozen native prompt/body, image feedback, stop parameters and four affine
buffers remain unchanged. Every book case has all 131 actual Adam counters at
one and changed elements in all 131 tensors, followed by exact restoration.
These are bound execution assertions, not an independent second neural replay.
No trial update, optimizer state, generated glyph or answer enters a later fit.

Worker execution takes 86.652 seconds, including zero pressure waiting, at
17.795 GiB peak allocated VRAM and 55.619 GiB minimum available host RAM. The
outer process takes 87.481 seconds and exits zero; its owner and worker are
terminal and GPU0 is released. No automatic retry or duplicate weight copy is
used. All 107 bound source files, including 101 preserved earlier sources, remain
unchanged. The whole model still has 7,553,448,370 parameters, 5,005,760 trainables
and 7,522,463,984 inherited parameters. This does not establish a lightweight
foundation or matched-quality efficiency.

The [public export](evidence/white_target_preflight_result_20261115.json) binds
the registration, actual case receipts, worker end and outer process receipts.
Its CPU validation rechecks the recorded gradient errors, full counts, resets
and ceilings without a neural model call:

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  MKL_NUM_THREADS=2 PYTHONPATH=. \
  python scripts/summarize_white_target_preflight.py --verify
```

This pass permits preparing a separately registered complete matched comparison
under the [unchanged learning decision](white_target_ranking_decision.md). Keep
the original fresh initialization, full 13,766-source/41,298-update schedule,
five snapshots and saved ranked-control pixels. All seven readable groups,
credited ink, at most 11/252 false-ink whitespace outputs and at most 37 all-case
false blanks must pass before separately binding autonomous evaluation. The
earlier failed gate, frozen readings and rejected decoding rule remain intact.
Independent useful visual language and the full engineering goal remain open.

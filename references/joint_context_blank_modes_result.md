# The learned density often prefers empty writing

The separately registered probability probe is complete. It restores every one
of the final actor's 131 learned context/head tensors and recomputes the actual
supplied-history states for all 124 cases and 12,597 positions. It scores only
existing source, empty and saved-output images. There are no optimizer updates,
new generated glyphs or autonomous answers. The failed physical gate and all
frozen readings remain unchanged.

| Book stratum | False blanks | Source beats blank by ≥0.001 nat | Required | Certified empty global modes |
|---|---:|---:|---:|---:|
| Development, fixed | 93 | 4 | 47 | 21 |
| Training, fixed | 307 | 26 | 154 | 43 |
| Training, proportional | 289 | 27 | 145 | 38 |

All three prospective source-preference thresholds fail. Only 57 of the 711
false blanks across all seven groups prefer their source image; all 57 are in
the book strata above. The first-nonzero-prefix bound certifies the all-white
image as the global density mode for 108 false blanks, including 102 book
positions. The remaining loose bounds do not establish a better mode. A
source/reference mismatch does not by itself classify every alternative as bad
language, but an empty tile at a nonempty source position is the measured failure.

All saved physical argmax choices reproduce with **zero disagreements** across
the cohort. The largest source-mask mean NLL difference from original
qualification is 6.222e-9 nat, below the registered 2e-5 tolerance. These checks
support the actual state/weight reproduction; no stale head-only cache was
substituted. The global-mode certificate follows from a normalized conditional
probability bound, not from enumerating every possible glyph. See the
[registered protocol](joint_context_blank_modes_protocol.md) for its assumptions
and exact formula.

The registered decoding-only eligibility rule is rejected. Exact density
maximization cannot choose a nonempty image at the 108 certified positions.
Other search procedures may return different writing, but this experiment does
not support search alone as the next repair. The selected
[next objective comparison](global_glyph_ranking_decision.md) changes learning
while keeping the pixel representation and greedy decoder fixed. It remains
untrained and requires a fresh real-source gradient/memory preflight and exact
execution binding before a new fit.

Worker execution reports 394.445 seconds, including 40.038 seconds of pressure
waiting, at 18.033 GiB peak allocated VRAM and 52.207 GiB minimum available host
RAM. The single worker exited normally; there was no automatic retry. The full
model remains 7,553,448,370 parameters with 7,522,463,984 inherited parameters.
This is not a compact foundation, matched-quality efficiency result or language
success. The original 28 blank-flag review inconsistencies, including six credited
empty images, are preserved and disclosed by the preceding physical result.

The [completed export](evidence/joint_context_blank_modes_result_20261114.json)
binds all 124 probability arrays, source registration, worker report and attempt
receipts. Its independent CPU recount rechecks original pixel blank flags,
case likelihood reproduction, all group totals and the fixed decision. It makes
no neural call and does not repair the review:

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  MKL_NUM_THREADS=2 PYTHONPATH=. \
  python scripts/summarize_joint_context_blank_modes.py --verify
```

The Nature-style PDFs still predate this measurement and the completed
context-gradient failure. Their next bounded results revision must show the
loss/output disconnection, certified modes, original samples and review limits
in editable native figures. Complete visual language remains unproved.

# A blank-image comparison does not repair the whitespace regression

The separately registered diagnostic is complete on the final ranked actor:
all 124 supplied-history cases, all 12,597 positions and all 131 actual learned
tensors. The fixed two-image rule fails its prospective eligibility conditions.
It would remove only three of the 25 false-ink outputs, leaving **22/252**
source-white positions incorrectly inked against a limit of eleven. It would
also erase one additional nonempty-source position, increasing false blanks
from 37 to 38. No output image, frozen reading or original gate was changed.

The rule compares complete log probabilities of the original emitted image
and a blank image. It selects blank only when its score exceeds the original
image's by more than 0.001 nat. Source labels assess costs after the comparison;
they do not select the output. The full cohort, including all nonempty positions,
prevents interpreting three corrected spaces alone as a successful decoder.

| Group | Original false ink | Fixed by blank comparison | Remaining false ink | Added false blanks |
|---|---:|---:|---:|---:|
| Development book, fixed | 5 | 0 | 5 | 0 |
| Development QA, fixed | 1 | 0 | 1 | 0 |
| Development QA, proportional | 1 | 1 | 0 | 0 |
| Training book, fixed | 9 | 1 | 8 | 0 |
| Training book, proportional | 9 | 1 | 8 | 1 |
| Training QA, fixed | 0 | 0 | 0 | 0 |
| Training QA, proportional | 0 | 0 | 0 | 0 |
| Total | **25** | **3** | **22** | **1** |

No selected readable credit would be removed, so all seven predicted readable
fractions remain unchanged. The rule nevertheless fails both whitespace and
no-added-false-blank conditions. These are counterfactual counts, not actual
new physical qualification, a revised gate or an autonomous language result.
They reject this particular fixed decoding change, not every possible search.

All saved physical-mask argmax choices reproduce with zero disagreements.
Maximum source-mask mean NLL drift is 9.154e-9 nat, below the fixed 2e-5
tolerance. Actual target-preceding states are recomputed from supplied source
images; no old cached continuation state is substituted. Every model parameter
is frozen, with mutation checks during execution. The first-nonzero bound
certifies 238 blank global modes in the measured arithmetic, of which 218 are
true whitespace and none are among the 25 false-ink outputs. Its floating-point
and loose-bound limitations remain those in the
[registered decision](global_glyph_ranking_whitespace_decision.md).

The single worker takes 261.322 seconds, with no resource waiting, at 18.033 GiB
peak allocated VRAM and 54.718 GiB minimum available host RAM. The outer process
takes 262.653 seconds and exits zero. No retry, optimizer update, new glyph,
answer or review is performed. GPU0 is released after completion. The deployed
7,553,448,370-parameter model and inherited pretraining remain separate from
this diagnostic's cost.

The [completed CPU export](evidence/ranked_writer_whitespace_result_20261115.json)
binds the registration, final snapshot, all probability arrays, source/output
pixels, frozen selected readings and execution receipts. An independent CPU
recount reproduces every group and the fixed decision without a neural call:

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  MKL_NUM_THREADS=2 PYTHONPATH=. \
  python scripts/summarize_ranked_writer_whitespace.py --verify
```

The selected [next learning comparison](white_target_ranking_decision.md)
addresses true-whitespace supervision while retaining the nonempty ranking
term and the original decoder. It is untrained and requires a new gradient
proof, real-source reset preflight and exact registration. Do not repeat this
completed diagnostic or launch the failed actor's autonomous evaluator. The
engineering goal and the manuscript's distinction between glyph quality and
complete language remain unchanged.

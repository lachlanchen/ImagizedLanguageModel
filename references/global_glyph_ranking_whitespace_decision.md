# Resolve the ranked actor's whitespace failure before another fit

The [complete ranking result](global_glyph_ranking_physical_result.md) passes all
seven conditional-readability groups and credited-ink checks, but produces ink
at 25/252 true-whitespace positions, above the fixed limit of 11. Its 37 false
blanks are fewer than the control's 711. This is a new measured failure of a new
trained density. The completed probability studies of the earlier false-blank
actors remain terminal; their conclusions and frozen reviews are unchanged.

## One bounded diagnostic

Before another full fit, distinguish a density that prefers the emitted ink
from a greedy path that loses to the complete blank image. Restore every one of
the ranked actor's final 131 learned tensors and recompute actual source-history
states for the original 124 cases and all 12,597 positions. Freeze all parameters.
Use the original source, saved emitted pixels and complete blank image only.
There is no optimizer, new glyph, new answer, font, OCR or lexical output head.

For every position, measure `B(h) = log q(blank|h)` and
`E(h) = log q(saved_image|h)`, summing the normalized physical-mask log
probabilities over the complete image. Reproduce the saved greedy path with
zero argmax disagreements and source-mask mean NLL within 2e-5 nat of the
original qualification before interpreting densities. A mismatch makes the
probe inconclusive and requires restore diagnosis, not a new learning claim.

Also retain the existing first-nonzero-prefix upper bound on the probability of
any nonempty image. If `ell_i(a)` is mask log probability after an all-zero
prefix, then

```
U(h) = max_j [ sum_(i<j) ell_i(0) + max_(a!=0) ell_j(a) ].
```

The unscored suffix has log probability at most zero. Thus `B > U + 0.001` is
a sufficient blank-global-mode certificate, subject to floating-point accuracy.
Loose bounds do not prove a better nonempty mode exists.

Evaluate one fixed counterfactual in arithmetic only: choose the blank if
`B > E + 0.001` nat, otherwise retain the exact saved image. Do not change image
files, tune a threshold, use sources to choose output, or revise the first gate.
Assess all 252 white and 12,345 nonempty source positions, not only affected
whitespace cases. Count by all seven groups. This prospective decoding comparison
is eligible only if **all** conditions hold:

1. At most 11/252 source-white positions remain falsely inked.
2. No additional source-nonempty position becomes blank: at most the existing 37.
3. Every currently credited candidate at a nonempty source position is retained.
4. After removing any selected credits changed to empty, each of the seven
   predicted readable fractions remains at least 90%.

These counterfactual counts are diagnostic eligibility, not new physical review
or a language score. If all pass, separately register a fixed-weight,
two-candidate physical-decoding comparison with the same control and 124 cases.
Actual new qualification, all-board review and a new freeze must pass the full
physical rule before any autonomous language binding can be considered. The
current actor's failed gate remains immutable.

## Actions for rejection and resources

If the fixed rule fails, reject this bounded decoding change. Prepare one
whitespace-learning objective comparison while preserving the current density,
greedy decoder and full source schedule. A candidate for separate preflight is
a white-target penalty `softplus(1 + U(h) - B(h))/128`, keeping the current
nonempty source-versus-blank term. This would train a sufficient condition for
blank global optimality. The bound can be loose, concentrate excessive blank
mass and damage nonempty forms; this is an untested proposal. It requires exact
CPU gradients and a real complete-source preflight containing actual whitespace,
with all trial updates reset, before any full-corpus registration. Do not launch
a fit from this decision alone.

The density probe requires its own implementation, source/input/final-weight and
runtime hashes, committed and pushed before neural execution. One GPU0 owner
under the project mutex; two CPU threads; at most 900 seconds total, including
120 seconds of pressure waiting; 48 GiB available host RAM and 20 GiB free GPU
at entry; retain 24 GiB available host RAM, 48 GiB disk and at most 22 GiB Torch
allocated VRAM. Limit diagnostic output to 256 MiB. Preserve partial costs and
failures; no automatic retry, duplicate worker or foreign-process cleanup.

The [diagnostic implementation](../scripts/analyze_ranked_writer_whitespace.py)
reuses the completed physical path scorer and restores through the ranking
checkpoint inventory. New CPU checks cover strict-margin choice, whitespace
and nonempty erasure costs, all seven prospective readability conditions,
missing-binding rejection before Torch import and first-attempt preservation.
These checks do not measure the inherited body's neural behavior. The
[independent CPU export](../scripts/summarize_ranked_writer_whitespace.py)
recounts every saved probability vector against original source/output arrays
and frozen selected readings without loading a neural model.

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  MKL_NUM_THREADS=2 PYTHONPATH=. \
  python scripts/analyze_ranked_writer_whitespace.py register
# Commit and push the exact public registration before GPU execution.
```

The external timeout sends TERM at 880 seconds and allows twenty seconds before
KILL, bounding setup and a stalled constructor as well as monitored forwards.
Keep the first console, process identity, start/end receipts and partial vectors.

Keep complete prompt dependence, paired questions, erasures, coherent books,
natural stopping, untouched confirmation, independent execution, historical
forms and whole-model matched-quality costs as separate outstanding criteria.
Neither a blank-density certificate nor readable conditional glyphs fulfils
the engineering goal. Preserve the full goal as active.

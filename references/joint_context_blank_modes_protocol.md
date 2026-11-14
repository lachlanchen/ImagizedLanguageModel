# Probability test on the final learned-context writer

The completed context-gradient fit lowers source-mask NLL but increases book
false blanks from 324 to 689. The frozen physical gate fails, and six empty
candidate/control images were incorrectly credited during review. The first
readings and gate remain unchanged. This probe follows the
[fixed failure decision](joint_context_failure_decision.md); it is neither a
new fit nor permission to run the failed actor's autonomous evaluation.

## One question and unchanged evidence

Does the new conditional density favor the observed empty images, or does its
greedy physical-mask path miss better source forms? Recompute the **actual new
context states**, restoring all 131 tensors from final update 41,298. Old
head-only cached states cannot answer this question. Use all original 124 cases
and 12,597 complete positions, including legitimate whitespace and all 711 false
blanks. The original control, source/output arrays, reports, frozen review and
failed gate are bound by hash; nothing in this probe changes them.

The probe scores three existing images per position: the source, the all-white
tile and the saved candidate output. Each complete glyph probability is the
sum of its 128 conditional log probabilities for eight-pixel masks. The alphabet
enumerates physical bit patterns, with no character IDs, font table or lexical
output head. No new glyph is sampled or written, and no optimizer is created.
Reference history is supplied explicitly; these are conditional image-density
measurements, not autonomous language scores.

For the all-zero physical history, let `ell[j,k]` be the log probability of
mask `k` at physical position `j`. An upper bound on every nonempty completion is

```
U = max_j (sum_{i<j} ell[i,0] + max_{k!=0} ell[j,k]).
```

Every nonempty image has a first nonzero mask, and later log probabilities are
at most zero. Thus `U < log p(blank) - 0.001` certifies an empty global mode
with the stated numerical margin. A loose bound establishes nothing about a
better mode. Source preference is a separate observed complete-image comparison.
The old head-only result is retained; it is not rerun or substituted for this
different learned-context actor.

## Fixed validity and decision rules

- Match prompt geometry to the original qualification and require mean source
  mask NLL agreement within 0.00002 nat. Restore every actual context/head tensor
  by name even after inference freezing, and verify its tensor inventory.
- All saved blank paths must still choose zero at every conditional argmax.
  Score every saved path and retain its parallel argmax disagreement count.
  Abort if its worst single-mask log-probability deficit exceeds 0.0001 nat.
  This tolerance concerns numerical reproduction, not language quality.
- Legitimate source whitespace remains in the complete cohort. A false blank is
  a saved empty output at a nonempty source position. Require exactly the bound
  711 false blanks, with book strata of 93, 307 and 289. Source/blank identical
  paths must have identical probability within 0.00001 nat.
- A source advantage of at least 0.001 nat for a complete glyph qualifies as
  source preference. Require at least 47/93 development-fixed, 154/307
  training-fixed and 145/289 training-proportional book false blanks. All three
  strata must pass to make a separately registered whole-glyph decoding
  comparison eligible. Otherwise redirect the conditional representation or
  objective. No pooled majority, intermediate checkpoint, forced ink, extra
  epoch, repaired reading or changed gate can substitute.

The report retains per-position probabilities, bounds and original blank flags,
all case hashes, whole-model parameters, the 131-tensor restored inventory and
resource accounting. A failed validity check preserves partial evidence and
does not select an architectural explanation. Neither decision proves usable
language, compactness, historical-form generalization or matched-quality
efficiency.

## Execution and resource contract

Commit implementation, math checks and this protocol. CPU-only registration then
binds sources, actual final checkpoint and all inputs; commit and push that exact
registration before the single neural execution. Registration and failed binding
admission do not import Torch. Use the existing process-local NVIDIA wrapper and
deterministic CUBLAS setting. The original pipeline is terminal; do not restart
it. Check actual owners, tmux, ports, RAM and device memory before launch.

One GPU0 job holds the existing project mutex. The cumulative wall ceiling is
1,200 seconds, including imports, input validation, model construction and any
failed attempt. Pressure/admission waiting is limited to 180 seconds. Use two
CPU threads, at most 22 GiB allocated VRAM, 48 GiB available host RAM and 20 GiB
free GPU memory at admission, and retain 24 GiB host RAM and 48 GiB free disk.
The reused admission observer adds a 2-GiB host headroom before the final entry
check. No automatic retry is supported. Preserve original and partial files
before a separately justified manual recovery; an observation timeout never
authorizes another worker.
The outer process timeout sends TERM at 1,180 seconds and allows at most twenty
seconds before KILL, bounding setup or a stalled constructor as well as monitored
forwards. Preserve the console, signal trace and first attempt receipts.

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  PYTHONPATH=. python scripts/analyze_joint_context_blank_modes.py register
# Commit and push the resulting exact public registration before run.
```

The complete ILM objective remains active. Update the manuscript and native
figures with the completed context-gradient failure and this probe's eventual
measured outcome. Existing PDFs predate these results; no result is implied by
the existence of the probe implementation.

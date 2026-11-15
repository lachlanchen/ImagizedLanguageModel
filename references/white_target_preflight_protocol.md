# Complete-source preflight for white-target ranking

The [completed ranked-writer diagnostic](ranked_writer_whitespace_result.md)
rejects the fixed blank-versus-emitted-image rule. This preflight implements the
single additional white-target term in the [selected learning decision](white_target_ranking_decision.md).
Its purpose is gradient, complete-source execution, frozen-state and memory
validation. It generates no glyphs or autonomous answers and measures no language.

The new module keeps source-image likelihood, the current nonempty ranking,
balanced stop loss, full model, 131 trainables and greedy decoder. At genuinely
white source images it adds `softplus(1 + U - B)/128`, where `B` is complete blank
log probability and `U` the first-nonzero upper bound. White targets' source-path
logits already provide the zero-prefix distribution, so their bound requires
no additional writer pass. All loss terms are averaged over complete targets.
The source-to-blank term remains zero at white targets; the new term remains
zero at nonempty targets. It is neither a blank ban nor a lexical output head.

Eight CPU checks compare the real physical writer's complete gradients with an
independent unsplit formula for mixed, all-white, all-nonempty and non-divisible
batches; zero added strength reproduces existing ranking gradients exactly.
An enumerable four-position, three-mask autoregressive density verifies the
upper bound against every nonempty path, including tied probabilities. These
fixtures do not establish full-body neural behavior or language generalization.
A ninth check rejects a missing preflight binding before neural-runtime import
or output creation.

## Six fixed complete cases

Retain the original complete five-glyph QA source and maximal 513-glyph book,
both fixed-cell and proportional layouts. Add both layouts of the first
original qualification training-book source containing white and nonempty
targets in that unchanged qualification order: source index 4,447, 314 target
glyphs, two white targets. Bind the actual source record, prompt, target and
original qualification arrays before neural execution. Use full source history,
with no crop, inserted whitespace or cached predictor replacement.

Initialize the identical original fresh head and point-final context. For both
QA layouts compare every actual context/head gradient with ordinary unsplit
autograd: relative L2 <=.01 or maximum absolute <=1e-5, and maximum absolute
<=1e-5+.01 times the reference maximum. Require loss agreement within2e-5.
The QA sources do not contain whitespace; new-branch autograd is separately
covered by CPU fixtures and actual complete mixed-white book execution.

For all four book cases, take one actual AdamW step at4e-7 with weight decay.01,
clip one. Require all131 finite, nonzero gradients, all131 counters exactly one,
and changed elements in all131 tensors. The new white term must be positive at
the two true-white targets and zero in the old nonwhite cases. Preserve frozen
native prompt/body, feedback, stop and four affine buffers. Reset every update
to the exact initial tensor inventory before the next case and at completion.
This is not a trained checkpoint; no preflight update may enter a future fit.

## Resource and outcome contract

Commit implementation and this protocol. CPU-only registration binds exact
sources, inputs, runtime and initial tensors; commit and push that registration
before the single GPU execution. Keep all101 earlier bound sources unchanged.
One GPU0 owner under the existing project mutex; two CPU threads;1,200 seconds
including180 seconds of pressure waiting, with external TERM at1,180 seconds
and twenty seconds of kill grace. Admit at48 GiB available host RAM and20 GiB
free GPU; retain24 GiB host RAM,48 GiB disk and at most22 GiB Torch-allocated
VRAM. Reuse shared weights and the existing per-process NVIDIA wrapper. Check
actual owned processes, tmux and ports before launch; preserve foreign projects.
No automatic retry; preserve partial output, console, identities and costs.

All six cases, all gradient/counter/reset checks and resource ceilings must pass.
A pass permits preparing a separately registered complete matched comparison;
it does not launch one or admit language evaluation. A failure preserves the
first attempt and redirects its actual mathematical, implementation or resource
problem before any separately justified recovery. Neither outcome changes a
completed trial, frozen reading, old gate or full engineering acceptance rule.

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  MKL_NUM_THREADS=2 PYTHONPATH=. \
  python scripts/preflight_white_target_ranking.py register
# Commit and push the exact registration before the GPU run.
```

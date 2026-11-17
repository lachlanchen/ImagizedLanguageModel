# Conditional image learning and competing objective gradients

The [full-context comparison](trainable_visual_context_result_20261117.md) is
complete and rejected. Both final actors produce zero readable complete normal
QA and no credited coherent complete long-book continuation. Treatment learns
to stop on all twelve normal long books without producing usable writing.
This diagnostic investigates one possible mechanism: interference between
conditional-flow learning and the other five equally weighted objectives.
It does not repeat autonomous generation, old first-span sampling or the earlier
update/reset checks, and it makes no optimizer updates.

## Fixed actors and complete training records

Load the final learned and frozen actors at update 8,172, including each retained
training-only teacher projection and ordinary AdamW moments. Match their complete
deployment tensors to the actual final files; bind both latest training files,
final deployment files and original fit audits. Keep the original source model,
codec, reader, writer, BF16 forward arithmetic, FP32 parameters and six losses.
The frozen context remains frozen. The original checkpoints are read-only.

Use the three previously selected complete training records in their existing
order: corpus index 458, a 513-glyph book sequence with 509 native units; index
4,762, an 846-glyph QA sequence with 740 native units; and index 9,420, the longest
native-unit training record, with 1,042 glyphs and 961 native units. They were
selected before the full comparison. Retain the whole source images, prompt,
answer, terminal and saved teacher states. Do not substitute shorter crops or
development examples. This selected-record diagnostic cannot establish a
corpus-wide mechanism or language generalization.

Bind the exact source shards, teacher shards, source preparation, model and
diagnostic code, package versions, CPU arithmetic evidence and this protocol
before execution. Commit and push the registration and sources. Use one serial
owner for learned context, then frozen context; no automatic retry.

## Gradient accounting

For each actor and complete record, reset seed 202611172 before each of three
forward/backward passes. Require all six scalar losses to match exactly across
these RNG replays. Measure the gradient of conditional flow g_F, the gradient of
the sum of the other five losses g_A, and the direct gradient g_T of their full
sum. Do not assume exact floating-point equality between g_T and g_F+g_A.
Record each pass's complete losses, nonfinite failures, gradient presence,
unclipped global norm and original clip scalar.

Report full-vector norms, inner products and cosines separately for the reader,
context, writer and training-only projection, and over all optimized parameters.
Absent gradients contribute zero when paired with an existing gradient; their
parameter dimensions must not be omitted from the other vector's norm. A zero
norm has undefined cosine and is reported as null. Frozen contextual parameters
have no optimizer participation and are not presented as measured zero gradients.

Retain only g_F on the host. Stream auxiliary and total gradient geometry rather
than keeping six full copies in device memory. Accumulate geometry in FP64.
The direct total pass uses the original global L2 clipping rule with maximum one.
For a flow-only direction, use that pass's own clip scalar over all parameters
with present flow gradients, including the original 1e-6 clipping denominator.

Gradient cosine alone is not an AdamW interference test. From the exact retained
moments m, v and step k, the clipped new gradient g gives

\[
 m'=.9m+.1g,\quad v'=.999v+.001g^2,\qquad
 w'=(1-\eta\lambda)w-
 \frac{\eta}{1-.9^{k+1}}
 \frac{m'}{\sqrt{v'}/\sqrt{1-.999^{k+1}}+10^{-8}}.
\]

Use the retained group's learning rate, betas, epsilon and weight decay rather
than replacing them with a gradient-descent surrogate. Compute the actual FP32
displacement d=w'-w on detached chunk copies, preserving the original arithmetic
order. A parameter with an absent gradient is skipped, including its moment and
decay update. Do not write w', m' or v' back to the actor or checkpoint.
Compare g_F dotted with the combined direction and the flow-only direction, both
starting from the same retained moment history. These are counterfactual new
batch directions, not replays of historical updates. First-order inner products
do not measure actual post-update loss or demonstrate language learning.

The CPU arithmetic check compares this copied calculation with installed PyTorch
AdamW at counters zero and 8,172, checks unchanged inputs and absent gradients,
and demonstrates why gradient alignment alone cannot establish descent with
retained momentum. Its four tests are numerical checks, not student-model results.

## Prospective decision rule

Interference is supported on these prescribed treatment records only if the book
and at least one QA record have g_F·d_combined >= 0 while g_F·d_flow_only < -1e-12.
Report all continuous values and the control results, including outcomes that do
not satisfy the rule. The rule concerns first-order directions on these records.
It cannot prove that loss interference caused every malformed generated glyph.

A supported result permits preparation of one separately registered objective
intervention and unchanged control, with real conditional-writing update/reset
preflight before any full fit. If unsupported, do not assume interference, tune
weights against development outputs, or rerun unchanged training. Preserve the
measured limitation before choosing the next bounded change. Execution failure
ends this diagnostic with partial evidence retained; no automatic retry or source
truncation follows. Neither outcome admits a full fit or qualifies an ILM.

## Resource and preservation limits

Use physical GPU0 and the existing project mutex. Require 48 GiB available host
RAM, 20 GiB free GPU memory, low memory pressure and 48 GiB free disk at admission;
retain at least 24 GiB host RAM and at most 22 GiB allocated VRAM. Use two CPU
threads and 262,144-element diagnostic chunks. The total monitored deadline is
1,200 seconds, including binding validation after the committed-source check,
both model loads, all six complete-record measurements and preservation checks.
At most 300 seconds of pressure waiting fit within that deadline. An external
1,260-second timeout with 30 seconds of termination grace bounds a hung process.
Protect foreign GPU1 services and other projects.

The first complete-record measurement is also the actual full-record support
check for its actor; no preliminary optimizer step or retained trial model is
created. Report resources and partial failures rather than silently reducing
record lengths. Compare exact model and teacher-projection tensor hashes before
and after each actor's complete three-record measurement. No generation, OCR,
reward judge, DRCD record or development target enters this execution.

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=. OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  python scripts/diagnose_visual_context_gradients.py register
# Commit and push the binding before the single serial diagnostic is launched.
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
  CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONPATH=. OMP_NUM_THREADS=2 \
  OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  timeout --signal=TERM --kill-after=30s 1260s \
  python scripts/diagnose_visual_context_gradients.py run
```

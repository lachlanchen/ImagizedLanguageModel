# Real-source proof for the whole-glyph ranking objective

This preflight tests the new objective in
[the ranking decision](global_glyph_ranking_decision.md). It is not a repetition
of the completed likelihood-only preflight and does not evaluate language.
All earlier sources, outputs and grades remain immutable.

CPU-only registration binds the original fresh writer and point-final context
initialization, all 131 expected initial tensors, shared normalization, corpus,
new implementation, runtime and exact source images. Use the original first
odd-length complete QA target with 5--63 glyphs (five glyphs) and the first
maximal complete book target (513 glyphs), each in fixed and proportional
layouts. Do not shorten answers or replace the actual body with a small model.

For each QA layout, independently differentiate the unsplit source NLL,
whole-glyph ranking and unchanged stop loss through the actual body. Recompute
its continuation graph and compare the new microbatched vector-Jacobian result
for every one of the 131 actual learned tensors. Require finite, nonzero
gradients. The relative-L2 tolerance is 0.01 or absolute tolerance 1e-5, together
with maximum error at most 1e-5 plus 0.01 times the largest reference magnitude.
Complete loss agreement must be within 2e-5. These are numerical execution
tolerances, not readability or language thresholds.

For each 513-glyph book, run the new exact microbatch accumulation at 64 glyphs
and take one actual clipped AdamW step at 4e-7 learning rate and 0.01 decay.
Every actual learned tensor must have finite, nonzero gradients, a changed
element and optimizer counter one. Preserve frozen native parameters, prompt
memory/feedback/stop weights and all four affine buffers. Reset every update;
no preflight state is carried into a future fit. QA uses microbatches of two,
including a non-divisible last batch. The scalar objective has ranking strength
one and whole-image margin one nat.

One GPU0 owner uses the existing project mutex and process-local NVIDIA wrapper.
Require 48 GiB available host RAM, low pressure and 20 GiB free GPU memory at
entry; retain 24 GiB host RAM and 48 GiB free disk, with at most 22 GiB allocated
VRAM and two CPU threads. The 900-second cumulative ceiling includes imports,
validation and construction; pressure waiting is limited to 180 seconds. An
outer timeout sends TERM at 880 seconds, with at most twenty seconds to KILL.
There is no automatic retry. Preserve any partial evidence before deciding on
manual recovery.

Commit sources and CPU checks, then create and commit/push the exact public
registration before neural execution. Only all four successful cases, both
actual reset book updates and unchanged frozen state permit preparation of a
separately registered full-corpus comparison. A successful preflight alone
establishes neither physical qualification nor useful language or efficiency.

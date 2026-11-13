# Complete image likelihood with a trainable continuation state

This controlled comparison addresses the empty conditional modes measured in
the [completed head-only trial](joint_writer_blank_density_result.md). Its
physical writer produced many clear forms but failed two of seven readability
groups. The new trial changes gradient reach: the continuation residuals and
initial query learn through the same proper complete-image likelihood as the
image writer. Prompt memory, inherited native weights, image feedback, stop
parameters and all four affine buffers remain frozen. This is conditional
maximum likelihood and the chain rule; no physical theory or language advantage
is inferred from the construction.

The real-source preflight is complete. Both five-glyph QA layouts match unsplit
autograd across all 131 trainable tensors; the largest relative gradient error
is 2.408e-6. Both complete 513-glyph book layouts produce finite gradients and
actual optimizer steps, with 17.795 GiB peak allocated VRAM. All updates were
reset. One original SIGTERM, with unknown sender, is preserved; one explicit
traced recovery completed without a termination signal. Cumulative preflight
wall time has a conservative 241.823-second upper bound. These measurements
establish a computational precondition, not complete language performance.

## Matched initialization and complete supervision

Use the exact original point-final step-41,298 continuation parameters and the
identical fresh joint-image-head draw used by the completed head-only control.
Reuse its training-only condition normalization and original schedule by hash.
Do not load the fitted head-only writer or carry preflight updates. Register
every initial trainable tensor independently on CPU before constructing the
actual model. The deployed model has 7,553,448,370 unique parameters, including
7,522,463,984 inherited visual/contextual parameters. The 5,005,760 trainables
comprise 2,363,392 continuation parameters and 2,642,368 image-head parameters;
this is not a lightweight foundation model.

Every one of 13,766 original eligible complete sources receives three updates:
one fixed-cell presentation and two proportional presentations. Retain all
41,298 updates and 6,108,741 target glyph exposures. Recompute prompt memory and
the differentiable, target-preceding continuation graph for each update. No
cached predictor state may replace that graph. There are 128 physical masks
per 32-by-32 glyph. Minimize their mean proper negative log likelihood over
the entire target sequence, plus 0.1 times the existing balanced stop loss.
Accumulate head microbatches of 64 glyphs through a detached state leaf, then
apply its accumulated gradient once to the actual continuation graph. The
preflight verifies this exact vector-Jacobian factorization.

Use FP32 AdamW over all 131 trainable tensors: learning rate 2e-4, 500-step
warm-up, cosine decay to 2e-5, weight decay 0.01, betas (0.9, 0.999), epsilon
1e-8 and gradient clipping at one. Keep snapshots at updates 0, 10,324, 20,649,
30,973 and 41,298. Save resumable current/previous checkpoints every 250 updates.
Restore update three and actually replay updates four and five. Retain both
saved step-five outcomes and require exact trainable, optimizer, RNG, loss and
gradient agreement. Count these two extra executed updates separately; they
do not add retained exposure. An independent CPU audit authenticates the full
source/target corpus, event hash chain, all five snapshots, actual Adam states
and both saved replay outcomes. Frozen-state checks are execution assertions,
not a second neural replay by the auditor.

## Serial execution and fixed decisions

One durable controller runs fit, independent CPU audit and qualification in
sequence, then exits awaiting opaque review. There is no automatic retry or
generation after a failed stage. Preserve failures, partial outputs, costs and
original sources before any separately justified manual recovery. Cumulative
limits include setup, imports and prior failed attempts: fit 108,000 seconds
with at most 7,200 seconds of pressure waiting; qualification 3,600 seconds
with at most 300 seconds waiting; CPU audit 900 seconds and 8 GiB peak RSS.
Use only physical GPU0 and two CPU threads. Entry requires 48 GiB available
host memory and 20 GiB free device memory; retain 24 GiB host memory, at most
22 GiB allocated VRAM and 48 GiB free disk throughout. Reuse the existing
verified process-local NVIDIA compatibility wrapper without changing system
libraries, network services or another project's GPU.

The physical cohort is the same 124 complete cases and selected positions,
including whitespace. Recompute their states using this trial's learned
continuation and supplied reference-image history. The unchanged control pixels
come from the original fitted joint head. Keep every candidate/control/source
image. Opaque review uses the unchanged physical workflow: all boards must
actually be viewed and readings frozen before the owner mapping is opened.
Every one of the seven candidate groups must meet the original 90% readable
nonblank threshold. Canonical bitmap similarity is a separate diagnostic.
Failure stops this intervention before autonomous evaluation; its language
remains unmeasured, rather than being assigned zero.

A physical pass permits the complete 708-stream autonomous comparison already
specified by `evaluate_joint_writer_language.plan`: 660 core-cap streams at
all five checkpoints and 48 initial/final long-book streams. Restore all 131
context/head tensors for each actual snapshot; the head-only evaluator's fixed
continuation construction must not be reused. Bind the execution implementation
and complete post-gate snapshots/review artifacts in a separate committed
registration before generation. Use only emitted-pixel feedback, a frozen stop
head at probability 0.5 and the original 128/640 caps, without reference answers
or lengths. Freeze all complete output readings before new reference scoring.
Retain paired QA, all four QA erasures, prefix erasure and all earlier required
successes. The pilot remains at least 12/32 exact, 4/16 both-correct pairs,
24/32 readable proportional answers and 4/12 coherent complete long books;
all eight proportional and two fixed successes must remain, each QA-erasure
arm may have at most two exact answers, and the dependence gap must be at
least eight. A pilot pass still requires untouched confirmation, standalone
independence, whole-model cost comparison and historical-form validation.

The Nature-style manuscript and editable figures must report completed controls,
the current trial's measured status and its inherited model cost. No autonomous
score, compactness claim or matched-quality efficiency claim follows from this
registration. The complete ILM goal stays active.

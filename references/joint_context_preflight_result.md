# Real-source context-gradient preflight passes

The [registered computational preflight](joint_context_preflight_protocol.md)
passes all four complete-source checks. In both prompt layouts, exact
microbatched image-likelihood backpropagation matches ordinary autograd across
all 131 trainable context/head tensors on a five-glyph QA target. The maximum
relative gradient error is 2.408e-6 and the maximum absolute error is 3.800e-7.
The real causal perturbation check also passes: changing a supplied target
image leaves its own and earlier predictors unchanged and changes later states.

Both complete 513-glyph book layouts produce finite, nonzero gradients for every
trainable tensor and complete an actual AdamW step. Peak allocated VRAM is
17.795 GiB; the minimum observed available host memory is 65.647 GiB. Frozen
parameters and all four affine buffers remain unchanged. All trainables are
restored exactly after each case. The two preflight updates do not enter the
full fit. No answer image is generated or graded in this check.

The original worker exited with SIGTERM while admitting, before any recorded
case or optimizer update. Its evidence is preserved; the sender and cause are
unknown. One separately recorded recovery used the existing process-local
NVIDIA compatibility wrapper and signal tracing. It completed in 86.823 seconds;
including a conservative 155-second debit for the original attempt, cumulative
wall time has a 241.823-second upper bound. The original attempt's peak memory
was not measured. A driver-library mismatch was observed, but no causal link
to the signal is established. See the [recovery record](joint_context_preflight_recovery.md).

The [public result](evidence/joint_context_preflight_result_20261114.json) binds
the original measurements. The actual whole model has 7,553,448,370 unique
parameters, of which 5,005,760 are trainable. This preflight supports proceeding
to the separately registered [complete context-gradient comparison](joint_context_learning_protocol.md).
It does not establish complete language, a lightweight foundation, matched
efficiency or historical-form generalization. Those requirements remain open.

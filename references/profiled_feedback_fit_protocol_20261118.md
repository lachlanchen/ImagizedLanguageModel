# Fixed-support variable-projection head comparison

Execute only after the separately registered
[complete-record preflight](profiled_feedback_preflight_protocol_20261118.md)
passes. This protocol is fixed before that execution; its later registration
must bind the actual passing receipt. A failed preflight does not admit this fit.

Retain all 33,941 source/slot rows and their original complete-corpus masses.
Load only the prescribed nonlinear ridge head, fixed query cache and BF16 source
embedding targets. Reproduce the legacy BF16 error within 1e-6, then separately
measure the same stored weights under explicit FP32 feedback execution.

Construct the common FP32 feature baseline and solve its output once with the
original ridge rule. Freeze lambda to that baseline value for the whole
comparison. Save this precision-matched fixed-feature control. Clone it for
the learner. Only expansion weight and bias learn; LayerNorm, raw queries,
source mean/scale, decoder, grouping, stopping and contextual body stay fixed.
The output coefficients are globally refit with the fixed ridge at every trial.

Perform exactly twelve accepted outer steps. Accumulate the partial feature
gradient on every row with its corpus mass before one step; no stochastic
minibatch step or frequency clipping substitutes for the full gradient.
Use the preflight's two-percent relative proposal, six-trial halving search and
Armijo constant 1e-4. Retain each proposed objective, actual error, step size,
normal-equation residual and acceptance. A failed finite search is terminal:
preserve the last accepted state, label incomplete execution and do not change
the step rule or select another checkpoint. There is no extra epoch, seed,
source, regularization, slot residual or hidden-layer-width change.

Report the old BF16 measurement, same-weight FP32 measurement, newly solved
FP32 control and final feature learner separately. Precision changes are common
to control and treatment and are not a feature-learning gain. Distinguish the
ideal profiled objective from actual FP32 stored-weight execution.

The reader support rule remains final actual native error <=.10, at least .10
below the common fixed-feature control, and <=.10 on every complete training
record. A pass permits only preparation and registration of live-image
conditional-writing qualification. It is not a language score or authorization
for a full language fit. On failure, preserve final/partial heads and redirect
the measured limitation; do not add unchanged steps or relax the gate.

No development/DRCD fitting, teacher inference, answer generation, font render,
lookup catalogue or lexical output head is introduced. Full source provenance
and inherited pretraining stay disclosed. Whole-model language, historical-form,
independence and matched-quality efficiency conditions remain unchanged.

Bind code, packages, source weights, cache, complete support and passing
preflight before launch, and commit/push. Use one serial GPU0 owner and the
existing project mutex, two CPU threads, 48-GiB host/20-GiB free GPU/48-GiB disk
entry, low PSI, 24-GiB runtime host reserve, 8-GiB allocated GPU cap, 1,800-second
monitored deadline and 300-second pressure wait. External deadline is 1,860
seconds with thirty-second cleanup grace. Preserve failures without retries;
terminate the exact completed owner and leave foreign services untouched.

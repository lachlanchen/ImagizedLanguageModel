# Direct readout fitting halves nonlinear alignment error

One full-support weighted ridge refit lowers the nonlinear reader's actual
native-coordinate error from **.542496895 to .271038527**. Refitting the affine
control lowers its error from **.578608694 to .536556525**. All 33,941 source
rows and original masses are retained. This is a measured improvement in a
training-only coordinate task, not a language result. The nonlinear reader
still fails the fixed .10 error limit and all complete-record limits. No
language fit, autonomous evaluation or complete successor deployment is admitted.

| Fixed final feature map | Previous AdamW error | Refit actual BF16 error | Ideal matrix error | Book actual error | QA actual error | Long QA actual error |
|---|---:|---:|---:|---:|---:|---:|
| Affine | .578609 | .536557 | .536338 | .501214 | .457258 | .541118 |
| Nonlinear | .542497 | .271039 | .269287 | .278250 | .203538 | .223079 |

Actual errors use the original FP32 parameter storage, BF16 forward operations,
native mean/scale and relative-error definition. The ideal matrix errors use
the fixed BF16 features with FP64 coefficients before writing them to the
deployed arithmetic. Their gap is .000218 for the affine map and .001751 for
the nonlinear map. This finite gap does not establish precision invariance for
other inputs, layouts or a complete recurrent language model.

The [prospective protocol](feedback_ridge_refit_protocol_20261118.md) fixes one
ridge rule and one solve per feature family. Native mean/scale, LayerNorm,
nonlinear expansion, query cache, targets and source weights remain fixed.
Only final output coefficients change. FP64 Cholesky normal-equation residuals
are 6.95e-15 and 2.67e-14. This solves the specified *regularized* matrix
problem, not an unrestricted nonlinear global optimum. Regularization and
precision differ from the prior AdamW procedure, so the improvement does not
uniquely attribute that procedure's failure to one optimization mechanism.

All six whole-record preflights improve actual BF16 fitting and reset exactly.
For example, the nonlinear QA trial reaches 6.87e-5 source-coordinate error
after fitting all 740 native positions of that one record. This is fitting on
known targets; it does not establish held-out reading or question answering.
Both previous full-support baseline errors replay within 1e-6. Original head
files and feature tensors remain unchanged; refits are saved separately.

The single pipeline takes **27.085 seconds**, with **1.135 GiB** peak allocated
VRAM, no pressure waiting and **55.781 GiB** minimum available host RAM. Its
supervisor scope is 35.036 seconds. These are nested scopes. It performs no
optimizer step but does fit readout weights algebraically; calling it
inference-only would be incorrect. These costs exclude prior source extraction,
the twenty-epoch feature calibration and inherited pretraining. The complete
prospective model still has 1.475B parameters. No matched-quality language
efficiency gain follows from head-only fitting time.

The [public evidence](evidence/feedback_ridge_refit_result_20261118.json)
binds registration, both final heads, all preflight solves, normal-equation
checks, source-head preservation and terminal receipt. The worker, supervisor
and project tmux exit, the mutex is free and GPU0 returns to 16 MiB used.
Foreign services are unchanged. No teacher body, rendering, new question or
frozen transcription is used.

## What the saved queries permit

The [separate exact-query analysis](frozen_query_identifiability_result_20261118.md)
finds a .016774741 free-function error floor from identical query inputs with
different native targets. It rules out perfect alignment on those inputs, but
does not rule out the unchanged .10 gate. Neither this lower bound nor the
current ridge error proves what a finite nonlinear map can attain.

Do not repeat unchanged epochs, tune lambda or lower the gate. The next bounded
candidate is feature learning with a solved linear readout, using variable
projection. It must preserve full source weighting, qualify its gradient and
numerical execution, use a fixed ridge and retain an unchanged precision-matched
control. It is an untested direction. Useful complete language and historical
forms remain unproved.

# Full-support variable-projection reader result

The prescribed twelve-step fit completes but fails every reader-support
condition. On all 33,941 source/slot rows, actual native-coordinate error falls
from **.269500554** for the common FP32 fixed-feature control to **.228074131**
for the final learned features. The reduction is .041426423; the required
absolute error is at most .10 and the required reduction is at least .10.
No language training or autonomous evaluation is admitted by this result.

The output layer is solved globally at every trial. Only the expansion weight
and bias learn: 1,576,960 feature parameters. Lambda remains
3.222739942390491e-6 throughout. All twelve first proposals satisfy the fixed
Armijo condition; the profiled objective decreases from .298766637 to
.252951782. Final ideal error is .228074127 and actual stored-weight execution
is .228074131. The final normal-equation relative residual is 3.13e-14.
These results establish an executed, numerically consistent optimization
improvement, without establishing adequate reading or useful writing.

| Complete training record | Fixed-feature error | Final error |
|---|---:|---:|
| 513-glyph book, 509 native positions | .276759014 | .242945984 |
| 846-glyph QA, 740 native positions | .202161983 | .167086985 |
| 1,042-glyph QA, 961 native positions | .221935019 | .168436389 |

Every complete-record limit also fails. These records belong to the fitting
support; they do not measure held-out generalization. The earlier BF16 head
replays at .271038527, while the same weights in FP32 give .271042808. The
precision-matched solved baseline, rather than the legacy score, is the control
for feature learning. No intermediate checkpoint replaces step twelve.

The head-only run takes **75.782 monitored seconds**, with zero pressure wait,
**1.246 GiB** peak allocated GPU memory and 53.318 GiB minimum sampled available
host memory. The outer supervisor records 85.086 seconds, which includes the
monitored interval. These are not whole-model training costs. A prospective
actor with this reader has 1,475,346,710 parameters and inherited language
pretraining; it is not a demonstrated lightweight successor.

The [hash-bound evidence](evidence/profiled_feedback_fit_result_20261118.json)
preserves both heads, every step/trial record and the terminal execution receipt.
The [registration](evidence/profiled_feedback_fit_registration_20261118.json)
and [protocol](profiled_feedback_fit_protocol_20261118.md) remain unchanged.
The source checkpoints, query cache, physical aliases and original readings
are preserved. No optimizer owner remains active.

The next [read-only image-consequence diagnostic](reader_writing_consequence_protocol_20261118.md)
compares these two final heads inside the same frozen actor, with identical
sampling noise and supplied source-image history. It localizes the consequence
of the coordinate change; it does not overturn the failed reader gate or
qualify autonomous language. No additional unchanged fit is selected.

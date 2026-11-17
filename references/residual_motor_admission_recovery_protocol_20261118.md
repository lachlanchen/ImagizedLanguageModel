# Separate admission recovery for the unexecuted correction fit

The [original execution](evidence/residual_motor_admission_failure_20261118.json)
ends at the unchanged 48-GiB host-entry guard. It constructs no model, performs
no update and produces no image. Its directory contains only status and failure
records. Preserve that failure; it is not a model-quality measurement.

Use a separately registered wrapper and new output directory. Reuse the exact
committed trainer, physical sources, frozen codec, initialization, objective,
6,000-update schedule, unchanged control, probabilities, quality criteria and
resource ceilings. Only the registration/output locations and entry scheduling
change. No source weight, threshold or original run directory is modified.

Before importing the neural worker, observe at least 50 GiB available host,
low pressure, 20 GiB free GPU0 and 48 GiB free disk on two observations ten
seconds apart. This stricter scheduling reserve accommodates imports; the
worker's original 48-GiB entry / 24-GiB runtime guards still apply. Wait at most
300 seconds; do not start a worker if the observations fail. Start at most one
worker and never automatically retry any worker exception.

The 2,400-second total deadline includes scheduling wait and is enforced by
an independent alarm around the unchanged worker. Keep the external 2,460-second
timeout and 30-second termination grace. Record the scheduling observations and
wait cost separately. No other project's processes are stopped. The existing
project GPU mutex remains the admission authority; an idle observation is not
permission to duplicate another owner.

An entry-only abort does not count as model fitting or a quality failure. After
an actual terminal comparison, audit its complete images, causal traces, data
order and original criteria before deciding the next intervention. The full
language goal remains active and unproved.

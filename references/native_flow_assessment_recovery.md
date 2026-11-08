# Recovering a failed offline assessment without regenerating images

All eight fixed cohorts completed, producing 384 original images and their
eight OCR reports. The final CPU aggregation failed before producing its
report: a local checksum-loop variable named `digest` shadowed the imported
schedule-digest function. The error is recorded in
`assessment.attempt_001.log`. The two downstream diagnostics stopped before
any model call because their assessment prerequisite was incomplete.

The bounded recovery keeps all 19 registered training, assessment and probe
source files unchanged. It changes neither an image, reference, transcript,
metric, developmental gate nor model parameter. The recovery utility accepts
only the known assessor source hash, aliases its imported schedule function
to `schedule_digest`, and changes the corresponding call to that alias.
It saves the exact corrected source and both hashes in a separate attempt
directory. The original source and failure logs remain available. This is an
explicit corrected execution, not execution of the unmodified assessor.

Before execution, the utility requires all 17 preceding stages to have
completed with unchanged primary reports, all 384 image-index entries to be
present, the exact recorded Python exception, three terminal failed
coordinators, and no existing field or positive-sampling arm. It binds all
19 sources, three preparations, input indexes and output/score indexes.

The corrected CPU assessor must then authenticate every image against its
actual decoder tensor, validate the completed optimizer and schedule,
recompute the scores, and apply the original gates. The recovery records its
execution hash in that result and adds only the completed aggregation stage
to the stage ledger. The original assessment coordinator's explicit
`--resume` path verifies the preceding stages and reaches its normal terminal
state without calling an image generator again.

Only after that success may the two registered diagnostics run. Their
existing arm and collection entry points are called in their original order:
144 field forwards, followed by the 10-image guidance diagnostic. Their
original preparations, cases, seeds, model bindings and code stay fixed. The
failed prerequisite-only runtime records are archived before the recovery
controller writes successor runtime states. Each child preserves a separate
log; a failed child stops the recovery, with no automatic retry or reroll.

Verification consists of the actual complete offline assessment, unchanged
source and stage bindings, the preserved original failure, and child reports
from any subsequently completed diagnostics. Structural preparation alone
does not establish model learning. Raw image review remains a separate task.
No manuscript result or figure is promoted by this recovery.

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=. python scripts/recover_native_flow_assessment.py \
  --stage prepare --output artifacts/native_flow_recovery_v1
CUDA_VISIBLE_DEVICES='' PYTHONPATH=. python scripts/recover_native_flow_assessment.py \
  --stage run --output artifacts/native_flow_recovery_v1
```

Use the existing lightweight Python for the coordinator. Its model children
use the existing model environment and shared GPU-0 lock. No weights are
downloaded and GPU 1 is preserved.

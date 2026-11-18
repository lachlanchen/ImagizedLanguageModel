# Learned pixel corrections degrade the frozen visual-plan motor

The complete 6,000-update correction fit fails seven of nine fixed conditions.
It preserves the frozen spatial-256 codec and learns 615,314 new parameters;
the whole physical motor has 2,385,763 parameters and no contextual foundation.
All 768,000 retained training draws, initial/final generations and strict
optimizer/probability checks complete. This is a failed physical intervention,
not a trained language result.

| Measurement | Frozen base / initial motor | Trained correction motor |
|---|---:|---:|
| Training pooled F1 | .981661645 | .971271145 |
| Validation pooled F1 | .943690071 | .933584204 |
| Validation macro-form F1 | .928693592 | .916099307 |
| Validation exact cells / 4,907 | 94 | 0 |
| Validation exact forms / 952 | 3 | 0 |
| False ink at 3,206 source blanks | 0 | 3,206 |
| Actual mean plan + generated events | 33.000000 | 45.998129 |

The trained motor has no false blanks, but introduces ink at every source
space. Its source mean generated correction length is 13.998129 events;
validation averages 17.598125. Remaining validation error-tree work increases
from 40.474424 to 46.787854 events. Generated work and remaining error complexity
are separate quantities. Fewer events than the direct-image tree do not qualify
writing when fidelity and whitespace fail.

The [independent CPU audit](../scripts/audit_residual_plan_motor.py) recounts all
137,896 actual generated images, every emitted correction action and physical
region, all event counts, all 6,000 ledger entries, source order and checkpoint
hashes. The bound GPU execution compares every emitted-prefix probability with
parallel evaluation; the CPU audit does not repeat neural inference. All source
arrays, frozen-codec weights, original criteria and failed earlier attempts are
preserved. The [complete result](evidence/residual_plan_motor_comparison_result_20261118.json)
and fixed [validation](evidence/residual_plan_motor_comparison_20261118/held_fixed_images.png)
and [training](evidence/residual_plan_motor_comparison_20261118/train_fixed_images.png)
plates contain the original pixels without repair or quality selection.

Monitored execution takes 434.507 seconds, including 140.133 seconds waiting
for pressure to clear, with .485 GiB peak allocation and 43.056 GiB minimum
sampled available host memory. Supervisor time is 445.453 seconds. Independent
CPU recount takes 24.048 seconds at 1.948 GiB peak RSS. These scopes exclude
earlier codec training, data preparation, failed runs and any language body.
They establish no matched-quality or 1,000-fold language-efficiency gain.

The numerical determinism repair passes, but the learned correction does not.
Do not extend this fit, choose an intermediate head, install the failed motor
or relax its physical conditions. The better frozen base and faithful continuous
codec remain separate retained results. Further work must test useful causal
language-to-image prediction rather than assume that optimizing a supplied-plan
pixel repair can create language understanding. The complete language goal,
untouched confirmation and original positive cases remain unchanged.

# Complete-sequence fit: execution audit and evaluation recovery

Date: 2026-11-13. Status: training complete; language generation and scoring
incomplete. The useful independent visual-language objective remains unproved.

The registered complete-sequence continuation fit completed all three epochs.
The [CPU audit](../scripts/audit_visual_continuation_completion.py) verifies the
registered source files, manifest, schedule, every scheduled ledger event, all
fixed snapshot bytes, final trainable tensors and final Adam states. Its
[result](evidence/visual_continuation_completion_20261113.json) is an execution
measurement, with no new neural inference or language score.

| Measurement | Verified result |
|---|---:|
| Sources in each epoch | 13,766 |
| Scheduled optimizer updates | 41,298 |
| Scheduled complete-target glyph positions | 6,108,741 |
| Book / QA updates | 14,286 / 27,012 |
| Fixed / proportional presentations | 13,766 / 27,532 |
| Per-source presentations | One fixed, two proportional |
| Discarded recovery-verification updates | Two: replayed steps 5 and 6 |
| Fixed snapshots | 0, 10,324, 20,649, 30,973, 41,298 |
| Final trainable parameters / tensors | 20,212,225 / 85 |
| Changed trainable tensors | 85 / 85 |
| Final Adam states at step 41,298 | 85 / 85 |
| Final trainables and Adam moments | Finite |

The complete ledger's hash chain and source/layout/target assignments match the
registered schedule. Each epoch contains every source exactly once. Learning
rates and finite training measurements are checked per update. The audit loads
only the initial and final training checkpoint payloads on CPU. It does not load
the inherited body independently; the frozen-body version check remains a
training-runtime report. It initializes no CUDA context and makes no model call.

The training report records 18.567853 GiB peak allocated VRAM, a minimum observed
37.628605 GiB available host RAM, and 1,771.684826 seconds of resource-pressure
waiting. Cumulative training execution is 75,209.326020 seconds, about 20.89
hours, including the earlier failed training child. These are reported training
costs, not a comparison against another language model. The complete deployed
actor still has 7,578,390,003 parameters, including inherited foundation weights.

## Evaluation admission failure

The serial controller next launched the original evaluator. Its host-RAM entry
check failed before model loading or output generation. The original
`evaluation/` contained only the failed status record: zero generated images.
Available host RAM on inspection was about 42 GiB, below the registered 48-GiB
entry requirement. GPU0 was free; foreign GPU1 services were left untouched.
No obsolete project-owned GUI or model runtime was found to clean up.

The failed controller/evaluator status files, consoles and child journal are
preserved under the private run's `admission_failure_20261113_001/`, with a
byte-hashed receipt. The original evaluator child's complete wall cost is
6.5200451500713825 seconds. This cost is retained for cumulative evaluation
accounting; a failed admission is not a zero-cost attempt.

## Resource-gated recovery

The [recovery launcher](../scripts/recover_visual_continuation_evaluation.py)
imports no model. It holds a separate project recovery lock and waits for
admission. Its prelaunch filter requires 50 GiB available host RAM, giving 2 GiB
of headroom before the unchanged evaluator checks its original 48-GiB guard.
The registered free-GPU0 requirement, pressure ceilings and GPU mutex remain in
force. The launcher never lowers these limits or stops foreign services.

Once admitted, it invokes the original candidate evaluator, then the original
parent long-book evaluator, one child at a time. It never invokes training.
Existing completed outputs retain their original verification behavior. It
requires the registered 708 candidate and 24 parent streams before preparing
opaque review boards through the original review program. Generation rules,
model sources, snapshot selection, stopping, caps and scores are unchanged.

Each action keeps its original six-hour cumulative child ceiling. All preserved
attempts count, including the failed admission. A ten-second margin permits
termination of the exact owned child process group before exhausting that
ceiling. An unaccounted interrupted child fails closed. Resource-queue time is
reported separately from child execution and must remain visible in total cost
reporting. Another failed child stops the launcher; it is not silently retried.

Five tests check failed-attempt accounting, unmatched interrupted children,
exhausted budgets, independent parent costs, multiple attempts, resource
boundaries and cleanup of real owned subprocesses with refusal to kill an
unowned process group. These tests exercise recovery control; they are not model-quality
experiments. The prospective
[recovery registration](evidence/visual_continuation_evaluation_recovery_20261113.json)
binds the new launcher and tests before new generation.

```bash
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=. \
  python scripts/audit_visual_continuation_completion.py \
  --run artifacts/visual_continuation_full_logger_recovery_v2 \
  --out artifacts/continuation_completion_fresh.json
CUDA_VISIBLE_DEVICES= PYTHONPATH=. \
  python -m pytest -q tests/test_continuation_recovery.py
PYTHONPATH=. python scripts/recover_visual_continuation_evaluation.py \
  --run artifacts/visual_continuation_full_logger_recovery_v2 \
  --name evaluation_recovery_001
```

Check the private owner receipt before running the launcher; a live existing
recovery owns the wait. Do not duplicate it because an observation handle
expired. After generation, review every one of the 732 opaque outputs before
opening the new reference mapping. Apply original scoring and the prospective
four-QA-erasure audit. A development-pilot pass still requires untouched
source-group confirmation and does not complete the full historical-form,
independence and whole-model-efficiency objective.

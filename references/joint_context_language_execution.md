# Autonomous evaluation of the learned context and image writer

The complete context-gradient fit has one serial owner recorded in the private
handoff. This evaluator prepares its post-qualification language comparison;
implementation is not a fitted result. It must not run another GPU job while
the original fit/audit/qualification pipeline remains live. Preserve all 59
registered training sources, original failures, outputs, scores and readings.

Unlike the head-only evaluator, `evaluate_joint_context_language.py` restores
every one of the 131 learned tensors at each actual snapshot: the continuation
query, all answer residuals and the joint image head. It verifies their tensor
inventory after restoration, even when a preceding arm has frozen the actor for
inference. Inherited body parameters, feedback, normalization and stop parameters
remain unchanged. Checkpoint steps measure joint context/head updates, with the
same actual fresh head and point-final continuation initialization as training.
No intermediate checkpoint replaces the final model.

## Physical evidence before language generation

The complete fit and independent source/optimizer audit must finish, followed
by all 124 qualifications and their opaque visual review. Keep the original
90% readable-nonblank threshold in every one of seven groups gate, source positions and whitespace. Freeze
readings before opening the owner mapping. The evaluator reauthenticates the
sealed bundle, all original arrays, PNGs and boards and recounts the gate.

The [old blank-flag audit](joint_writer_blank_flag_audit.md) motivates an
additional check of a necessary fact: a credited nonblank candidate must
contain ink. The new post-gate evaluator checks all 2,080 selected PNGs against
their sealed image hashes and records blank-flag disagreements without changing
a reading or score. Any credited empty candidate stops admission. Nonempty
images still need actual visual legibility judgments; ink alone does not prove
readability. Control inconsistencies are disclosed. The original threshold and
failed control verdict remain unchanged. A contradictory frozen review is
preserved, not repaired after unmasking.

An absent or failed physical gate refuses registration before importing Torch
or creating the language-output directory. A failed ink audit is preserved in
`qualify/credited_ink_audit.json`. No new generation follows either failure.

## Complete fixed comparison

| Joint updates | Core streams, cap 128 | Long-book streams, cap 640 |
|---|---:|---:|
| 0 | 216 | 24 |
| 10,324 | 76 | 0 |
| 20,649 | 76 | 0 |
| 30,973 | 76 | 0 |
| 41,298 | 216 | 24 |

All 708 streams retain paired passage questions, both prompt layouts, question and
passage erasures, all normal/erased book prefixes and every fixed checkpoint.
Generation predicts physical masks greedily, rereads the actual emitted pixels
and uses the unchanged stop head at probability 0.5. Reference answers and output
lengths do not enter generation. There is no lexical output head, OCR, font
lookup or live teacher in the answer path.

The actual step-zero snapshot is the within-trial control. The original point
model's complete final streams and frozen scores are reused explicitly.
The fitted head-only control has no autonomous language score because it failed
its physical prerequisite; do not regenerate that actor or claim a measured
language advantage over it. Its conditional physical comparison remains intact.

Before generation, CPU registration binds all five actual 131-tensor snapshots,
completed fit/audit, frozen physical review, the ink audit, complete prompts,
scoring-reference file hashes and all evaluator/review source files. Reference
answer content is not opened by the binder. Commit and push this exact
post-gate registration before the GPU command. There is no automatic retry.
Manual resume authenticates existing complete streams and debits every prior
attempt's elapsed time and pressure wait; an unterminated attempt must first
be reconciled. The cumulative ceiling is 28,800 seconds, including validation,
imports and setup, with at most 1,800 seconds of resource waiting. Use only GPU0,
two CPU threads, 48 GiB available host entry, 20 GiB free VRAM entry, 24 GiB host reserve,
22 GiB allocated VRAM maximum and 48 GiB free disk. Reuse the existing process-local
NVIDIA compatibility wrapper and exclusive GPU mutex.

The pure CPU plan is available while training runs:

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=. \
  python scripts/evaluate_joint_context_language.py plan
```

The following commands remain gated until the original pipeline and physical
review are complete:

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONPATH=. \
  python scripts/evaluate_joint_context_language.py register
# Commit and push the exact generated registration before generation.
CUDA_VISIBLE_DEVICES=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH=. \
  ~/.local/bin/with-current-nvidia python -u scripts/evaluate_joint_context_language.py generate
```

After all 708 raw streams finish, `review_joint_context_language.py` scopes the
unchanged complete-output workflow to the new run. Inspect all 177 opaque boards
and 84 visible normal-book prefixes, retain uncertain readings and freeze the
complete bundle before new owner/reference scoring. The original 12/32 exact,
4/16 both-correct pairs,24/32 readable and 4/12 coherent-long-book pilot, all 8+2
retention cases, all four QA erasures and dependence-gap requirements remain.
Untouched confirmation, cold execution independence, whole-model matched-quality
cost and historical/unencoded forms remain required after a pilot pass. The
current 7.553B whole model is not a compact foundation because only 5.006M of its
parameters are trainable. No implementation or planning check establishes
complete language or closes the ILM goal.

## Implementation checks and remaining execution

The evaluator's CPU checks use the actual fixed cohort, initial checkpoint and
old frozen physical-review corpus. They cover complete planning, refusal before
neural import, scope isolation, provenance rejection, all seventeen original
blank-flag errors and restoration of every real learned tensor after inference
freezing. The restoration test uses a storage container with the actual 131
tensors and makes no model forward call; the full inherited body is not tested
by that CPU fixture. Actual full-body checkpoint restoration, autonomous output
and language quality remain unexecuted until post-qualification generation.

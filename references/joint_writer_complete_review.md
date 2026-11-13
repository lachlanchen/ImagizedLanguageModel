# Complete autonomous answer review for the joint writer

This workflow follows the existing physical prerequisite and the prescribed
708-stream autonomous comparison. It changes no model, fitting schedule,
reference, input, stopping rule or language threshold. Its implementation is
`scripts/review_joint_writer_language.py`, included in the future post-physical-
gate generation source bindings. Preparing this code does not qualify the
writer or authorize early language generation.

## Required order

Finish the original cache/fit/audit/qualification pipeline, review physical
forms with sealed readings, and require every candidate physical group to pass.
Only then bind/commit the actual fitted heads and run all 708 autonomous streams
as described in `joint_writer_language_execution.md`. The complete-output
review refuses missing generation, a live evaluator, changed registration or
incomplete arms. It verifies the original emitted tensors, rasters and display
grids without another model call.

After generation, prepare opaque review pages with the actual reviewer name/kind:

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=. python scripts/review_joint_writer_language.py \
  prepare --reviewer Codex --reviewer-kind model
```

The new review contains 708 complete response images on 177 boards. All 84 normal
book outputs have their original visible prefix linked beside the opaque pages;
the reviewer must also view each associated prefix. Outputs are never resized,
cropped, repaired or replaced. Blank/short/capped streams remain in the inventory.
Checkpoint/case ownership and canonical references stay closed during reading.
Model review retains the disclosed prior cohort familiarity and cannot be
described as independent human adjudication.

Inspect every complete board and each listed prefix. Fill its template with
visible text, uncertainty and, for normal books, a coherence judgment against
that prefix. Record each board exactly once:

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=. python scripts/review_joint_writer_language.py \
  record --readings path/to/completed_board_001.json
```

Actual inspection is required. A JSON declaration alone does not establish it.
After all boards, freeze readings and write the separate SHA seal before
permitting owner-map or reference access:

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=. python scripts/review_joint_writer_language.py freeze
CUDA_VISIBLE_DEVICES='' PYTHONPATH=. python scripts/review_joint_writer_language.py score
```

Freeze requires all responses, boards and prefixes, typed judgments and exact
original page/board pixels. Scoring first authenticates the seal and every board
receipt, then opens ownership and references. It reuses the original complete-
answer scorer and the separate full four-QA-erasure pilot audit. All 660 core
and 48 long-book responses must remain. Earlier point-writer controls and their
scores are authenticated and explicitly reused, without generation or regrading.
Existing files are never overwritten. Preserve and reconcile any partial
operation before a manual recovery; there is no automatic retry.

## Interpretation and resources

The full final pilot retains at least 12/32 proportional exact answers,
4/16 both-correct pairs, 24/32 readable responses and 4/12 coherent complete
long-book continuations. Preserve all eight proportional and both fixed exact
retention cases. Each of all four QA-erasure arms must have at most two exact
answers, with a proportional normal-minus-better-erased gap of at least eight.
The older proportional-only route is recorded separately and cannot override
a failed full audit. Intermediate snapshots cannot replace the final one.
Canonical mismatch alone is not semantic falsity; uncertain or unreadable
responses receive no credit and cannot be repaired using references.

Each CPU preparation/freeze/scoring operation has a 900-second ceiling,
8 GiB peak RSS limit, 24 GiB available-host reserve and two CPU threads. There
are no new model calls, optimizer updates or GPU workers. A development pass
still requires untouched source-group confirmation, standalone execution and
whole-model resource accounting. The seven new tests exercise review invariants
with metadata/bitmap fixtures; they provide no trained-model language result.

# Opaque review before the joint-writer physical gate

The registered conditional-writing comparison retains 124 complete cases and
seven split/kind/layout groups per writer. Its prerequisite to autonomous
language evaluation is at least 90% confidently readable nonblank selected
forms in **every candidate group**. Readable alternatives to canonical book
glyphs can pass this component gate. Complete answers, coherent books and the
full language pilot remain separate requirements.

The original bound trainer contains a physical scorer that opens the owner map
before writing its frozen-readings file. Its coverage check also relies on the
generated map rather than independently enumerating every registered position.
These are prospective review-order and audit limitations, not evidence that the
still-unmeasured writer passed or failed. Preserve the running trainer and its
45 bound source files. Use the separate CPU workflow
`scripts/review_joint_writer_physical.py` after qualification finishes; it does
not change model weights, generation, data, selected positions or thresholds.

## Review sequence

1. Confirm that the original coordinator has reached
   `awaiting_opaque_physical_review`, both coordinator and qualification worker
   are absent, and complete fit, independent audit and all 124 qualifications
   are source-bound. Commit the review code before using it. Preparation reads
   the opaque board manifest and report hashes, without opening owner rows.
2. Create a named model- or human-review session. Inspect each complete actual
   board, transcribe the visible forms into its template, and retain blank or
   uncertain readings. Set the actual-view declaration only after inspection.
   Record each board once while the mapping remains unopened. A template, log
   or declaration alone cannot establish that visual inspection happened.
3. Freeze all board readings and page hashes, then write a separate SHA-bound
   seal. No owner-map read occurs in this operation. Incomplete boards,
   duplicate or missing IDs, invalid flags and changed images cannot be frozen.
   Preserve the first reading files; no target may repair them after unmasking.
4. Only then score. Authenticate the sealed readings and each board receipt
   before opening the mapping. Re-enumerate both roles at every registered
   case/position, verify all seven candidate groups, and compare original
   candidate/parent arrays with page pixels and the actual board crops. Do not
   use the canonical reference array to decide whether a form is readable.
5. Write the first `qualify/gate.json` once. Count the threshold with integer
   cross-multiplication so rounding cannot promote a failing group. Preserve
   all parent groups as controls. A failed candidate group stops this writer
   before a full autonomous run. A pass permits that comparison but does not
   qualify language, compactness, generalization or efficiency.

`frozen_readings.json`, `review/freeze_receipt.json`, session/board receipts and
`gate.json` remain private under the existing run. Future autonomous generation
must bind their hashes and the exact final writer weights before launch. The
workflow records reviewer kind, prior cohort familiarity and the absence of
independent human adjudication. It never relabels model review as human review.
Existing review files are not overwritten or automatically retried; reconcile
any partial operation while preserving its files before a manual recovery.

## Resource and verification scope

Preparation and recording import neither Torch nor NumPy. Bitmap verification
imports NumPy/Pillow only after freeze, uses two CPU threads, retains 24 GiB of
available host RAM, and stops at 900 seconds or 8 GiB peak RSS. There are no new
model calls, optimizer updates or language scores. The running GPU owner is
neither restarted nor interrupted by implementing this workflow.

The six tests in `tests/test_joint_writer_physical_review.py` check exact
thresholds, all-group coverage, owner-row tampering, freeze-before-map access,
sealing, incomplete/changed evidence, no-overwrite publication and raw bitmap
identity. Their numerical image fixtures test the review mechanism only. They
are not generated Chinese, trained-model evidence or substitutes for actual
opaque review of the forthcoming qualification outputs.

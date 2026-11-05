# V68: preserve actor coordinates under question blanking

Before any final adaptation or autonomous answer exists, a CPU inspection finds
that independently re-folding an erased prompt changes its actor coordinates
in 31 of 128 Chinese controls and one of 64 English controls. The English case
also changes page height. Flat source/control strips have identical dimensions;
the confound occurs when a new blank column changes a later wrap boundary.

The normal prompt determines the fold ranges once. Place the existing blank
strip into those exact ranges, with identical right/bottom padding. Return
owning pixel storage. Verify both routes with an independent NumPy recreation
of the normal wrap and control placement. Normal actor pixels must remain
bit-identical; every control must have the normal shape and differ only where
the prepared flat control erases ink.

This is a pre-generation measurement correction. It changes no training/model
source, optimizer update, prepared data file, case, reference, maximum output
capacity, metric, bootstrap setting or original quality threshold. Prepared
data report remains SHA-256
`4f90406b51670e4e82327d1396e1c954e62bfb5d7b31c437b5c59e88a9180fe5`.
No generation directory exists at discovery. Training continues in the same
process; retire only the lightweight waiting coordinator before editing its
guarded sources, then requeue the corrected evaluator behind that trainer.

Preserve the original evaluator preflight (`audit_preflight/attempt01`) and
its committed source history. Record the discovery and old coordinator under
`artifacts/codec_language_v68_20261105/audit_revision_r2/`. The corrected CPU
evaluator preflight uses `audit_preflight/attempt02`; commit its implementation
before execution and freeze it again before any answer generation.

Update the file-interface preflight as well. A fully blank raw strip does not
carry its original wrap boundaries. Such a control must enter the actor as its
precomposed canonical page. Ordinary explicit strip folding remains valid,
and all normal strip inputs remain unchanged. Check canonical pages for all
384 routes and explicit raw-strip folds separately, without implying that
re-folding a blank raw strip preserves the intervention layout.

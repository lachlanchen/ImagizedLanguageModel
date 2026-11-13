# Full-corpus joint glyph writer

This comparison replaces the final continuation model's point image writer with
a normalized joint distribution over all 1,024 physical pixels. It preserves
the trained reader, continuation dynamics, query, image feedback and stop head.
The new head has 2,642,368 parameters; the inherited full system remains large.
Neither a working trainer nor a low likelihood establishes a usable ILM.

The [protocol](../references/continuation_joint_writer_protocol.md) and
[registration](../references/evidence/continuation_joint_writer_registration_20261113.json)
fix the complete corpus, source exclusions, old control, initialization, schedule,
checkpoints, physical review and language criteria. The local parent checkpoint,
authenticated visual corpus and shared pinned model installation are required.
This is a continuation experiment on those assets, not foundation training from
scratch. Do not download or duplicate another copy of the shared weights.

Preparation caches both layouts for all 13,766 eligible sources, with 4,072,494
target-predicting states and about 63 GiB of lossless private arrays. Each state
precedes its own target; terminal states are excluded from image loss. The head
then trains for three complete epochs and 41,298 updates. A complete source gets
one optimizer update regardless of its length; microbatches retain that reduction.

From an unprepared checkout with the required local assets and committed source:

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH=. \
  python scripts/prepare_joint_continuation_writer.py
```

Commit and push the generated registration before starting neural work. Preserve
an existing study; do not run registration again over its files. The coordinator
sets physical GPU0, deterministic CUDA workspace and two CPU threads for workers:

```bash
PYTHONPATH=. python scripts/run_joint_continuation_writer.py
```

Use one durable session for a long run. Read
`artifacts/continuation_joint_writer_v1/controller_status.json` and the active
stage's `status.json`. Authenticate live PID/start-ticks/boot-ID ownership before
restarting anything. A polling timeout does not establish that a worker stopped.
The stages run serially: full frozen-reader cache, head fit, CPU execution/data
audit, then conditional writing qualification. An error stops the coordinator;
there is no automatic retry. Inspect and preserve its attempt before an explicit
`--resume`, and account for all discarded/repeated work within the original cap.

The fit saves five fixed snapshots and exact periodic resume state. A real
step-three restore replays steps four and five, checking losses, gradients,
weights, optimizer and random state. The CPU audit checks the retained event
chain, every source exposure, packed target pixels, normalization and actual
checkpoint counters. It does not repeat a neural reader run.

Qualification writes complete supplied-history predictions and opaque physical
review boards under `artifacts/continuation_joint_writer_v1/qualify/`. Inspect
every selected candidate/control glyph before opening the owner mapping or
target, then freeze its visible reading, uncertainty and blank label. Correct
history and supplied lengths make this a physical component test. A legible
alternative book glyph is not a canonical error for this gate. The subsequent
complete-answer comparison still requires the fixed QA, erasures, paired
questions, book continuations, natural stops, retained successes and untouched
confirmation. No unexecuted score may be recorded as zero.

All arrays, private logs and weights remain ignored under `artifacts/`. Public
summaries must report cache/fit/qualification costs as well as the inherited
body and earlier preparation. Keep the full independent image-input/image-output
language objective open until those complete-language requirements are verified.

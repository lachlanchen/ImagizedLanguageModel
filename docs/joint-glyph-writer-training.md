# Full-corpus joint glyph writer

**Current result:** the fit, independent audit and all 124 qualifications are
complete. All 33 boards and 2,080 forms were reviewed and sealed before unmasking.
The [physical gate fails](../references/joint_writer_physical_result.md) in two
groups. Do not resume this run or launch its autonomous evaluator. The commands
below document its reproducible recipe. The
[empty-image diagnosis](../references/joint_writer_blank_density_result.md) is
also complete; current work follows the separate
[context-learning decision](../references/joint_writer_context_learning_decision.md).

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

The [qualification-cache API recovery](../references/joint_writer_cache_tuple_recovery.md)
records a pre-fit caller failure and the verified reuse of all saved training
arrays. Its amended source binding preserves the original attempt and debits
CPU verification to the existing cache budget; it changes no neural or language
acceptance rule. Use the private handoff for its current execution state.

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

Use the [separate opaque-review workflow](../references/joint_writer_physical_review_workflow.md)
after the coordinator finishes qualification. It freezes a complete set of
readings and a SHA-bound seal **before** opening owner rows, then independently
recounts all registered positions and verifies source/page/board pixel identity.
It leaves the bound trainer and its original registration unchanged. For a
model review by Codex, prepare the session with:

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=. python scripts/review_joint_writer_physical.py \
  prepare --reviewer Codex --reviewer-kind model
```

Use the actual reviewer name and kind. View every complete opaque board, fill
its corresponding JSON template in `qualify/review/templates/`, and retain all
uncertain or blank readings. Set `all_outputs_actually_viewed` only after actual
inspection; leave `owner_mapping_opened` false. Record each completed board with:

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=. python scripts/review_joint_writer_physical.py \
  record --readings path/to/completed_board_001.json
```

After every board is recorded, freeze first and score separately:

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=. python scripts/review_joint_writer_physical.py freeze
CUDA_VISIBLE_DEVICES='' PYTHONPATH=. python scripts/review_joint_writer_physical.py score
```

Preparation refuses a live or unfinished pipeline. Missing boards, altered
receipts, changed pixels or omitted candidate groups cannot yield a pass.
The first frozen review and verdict are never overwritten. A failed physical
gate stops this replacement; a pass still requires the complete autonomous
comparison and does not establish a usable language model.

The [autonomous execution guide](../references/joint_writer_language_execution.md)
provides the complete 708-stream plan and the evaluator for the existing
image-feedback generator with the new joint bridge. Its read-only `plan` action
is available during training. Registration and generation require the completed
fit and sealed passing physical review; they do not launch early or change the
running pipeline. Bind and commit the actual five fitted snapshots before
generation. GPU integration and complete language quality remain unmeasured
until execution and output review finish.

After generation, use the [complete-output review guide](../references/joint_writer_complete_review.md)
for all 708 answers and 84 visible normal-book prefixes. It seals readings
before unmasking and retains the full four-arm erasure rule. Its checks and
fixtures validate the review mechanism, not trained-model language quality.

All arrays, private logs and weights remain ignored under `artifacts/`. Public
summaries must report cache/fit/qualification costs as well as the inherited
body and earlier preparation. Keep the full independent image-input/image-output
language objective open until those complete-language requirements are verified.

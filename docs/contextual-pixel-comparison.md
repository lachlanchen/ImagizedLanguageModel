# Contextual image-span comparison

The registered comparison tests whether contextual adaptation improves
answer-image selection beyond continued head-only training. It is an
extractive image model: it selects original source columns and cannot compose
an answer absent from the input. The complete independent ILM remains an
unachieved research objective.

The [protocol](../references/contextual_pixel_adaptation_protocol.md) fixes
3,193 Chinese QA images, 64 copying images, four complete passes and the same
13,028-draw schedule in both arms. Only the adapted arm trains contextual
query/value/output adapters. Both arms retain a 7.531B-parameter externally
pretrained runtime. The [execution preflight](../references/contextual_pixel_learning_preflight.md)
passes; no production language-quality result follows from that check.

The current run is `artifacts/contextual_pixel_adaptation_v1`. Its controller
is `scripts/compare_contextual_pixel_learning.py`; `status.json` records the
current phase and PID. Check the live process or existing tool session before
considering resume. Do not run a second GPU job or change bound source/protocol
files while it is active. Checkpoints retain current/previous learned tensors,
Adam moments, cursor and CPU/CUDA RNG. An explicitly resumed run must use the
same root and unchanged source/data identities.

The controller first authenticates the data and caches initial source
features, then checks native/cached head gradients, trains both arms, and
generates all 512 final assessment images through the complete native path.
It never selects a checkpoint using development answers. Source-feature cache
time and external pretraining are separate from arm optimization time.

## Independent checks

`scripts/audit_contextual_pixel_data.py` runs on CPU. It authenticates every
training input/reference and searches exact physical column bytes for all
reference occurrences. It also measures whether question/passage erasure
changes folding ranges, retained ink positions or observer dimensions.
Erasure controls should not automatically be interpreted as isolating a
unique semantic mechanism. This audit does not invoke or train a model.
The completed [physical data audit](../references/contextual_pixel_data_audit.md)
finds all 3,257 training references admissible. Question erasure preserves
answer coordinates, while moving other retained ink in 11/64 cases. Passage
erasure removes every possible correct source span, so zero exact copying
there is mechanically forced and does not establish understanding.

After both arms finish, `scripts/export_contextual_pixel_comparison.py`
independently checks the complete four-pass schedule, every recorded Adam
step, parameter changes, source-feature payloads and all 512 answer images.
It reconstructs every answer from its declared source columns and recomputes
pixel equality, unordered column F1, paired-question outcomes and the fixed
decision rules. It retains raw images and journals rather than replacing
wrong answers with rendered reference text.

## Standalone PNG inference

`scripts/infer_adapted_pixel_spans.py` accepts the final `checkpoint.pt`, its
SHA256 from `training_report.json`, supplied image paths and a fresh output
directory. It rejects incomplete checkpoints. The runtime requires the shared
source archive and fixed numeric task controls, but does not open training
manifests, reference images, feature caches or the original trained head.

Inputs must currently be complete binary writing strips of height 32 pixels.
Raw scanned-page segmentation, arbitrary image prompting and free writing
remain unsupported. Any modern or historical ink present in a selected span
is copied directly; this does not establish understanding of that script.
The output is a real PNG, not a predicted Unicode string rendered through a
font table. No tokenizer, OCR or external model service runs in this path.

Use the existing Python 3.12 environment with `CUDA_DEVICE_ORDER=PCI_BUS_ID`,
`CUDA_VISIBLE_DEVICES=0`, `CUBLAS_WORKSPACE_CONFIG=:4096:8`, four CPU threads,
offline Hugging Face settings and `PYTHONPATH=.`. The renderer's documented
system-library preload is needed when preparing text images. No font renderer
is invoked by standalone PNG inference.

Before final export, run the first two fixed normal development questions
and first Chinese/English development copies from each arm in separate
standalone invocations, saving to `<arm>/standalone_replay`. The exporter
requires these eight native calls to reproduce the controller's answer pixels
and selected events exactly. Those deployment checks are pending until the
training controller has finished and released GPU0.

`scripts/complete_contextual_pixel_comparison.py` can wait for an already
running controller and chain these checks after its actual process exit.
Pass the comparison root, its verified controller PID and a fresh work
directory. The supervisor records the PID's start time, binds the source
hashes and checks host/GPU resources before launching each standalone call.
It keeps one child at a time, preserves failed logs/partial outputs and saves
the independent export inside its work directory. While waiting it uses only
the Python standard library. Stopping it terminates only its own child, never
the original training controller. The verified export still requires review
before publication or any language-quality claim.

The original component requirements remain 30/32 new copies, 48/64 normal
answers, 20/32 both-correct pairs and a 25-point question-erasure gap. Passing
them would qualify this prepared-strip extraction component only. It would
not demonstrate a lightweight model trained from scratch, free image-language
generation, historical-script semantics or efficiency at matched task quality.

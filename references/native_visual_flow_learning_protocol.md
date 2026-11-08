# Complete mixed image-answer flow-learning experiment

This experiment trains the self-contained visual-semantic candidate on all
384 prepared training image pairs. It is a new pretrained-model adaptation
route, separate from the 128-update FLUX experiment. The model receives
changing questions only as pixels and produces answer pixels. Existing
language and image pretraining is retained and counted; this is not a claim
of foundation training from scratch, historical-script competence or parity
with a conventional large language model.

## Source, data and admission

Use the pinned complete DeepGen release and fixed-task compiled condition
already registered for native replay. Keep the unused lexical table, word
head and tokenizer removed from the inference graph. The image codec and
visual/language body remain frozen; adapt the generator and connector using
the same native-dtype low-rank branches as the actual one-step preflight.

CPU preparation can precede model execution. A longer training job requires
the complete actual codec/gradient preflight: exact zero-adapter native
generation, eight real codec decodes, one actual optimizer update and two
autonomous answer generations. Inspect those original pages before deciding
to spend the longer training budget. Numerical projection checks cannot
replace this model-level evidence.

Training uses all 256 QA and 128 copying pairs in the prepared image corpus.
QA training spans 128 articles; the 16 validation articles are disjoint.
The 64 validation pairs never enter the optimizer. Input pages remain exact
1024-square originals; answer images are padded with white to that geometry,
preserving all original answer pixels. No target transcription, word index,
question-specific text embedding or OCR loss is supplied to optimization.
Grouping metadata controls bookkeeping only, not neural conditioning.

## Fixed learning recipe

Start fresh from the pinned source and zero-output adapter branches; do not
restore the single-step preflight checkpoint. Rank is 16, with 37,380,992
trainable FP32 adapter masters. Their multiplications use the native output
dtype. Train four independent permutations of all 384 examples: 1,536
batch-one updates. Seeds are 20269401 for adapter initialization, 20269402
for order, 20269403/20269404 for time/noise generators and 20269405 for the
global codec RNG. Save the exact complete schedule before execution.

Compute input-image semantic features and reference latents first. Load the
answer image afterwards as supervised codec data. The fixed positive-task
branch supplies the connector and generator conditions. Use the already
checked native unweighted velocity MSE and a fresh 1000-step fixed-shift
training scheduler. One logical generator forward contributes each update;
native activation checkpointing can recompute internal layers in backward.

AdamW uses rate 5e-5, betas (0.9, 0.95), decay 0.05 and norm clipping at one.
Only adapter parameters enter the optimizer. Enable native non-reentrant
checkpointing on the generator and connector. Keep the reader and codec in
evaluation mode, without gradients. One GPU0 model job may run at a time;
preserve GPU1 and the workstation resource policy. Stop on nonfinite loss,
gradients or saved parameters, source/data changes, or allocated VRAM above
23 GiB. Do not automatically retry a smaller canvas or tune against results.

## Durability and monitoring

Write each attempt to a fresh log directory. Commit a checkpoint every 32
completed updates and at the fixed final update. Preserve a readable current
checkpoint throughout atomic replacement and retain one predecessor. Save
adapter parameters, actual AdamW state, completed history and all used CPU,
CUDA, time-generator and noise-generator RNG states. Resume only explicitly,
with identical implementation, data, source and schedule. The committed
checkpoint is authoritative if a failed attempt logged later unsaved steps;
retain those failed logs and their compute cost instead of overwriting them.

After every 128 updates, generate both fixed first training questions with
fixed seeds 20269490/20269491, 50 native steps and guidance 4. Preserve complete
unrepaired pages, real decoder tensors and input/seed/update bindings. Save
and restore RNG states around monitoring so it cannot change subsequent
training draws. These are training previews, not held-out evaluation or a
checkpoint-selection rule. Do not choose an intermediate checkpoint by its
appearance or answer scores. Keep all previews, including failures.

The final export contains only adapter tensors and deployment source bindings.
The separate inference process accepts an index with exactly `id`, `input`,
`input_sha256` and `seed`. It loads no training manifest, optimizer, expected
answer, OCR engine or varying text condition. Authenticate every output PNG
against its actual native decoder tensor.

## Assessment and interpretation

After training, assess the fixed initial source and final model with matched
input pixels and seeds. Use the first 32 natural training questions, all 32
article-disjoint development questions, both 32-case held-out erasure cohorts,
and the first 32 training plus all 32 validation copying pages. All original
pages and outputs remain complete. This is 192 outputs per model, 384 total;
none is selected for legibility or correctness. Reused development examples
are not a new blinded test set. Independently calibrate reference-image OCR
at the new canvas geometry, retain source-accepted answer matching and inspect
all original pages. OCR limitations do not permit silent transcript repair.

A bounded developmental signal requires seen QA at least 24/32; held-out QA
at least 12/32 and four above initial; both questions correct in at least
4/16 held-out pairs and two above initial; normal held-out QA four above each
erasure; and held-out copying no more than two below initial. Report every
condition, including failures. The existing reliable-writing requirement and
broader independent-ILM goal remain separate and unachieved by these gates
alone. No English QA or historical-form result is implied by Chinese QA and
English copying. Measured runtime/memory and complete autonomous answer
quality, rather than loss reduction, determine the next research action.

The [CPU persistence check](evidence/native_visual_flow_resume_cpu.json) uses
the released BF16 `context_embedder` projection. Saving after one real
adapter update, then restoring and repeating the second, reproduces the
loss, adapter weights, AdamW tensors and CPU/time/noise RNG states exactly.
The predecessor checkpoint is retained byte-for-byte. This checks persistence
on one actual projection; it does not exercise the full model, CUDA replay or
language learning. The reproducible check is
`scripts/check_native_flow_resume.py --output <new-folder>` with CUDA hidden.

CPU preparation freezes four input-only indices in `evaluation/`, together
with separate reference manifests for the offline evaluator. They contain
64 normal, 32 question-erased, 32 passage-erased and 64 copying cases. The
generator never reads the reference manifests. Recalibrate
`scripts/score_paired_writing.py` on each padded reference manifest before
scoring the corresponding output index. Source-accepted QA matching remains
a separate check against the original CMRC records.

## Entry points

Use `scripts/train_native_visual_flow.py --stage prepare --output <run>` with
CUDA hidden. Once the actual preflight is complete and reviewed, run
`--stage train` on GPU0 through the existing model Python. `--resume` restores
only a committed interrupted run with identical sources. `--stage export`
writes the final `deployment/` folder. Run
`scripts/infer_native_visual_flow.py --deployment <run>/deployment --inputs
<input-only-index> --output <new-folder>` separately for image generation.
Use `--initial-source` with the same deployment bindings and input indices
for the matched source control; this arm keeps zero adapters and loads no
learned adapter weights.
No long training or autonomous quality result has been observed when this
protocol is registered.

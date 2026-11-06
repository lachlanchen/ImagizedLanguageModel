# Native-context image-language training

The earlier full-corpus image writer (V75) failed to qualify as a usable
language model: 2/128 exact Chinese development answers and 0/12 exact new copy
answers.
Retaining full contextual history is the next tested design direction; it has
not yet established language capability. The [diagnostic results](../references/native_context_v80_result.md)
explain the distinction between source-reader content, glyph reconstruction and
actual autonomous image answers.

The new architecture accepts prompt pixels, retains the frozen native visual
and contextual body, and learns continuous image-history and writing layers.
Each emitted block is decoded to pixels and re-encoded before continuing.
This keeps the actual generated image in the feedback loop. It uses externally
pretrained weights and has 7.583B complete parameters, including 46.900M
trainable interface parameters. No tokenizer, OCR or teacher service is needed
by the deployed actor. These properties alone do not guarantee correct answers.

An exact image answer requires a learned stop and the same complete binary
raster dimensions and pixel values as the reference. Foreground F1 measures
aligned ink-pixel overlap. These training snapshots do not include a separate
semantic-transcription accuracy score; the raw images remain available for
inspection. The original physical-image evaluation gates are unchanged.

A [fixed CPU recognition calibration](../references/native_cpu_transcription_v80_result.md)
fails on the current format: 30/128 Chinese and 78/128 English reference
transcripts are exact, with additional blank/noise-control failures and one
oversized-image error. No actor images are transcribed by this unqualified
instrument. These measurements add no ILM semantic score.

A separate [neural reading protocol](../references/native_neural_transcription_v80_protocol.md)
is prepared for execution after the original training, fixed audit, physical
verification and standalone inference checks finish. Its
[CPU preparation receipt](../references/evidence/native_neural_transcription_v80_preparation.json)
verifies all 288 calibration cases, exact preservation of every source pixel,
repeatable persisted layouts and refusal to start while the original pipeline
is active. Neural recognition has not run. Calibration must qualify separately
for each language before any corresponding actor answer is transcribed. The
reader shares the actor's pretrained foundation, so agreement remains an
auxiliary measurement with correlated errors.

The [native prediction diagnosis](../references/native_corpus_fit_v80_protocol.md)
reuses the same 32 Chinese training questions previously evaluated with V75.
It compares free generation, generation with correct preceding image codes
and supplied answer length, and fixed-codec reconstruction. These controls
will help distinguish errors that persist under correct history from errors
introduced by free continuation. They provide no autonomous-answer credit.
The [preparation receipt](../references/evidence/native_corpus_fit_v80_preparation.json)
authenticates all 32 source images and code records and independently rebuilds
their targets and prompt layouts. Model evaluation remains pending.

Prepare on CPU with `CUDA_VISIBLE_DEVICES='' PYTHONPATH=. python
scripts/probe_native_corpus_fit_v80.py --run artifacts/native_chinese_v80_20261106
--stage prepare` in the compatible environment. After the original training,
fixed audit, physical verification and three standalone calls finish, use
`--stage evaluate` with the GPU environment below, then `--stage verify` with
CUDA disabled. Run this diagnosis and neural reading sequentially. Preparation
alone measures neither model predictions nor language capability.

The [update-1,000 snapshot](../references/evidence/native_chinese_v80_update01000/receipt.json)
contains all four scheduled training-example answers: 0/4 exact, 35.217% full
foreground F1, 69.845% first-block F1 and four stopped streams. Only 256 of the
3,193 training records had been visited at this point. The
[unaltered output/target comparison](../references/evidence/native_chinese_v80_update01000/all_outputs_and_targets.png)
shows malformed glyphs and incorrect answer lengths. This is an early training
diagnostic, not a completed-run or held-out result. Later diagnostics use the
current replay pool, so their four-example cohorts differ.

At the [update-5,000 snapshot](../references/evidence/native_chinese_v80_update05000/receipt.json),
all four scheduled answers still fail: 0/4 exact, 19.501% full foreground F1,
41.286% first-block F1 and four stopped streams. Generated lengths are
5/2/12/5 blocks versus target lengths 2/5/3/2. All four
[raw outputs and owning targets](../references/evidence/native_chinese_v80_update05000/all_outputs_and_targets.png)
are retained. The run had visited 1,280 of 3,193 training records, with 20,000
optimization draws. This is another failed training diagnostic, with no
held-out or language-success claim. Its different four-example cohort prevents
interpreting the two snapshots as a matched learning curve. The declared
40,000-update run and final full audit remain unchanged.

The [update-10,000 snapshot](../references/evidence/native_chinese_v80_update10000/receipt.json)
also gives 0/4 exact training answers, with 48.319% full foreground F1,
67.056% first-block F1 and four stopped streams. Output lengths are 3/2/4/2
blocks versus targets of 3/2/2/5. The
[unmodified images and owning targets](../references/evidence/native_chinese_v80_update10000/all_outputs_and_targets.png)
show recognizable initial characters in some outputs, followed by damaged
writing, incorrect continuations or incomplete answers. At this point 2,528 of
3,193 records had been used, with 40,000 optimization draws. The
[exposure replay](../references/evidence/native_chinese_v80_update10000/case_exposure.json)
matches all 1,001 logged minibatches through update 10,000 and finds two
optimization exposures for each diagnostic case. These are different cases
with different exposure counts from the earlier snapshots; their scores do
not form a matched learning curve. No held-out or usable-language result is
established. Training continues under the original 40,000-update protocol.

At [update 20,000](../references/evidence/native_chinese_v80_update20000/receipt.json),
the four scheduled training answers give 0/4 pixel-exact images, 70.926% full
foreground F1 and 80.163% first-block F1. All four streams stop. Output lengths
are 1/2/2/2 blocks versus targets of 1/3/2/2. The
[unaltered comparison](../references/evidence/native_chinese_v80_update20000/all_outputs_and_targets.png)
contains recognizable writing such as “农业” and “1999年”, with visible
defects, stray marks and a damaged book-title response. This is an unblinded
qualitative observation, not a separately measured semantic accuracy score.
The [exposure replay](../references/evidence/native_chinese_v80_update20000/case_exposure.json)
matches all 2,001 logged minibatches and finds twenty optimization exposures
for each of these four cases. The run has used all 3,193 training records and
80,000 optimization draws. Different cases and exposure counts again prevent
a matched comparison with earlier snapshots. No held-out language capability
is established; the original final audit remains pending.

An [exposure audit](../references/evidence/native_case_exposure_v80.json) replays
the declared sample order and matches all 501 logged minibatches through update
5,000. Each case in the 1,000-update diagnostic had thirteen optimization
exposures; each case in the 5,000-update diagnostic had one. The two checkpoints
were sampled at different positions within their rotating pools, as well as on
different questions. Global update count is therefore insufficient to compare
their case-level learning. The malformed outputs remain failures; the exposure
audit adds no language-success claim. Reproduce it with
`PYTHONPATH=. python scripts/audit_native_case_exposure_v80r2.py --run artifacts/native_chinese_v80_20261106`
in the compatible CPU environment. The revised replay preserves the original
producing commit when unrelated documentation commits change repository HEAD;
different inputs or computations still require a separate record. The original
audit source and receipt remain unchanged.

The [declared training protocol](../references/native_chinese_v80_protocol.md)
uses all 3,193 existing Chinese training records, batch four, 40,000 updates and
a bounded 32-prompt RAM pool. The original data, codec, numeric controls and
inherited weights are authenticated local prerequisites. The scripts refuse
missing or changed inputs rather than downloading substitute models or using
synthetic execution fixtures.

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
CUBLAS_WORKSPACE_CONFIG=:4096:8 OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 \
PYTHONPATH=. artifacts/external/mixar_env_py312/bin/python \
  scripts/train_native_chinese_v80.py --run artifacts/native_chinese_v80_20261106
```

The same command resumes the latest checkpoint. Source and protocol changes
require a new run. The real-data preflight verifies prompt-memory identity,
finite gradients and exact optimization-state replay before resetting new
parameters for production. It is not a language evaluation.

Outputs stay under the selected run directory:

- `status.json`: current stage and update; `failure.json` records a failed run.
- `preflight/preflight.json`: actual-data execution and resume checks.
- `training/metrics.jsonl`: loss, optimizer and sample counts.
- `training/checkpoint_latest.pt` and `checkpoint_previous.pt`: new parameters,
  optimizer, RNG and replay cursors. Frozen source weights are not duplicated.
- `diagnostics/update_*/`: raw autonomous training-example images, not held-out
  evidence. Short image blocks form a complete generated writing strip.
- `development/`: all 384 final normal/question-erased image responses and scores.
- `verification.json`: independent physical-pixel scoring after the final audit.

After training completes, run `scripts/audit_native_chinese_v80.py --run ...`
with the same GPU environment, followed by the same command with `--verify`
and `CUDA_VISIBLE_DEVICES=''`. The fixed audit reports Chinese gates and English
as untrained transfer. The optional `scripts/continue_native_chinese_v80.py`
waits for the owned training job and performs those two phases sequentially.
Do not launch another training or generation process while this job owns GPU0.
The coordinator also scores each completed four-answer training diagnostic on
CPU as it appears, saving `pixel_scores.json` and an unchanged output/target
contact sheet. This starts no additional model-generation job.

## Standalone image and typed prompts

`scripts/infer_native_chinese_v80.py` accepts `--prompt page.png`, `--text`, or
`--text-file prompt.txt`. Typed text uses the same optional host renderer as
the prepared corpus; only its pixels reach the neural model. Explicit line
breaks and empty lines are preserved. Unknown font glyphs and oversized inputs
raise an error. For marks outside available fonts, supply their actual image.
The image path accepts grayscale pages without recognizing their characters.

Input preparation can run while training is active:

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=. \
artifacts/external/mixar_env_py312/bin/python scripts/infer_native_chinese_v80.py \
  --text '材料：甲有三本书，乙有五本书。问题：谁有五本书？' \
  --prepare-only --output artifacts/my_prepared_prompt
```

This creates `input/prompt.png`, the exact input tensor and an input receipt.
It does not load a checkpoint, allocate CUDA or produce an answer. The CLI
configures the existing Linux native font libraries before using the optional
text renderer. Image-only input does not need those rendering libraries.

After the declared training and queued benchmark have finished, generate from
an image with an optional visible instruction:

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
CUBLAS_WORKSPACE_CONFIG=:4096:8 OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 \
PYTHONPATH=. artifacts/external/mixar_env_py312/bin/python \
  scripts/infer_native_chinese_v80.py --prompt page.png --layout page \
  --instruction '请执行下方图片中的要求。' --output artifacts/my_image_answer
```

Replace `--prompt`, `--layout` and `--instruction` with `--text '…'` or
`--text-file prompt.txt` for typed input. Instructions are rendered above the
image with a white gap, preserving the prepared image pixels exactly. The
fixed input budget is 512 visual cells; the output cap defaults to 128 blocks.
The default checkpoint is the native run's final `checkpoint_latest.pt`.
The loader rejects partial training and untrained execution fixtures.

The result contains `output/answer.pt`, its exact `output/answer.png` when
nonempty, input rasters and a provenance/timing report. No glyph repair,
reference answer or OCR is used during generation. Completion of the training
budget alone does not qualify language quality.

The sequential coordinator also runs
`scripts/check_completed_native_prompt_routes_v80.py` after the complete fixed
audit. It calls the actual CLI separately for the existing first copy image,
its exact typed prompt and that image with a visible instruction. Typed/image
pixel and generated-output equivalence is checked with the same random seed.
All three raw answers and their copy scores are retained under
`standalone_inference/`. These interface calls are additional diagnostics; the
original 384-output quality gates remain unchanged.

# End-to-end native pixel-language training

V61 fine-tunes every parameter of the released 477M MIXAR pixel-language model
on prompt/answer images. The output is a newly generated binary image stream,
with no source-span selector, glyph vocabulary, Unicode IDs, OCR or external
model call. This is supervised adaptation of an externally pretrained model;
its original pretraining is not single-4090 work. The completed reading
qualification must be consulted before treating it as a useful assistant.

The frozen [experiment protocol](../references/native_pixel_instruction_v61_protocol.md)
defines 1,000 optimizer updates, eight accumulated image examples per update,
answer-only causal pixel likelihood, EMA and strict raw-image evaluation. The
training sources are pinned. Full native backward uses about 9.91 GiB allocated
GPU memory in the completed preflight. Interrupted/resumed model, EMA,
optimizer and all recorded RNG states agree exactly. Future answer pixels do
not change earlier full-model predictions. These checks establish real training
and causality, not language quality.

## Data and environment

The existing CMRC subset has 3,211 train and 1,321 development questions with
article overlap removed. Its provenance and CC-BY-SA-4.0 attribution are in the
[reading guide](visual-reading-language.md). Each complete prompt and first
answer are rendered as Noto CJK 26-pixel forms inside 32-pixel binary cells.
This differs from the author's native renderer and is disclosed as a domain
shift. Odd cell counts get a blank padding cell; two blank native blocks teach
termination. Only the answer and termination pixels contribute label loss.

```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONPATH=. \
  python scripts/prepare_native_pixel_qa_v61.py
```

Losslessly packed training image bits and their hashes are stored in
`artifacts/native_pixel_instruction_v61_20261105/data/`. The source JSONLs
remain in `data/visual_qa/cmrc2018_short/`. Training labels never enter the
inference function. The inference output is scored directly as pixels; it is
not snapped to a font or converted through an output character inventory.

The existing `artifacts/external/mixar_env_py312/` environment reuses the shared
Python 3.12 / Torch 2.3 CUDA installation. It adds Transformers 4.57.3 without
duplicating Torch, CUDA or release weights. The Apache-2.0 foundation is
`TYTTYTTYT/llama-pixar-477m-p32-stage2`, pinned revision
`c5d5bcfbad2f290296fd83810558f396c6a6333a`; its SHA is checked before training.
Canonical Noto rendering does not invoke pixar-render or its environment
installer. No new GI/Cairo setup is needed by this route.

## Training and recovery

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONPATH=. \
  artifacts/external/mixar_env_py312/bin/python \
  scripts/train_native_pixel_instruction_v61.py \
  --out artifacts/native_pixel_instruction_v61_20261105/training
```

Only physical GPU 0 is authorized for this run. The process requires that
explicit selection. It stores `training_metrics.jsonl`, a resumable
`checkpoint_latest.pt`, and on completion a weight-only EMA `model_final.pt`
plus `training_report.json`. Resume with the identical command and
`--resume path/to/checkpoint_latest.pt`. A 20-GiB allocation or four-hour
training limit saves a checkpoint and stops the job. Training time excludes
model loading, image preparation and the author's original pretraining.

Use `--smoke` only for the registered four-update backward/resume preflight;
these weights are rejected by normal inference. The actual run uses the full
production schedule. Completed final outputs are protected from replacement.
Weights, optimizer states and image caches remain ignored by Git.

## Native image inference and deployment

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONPATH=. \
  artifacts/external/mixar_env_py312/bin/python \
  scripts/infer_native_pixel_language.py \
  --checkpoint artifacts/native_pixel_instruction_v61_20261105/training/model_final.pt \
  --prompt-image prompt.png --out answer.png --export ilm_native_bundle
```

The prompt PNG is a grid of aligned 32-pixel cells, read left to right then top
to bottom, with no gutters or unused trailing cells. Optional `--prompt-text`
performs host-side rendering; the actor still receives only image tensors.
The trained layout is `资料：{passage}问题：{question}回答：`. Standalone general
questions, raw-page segmentation and new handwriting are not trained tasks.

Inference greedily writes native 32-by-64 blocks and feeds the actual pixels
back into its causal state. The first exactly all-white block terminates the
answer. Ten blocks is the fixed safety cap, independent of references. Only
completely blank trailing layout cells are removed; malformed nonblank glyphs
remain. `answer.png`, the raw `images` tensor, exact prompt PNG and a JSON
stopping/cap receipt are saved. An empty answer has zero raw cells and a white
PNG display placeholder.

The optional deployment directory contains learned weights, config, author
model code, Apache license and an attribution/hash manifest. It can be loaded
through the same `--checkpoint` argument without the original release weights.
An actual exported-model native PNG replay is required before the quality
audit. The audit then compares final EMA and unchanged base on the fixed
questions, blank controls and paired questions:

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONPATH=. \
  artifacts/external/mixar_env_py312/bin/python \
  scripts/eval_native_pixel_instruction_v61.py
```

The audit requires `standalone/verification.json` binding the final checkpoint
hash to exact tensor, PNG and event replay. Reports, raw answers and immutable
audit identities go under `artifacts/native_pixel_instruction_v61_20261105/development/`.
Development is explicitly reused. Exact raster match is strict; foreground
pixel fidelity is descriptive and cannot substitute for correct answers.

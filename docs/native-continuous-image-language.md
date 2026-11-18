# Native continuous image-language comparison

This research model reads binary 32-by-32 writing images and predicts answer
images through a frozen neural decoder. It has 23.55 million total parameters,
16.13 million trainable in the eight-component candidate. A one-component
control shares the same native causal reader and local writing architecture.
No pretrained language body, tokenizer or glyph inventory is loaded. The
previously trained 7.42M image codec and corpus preparation still count as costs.

The [algorithm and probability law](../references/continuous_autoregressive_language_direction_20261118.md)
separate continuous code density from exact image likelihood. Actual generated
pixels are re-encoded as history; an unrendered code is never substituted for
that feedback. Stopping is learned and receives no expected output length.
Three numerical tests and a six-case real-GPU update/reset preflight pass.
The independent audit checks all 2,208 trial images. This establishes execution,
not trained language quality. The full comparison must still be evaluated.

The [fixed protocol](../references/continuous_autoregressive_full_comparison_protocol_20261118.md)
uses 13,785 complete records, 2,571,477 next-image targets per arm and five fixed
snapshots. It schedules 792 autonomous streams, including erased-input controls
and complete long-book continuations. Whole-system matched-quality efficiency,
historical-form explanations and broad language understanding are unproved.

With the local authenticated source pixels and codec already installed, the
registered fit runs once:

```bash
env CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
  CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  PYTHONPATH=. python scripts/train_continuous_image_language.py --stage run
```

The private run directory is `artifacts/continuous_autoregressive_full_v1`.
It contains the fixed source order, each arm's full ledger, five model snapshots,
latest optimizer checkpoint and completion report. Existing execution directories
are refused. An incomplete run stays preserved; the command does not resume it
or create an automatic retry.

After successful completion and process exit, verify the full source/update
schedule before generating any language outputs:

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=. \
  python scripts/evaluate_continuous_image_language.py --stage audit

env CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
  CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  PYTHONPATH=. python scripts/evaluate_continuous_image_language.py --stage evaluate
```

Generation preserves the raw binary arrays, PNGs, natural-stop/cap receipts and
checkpoint/input hashes for every stream. It does not run OCR or repair output.
PNG layout only places the emitted pixels in rows. Complete image review and
frozen readings precede reference scoring. A passed fixed-cell development test
still requires proportional-layout retention, independent execution, untouched
source confirmation and historical transfer before the full goal is achieved.

A complete snapshot can also run outside the training directory on a new aligned
prompt image. It contains the neural codec weights, so this path reads no corpus
or separate decoder archive:

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=. python scripts/infer_continuous_image_language.py \
  --checkpoint artifacts/continuous_autoregressive_full_v1/k8/model_0013785.pt \
  --image prompt.png --output artifacts/my_visual_answer --device cpu
```

Use `--input-glyphs N` to exclude white grid padding. Typed input is supported
with `--text '問題……' --font /path/to/input-font.otf` in place of `--image`;
the font renders input only. The model writes all output pixels itself. Output
contains `answer.png`, exact binary arrays and a receipt with stopping, cap and
checkpoint identity. Existing output directories are refused. These commands
do not establish that the still-unqualified research model answers correctly.

## Reading the held-out outputs

After the evaluator finishes all 792 streams, prepare the anonymous reading packet.

```bash
PYTHONPATH=. python scripts/review_continuous_image_language.py prepare
```

The packet is under `artifacts/continuous_autoregressive_full_v1/review/boards/`.
Each board contains complete output images at their original pixel scale. For
book continuation, the visible book prefix appears above the answer. Work from
this directory alone. Copy `readings_template.json` to a submission file and
record the text, uncertainty and, for each visible book prefix, whether the
complete continuation is coherent. Use an empty transcription for a blank.
Record recognizable fragments of unclear writing and flag the reading as
uncertain. Identify the reviewer and disclose prior familiarity with the cases.

```bash
PYTHONPATH=. python scripts/review_continuous_image_language.py freeze --readings submission.json
PYTHONPATH=. python scripts/review_continuous_image_language.py score
```

Preparation checks every PNG against its raw emitted pixels, including padding.
Freezing checks the board hashes and copies all readings into a file that cannot
be replaced by the command. Scoring subsequently opens the reference answers.
The report distinguishes missing intermediate book or erasure trials from
observed zero scores. The development result applies to aligned, fixed-size
glyph images.

# Natural first-image writer experiment

This is an ongoing repair experiment. It expands real first-image training
after stronger training-set fitting and broad synthetic transfer both fail to
improve useful writing. It has not established a robust language model or a
matched-quality efficiency advantage. The deployed research checkpoint remains
unchanged while this experiment runs.

The prepared dataset contains 30,030 complete prompt images from the pinned
CMRC training partition: 27,099 training images from 9,033 questions and 2,132
paragraphs, and 2,931 held-context images from 977 questions and 242 paragraphs.
Three fixed prompt styles explain the repeated questions. Official development
titles/paragraphs are excluded. Sixty training images have unsupported first
targets and do not enter writer fitting; held unsupported references remain
failures in the denominator. The [protocol](../references/natural_first_image_protocol.md)
fixes selection, losses, thresholds, resources and claim boundaries.

All eight fixed input previews have been inspected. Eight inherited prompts
also reproduce their previous physical input pixels exactly. Source literal
markup is preserved; lossless line wrapping can split numbers across lines.
The data is a language-training source, not a cleaned book edition.

## Run the stages

The prepared images are under `data/visual_qa/natural_first_images_v1/`.
The current run is `artifacts/natural_first_image_writer_v1/`. CPU preparation
uses the existing native renderer environment documented in
[complete-writing-data.md](complete-writing-data.md); it does not install a
second model or environment.

The cache and training commands use physical GPU0:

```bash
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONPATH=.
python scripts/cache_natural_first_states.py --stage cache
```

Run preparation first only for a new run, using
`CUDA_VISIBLE_DEVICES='' python scripts/cache_natural_first_states.py --stage prepare`
after recording the actual fixed-input review. Completed directories are
immutable. Cache shards contain final states from the unchanged full reader;
this is legitimate because only G changes in the subsequent fit. They cannot
be treated as fixed final states if the contextual body is later updated.

After the cache completes, the sequential commands are:

```bash
CUDA_VISIBLE_DEVICES='' python scripts/train_natural_first_writer.py --stage prepare
python scripts/train_natural_first_writer.py --stage fit
python scripts/train_natural_first_writer.py --stage qualify
```

For an already running cache, `python scripts/run_natural_first_writer_pipeline.py`
waits for its terminal success and then runs those three stages. It launches
no second heavy job during caching. It can resume only time/pressure-cap cache
interruptions with matching frozen input/model/code hashes, up to four cache
attempts. Other failures stop the pipeline for inspection. Use one pipeline per
run. The first attempt has a 10,800s wall cap; subsequent fitting has its own
fixed limits. Cache work is part of the experimental cost, not free preparation.

Fit updates G alone for 36,000 steps. E, all D, stop and body adapters stay
fixed. Qualification retains every predicted binary image and counts premature
stops as failed first emissions. It reports paragraph/question summaries and
separates held primary forms absent from natural first-target training. Those
forms may still occur in copy or inherited training; they are not claimed as
unseen characters. Inspect the ten fixed parent/candidate qualification boards
before interpreting the result.

## Use a completed candidate

Once the registered fit finishes, the separate inference entry point accepts
its actual 36,000-update checkpoint:

```bash
python scripts/infer_natural_first_language.py \
  --checkpoint artifacts/natural_first_image_writer_v1/fit/checkpoint_latest.pt \
  --image prompt.png --output artifacts/natural_first_inference_01 \
  --maximum-glyphs 64
```

It reads the prompt image and emits `output.png`, raw tensors and stop/feedback
metadata. Generation feeds the actual emitted pixels through E. No runtime
lexical classifier, tokenizer, font or answer bank supplies the output. The
interface also delegates preserved 18,000-update parents to their original
loader. A real incomplete 17,000-update checkpoint has been checked and
rejected without CUDA initialization. A full replay of the new checkpoint is
pending its completed training; compilation and schema rejection do not prove
that it generates useful language.

Passing the first-image test would qualify a physical readout improvement only.
Complete answers still require actual image-history continuation, learned
stopping, a new reserved context cohort and unrepaired output inspection before
reference scoring. See the [readout/clock analysis](../references/glyph_readout_and_clock_math.md)
for that distinction. Historical writing, unknown forms, etymological
explanations and general language competence remain part of the full objective.

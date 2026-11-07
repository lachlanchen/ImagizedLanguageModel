# A learned copying component for image-language modelling

This experiment learns to select an answer from the pixels of a writing
image. Its output is an original source crop with a two-pixel white frame.
It cannot write an answer absent from that image. The full ILM objective
still requires independent free image-language generation; this component
does not establish that capability.

The earlier image writer fails even when the source decoder can answer
the same request. The [V77 control](../references/reader_content_v77_result.md)
gives 19/32 exact source-helper answers on seen questions and 8/12 new-copy
answers; the paired image writer gets zero on both sets. The earlier
[V58 image-block extractor](../references/visual_answer_blocks_v58_result.md)
also fails paired-question qualification. The new experiment tests a readout
from the calibrated contextual source, rather than claiming copying itself
as a new language-learning algorithm.

Copying and generation have established precedent in
[See, Liu and Manning (ACL 2017)](https://aclanthology.org/P17-1099/).
The foundation is the existing attributed
[Qwen3-VL-8B-Instruct model](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct),
reused through this repository's authenticated archive loader. Its full
externally pretrained image/context body remains part of the runtime.
The new 2,378,370-parameter span head is an adaptation count, not the full
model size. External pretraining is outside the local training measurement.

## Image boundary and probability law

Prepared input is a binary 32-pixel-high strip. Consecutive columns containing
ink form anonymous units; these can be radicals or parts of letters and do
not identify Unicode characters. Folding and native image-budget resizing
use the existing calibrated path. Original strip pixels remain available
to the copy motor. This interface does not segment arbitrary scanned pages.

The frozen encoder supplies contextual visual features at each unit and the
final numeric task-frame states. The learned head scores every visible
contiguous span of 1-128 units. Its normalized probability sums the masses of
duplicate complete images. Target supervision consists of source-rendered
answer images. No gold start/end offset or reference whitelist enters neural
conditioning. Pango and original source annotations prepare the images
offline; they are not used to choose an answer at inference.

The model has no language vocabulary module or text output head. New-input
encoding and autonomous span selection use no OCR, tokenizer, source text,
reference answer or teacher service. These boundaries do not make a
source-copy distribution a density for arbitrary new writing images.

## Run the complete experiment

Use the existing native Python 3.12 runtime and shared source archive. From
the repository root:

```bash
env LD_PRELOAD='/usr/lib/x86_64-linux-gnu/libglib-2.0.so:/usr/lib/x86_64-linux-gnu/libgobject-2.0.so:/usr/lib/x86_64-linux-gnu/libgio-2.0.so' \
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
  CUBLAS_WORKSPACE_CONFIG=:4096:8 OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 \
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=. \
  artifacts/external/mixar_env_py312/bin/python \
  scripts/run_contextual_pixel_spans.py --stage pipeline \
  --run artifacts/contextual_pixel_spans_v1
```

The prospective [protocol](../references/contextual_pixel_spans_protocol.md)
fixes 320 training images, 224 development inputs, 3,200 optimizer updates
and 256 final readouts including 32 training-fit questions. The encoder must
first reproduce four saved V77 contextual memories exactly. The parent runs
preparation, encoding, training, evaluation and scoring sequentially, with
one GPU0 child at a time. It never uses GPU1. The raw implementation is
[the stage runner](../scripts/run_contextual_pixel_spans.py) and
[the span distribution and pixel motor](../ilm/visual_lm/contextual_pixel_spans.py).

The selected run directory must not exist. Each stage preserves completed
and incomplete output rather than silently overwriting it. During a live run,
inspect the existing process and `status.json`; an observation timeout is not
permission to launch another model. Bound implementation files must remain
unchanged while a run is active. A checkpoint contains head weights,
optimizer state, RNG states and the exact schedule cursor; automatic resume
of an incomplete pipeline is not implemented.

## Artifacts and assessment

- `prepared/inputs.json` and input PNGs contain the image inventory.
  `prepared/references.json` and reference PNGs are separate supervision.
- `features/report.json` authenticates source replay and every BF16 feature
  file. It reports setup, encoding cost and the complete retained source size.
- `training/updates.jsonl`, `schedule.json`, `checkpoint.pt` and `report.json`
  retain all 3,200 updates, 25,600 image draws and 80 complete passes.
- `assessment/` retains every raw answer PNG, chosen source boundary and
  output hash. Evaluation does not open references.
- `result.json` scores all normal, question-erased, passage-erased and fit
  outputs. Exact image agreement removes only all-white horizontal exterior;
  every vertical row stays unchanged. Unordered column F1 is a separate
  diagnostic and cannot establish the correct answer order.

The fixed component requirements are at least 30/32 new copies, 48/64 natural
answers, 20/32 both-correct paragraph pairs and a .25 exact-accuracy gain over
question erasure. All criteria must pass. Even a pass qualifies extraction
from this prepared format only. It does not establish a new general language
model, historical-script semantics, raw-book reading, matched-quality
efficiency or an independently trained small foundation.

CMRC 2018 retains Cui et al. (2019), CC-BY-SA-4.0 attribution. The source
Qwen model retains Apache-2.0 attribution. Copy strings are synthetic.

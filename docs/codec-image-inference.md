# Prompt images and generated writing

`scripts/infer_codec_language_v68.py` accepts an observed image and saves the
codec-language actor's physical output stream. The file boundary passes nine
CPU checks, including exact round trips for all 384 canonical actor pages and
separate checks of all 384 explicit raw-strip folds. Final-model inference is not yet tested:
the same V68 instruction stage is running, followed by its queued autonomous
audit. See the [interface receipt](../references/evidence/codec_image_interface_preflight.json).

## Prepare an image on CPU

From the repository root, select a new output directory:

```bash
CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 PYTHONPATH=. \
artifacts/external/mixar_env_py312/bin/python scripts/infer_codec_language_v68.py \
  --image prompt.png --layout page --out artifacts/my_prompt_input --prepare-only
```

Page mode applies EXIF orientation, composites transparency on white, converts
to grayscale and pads the right/bottom edges to a multiple of 32 pixels. It
does not resize, threshold, segment characters or infer a reading order. A
single image is required; crop a large scan explicitly before submission.
Admission is limited to 512 native visual cells. The complete actor has been
exercised on a 512-by-1024 page, but that execution check does not establish
language capability on arbitrary scans or every admitted aspect ratio.

For a binary 32-pixel-high writing strip, use `--layout strip`. This reproduces
the training layout, wrapping at white columns without losing pixels. The
default page width is 1024 pixels. Such wraps need not coincide with word
boundaries. Do not select strip mode for an ordinary scanned page.

For a matched blanking intervention, pass the already composed control page
with `--layout page`. Independently re-folding an erased raw strip can change
wrap boundaries and reflow the control. The [audit correction](../references/codec_language_v68_layout_control_correction.md)
retains each normal prompt's source ranges for its blank control. Normal
strip inputs remain unchanged; a white strip alone cannot recover its original
wrap boundaries.

Preparation writes `prompt.pt`, the exact owning FP32 actor-input tensor;
`prompt.png`, its physical grayscale image; and an input receipt recording
source and pixel hashes, orientation, color conversion and padding. It makes
zero model calls. Two typed Chinese/English examples have also been rendered
and prepared through the real commands under
`artifacts/codec_image_interface_demo_20261105/`; these are inputs, not answers.

## Type a prompt using the existing host adapter

The optional text-to-image adapter uses the same Pango/font configuration as
the instruction images. Unicode is confined to this host-side rendering step;
the actor subsequently receives pixels alone. It needs no renderer for input
images that already exist.

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=. \
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libglib-2.0.so.0:/usr/lib/x86_64-linux-gnu/libgobject-2.0.so.0:/usr/lib/x86_64-linux-gnu/libgio-2.0.so.0 \
artifacts/external/mixar_env_py312/bin/python scripts/render_complete_writing.py \
  --text '短文：言可以表示说话，也可以表示说出的话。问题：言可以表示什么？回答：' \
  --out typed_prompt.png
```

Then pass `typed_prompt.png` to the image command with `--layout strip`.
Rendering validates that every source glyph exists in the selected fonts and
retains the complete line. It cannot synthesize an unencoded historical form;
an observed image can supply such a form, with model understanding still to be
demonstrated.

## Generate after training and the fixed audit finish

```bash
CUDA_VISIBLE_DEVICES=0 CUDA_DEVICE_ORDER=PCI_BUS_ID \
CUBLAS_WORKSPACE_CONFIG=:4096:8 OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 PYTHONPATH=. \
artifacts/external/mixar_env_py312/bin/python scripts/infer_codec_language_v68.py \
  --image typed_prompt.png --layout strip --out artifacts/my_image_answer
```

The command requires the final 10,000-update adaptation and its matching
independent 384-generation audit receipt. It refuses unfinished training
before initializing CUDA, and refuses an occupied GPU 0. A failed language
audit remains a failed result; the command may inspect that actor without
relabeling it as useful language. All previous output directories are retained.

Only the owning prompt tensor enters `actor.generate`. Generation uses its
predicted continuous writing codes and rereads its own generated pixels.
Reference images, input filenames, source text, OCR and teacher responses do
not enter that call. The current loader authenticates inherited fixed weights
from the shared local archive at startup. An isolated standalone weight export
has not yet been qualified.

The output contains:

- `generation/raw_generation.pt`: all original binary blocks and stop events.
- `generation/answer.png`: the untrimmed, 32-pixel-high image stream.
- `generation/blocks/`: each generated 32-by-64 block, without repair.
- `generation/output_receipt.json`: hashes, width, probabilities and termination.
- `report.json`: checkpoint/audit identity, input/output receipts and measured
  setup, generation time and allocated GPU memory.

An immediate stop has a real empty stream and no placeholder PNG. Reaching
the default 128-block cap is recorded as capped, not successful termination.
No automatic transcription or text fallback is performed.

## Verification scope

The current CPU check runs in 11.859 seconds excluding interpreter startup and makes
zero model calls or GPU allocations. Its source identity is commit
`c998db70deb6bc04f467c0c57ff3f5d989766cf2`. It checks the existing two-block
preflight execution image and its actual probabilities through serialization;
it does not rerun that actor or claim the fragmented writing has meaning.
Grayscale/alpha, empty-output and malformed-event fixtures check file formats
only. Earlier attempts remain preserved, including the first reader type
failure and the later check predating the matched-control layout correction.

The next language decision remains the frozen autonomous audit. Interface
correctness, copied pixels and low training loss cannot replace that result.

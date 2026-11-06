# Image-only generator baseline

This is an unqualified candidate using an external pretrained image generator.
Its [CPU preparation checks](../references/evidence/image_edit_baseline_preparation.json)
cover real pinned source metadata, 20 questions rendered as 60 page images,
541 source characters with no missing glyphs, and unfinished-experiment
rejection. The new package imports without the legacy visual package's
Torchvision dependency. It has not yet passed a model-loading, codec,
generation or language test.
It does not replace the unfinished V82 comparison. The
[protocol](../references/image_edit_baseline_protocol.md) preserves the full
ILM objective and distinguishes each preparation/execution stage from learning.

Use the existing image-generation Python environment from LocalLLM. It already
contains Diffusers 0.39.0; no framework or CUDA installation is added. From the
ILM repository root:

```bash
ILM_IMAGE_PYTHON=/home/lachlan/ProjectsLFS/LocalLLM/.local/image-generation/venv/bin/python
env CUDA_VISIBLE_DEVICES= PYTHONNOUSERSITE=1 PYTHONPATH=. "$ILM_IMAGE_PYTHON" scripts/image_edit_language_baseline.py --stage inspect --output artifacts/image_edit_baseline/source_inspection
env CUDA_VISIBLE_DEVICES= PYTHONNOUSERSITE=1 PYTHONPATH=. "$ILM_IMAGE_PYTHON" scripts/image_edit_language_baseline.py --stage prepare-data --output artifacts/image_edit_baseline/pages
```

These two commands inspect real pinned metadata and render the fixed 20
questions, blank controls and separate reference pages. They load no model.
All source characters must be present in the selected font and within the
page bounds. Preparation records file, source and font hashes. Rendered
questions and references are data artifacts; no character labels or target
pages are sent through the generator.

The commands below reject an unfinished V82 experiment before acquiring
weights or initializing a neural model. They also reject a completed
comparison if either existing arm meets absolute training fit: its broader
language evaluation takes precedence. After an eligible failed comparison:

```bash
env CUDA_VISIBLE_DEVICES= PYTHONNOUSERSITE=1 PYTHONPATH=. "$ILM_IMAGE_PYTHON" scripts/image_edit_language_baseline.py --stage acquire --output artifacts/image_edit_baseline/acquisition
env CUDA_VISIBLE_DEVICES=0 PYTHONNOUSERSITE=1 PYTHONPATH=. "$ILM_IMAGE_PYTHON" scripts/image_edit_language_baseline.py --stage condition --output artifacts/image_edit_baseline/condition
env CUDA_VISIBLE_DEVICES=0 PYTHONNOUSERSITE=1 PYTHONPATH=. "$ILM_IMAGE_PYTHON" scripts/image_edit_language_baseline.py --stage codec-cohort --data artifacts/image_edit_baseline/pages --output artifacts/image_edit_baseline/codec
```

Acquisition uses an explicit pinned allowlist of transformer/VAE weights and
configuration. It does not download the checkpoint's text encoder. A complete
verified shared snapshot is reused if already available. Otherwise weights
go into the project's ignored artifact cache; acquisition must leave 100 GiB
free. The acquisition report records the actual snapshot path. If it reuses
a shared snapshot outside the default project cache, pass that path with
`--snapshot` to the codec and generation commands.

The offline condition stage temporarily loads the verified shared Z-Image
Qwen3-4B encoder/tokenizer, compiles one generic instruction and one empty
negative condition, and exits. The same numeric cache is used for every
question. This composition is explicitly different from claiming byte-exact
reproduction of the FLUX release's own text encoder. Its setup cost and all
external pretraining remain attributed.

The codec stage retains raw RGB and decoded PNGs for all prepared pages,
reusing results only for byte-identical duplicate pages. Its image-agreement
scores describe supplied-image reconstruction, not language. Review actual
stroke preservation before considering any local language training.

After those codec observations, run the complete exploratory generation cohort:

```bash
env CUDA_VISIBLE_DEVICES=0 PYTHONNOUSERSITE=1 PYTHONPATH=. "$ILM_IMAGE_PYTHON" scripts/image_edit_language_baseline.py --stage cohort --data artifacts/image_edit_baseline/pages --condition artifacts/image_edit_baseline/condition/condition.pt --codec-report artifacts/image_edit_baseline/codec/report.json --output artifacts/image_edit_baseline/generations
```

The generator loads once, with its tokenizer and text encoder absent, and
produces all 40 normal/blank answers under fixed settings. Only prompt images
vary; no reference answer, OCR, per-question text embedding or teacher call
enters generation. Every output retains raw latents, raw decoded RGB and an
unrepaired PNG. This stage has no optimizer and makes no language-quality
claim. All outputs still require separate readability/correctness assessment,
calibrated reference recognition and eventual genuinely held-out language tests.

Every output directory must be fresh. Partial or completed artifacts are
preserved on failure; they are not silently overwritten or counted as a
completed cohort. One project GPU lock and the shared workstation memory
limits apply to every actual model stage. The running model experiment,
source weights, checkpoints and other projects' services remain intact.

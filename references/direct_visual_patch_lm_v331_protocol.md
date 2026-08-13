# Direct Visual Patch Language Model V33.1: Calibration Amendment

Date frozen: 2026-08-13

Status: frozen before V33.1 implementation and training

## Reason for the Amendment

The preregistered V33 Stage-A run completed 2,000 adapter-only updates and
correctly stopped before semantic training. Its raw checkpoint achieved:

- binary ink F1 `0.8402523531`;
- edge F1 `0.8368313290`;
- `chi_sim` OCR character accuracy `0.3247279077`;
- blank-patch false-ink rate `0`; and
- a byte-identical frozen transformer core.

The run is permanently classified `visual-interface-failure` under the V33
protocol. Its checkpoint SHA-256 is
`cacb0215634a23bd7801fd7544c4b3a59e68274d41e39f2d44a68fb909a39696`.

A post-run evaluator audit found that the same `chi_sim` Tesseract pipeline
scores untouched target strips at only `0.5683845029` character accuracy. The
installed `chi_tra` model raises that reference score to `0.7359567901`, still
far below V33's absolute `0.95` OCR gate. That gate was therefore unreachable
for this held-out classical/traditional Chinese distribution and cannot be
reused as a valid legibility test. This amendment does not reinterpret V33; it
defines a new experiment.

## Fixed Question

Can additional adapter-only optimization align the `32 x 32` direct raster
interface with the frozen PIXAR transformer strongly enough to preserve held-
out Chinese glyph strokes and most of the legibility available to a fixed OCR
evaluator, before any causal language training is attempted?

## Inherited Student Boundary and Architecture

V33.1 inherits the complete student boundary, architecture, data sources,
rendering rules, train/development partitions, and forbidden runtime operations
from `references/direct_visual_patch_lm_v33_protocol.md`.

In particular, the student still receives only raster pixels and a patch mask,
emits direct pixel and stop logits, and contains no strings, token IDs, Unicode
IDs, vocabulary, OCR, glyph table, codebook, VAE, retrieval system, or runtime
teacher. Evaluator OCR remains outside the student and cannot affect model
inputs, gradients, checkpoint selection, or generation.

## Fixed Starting State

- source checkpoint:
  `artifacts/direct_visual_patch_v33_calibration_20260813/checkpoint_latest.pt`;
- required source SHA-256:
  `cacb0215634a23bd7801fd7544c4b3a59e68274d41e39f2d44a68fb909a39696`;
- source stage/update: `visual-calibration` / `2000`;
- model route used for training and the gate: raw weights;
- optimizer and EMA state: restored from the source checkpoint; and
- transformer core: frozen and required to remain byte-identical.

The EMA route is evaluated only as a diagnostic and cannot replace the raw
route for the mandatory gate.

## Fixed Extension

- additional updates: `6,000` (total calibration updates: `8,000`);
- physical batch: `8`;
- gradient accumulation: `8` (effective batch: `64`);
- objective: same-patch pixel BCE plus `0.25` Sobel edge loss plus `0.25` ink
  Dice loss; stop loss is inactive;
- optimizer: restored AdamW state, betas `(0.9, 0.95)`, weight decay `0.05`;
- adapter learning rate: cosine decay from `3e-5` to `3e-6`, without warmup;
- transformer-core learning rate: exactly `0`;
- gradient norm clip: `1.0`;
- precision/device: BF16 on one RTX 4090;
- seed and deterministic data order: `20_263_300`;
- first extension example index: `128,000`, exactly the number consumed by V33;
- checkpoint overwrite interval: every `1,000` updates and on a clean signal;
- evaluation timing: once after all 6,000 additional updates; and
- maximum allocated CUDA memory: below `20 GiB`.

The continued dataset must preserve the V33 record order and variant function.
It must begin after the examples consumed by V33 rather than silently replaying
the first 128,000 rendered examples.

## Fixed Evaluator

Evaluate at least 2,048 development patches with the same deterministic held-
out records and unseen Noto Serif CJK font used by V33. For every strip, run the
same installed Tesseract executable with language `chi_tra`, page segmentation
mode `7`, and nearest-neighbor `2x` scaling on both:

1. the untouched target strip; and
2. the thresholded reconstruction from the raw student.

Record the executable version and SHA-256 of the selected traineddata file.
Use a fixed white-pixel threshold of `0.5`; threshold search is diagnostic only
and cannot affect the gate.

Report:

- binary ink F1 and Sobel edge F1 against target pixels;
- expected-text accuracy of target OCR;
- expected-text accuracy of reconstructed OCR;
- OCR retention = reconstructed accuracy / target accuracy;
- paired OCR agreement between the target-OCR and reconstruction-OCR strings;
- blank-patch false-ink rate;
- raw and EMA diagnostics; and
- deterministic, uncurated target/reconstruction gallery rows.

## Mandatory V33.1 Gate

All conditions must hold on the raw route:

- finite model outputs and checkpoint;
- mean binary ink F1 at least `0.90`;
- mean edge F1 at least `0.90`;
- target OCR character accuracy at least `0.60`;
- reconstruction OCR retention at least `0.90`;
- paired target/reconstruction OCR agreement at least `0.80`;
- blank-patch false-ink rate below `0.01`;
- transformer core byte-identical to the source; and
- all 6,000 extension updates completed within the VRAM limit.

No semantic or instruction-training result may be claimed from this run. A
passing result permits a separately frozen causal-language protocol. A failure
means the direct linear `32 x 32` raster interface needs an architectural or
objective revision before semantic compute.

## Decision Labels

- `raster-interface-qualified`: every V33.1 gate passes;
- `readable-below-gate`: target/reconstruction examples are visually legible
  but one or more quantitative gates fail;
- `raster-interface-failure`: fidelity or OCR retention remains materially
  inadequate; and
- `invalid-run`: source hash, data cursor, frozen-core integrity, finite state,
  update count, evaluator identity, or resource requirements fail.

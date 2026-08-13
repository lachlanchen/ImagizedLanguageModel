# Direct Visual Patch LM V33/V33.1 Result

Date completed: 2026-08-14

Decision: `readable-below-gate`; semantic training not authorized

## Result in One Sentence

A 114.84M-parameter pixel-only student can reconstruct recognizable held-out
traditional Chinese after adapter-only training on one RTX 4090, but its single
linear `768 -> 1024` raster projection still changes too many fine strokes to
qualify as a reliable writing interface.

## Reproducible Route

V33 initialized all 109 transformer-core tensors from the audited PIXAR
checkpoint and resized its direct raster projections from `8 x 16` to `32 x
32`. It trained the 1.57M raster-adapter parameters for 2,000 updates while the
transformer core remained frozen. The preregistered V33 gate failed and stopped
the causal stages as required.

V33.1 then restored the exact V33 checkpoint, optimizer, and EMA state and ran
the separately frozen 6,000-update calibration extension:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH=. \
python scripts/continue_direct_visual_patch_v331.py \
  --device cuda:0 \
  --out artifacts/direct_visual_patch_v331_calibration_20260813 \
  --log-every 50
```

The deterministic training cursor started at example `128000` and ended at
`512000`; the extension did not replay V33's first 128,000 rendered examples.

## Frozen-Gate Measurements

| Measurement | V33 raw | V33.1 raw | V33.1 gate |
|---|---:|---:|---:|
| Ink pixel F1 | 0.8403 | **0.8919** | >= 0.90 |
| Sobel edge F1 | 0.8368 | **0.8807** | >= 0.90 |
| Reconstruction OCR accuracy | 0.3247 (`chi_sim`) | **0.4823** (`chi_tra`) | diagnostic |
| Untouched-target OCR accuracy | 0.5684 (`chi_sim`) | **0.7360** (`chi_tra`) | >= 0.60 |
| OCR retention | not validly measured | **0.6553** | >= 0.90 |
| Target/reconstruction OCR agreement | not measured | **0.6190** | >= 0.80 |
| Blank-patch false ink | 0 | **0** | < 0.01 |

The V33.1 EMA route was slightly worse than raw: ink F1 `0.8888`, edge F1
`0.8778`, reconstruction OCR accuracy `0.4695`, OCR retention `0.6380`, and
paired OCR agreement `0.6096`.

The OCR evaluator was Tesseract `5.3.4`, language `chi_tra`, PSM `7`, with
traineddata SHA-256
`529c5b5797d64b126065cd55f2bb4c7fd7b15790798091b1ff259941a829330b`.
Target and reconstruction strips used the same evaluator and fixed `0.5`
white-pixel threshold.

## Integrity and Resources

- completed updates: `8,000` total (`2,000 + 6,000`);
- effective batch: `64`;
- extension elapsed time: `1,456.06 s`;
- peak allocated CUDA memory: `1,486,761,472` bytes (`1.38 GiB`);
- device: NVIDIA GeForce RTX 4090 D;
- source checkpoint SHA-256:
  `cacb0215634a23bd7801fd7544c4b3a59e68274d41e39f2d44a68fb909a39696`;
- final checkpoint SHA-256:
  `e4c100fe0b253c9ed2d1d4bc9ce464d6c2e2f34ae268c7169eb7ce6847f9aa0b`;
- transformer core before and after:
  `2f420abd2d75950278d2104e18f739bec2657f7241f5f7ee80729febe8d88293`;
  and
- protocol SHA-256:
  `9ed5873293d4249cb120204eb7b13074371fbdd0785b13e94ed309512a0767ab`.

Local evidence is under
`artifacts/direct_visual_patch_v331_calibration_20260813/`, including raw and
EMA reports, uncurated galleries, all update metrics, receipt, summary, and the
recoverable checkpoint. The artifact directory is intentionally ignored by
Git because the checkpoint is approximately 931 MB.

## Interpretation

The extension materially improved every raster metric, and the deterministic
gallery is visually readable. It did not close the stroke-identity gap: small
component changes turn one Chinese character into another even when the line
looks plausible at page scale. More updates on the same linear interface are
therefore not justified by this result.

This is not evidence that the model learned prompt-conditioned language. The
mandatory raster gate failed, so no public-continuation or instruction stage
was run and no semantic claim is made.

## Next Falsifiable Step

Replace the single raw-pixel projection with a compact **continuous**
convolutional writing codec:

1. encode each `32 x 32` raster patch into a continuous latent vector;
2. decode that vector through a spatial convolutional decoder;
3. use a learned continuous bridge from causal transformer states to codec
   latents; and
4. feed decoded raster patches back through the same visual encoder during
   autonomous generation.

The codec must contain no vocabulary, character IDs, Unicode IDs, OCR,
retrieval, codebook, quantization, or runtime teacher. It must first demonstrate
high-fidelity reconstruction of unseen fonts and unseen glyph forms. Only then
is causal language training a rational use of GPU compute.

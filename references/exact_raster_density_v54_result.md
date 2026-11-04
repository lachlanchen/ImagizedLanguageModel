# Exact raster density: measured outcome

Date: 2026-11-05. Protocol frozen before training. Decision: **reject as a
complete visual language mechanism**; retain the exact copy path and density
implementation as working components.

## Artifact and numerical checks

- Run: `artifacts/exact_raster_density_v54_20261104/`.
- Production checkpoint SHA-256:
  `f2c3fab74981541243a963c8cffab6bbbd8c7f9d051512b228347db04a57a9ee`.
- 18,711,841 runtime parameters; 2,434,592 trained density parameters. The
  16,277,249-parameter V42 reader stays frozen.
- All 8,000 updates finished in 207.849 s, peak allocated CUDA memory
  0.40016 GiB, physical GPU 0 only. Reader pretraining is excluded.
- Audit: 52.164 s, peak allocated memory 0.23100 GiB.
- Five unit checks cover strict intra-row and inter-row pixel causality,
  exact initial copying, density-only gradients, and normalized four-bit
  enumeration. Interrupted versus continuous smoke training matches every
  model/EMA tensor and the saved generator.
- Pre-production binarization correction: rendered source images have gray
  antialiasing; observations are thresholded before binary NLL. The protocol
  already specified a binary sensor. No loss or audit threshold changed.

## What works

The inverse-field logit path preserves arbitrary supplied binary images at
initialization. After training, all 256 development motor targets still copy
exactly in greedy mode, including punctuation. Stochastic-copy F1 is 0.99949
and exact-image match is 81.25%. Deranged-image greedy F1 is 0.35769. This is a
strong supplied-image motor result, not autonomous language or learned script
understanding.

The model learns an exact normalized raster distribution. Ordered target NLL
is 15.21873 nats/image (0.0214414 bits/pixel), versus 846.09412 nats for the
independent-pixel distribution using the same predicted plan. This comparison
supports dependencies within images. It does not isolate language learning.

## What fails

| Measurement | Ordered | Shuffled |
| --- | ---: | ---: |
| Target NLL, 512 windows | 15.21873 | 15.25424 |
| First-sample nearest identity, 256 windows | 4/256 | 3/256 |
| First-sample pixel F1 | 0.35033 | 0.35124 |
| First-sample exact image match | 2/256 | 1/256 |
| Four-step feedback mean identity | 4.2969% | 3.1250% |

The ordered NLL advantage is 0.03551 nat, source-record bootstrap 95% CI
[-0.10142, 0.16932], 169 source records, 2,000 draws. The paired identity
advantage is 0.3906 percentage points, CI [0, 1.2097], 134 records. The earlier
history effect is unresolved. Zero condition is an out-of-distribution
intervention, not a fitted unconditional baseline.

Greedy continuation matches 14/256 images but reaches only 0.21454 F1 and
0.32981 ink ratio; visible outputs favor sparse/common forms. Sample isolated
ink is 0.01404, above the registered 0.01 limit. Four of eight gates pass;
likelihood context effect, sampled identity, sample quality and surface gates
fail. No sealed records or independent training seed have been evaluated.

The evaluator is constructed **after generation** and contains 9,932 distinct
binary rasters from all training forms plus audit targets. Coverage is 100%.
This bank includes punctuation and differs from previous 1,024-Han banks.
Nearest-identity results cannot be directly compared across those evaluators;
raw images and exact-image match remain available.

## Independent deployment verification

`scripts/infer_exact_raster_density_v54.py` accepts native aligned image grids,
produces rasters, and exports a self-contained checkpoint. The 75,225,847-byte
export is at `standalone/ilm_v54_standalone.pt` beneath the run directory,
SHA-256 `9b7f887a5e0257fe95a17fbb45b3855815bbc140ba5c12bbe3916e6b04b26370`.
The exported native-image continuation exactly reproduces the production
checkpoint's eight-cell greedy result. Native-image motor copying matches the
binarized prompt exactly. Receipts and PNGs are in `standalone/`.

No tokenizer, Unicode ID, OCR, image-bank selection or external model is used
to select output pixels. The checkpoint remains an **unqualified research
model**, not a usable general ILM.

## Scientific implication

Another writer surface change is insufficient. The aligned copy path solves
image retention, but teacher-forced NLL can exploit already-observed strokes
inside the target instead of selecting the future form from language history.
This is a plausible mechanism, not a proven unique cause. The next read-only
diagnostic should attribute NLL and context effects by image row and compare
natural conditioning with supplied-image conditioning. Then test whether the
reader state is actually used before expensive data expansion or retraining.

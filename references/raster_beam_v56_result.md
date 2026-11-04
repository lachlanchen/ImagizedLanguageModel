# Raw-pixel beam decoding: measured result

Date: 2026-11-05. No training or weight change. Decision: **reject as the
decoder for a qualified language model**, because the registered feedback
repetition criterion fails. Retain the measured next-image improvement and
exact-form generation as components; keep all prior decisions unchanged.

The unchanged 18.71M V55 joint checkpoint is decoded by width-eight raw-pixel
beam search or the original local greedy rule. Both use highest FP32 matmul
precision under the documented pre-production numerical amendment. There is
no output glyph inventory, OCR, external model, discriminator or image repair.

| 2,048 fixed development prefixes | Ordered beam | Earlier-shuffled beam | Ordered greedy |
| --- | ---: | ---: | ---: |
| Exact observed next image | 238/2048 (11.621%) | 202/2048 (9.863%) | 169/2048 (8.252%) |
| Exact known canonical form | 100% | 100% | 99.561% |
| Pixel F1 to observed target | 0.31944 | 0.28629 | 0.23071 |
| Distinct output rasters | 127 | 108 | 116 |
| Mean generated-image NLL | 3.10763 | 3.07762 | 3.50761 |

Ordered beam improves exact target-image match over greedy by **3.369
percentage points**, paired source-record 95% interval **[2.351, 4.412]**.
Its earlier-history advantage is **1.758 points [0.966, 2.552]**. These are
positive conditional generation effects under the registered test, without
bank-selected output pixels. They are stronger than matching a nearest identity
to an approximate glyph. The bank is constructed after all generation,
contains 9,934 distinct rasters and covers every target.

## Visible feedback remains inadequate

The first 128 prefixes continue for eight generated-image feedback steps.
Ordered beam exactly matches 68/1024 observed cells (6.641%), versus 50/1024
(4.883%) for greedy. The mean difference is 1.758 points, interval [0, 3.770].
Earlier-shuffled beam matches 51/1024. Mean exact-target improvement alone is
not enough: adjacent equal-image repetition is **25.446%** ordered beam,
32.366% shuffled beam and 17.746% greedy, compared with **0.112%** in the real
target streams. Mean distinct forms per eight-cell stream are 4.711, 4.117,
3.883 and 7.789, respectively. No repetition filter is applied.

All 256 answer-supplied motor images copy exactly. This reveals the target and
does not qualify language. Four implementation tests cover width-one equality,
enumeration, an example where greedy misses the global mode, parent history
across rows and score agreement. The checkpoint preflight also matches width
one to the original greedy decoder. Production maximum accumulated-score/NLL
error is 0.00002104 nat, below the unchanged 0.001 tolerance.

Five of six bounded gates pass: implementation, boundary/resources, exact-image
gain over greedy, earlier-history gain, and surface/motor quality. Feedback
fails from repetition. The model therefore remains unqualified as a usable
general ILM, despite clear next-form and conditional-generation progress.

## Reproduction and limits

Run directory: `artifacts/raster_beam_v56_20261105/`. Audit seed 20265620;
2,000 source-record bootstrap draws, seed 20265719. First-image comparisons
cover 188 source records and feedback covers 89. The reused development
partition is not a sealed independent generalization test.

The complete audit takes **269.33 seconds**, peak allocated CUDA memory
**0.15368 GiB**, on physical GPU 0 only, excluding earlier model training.
The checkpoint SHA-256 remains
`ae4598c609648ba92a4eaa8680fa0842966a1f11b0757ed4a14b83038c048394`.
All raw outputs, per-example scores, provenance and fixed galleries are saved.
The original high-matmul smoke failure and corrected FP32 smoke remain retained.

The result isolates meaningful image choice from clear rendering. Mode search
improves next-image correctness and reaches clean known forms, but amplifies
some repetitive trajectories. A new training task should require question-
dependent, grounded answers and retain the free-writing path. The prepared
natural CMRC questions and same-paragraph counterfactual pairs provide that
next test; they are data-ready, not yet a trained language capability.

# Matched conditional bitmap density: completed primary result

5 November 2026. Both registered 1,000-update arms and their locked audit are
complete. The decision is `reject_density_intervention`; neither actor passes
the separate bounded free-reading gate. General image-language generation
remains unqualified.

| Measurement | K=1 | K=8 |
| --- | ---: | ---: |
| Fully trained parameters | 477,145,984 | 489,998,208 |
| Recorded training through final update | 1,887.493 s | 1,972.752 s |
| Peak allocated training memory | 9.904 GiB | 10.174 GiB |
| All-development answer/termination-pixel NLL | 0.154201 | 0.124809 |
| Exact primary answers | 0/256 | 0/256 |
| Strict-raster multiset F1 | 0.234% | 0.815% |
| Aligned foreground F1 | 12.205% | 15.827% |
| Both questions correct in a paragraph pair | 0/393 | 0/393 |
| Normal outputs reaching cap | 0/256 | 0/256 |

The strict-raster gain is 0.580 percentage points, article-grouped 95%
bootstrap interval [0.089, 1.094]. Foreground gain is 3.623 points
[1.815, 5.554]. These gains do not establish useful answers. The K=8
normal-minus-blank-question raster difference is 0.295 points
[-0.006, 0.623], and its foreground difference is -0.044 points
[-0.629, 0.509]. Question dependence remains unresolved. Seven of eight
intervention gates pass; each actor passes six of ten reading gates.
The gates, source articles, final EMA choice and decoder are unchanged.

The K=8 gate has mean entropy 1.812 nats; experts 0, 1, 2, 4 and 7 are selected
223, 4, 106, 605 and 3 times across primary continuation steps, including
termination. These counts describe use, not semantic roles. The first two
fixed outputs show more coherent digits than the single-head fragments, but
do not answer the source questions. No output was repaired or selected for
appearance.

The complete training sample orders match byte for byte. Real four-update
preflights match uninterrupted and resumed weights, EMA, optimizer and RNG
exactly, pass full-backbone gradient checks and show zero future-pixel causal
effect. Strict-FP32 cached/prefill checks pass; larger TF32 discrepancies
remain in the preflight receipt. Native PNG inference and fresh offline
standalone bundles reproduce tensors, PNG bytes and decoder events exactly.
The audit takes 527.439 s excluding loading, at 1.904 GiB allocated memory.
All phases use physical GPU 0.

The run uses external MIXAR pretraining and a shifted Noto-26 interface.
Only one training seed is tested on explicitly reused development data.
K=8 has additional parameters and output computation, so the comparison
does not establish efficiency at equal capacity or equal language quality.
Training timing excludes loading, final serialization, renderer preparation
and external foundation pretraining.

Checkpoint SHA-256:

- K=1: `d2112540cdba0f7405492995636f638df327ac378be3970c0c27969b7085a386`
- K=8: `9e22ebccbd769cc44e7de026ca89e433ae934ddeaf808af01020552f24d1b1b9`

The immutable primary receipt is
`artifacts/correlated_native_density_v62_20261105/development/report.json`.
The [protocol](correlated_native_density_v62_protocol.md) and
[training/inference guide](../docs/correlated-image-language.md) specify the
actual code and probability law. Figure 7 and Supplementary Note 21 report
the completed comparison, preserving V61 and every earlier rejection.

A separately preregistered answer-only vision reader passes its calibration:
122/128 references exact, all 16 blanks empty, all 16 random-ink controls empty
and valid JSON throughout. At this checkpoint, 1,015 unique anonymous actor
images are being transcribed on an owned GPU-0 evaluator, outside the actor.
Its auxiliary scores are not yet available and cannot alter the primary
decisions. The owned evaluator will be stopped after evidence capture.

The result narrows the next question: more expressive local image likelihood
helps bitmap fit at this budget, but a training signal that preserves the
connection between prompt meaning and generated writing is still needed.
Neither the mixture property nor recent physical sampling results establish
a singularity or a substitute for that language test.

# Wider velocity prediction improves glyphs but fails repeatability

The fixed known-image comparison uses the first 32-by-64 answer block of
the first CMRC record in the authenticated cache order. Both heads train
from scratch for 4,000 updates on the same single image, using common
noise/time draws and the fixed V34 renderer.

| Velocity head | Parameters | Exact / 64 | Mean foreground F1 | Worst F1 |
| --- | ---: | ---: | ---: | ---: |
| Width 768 | 10,534,656 | 0 | 90.651% | 0% |
| Width 1,536 | 36,403,200 | 63 | 99.954% | 97.053% |

The supplied-code render reproduces the original image exactly.
The wider arm still fails the predeclared requirement that every seed reach
99% F1. Its mean and exact-fraction requirements pass. All 128 output images
receive independent scoring and exact CPU replay of the fixed decoder.
This is a known-image denoising experiment; no prompt reader is trained or
evaluated.

The subsequent registered post-hoc diagnosis replays the original sampler
exactly, then tests FP32 arithmetic and/or 64 midpoint steps on the same
64 noise draws. BF16 at 64 steps is the first passing calibration setting.
Its one fresh-noise confirmation gives 116/128 exact renders, mean F1
99.788% and worst F1 87.607%. The unchanged worst-sample requirement fails.
All calibration and confirmation images receive independent physical
scoring and fixed-decoder replay. No second setting is selected from those
confirmation outcomes.

These results remove one structural noise-retention restriction but do not
establish reliable glyph generation or a usable image-language model.
The [clean-endpoint control](clean_endpoint_v74_protocol.md) tests a change
of prediction parameterization and noise contraction before further
language training.

Raw controls are retained under artifacts/full_rank_flow_v73_20261106 and
artifacts/flow_precision_v73_20261106. Their original failed outcomes are
not replaced by later experiments.

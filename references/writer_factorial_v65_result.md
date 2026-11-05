# Direct writing isolates the tested flow-route failure

5 November 2026. The [registered factorial experiment](writer_factorial_v65_protocol.md)
completes all four 2,000-update arms, with identical initial weights and batch
indices. It tests 64 training-only images, deliberately allowing memorization.
It does not test language, unseen-image generalization or autonomous answers.

| Condition and objective | Foreground F1 | Exact / 64 | Shuffled F1 | Conditioning gain | Fitting criterion |
| --- | ---: | ---: | ---: | ---: | --- |
| Inherited features, flow | 34.567% | 0 | 26.716% | 7.851 points | fail |
| Inherited features, direct | 99.995% | 63 | 27.466% | 72.529 points | pass |
| Lossless pixels, flow | 33.648% | 0 | 26.581% | 7.068 points | fail |
| Lossless pixels, direct | 100.000% | 64 | 27.464% | 72.536 points | pass |

The fixed fitting criterion is foreground F1 >=95% and conditioning gain
>=50 points. Shuffling stays within source language. Every case is retained,
including duplicate target rasters. The lossless pixel condition supplies the
answer image and is an oracle control; it cannot predict an unknown answer.

Direct reconstruction can recover these images from the inherited features.
The failed flow arms also fail when given all source pixels. Thus the tested
flow learning/sampling route is an immediate obstacle; this experiment does
not support blaming the inherited encoder for irreversible loss of these
particular fitting images. It does not identify the precise cause within flow
optimization, time/noise conditioning or numerical sampling, or rule out flow
methods generally. Broader-corpus learning and complete prompt-to-answer
generation still require independent tests.

The unchanged 19,374,096-parameter writer is common to all arms. Direct writing
sets its image/time inputs to zero and regresses the physical output. Flow
uses Gaussian interpolation and velocity regression. All four early exact
optimizer/EMA/global/random-state replays pass, with one counted extra update
each. The run completes in 422.003 s including source extraction and evaluation.
Arm end-to-end times are recorded separately in the raw reports; sequential
timings are not evidence of a capability-matched language-training advantage.
Inherited extraction peaks at 15.251 GiB; fitting after releasing unused
inherited modules peaks at 0.660–0.665 GiB. These fitting peaks are not full
language-model memory measurements.

The original auxiliary V34 codec comparison used inverted polarity. Its
invalid 81.364% score and all raw outputs are preserved. The documented
[correction](writer_factorial_v65_codec_correction.md) runs the same fixed
checkpoint and data through the native white-positive interface on CPU,
without learning. It reconstructs 64/64 blocks exactly. This confirms the
existing visual codec on these images and establishes no language capability.

The [independent verifier](../scripts/verify_writer_factorial_v65.py) checks
all 512 main generated images, 128 auxiliary/corrected images, original 64
training-image sources, every batch index for all 8,000 fitting updates,
common initialization and conditions, final EMA hashes, committed source
bytes, quality decisions and exact replay receipts. Its
[receipt](evidence/writer_factorial_v65_result.json) binds the complete evidence.
First-eight aligned/shuffled galleries were inspected without quality selection.

Source commit: `9ca61e3`. Inherited/direct final EMA SHA-256:
`f9c5d846a280a9340237c0a52b1fb7f893cd533e4a1ce8bc3c00f1262fc1a3a1`.
Artifacts remain in `artifacts/writer_factorial_v65_20261105/`. The owned
process/tmux session have exited and GPU 0 is released. No GPU-1 service
was used. The next candidate is direct-writing bootstrap over the complete
Chinese/English corpus, followed by a fixed reused-development audit. The
general independent image-language goal remains active.

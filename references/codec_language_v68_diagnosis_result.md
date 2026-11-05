# V68: conditional writing fails even with correct history

The [registered diagnostic](codec_language_v68_diagnosis_protocol.md) applies
the unchanged final actor to 32 fixed Chinese/English examples. It performs
no optimization and opens no sealed data. It separates supplied-code image
reconstruction, next-block prediction with exact reference history, and
autonomous writing. The
[offline measurement receipt](evidence/codec_language_v68_diagnosis_result.json)
reloads all 32 source measurements and verifies their retained physical scores.

Each row below contains eight examples. F1 values are case averages in percent.
The exact-history route receives true previous answer blocks and the reference
length, continues even past a predicted stop, and therefore is an assisted
diagnostic. Its comparison with autonomous full-image F1 also includes this
length assistance; it is not a pure estimate of the effect of history alone.

| Split / source | Supplied-code F1 | First predicted block F1 | Exact-history F1 | Autonomous F1 |
| --- | ---: | ---: | ---: | ---: |
| Training / Chinese | 99.9468 | 38.6338 | 35.5543 | 30.0053 |
| Training / English | 100.0000 | 34.4160 | 21.8050 | 8.1448 |
| Development / Chinese | 99.9874 | 30.8801 | 29.1315 | 25.6324 |
| Development / English | 100.0000 | 36.4205 | 24.4436 | 10.5421 |

No first predicted block is exact, and none of the next blocks predicted with
exact history is exact. Normalized writing-code MSE is 1.0635 and 1.2776 in
the two training groups, and 1.2329 and 1.2260 in development. None satisfies
the recorded decoded-margin sufficient condition. Minor supplied-code errors
remain in Chinese; their scale is much smaller than the prediction error.

These measurements localize an unresolved problem to the conditional
prediction path. Failed initial blocks occur before any generated-history
drift. Prediction remains poor on training examples even when previous
answer images are supplied. Generated history may worsen later writing, but
cannot alone explain the failure. This does not determine whether feature
information, readout capacity, the regression objective, optimization or their
interaction is responsible. A different pixel decoder alone is not supported
as the next repair by this experiment.

## Execution and limits

All 32 initial stop probabilities and initial output blocks agree between the
two paths starting from identical prompts. The 16 development autonomous
streams are reused byte-for-byte from the original
[384-generation audit](codec_language_v68_result.md); only the 16 training
autonomous streams are newly generated. Exact-history predictions use the
actual cached inference path and append a reference block only after its
prediction has been recorded. No current or future reference image conditions
its own contextual query.

The experiment takes 208.112 seconds including native actor initialization and
peaks at 15.816 GiB allocated GPU memory. Its GPU process has exited and
released GPU 0. CPU verification checks source pixels, retained tensors,
reused streams, thresholding, margins and aggregation; it does not independently
replay the native neural forward pass. No optimizer update, OCR, font lookup,
source-model call or standalone weight export is performed. This small,
fixed diagnostic is not a population-level language benchmark.

Run: `artifacts/codec_language_v68_diagnosis_20261105/`.
Source commit: `c2260ff7ae035ec58b18470d9d882a4f8015225f`.
Generation-report SHA-256:
`424885b9dfcbc235c95344780cf9cd2dbb1b72edc239fabe63b220513290bef9`.
Measurement-receipt SHA-256:
`69446b336ee39016758ad9229bbb5d48788dfcf931a44303fa69e83275c2b461`.

The next learning experiment must improve image-conditioned writing-code
prediction and verify autonomous answers again. A conditional density or a
more expressive readout is a hypothesis to test, not a demonstrated solution.
The original language gates remain failed and the full ILM goal remains open.

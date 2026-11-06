# Decoder-sensitive loss passes derivative checks

The state-only writing metric is implemented and passes its declared checks
on the actual frozen 7,423,361-parameter codec, on CPU and GPU0. These are
derivative and execution measurements. No optimizer update or new generated
answer enters this test, and no language improvement is established.

Every block from fixed cohort positions 00–03 is used: ten blocks, three
noise times and a shared seeded noise draw. The direction is the completed
V81 model's retained predicted-code error. Its squared derivative through the
normalized decoder is 11,929–29,869 times its code MSE across these times.
The declared strength 0.001 therefore makes the new component substantial;
that does not show it points toward a better learned generator.

| Actual check | CPU | RTX 4090 D |
| --- | ---: | ---: |
| Maximum gradient relative L2 error against reverse-mode adjoint | 3.58e-7 | <1e-6 |
| Maximum JVP relative L2 error against FP64 central differences | 4.11e-6 | 4.12e-6 |
| First loss + backward invocation, ten blocks | 0.337 s | 0.310 s |
| Subsequent invocations at the other two times | 0.095–0.111 s | 0.017–0.018 s |
| Peak allocated VRAM | none | 0.163 GiB |

Strength zero reproduces the original MSE and gradient exactly. The metric
point receives no gradient, all loss/gradient values are finite, and every
codec parameter remains frozen and bit-identical. Direct normalized decoder
logits equal the physical codec logits exactly in FP32. The separately copied
FP64 decoder is a numerical reference, not a deployed inference component.
No full-actor speed or memory conclusion follows from these small-decoder
costs. Both full reports and source digests are in the
[receipt](evidence/decoder_pullback_v82_preflight.json); retained derivatives
remain under the report-bound artifact paths.

The [next matched protocol](decoder_metric_v82_protocol.md) fixes all tensors,
sampling, data, seeds and stopping, and changes only this training loss. It
requires an actual complete-actor preflight before optimization and autonomous
image evaluation afterward. Neither a passing derivative check nor a lower
training objective can substitute for correctly generated answers.

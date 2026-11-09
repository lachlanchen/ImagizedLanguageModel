# Captured pixel-cache discrepancy is consistent with FP32 roundoff

The controlled reconstruction reproduces the failed numerical gate: maximum
cached/full-prefix logit difference is 0.0001249313, exceeding the original
0.0001 limit. All 512 teacher-prefix argmax decisions and both complete
autonomous 256-piece trajectories agree. Target-suffix causality error is
exactly zero. These are two physical blocks under actual image-derived
writing conditions, not complete autonomous language answers.

The independent FP64 CPU implementation gives a maximum cached/full-prefix
difference of 1.82965e-13. Maximum differences from that reference are
7.28034e-5 for the FP32 full path and 6.45551e-5 for the cached path. Their
opposite rounding errors can exceed 1e-4 when the two FP32 paths are directly
subtracted. The FP32 pair has relative normwise error 7.20265e-7 and maximum
total variation 1.72975e-5. This evidence supports numerical accumulation as
the explanation for this captured comparison; it does not diagnose the
separate failure to generate correct language.

The [registered capture](native_pixel_numerics_protocol.md) reconstructs the
same recipe, seed, pretrained body, reset projection and five original prompt
PNGs. The preceding failed run saved no decoder snapshot, so its exact
writing-query tensors cannot independently be compared with this capture.
The new snapshot, actual FP32 traces, FP64 reference, initial actor state,
five native prefixes and image encodings remain under
`artifacts/native_pixel_numerics_v1/` with
[recorded hashes](evidence/native_pixel_numerics_result/retained_tensors.json).
The [complete analysis](evidence/native_pixel_numerics_result/analysis.json)
and [capture receipt](evidence/native_pixel_numerics_result/execution/report.json)
retain all measurements and source identities.

Capture takes 89.248 seconds including model load and five native prefills,
peaking at 17.762 GiB allocated VRAM on GPU0. Independent FP64 analysis takes
0.575 seconds on CPU. There are zero optimizer updates and zero complete
language answers. GPU0 returns to 16 MiB and GPU1 remains unchanged. The old
absolute gate remains failed; this result neither changes its tolerance nor
qualifies training or language understanding.

A bounded repair can improve numerical accumulation in normalization while
retaining FP32 model parameters. It must be tested on these exact captured
states against the unchanged gate and independent reference before another
full native learning preflight.

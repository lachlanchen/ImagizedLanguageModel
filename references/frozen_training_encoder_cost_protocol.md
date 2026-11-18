# Computing repeated training images once

The native comparison encoded each complete source record through the same
frozen, per-image convolutional encoder. Its previous source audit counted
7,255,223 image positions and 9,581 distinct physical masks. Group and layer
normalization operate within an image. With fixed parameters, an identical
bitmap therefore has the same mathematical code at every source position.
Floating-point kernels can still depend on batch shape.

This CPU systems comparison changes only reuse of a computed code for equal
binary training images. Use the final K=8 checkpoint from the audited fit, two
CPU threads, FP32, strict deterministic algorithms and the three complete
training records already prescribed for the native preflight. The ordinary
model and cached subclass both run the original training objective. Each
record receives one warm-up pass followed by three alternating paired
forward/backward timings. Report the first cold cached pass separately.
Measure code, code NLL, stop BCE and every participating parameter gradient.
Record the maximum absolute code and loss differences and the maximum relative
L2 gradient difference across parameter tensors with a 1e-8 denominator floor.
Numerical eligibility requires code and loss differences at most 2e-4 and
maximum relative gradient difference at most 1e-3 for every case.

Count all parameters and every encoded image. The cache is temporary, holds at
most 16,384 physical patterns and is excluded from deployment weights. Compare
cold and warm elapsed times separately for each whole training record. Three
fixed records and a warmed cache estimate a CPU execution effect only. Even a
large reduction in encoder calls cannot give a training-wide or matched-quality
speedup without timing the remaining computations. No optimizer step, new
answer, development input, reference score or reserved text enters this test.

Run once with a 300-second outer timeout, 48 GiB of host memory available at
entry, a 24-GiB reserve and a 4-GiB process RSS ceiling. Preserve discrepancies.
If numerical eligibility fails, keep the ordinary training path. A pass
qualifies the measured precision on these CPU cases only. GPU arithmetic,
whole-corpus wall time and held-out language remain separate measurements.

# V56 numerical correction before production

Date: 2026-11-05. The CPU implementation tests pass, including width-one
equality and exact enumeration. The first checkpoint smoke audit is retained
at `artifacts/raster_beam_v56_20261105_smoke/`; its score-consistency gate
fails (maximum error 0.00962 nat). It is not production evidence.

A numerical preflight uses two fixed random binary rasters, zero hidden states
and their exact image fields with the unchanged V55 weights. Under CUDA's
`high` float32 matmul setting, evaluating the same causal row as a partial
prefix versus a full teacher-forced image differs by up to 0.002923 in state;
beam accumulated log probability differs from parallel image NLL by 0.003501
nat. Under `highest` (full FP32 matmuls), those errors fall to 0.000001907 and
0.00001192 nat. Raw values are retained in
`artifacts/raster_beam_v56_numerical_preflight/precision_report.json`.

For production V56, supersede only the original protocol's `high` matmul
instruction with `highest`, for **both** beam and greedy controls and all
likelihood checks. Model tensors remain FP32 and weights remain unchanged.
Keep the original 0.001-nat score tolerance, width, windows, seed, context
controls and all quality/decision criteria. There is no decoding-quality or
hyperparameter sweep. This correction is made before production scoring.

The new audit and inference receipts must record the precision explicitly and
hash this amendment alongside the unchanged original protocol. Comparisons
with earlier V55 CUDA-high values are descriptive; the primary matched
comparison is entirely within the corrected V56 evaluation.
